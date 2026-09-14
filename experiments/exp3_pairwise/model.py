"""Compact pairwise verifier (task §8).

Pixel-pair stream: `[query, candidate, query - candidate, abs(query - candidate)]`
as a 4-channel 32x32 input to a small GroupNorm CNN (batch-size stable, so a
32-group overfit test and single-group inference behave identically).

Frozen-HardNet stream (arms D, E, F only): `[z_query, z_candidate,
abs(z_query - z_candidate), z_query * z_candidate]`, 512-D for 128-D descriptors.
The HardNet backbone is loaded once, frozen (`requires_grad_(False)`, eval mode,
never `.train()`-ed) and reused across the whole experiment; nothing in this
module ever updates its weights or its BatchNorm running statistics.

Heads:
  pair_logit           higher is better (matches Experiment 1's score convention)
  absent_logit         one per QUERY GROUP, from frozen query features + aggregate
                        candidate evidence (see `AbsentHead`)
  dx, dy               bounded residual correction, `MAX_OFFSET * tanh(raw)`
  offset_confidence     in [0, 1], sigmoid of a small head

Photometric normalisation is a single global affine per crop (subtract the crop's
own mean pixel, divide by a floor-clamped std), which cannot erase all evidence:
a uniform crop keeps a slightly attenuated response rather than becoming exactly
zero, since the divisor is floor-clamped rather than the numerator being masked.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import MAX_OFFSET
from experiments.exp1.models import DESCRIPTOR_DIM

NORM_STD_FLOOR = 8.0 / 255.0   # prevents a near-uniform crop from being amplified


def photometric_normalize(x: torch.Tensor) -> torch.Tensor:
    """Global affine brightness normalisation preserving saturation and structure.

    `x` is (B, 1, 32, 32) or (B, 32, 32) in [0, 1]. Per-crop mean-subtract, divide
    by std with a floor -- NOT per-pixel min-max, which would make every crop's
    darkest and brightest pixel identical and erase real contrast differences.
    """
    flat = x.reshape(x.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp(min=NORM_STD_FLOOR)
    shape = [x.shape[0]] + [1] * (x.dim() - 1)
    return (x - mean.view(*shape)) / std.view(*shape)


def pixel_pair_input(query: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """4-channel [query, candidate, query-candidate, abs(query-candidate)]."""
    q = photometric_normalize(query)
    c = photometric_normalize(candidate)
    diff = q - c
    return torch.stack([q, c, diff, diff.abs()], dim=1)


class PixelPairCNN(nn.Module):
    """Small GroupNorm CNN over the 4-channel pair input -> fixed-size embedding."""

    def __init__(self, out_dim: int = 64, groups: int = 8):
        super().__init__()
        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(min(groups, cout), cout),
                nn.ReLU(inplace=True))
        self.net = nn.Sequential(
            block(4, 16, 1),
            block(16, 32, 2),   # 32 -> 16
            block(32, 32, 2),   # 16 -> 8
            block(32, 64, 2),   # 8 -> 4
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = out_dim
        self.proj = nn.Linear(64, out_dim)

    def forward(self, query: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        x = pixel_pair_input(query, candidate)
        z = self.net(x).flatten(1)
        return self.proj(z)


class HardNetFusion(nn.Module):
    """[zq, zc, abs(zq-zc), zq*zc] -> a small embedding, from FROZEN descriptors."""

    def __init__(self, dim: int = DESCRIPTOR_DIM, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * dim, 64), nn.ReLU(inplace=True), nn.Linear(64, out_dim))

    @staticmethod
    def features(zq: torch.Tensor, zc: torch.Tensor) -> torch.Tensor:
        return torch.cat([zq, zc, (zq - zc).abs(), zq * zc], dim=1)

    def forward(self, zq: torch.Tensor, zc: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(zq, zc))


class AbsentHead(nn.Module):
    """Query/set head producing one absent logit per group.

    Inputs, explicitly: the query's own pixel-pair-CNN self-embedding (the query
    paired with itself, i.e. `PixelPairCNN(query, query)`, which is well-defined
    since the CNN takes two 32x32 crops) concatenated with the aggregate candidate
    evidence `[max(pair_logit), mean(pair_logit), std(pair_logit), n_valid/20]`.
    No scene identity, filename, coordinate or per-scene patch count is available.
    """

    def __init__(self, query_embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(query_embed_dim + 4, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1))

    def forward(self, query_embed: torch.Tensor, pair_logits: torch.Tensor,
               valid_mask: torch.Tensor) -> torch.Tensor:
        """`pair_logits`/`valid_mask`: (B, K) padded; returns (B,) absent logit."""
        neg_inf = torch.finfo(pair_logits.dtype).min / 2
        masked = torch.where(valid_mask, pair_logits, torch.full_like(pair_logits, neg_inf))
        n_valid = valid_mask.float().sum(dim=1, keepdim=True)
        has_any = (n_valid > 0).float()
        safe = torch.where(valid_mask, pair_logits, torch.zeros_like(pair_logits))
        agg_max = torch.where(has_any.bool().squeeze(1), masked.max(dim=1).values,
                              torch.zeros(pair_logits.shape[0], device=pair_logits.device))
        agg_mean = safe.sum(dim=1) / n_valid.squeeze(1).clamp(min=1)
        agg_std = torch.sqrt(
            (((safe - agg_mean[:, None]) ** 2) * valid_mask).sum(dim=1)
            / n_valid.squeeze(1).clamp(min=1)).clamp(min=0)
        agg = torch.stack([agg_max, agg_mean, agg_std, n_valid.squeeze(1) / 20.0], dim=1)
        return self.net(torch.cat([query_embed, agg], dim=1)).squeeze(1)


class OffsetHead(nn.Module):
    """Bounded residual centre correction, only meaningful for a likely positive."""

    def __init__(self, in_dim: int, max_offset: float = MAX_OFFSET):
        super().__init__()
        self.max_offset = max_offset
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 3))

    def forward(self, features: torch.Tensor):
        raw = self.net(features)
        offset = self.max_offset * torch.tanh(raw[:, :2])
        confidence = torch.sigmoid(raw[:, 2])
        return offset[:, 0], offset[:, 1], confidence


class PairwiseVerifier(nn.Module):
    """Full model for one arm; `hardnet_fusion`/`offset` toggle the optional streams.

    The HardNet backbone passed in must already be frozen: this module asserts
    `not any(p.requires_grad for p in hardnet.parameters())` at construction so a
    misconfigured caller fails loudly rather than silently fine-tuning it.
    """

    def __init__(self, hardnet_fusion: bool = False, offset: bool = False,
                hardnet_backbone=None):
        super().__init__()
        self.hardnet_fusion = hardnet_fusion
        self.offset_enabled = offset
        self.pixel_cnn = PixelPairCNN(out_dim=64)
        feat_dim = 64
        self.hardnet = None
        if hardnet_fusion:
            if hardnet_backbone is None:
                raise ValueError('hardnet_fusion=True requires a frozen hardnet_backbone')
            if any(p.requires_grad for p in hardnet_backbone.parameters()):
                raise ValueError('hardnet_backbone must be frozen (requires_grad=False)')
            self.hardnet = hardnet_backbone
            self.fusion = HardNetFusion(dim=DESCRIPTOR_DIM, out_dim=32)
            feat_dim += 32
        self.pair_out = nn.Linear(feat_dim, 1)
        self.absent_head = AbsentHead(query_embed_dim=64)
        if offset:
            self.offset_head = OffsetHead(in_dim=feat_dim)

    def hardnet_descriptors(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 32x32 crops with the frozen backbone. No gradient reaches it."""
        with torch.no_grad():
            return self.hardnet(x[:, None] if x.dim() == 3 else x)

    def pair_features(self, query: torch.Tensor, candidate: torch.Tensor,
                      zq: torch.Tensor | None = None,
                      zc: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.pixel_cnn(query, candidate)
        if self.hardnet_fusion:
            if zq is None:
                zq = self.hardnet_descriptors(query)
            if zc is None:
                zc = self.hardnet_descriptors(candidate)
            feat = torch.cat([feat, self.fusion(zq, zc)], dim=1)
        return feat

    def pair_logit(self, query: torch.Tensor, candidate: torch.Tensor,
                   zq: torch.Tensor | None = None,
                   zc: torch.Tensor | None = None) -> torch.Tensor:
        return self.pair_out(self.pair_features(query, candidate, zq, zc)).squeeze(1)

    def query_self_embed(self, query: torch.Tensor) -> torch.Tensor:
        return self.pixel_cnn(query, query)

    def offset(self, query: torch.Tensor, candidate: torch.Tensor,
              zq: torch.Tensor | None = None, zc: torch.Tensor | None = None):
        if not self.offset_enabled:
            raise RuntimeError('offset head disabled for this arm')
        feat = self.pair_features(query, candidate, zq, zc)
        return self.offset_head(feat)

    def trainable_parameters(self):
        """Parameters that receive gradients; excludes the frozen HardNet backbone."""
        params = list(self.pixel_cnn.parameters()) + list(self.pair_out.parameters())
        params += list(self.absent_head.parameters())
        if self.hardnet_fusion:
            params += list(self.fusion.parameters())
        if self.offset_enabled:
            params += list(self.offset_head.parameters())
        return params
