"""Batched forward pass of `PairwiseVerifier` over a padded (B, K) group batch."""
from __future__ import annotations

import torch

from .model import PairwiseVerifier


def score_batch(model: PairwiseVerifier, batch: dict, device: str) -> dict:
    """Pair logits (B,K), absent logit (B,), and optional offsets (B,K,2)+conf(B,K)."""
    query = batch['query'].to(device)          # (B, 32, 32)
    crops = batch['crops'].to(device)           # (B, K, 32, 32)
    b, k = crops.shape[0], crops.shape[1]

    q_flat = query[:, None, :, :].expand(-1, k, -1, -1).reshape(b * k, 32, 32)
    c_flat = crops.reshape(b * k, 32, 32)

    zq_flat = zc_flat = None
    if model.hardnet_fusion:
        zq_flat = model.hardnet_descriptors(q_flat)
        zc_flat = model.hardnet_descriptors(c_flat)

    feat_flat = model.pair_features(q_flat, c_flat, zq_flat, zc_flat)
    pair_logits = model.pair_out(feat_flat).squeeze(1).reshape(b, k)

    query_embed = model.query_self_embed(query)
    absent_logit = model.absent_head(query_embed, pair_logits, batch['valid'].to(device))

    out = {'pair_logits': pair_logits, 'absent_logit': absent_logit}
    if model.offset_enabled:
        raw = model.offset_head.net(feat_flat)
        dx = (model.offset_head.max_offset * torch.tanh(raw[:, 0])).reshape(b, k)
        dy = (model.offset_head.max_offset * torch.tanh(raw[:, 1])).reshape(b, k)
        conf = torch.sigmoid(raw[:, 2]).reshape(b, k)
        out.update({'dx': dx, 'dy': dy, 'confidence': conf})
    return out
