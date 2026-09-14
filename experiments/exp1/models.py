"""Backbones and the pairwise verification head (plan §3, §9).

Weights come from the local files under `cnn/`, loaded into the installed Kornia
architectures with `strict=True`. Nothing is silently swapped: if a named file is
absent the caller gets an error naming the exact expected path and digest.

These are patch-descriptor backbones used in Siamese/triplet training. They are
NOT pretrained sky-matching systems, and the fine-tuned results must be labelled
"backbone fine-tuned with the Exp1 loss", not a reproduction of the original
HardNet/HyNet/SOSNet recipes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .env import ROOT, sha256_file

EPS = 1e-8

# A patch whose standard deviation (on the 0-1 scale) falls below this carries no
# usable structure. HardNet and SOSNet normalise each input internally, so such a
# patch is divided by ~0 and its descriptor is numerically arbitrary: CPU and MPS
# can disagree, and the direction means nothing. Plan §6 requires flat patches to
# have "a finite, explicitly low-information outcome", so they are FLAGGED here and
# scored as low information rather than trusted or silently treated as matches.
LOW_INFO_STD = 1e-3


def low_information(x: torch.Tensor) -> torch.Tensor:
    """Per-sample flag for patches with no usable structure."""
    flat = x.reshape(x.shape[0], -1)
    return flat.std(dim=1) < LOW_INFO_STD


@dataclass
class ModelSpec:
    name: str
    kornia_class: str
    weights: str            # path relative to the repository root
    sha256: str
    source_url: str
    license_note: str
    checkpoint_key: str | None = None    # sub-key holding the state dict


SPECS = {
    'hardnet': ModelSpec(
        name='hardnet',
        kornia_class='kornia.feature.HardNet',
        weights='cnn/hardnet-hf/checkpoint_liberty_with_aug.pth',
        sha256='1e9a41b19f1dc93c986e91df9aaf5696d1a777ac1d67498492856a65d6f49c16',
        source_url='https://github.com/DagnyT/hardnet '
                   '(mirror: https://huggingface.co/kornia/hardnet)',
        license_note='HardNet author repository: MIT licence',
        checkpoint_key='state_dict'),
    'hynet': ModelSpec(
        name='hynet',
        kornia_class='kornia.feature.HyNet',
        weights='cnn/hynet-hf/HyNet_LIB.pth',
        sha256='d92559e90b8b228e7ed121935f64d2629a015c2bf9d7d7ee14b43798c5a2af13',
        source_url='https://github.com/yuruntian/HyNet '
                   '(mirror: https://huggingface.co/kornia/hynet)',
        license_note='HyNet author repository: MIT licence'),
    'sosnet': ModelSpec(
        name='sosnet',
        kornia_class='kornia.feature.SOSNet',
        weights='cnn/sosnet-local/sosnet_32x32_liberty.pth',
        sha256='baa575c55a5d1b7b140206ece4af56b735395409f2bb3c924069faf6a7ededa0',
        source_url='https://github.com/yuruntian/SOSNet/raw/master/'
                   'sosnet-weights/sosnet_32x32_liberty.pth',
        license_note='SOSNet author repository: BSD-3-Clause'),
}
BACKBONES = tuple(SPECS)
DESCRIPTOR_DIM = 128


def _kornia_ctor(spec: ModelSpec):
    import kornia.feature as KF
    return getattr(KF, spec.kornia_class.rsplit('.', 1)[1])


def weight_provenance(name: str) -> dict:
    spec = SPECS[name]
    path = ROOT / spec.weights
    entry = {
        'model': spec.name,
        'kornia_class': spec.kornia_class,
        'weights_path': spec.weights,
        'expected_sha256': spec.sha256,
        'source_url': spec.source_url,
        'license': spec.license_note,
        'present': path.exists(),
    }
    if path.exists():
        entry['actual_sha256'] = sha256_file(path)
        entry['sha256_matches'] = entry['actual_sha256'] == spec.sha256
        entry['bytes'] = path.stat().st_size
    return entry


def load_backbone(name: str, pretrained: bool = True, device: str = 'cpu'):
    """Instantiate a backbone. `pretrained=False` gives a scratch initialisation.

    Kornia constructors return EVAL mode (verified), so callers that optimise must
    call `.train()`. `build_for_training` does that for them.
    """
    if name not in SPECS:
        raise KeyError(f'unknown backbone {name!r}; expected one of {BACKBONES}')
    spec = SPECS[name]
    ctor = _kornia_ctor(spec)
    model = ctor(pretrained=False)
    info = {'model': name, 'pretrained': bool(pretrained),
            'ctor_training_mode': bool(model.training)}
    if pretrained:
        path = ROOT / spec.weights
        if not path.exists():
            raise FileNotFoundError(
                f'{name}: expected local weights at {path} (sha256 {spec.sha256}, '
                f'source {spec.source_url}). Run `download-models` or place the '
                f'file; checkpoint families are never substituted silently.')
        actual = sha256_file(path)
        if actual != spec.sha256:
            raise ValueError(f'{name}: {path} sha256 {actual} != expected {spec.sha256}')
        blob = torch.load(path, map_location='cpu', weights_only=False)
        state = blob[spec.checkpoint_key] if spec.checkpoint_key else blob
        model.load_state_dict(state, strict=True)
        info['weights_sha256'] = actual
        info['loaded_strict'] = True
    return model.to(device), info


class Descriptor(nn.Module):
    """Shared-weight encoder emitting L2-normalised 128-D descriptors.

    A finite epsilon is always applied before distance computation, so a zero-norm
    input yields a finite low-information descriptor rather than a NaN or an
    accidental perfect match.
    """

    def __init__(self, backbone: nn.Module, name: str):
        super().__init__()
        self.backbone = backbone
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return z / torch.clamp(z.norm(dim=1, keepdim=True), min=EPS)


def build_for_training(name: str, pretrained: bool = True, device: str = 'cpu',
                       freeze_bn: bool = True):
    """Training-ready encoder.

    `freeze_bn` is the initial policy for pretrained fine-tuning (plan §9): batch
    normalisation running statistics are held fixed while convolution gradients
    still flow. Scratch training uses normal training statistics. HyNet's final
    batch-normalisation layer is covered by the same policy; its FRN layers carry
    learnable parameters and are not affected.
    """
    backbone, info = load_backbone(name, pretrained=pretrained, device=device)
    model = Descriptor(backbone, name).to(device)
    model.train()
    applied = []
    if freeze_bn and pretrained:
        for mod_name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = False
                applied.append(mod_name)
    info['bn_frozen'] = applied
    info['bn_policy'] = ('running statistics frozen, convolution gradients active'
                         if applied else 'normal training statistics')
    info['n_parameters'] = int(sum(p.numel() for p in model.parameters()))
    info['n_trainable'] = int(sum(p.numel() for p in model.parameters()
                                  if p.requires_grad))
    return model, info


def restore_train_mode(model: nn.Module, freeze_bn: bool) -> None:
    """Re-enter training mode after validation, preserving the BN policy."""
    model.train()
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()


class PairHead(nn.Module):
    """Pairwise verifier over two descriptors (plan §9).

    Input is `[abs(zq - zc), zq * zc]`, so 256-D for 128-D descriptors, then
    256 -> 128 -> 32 -> 1 with ReLU and dropout 0.1 between hidden layers. No scene
    identity, source coordinate, filename or figure membership is available to it.
    """

    def __init__(self, dim: int = DESCRIPTOR_DIM, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(2 * dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1))

    @staticmethod
    def features(zq: torch.Tensor, zc: torch.Tensor) -> torch.Tensor:
        return torch.cat([(zq - zc).abs(), zq * zc], dim=1)

    def forward(self, zq: torch.Tensor, zc: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(zq, zc)).squeeze(1)


def pairwise_distances(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    """Euclidean distance matrix from dot products, with an explicit epsilon.

    Written as a small mathematically equivalent formulation rather than `cdist`
    so it is supported identically on CPU and MPS (plan §4).
    """
    sq = (a * a).sum(1, keepdim=True) + (b * b).sum(1)[None, :] - 2.0 * (a @ b.t())
    return torch.sqrt(torch.clamp(sq, min=eps))


def descriptor_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    """Row-wise Euclidean distance between paired descriptors."""
    return torch.sqrt(torch.clamp(((a - b) ** 2).sum(1), min=eps))
