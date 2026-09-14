"""Binary BCE baseline and masked listwise+absent objective (task §9).

Listwise, present group with >=1 positive (masked logsumexp, numerically stable):

    loss = -log( sum(exp(logit_j), j in positives)
               / (exp(absent_logit) + sum(exp(logit_k), k in valid candidates)) )
         = logsumexp([absent_logit] + valid_logits) - logsumexp(positive_logits)

Absent group:

    loss = -log( exp(absent_logit) / (exp(absent_logit) + sum(exp(logit_k))) )
         = logsumexp([absent_logit] + valid_logits) - absent_logit

Both are `logsumexp(denominator_terms) - logsumexp(numerator_terms)`, computed with
`torch.logsumexp` over a masked, -inf-padded tensor so invalid/padded candidates
never contribute (`exp(-inf) = 0`).

Offset loss (Smooth L1) applies ONLY to candidates labelled positive, weighted
separately from the classification loss so it cannot dominate matching.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NEG_INF = torch.finfo(torch.float32).min / 2


def _masked_logsumexp(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """logsumexp over `dim` counting only `mask`-True entries; -inf if none valid."""
    masked = torch.where(mask, logits, torch.full_like(logits, NEG_INF))
    return torch.logsumexp(masked, dim=dim)


def binary_bce_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    """Balanced BCE on individual positive/negative pairs (Arms A, B)."""
    targets = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    logits = torch.cat([pos_logits, neg_logits])
    return F.binary_cross_entropy_with_logits(logits, targets)


def listwise_absent_loss(candidate_logits: torch.Tensor, valid_mask: torch.Tensor,
                         positive_mask: torch.Tensor, absent_logit: torch.Tensor,
                         is_present: torch.Tensor) -> torch.Tensor:
    """Masked listwise cross-entropy with an explicit absent option.

    Shapes: `candidate_logits`, `valid_mask`, `positive_mask` are (B, K);
    `absent_logit`, `is_present` are (B,). `positive_mask` may mark MULTIPLE
    candidates per row (task §6: "treat them as multiple valid positives"),
    handled by the marginal-likelihood numerator `logsumexp(positive_logits)`.

    A present row with valid_mask all False, or with no positive under mask
    (pool_missing), contributes only through the denominator: its "positive
    numerator" is `-inf`, matching an unreachable target rather than silently
    treating a missing positive as a match. Such rows are excluded from the mean
    by the caller via `usable_present_mask` (defined in `training.py`); this
    function itself only implements the stable arithmetic.
    """
    denom_terms = torch.cat([absent_logit[:, None], candidate_logits], dim=1)
    denom_mask = torch.cat([torch.ones_like(is_present, dtype=torch.bool)[:, None],
                            valid_mask], dim=1)
    denom = _masked_logsumexp(denom_terms, denom_mask, dim=1)

    pos_num = _masked_logsumexp(candidate_logits, positive_mask & valid_mask, dim=1)
    absent_num = absent_logit

    numerator = torch.where(is_present, pos_num, absent_num)
    loss_per_row = denom - numerator
    return loss_per_row


def offset_loss(pred_dx: torch.Tensor, pred_dy: torch.Tensor,
               true_dx: torch.Tensor, true_dy: torch.Tensor,
               weight: float = 0.1) -> torch.Tensor:
    """Smooth L1 on the residual offset, positives only, weighted down."""
    loss = F.smooth_l1_loss(pred_dx, true_dx) + F.smooth_l1_loss(pred_dy, true_dy)
    return weight * loss
