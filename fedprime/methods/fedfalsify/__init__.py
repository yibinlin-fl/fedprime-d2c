"""Offline audit utilities for the FedFalsify research candidate.

The package is intentionally independent from every training runner.  Its first
job is to test whether receiver-side foreign-environment evidence exists before
FedFalsify is allowed to become a trainable method.
"""

from fedprime.methods.fedfalsify.evidence import (
    PairedAdvantage,
    classwise_accuracy_tensor,
    compute_classwise_paired_advantages,
    compute_paired_advantage,
    planned_stratified_audit_counts,
)
from fedprime.methods.fedfalsify.transfer import (
    conservative_margin_transfer_loss,
    direct_peer_kd_loss,
    fixed_margin_loss,
    gradient_cosine_from_losses,
    normalize_logits,
)

__all__ = [
    "PairedAdvantage",
    "classwise_accuracy_tensor",
    "compute_classwise_paired_advantages",
    "compute_paired_advantage",
    "planned_stratified_audit_counts",
    "conservative_margin_transfer_loss",
    "direct_peer_kd_loss",
    "fixed_margin_loss",
    "gradient_cosine_from_losses",
    "normalize_logits",
]
