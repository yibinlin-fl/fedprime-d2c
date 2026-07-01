from __future__ import annotations

import torch
import torch.nn.functional as F


def _flatten_normalize(feature: torch.Tensor) -> torch.Tensor:
    if feature.dim() > 2:
        feature = feature.view(feature.size(0), -1)
    return F.normalize(feature, dim=1)


class SARALoss(torch.nn.Module):
    """Skew-Aware Robust Alignment for AugMix local training.

    SARA keeps the stable structure of RAHFL DCL but makes two assumptions
    explicit under label-skew Non-IID data: not every local class should receive
    the same contrastive pressure, and not every strong AugMix view is equally
    reliable for relation alignment.
    """

    def __init__(
        self,
        num_classes: int,
        temperature: float = 0.2,
        relation_temperature: float = 0.2,
        beta: float = 1.0,
        class_weight_min: float = 0.75,
        class_weight_max: float = 1.5,
        class_weight_power: float = 0.5,
        reliability_tau: float = 1.0,
        reliability_min: float = 0.05,
        use_class_calibration: bool = True,
        use_view_reliability: bool = True,
        use_relation_alignment: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.relation_temperature = float(relation_temperature)
        self.beta = float(beta)
        self.class_weight_min = float(class_weight_min)
        self.class_weight_max = float(class_weight_max)
        self.class_weight_power = float(class_weight_power)
        self.reliability_tau = float(reliability_tau)
        self.reliability_min = float(reliability_min)
        self.use_class_calibration = bool(use_class_calibration)
        self.use_view_reliability = bool(use_view_reliability)
        self.use_relation_alignment = bool(use_relation_alignment)

    def forward(
        self,
        original_feature: torch.Tensor,
        weak_feature: torch.Tensor,
        strong_feature: torch.Tensor,
        labels: torch.Tensor,
        strong_logits: torch.Tensor | None = None,
        class_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        labels = labels.long()
        original = _flatten_normalize(original_feature)
        weak = _flatten_normalize(weak_feature)
        strong = _flatten_normalize(strong_feature)

        class_weights = self._class_weights(labels, class_counts, original.device)
        contrast_loss = self._skew_supcon(original, weak, labels, class_weights)

        if self.use_relation_alignment:
            relation_loss, reliability = self._reliable_relation(
                original=original,
                weak=weak,
                strong=strong,
                labels=labels,
                strong_logits=strong_logits,
                class_weights=class_weights,
            )
        else:
            relation_loss = original.sum() * 0.0
            reliability = torch.ones_like(labels, dtype=torch.float32, device=original.device)

        loss = contrast_loss + self.beta * relation_loss
        stats = {
            "sara_contrast_loss": float(contrast_loss.detach().cpu()),
            "sara_relation_loss": float(relation_loss.detach().cpu()),
            "sara_reliability": float(reliability.detach().mean().cpu()),
            "sara_class_weight": float(class_weights.detach().mean().cpu()),
        }
        return loss, stats

    def _class_weights(
        self,
        labels: torch.Tensor,
        class_counts: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.use_class_calibration:
            return torch.ones(labels.size(0), device=device)

        if class_counts is None:
            counts = torch.bincount(labels.detach().cpu(), minlength=self.num_classes).float().to(device)
        else:
            counts = class_counts.detach().float().to(device)
            if counts.numel() != self.num_classes:
                raise ValueError(f"class_counts must have {self.num_classes} values, got {counts.numel()}")

        present = counts > 0
        if not present.any():
            return torch.ones(labels.size(0), device=device)

        mean_count = counts[present].mean()
        selected = counts[labels].clamp_min(1.0)
        weights = (mean_count / selected).clamp_min(1e-8).pow(self.class_weight_power)
        weights = weights.clamp(self.class_weight_min, self.class_weight_max)
        return weights / weights.mean().clamp_min(1e-8)

    def _skew_supcon(
        self,
        original: torch.Tensor,
        weak: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = labels.size(0)
        features = torch.cat([original, weak], dim=0)
        anchor_labels = labels.repeat(2)
        anchor_weights = class_weights.repeat(2)

        logits = features @ features.T / max(self.temperature, 1e-8)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.ones_like(logits, dtype=torch.bool)
        self_mask[torch.arange(features.size(0), device=features.device), torch.arange(features.size(0), device=features.device)] = False
        positive_mask = anchor_labels[:, None].eq(anchor_labels[None, :]) & self_mask
        valid = positive_mask.sum(dim=1) > 0

        exp_logits = torch.exp(logits) * self_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        per_anchor = -((positive_mask.float() * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1))
        per_anchor = torch.where(valid, per_anchor, torch.zeros_like(per_anchor))

        per_sample = 0.5 * (per_anchor[:batch_size] + per_anchor[batch_size:])
        sample_valid = valid[:batch_size] | valid[batch_size:]
        if not sample_valid.any():
            return original.sum() * 0.0
        weights = class_weights[sample_valid]
        return (per_sample[sample_valid] * weights).sum() / weights.sum().clamp_min(1e-8)

    def _reliable_relation(
        self,
        original: torch.Tensor,
        weak: torch.Tensor,
        strong: torch.Tensor,
        labels: torch.Tensor,
        strong_logits: torch.Tensor | None,
        class_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference = torch.cat([weak, original], dim=0).detach()
        weak_sim = weak @ reference.T / max(self.relation_temperature, 1e-8)
        strong_sim = strong @ reference.T / max(self.relation_temperature, 1e-8)
        weak_prob = F.softmax(weak_sim, dim=1).detach()
        strong_log_prob = F.log_softmax(strong_sim, dim=1)
        per_sample = F.kl_div(strong_log_prob, weak_prob, reduction="none").sum(dim=1)

        reliability = self._strong_reliability(strong_logits, labels)
        weights = class_weights * reliability
        return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8), reliability

    def _strong_reliability(self, strong_logits: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
        if (not self.use_view_reliability) or strong_logits is None:
            return torch.ones_like(labels, dtype=torch.float32)

        logits = strong_logits.detach()
        true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, labels.view(-1, 1), -torch.inf)
        competitor = masked.max(dim=1).values
        margin = true_logits - competitor
        reliability = torch.sigmoid(margin / max(self.reliability_tau, 1e-8))
        return reliability.clamp(self.reliability_min, 1.0)
