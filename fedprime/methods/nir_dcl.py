from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F


class NIRDCLFeatureQueue:
    """Client-local feature memory for tail classes.

    The queue never leaves the client. It only provides additional positives and
    negatives for local contrastive learning when a Non-IID mini-batch has too
    few same-class samples.
    """

    def __init__(self, num_classes: int, max_size_per_class: int = 64):
        self.num_classes = int(num_classes)
        self.max_size_per_class = int(max_size_per_class)
        self._queues = [deque(maxlen=self.max_size_per_class) for _ in range(self.num_classes)]

    def enqueue(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        features = features.detach().cpu()
        labels = labels.detach().cpu().long()
        for feature, label in zip(features, labels):
            class_id = int(label.item())
            if 0 <= class_id < self.num_classes:
                self._queues[class_id].append(feature.clone())

    def all_features(self, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        features = []
        labels = []
        for class_id, queue in enumerate(self._queues):
            if not queue:
                continue
            stacked = torch.stack(list(queue), dim=0)
            features.append(stacked)
            labels.append(torch.full((stacked.size(0),), class_id, dtype=torch.long))
        if not features:
            return None, None
        return torch.cat(features, dim=0).to(device), torch.cat(labels, dim=0).to(device)


def _flatten_normalize(feature: torch.Tensor) -> torch.Tensor:
    if feature.dim() > 2:
        feature = feature.view(feature.size(0), -1)
    return F.normalize(feature, dim=1)


def _class_balanced_mean(
    values: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    class_losses = []
    for class_id in range(num_classes):
        mask = labels == class_id
        if not mask.any():
            continue
        selected = values[mask]
        if weights is None:
            class_losses.append(selected.mean())
            continue
        selected_weights = weights[mask].clamp_min(1e-6)
        class_losses.append((selected * selected_weights).sum() / selected_weights.sum())
    if not class_losses:
        return values.mean() * 0.0
    return torch.stack(class_losses).mean()


class NIRDCLLoss(torch.nn.Module):
    """Non-IID-aware robust variant of RAHFL DCL.

    It preserves the original DCL spirit, but changes the aggregation rule and
    the strong-view relation alignment to reduce head-class dominance under
    label-skew Non-IID data.
    """

    def __init__(
        self,
        num_classes: int,
        temperature: float = 0.2,
        relation_temperature: float = 0.2,
        beta: float = 1.0,
        reliability_tau: float = 1.0,
        reliability_min: float = 0.05,
        use_class_balance: bool = True,
        use_queue: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.relation_temperature = float(relation_temperature)
        self.beta = float(beta)
        self.reliability_tau = float(reliability_tau)
        self.reliability_min = float(reliability_min)
        self.use_class_balance = bool(use_class_balance)
        self.use_queue = bool(use_queue)

    def forward(
        self,
        original_feature: torch.Tensor,
        weak_feature: torch.Tensor,
        strong_feature: torch.Tensor,
        labels: torch.Tensor,
        strong_logits: torch.Tensor | None = None,
        feature_queue: NIRDCLFeatureQueue | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        labels = labels.long()
        original = _flatten_normalize(original_feature)
        weak = _flatten_normalize(weak_feature)
        strong = _flatten_normalize(strong_feature)

        cont_loss = self._contrastive_loss(original, weak, labels, feature_queue)
        rel_loss, reliability = self._relation_loss(original, weak, strong, labels, strong_logits)
        loss = cont_loss + self.beta * rel_loss
        stats = {
            "nir_cont_loss": float(cont_loss.detach().cpu()),
            "nir_rel_loss": float(rel_loss.detach().cpu()),
            "nir_reliability": float(reliability.detach().mean().cpu()),
        }
        return loss, stats

    def _contrastive_loss(
        self,
        original: torch.Tensor,
        weak: torch.Tensor,
        labels: torch.Tensor,
        feature_queue: NIRDCLFeatureQueue | None,
    ) -> torch.Tensor:
        batch_size = labels.size(0)
        anchors = torch.cat([original, weak], dim=0)
        anchor_labels = labels.repeat(2)

        contrast = anchors
        contrast_labels = anchor_labels
        if self.use_queue and feature_queue is not None:
            queue_features, queue_labels = feature_queue.all_features(original.device)
            if queue_features is not None and queue_labels is not None:
                contrast = torch.cat([contrast, queue_features.detach()], dim=0)
                contrast_labels = torch.cat([contrast_labels, queue_labels], dim=0)

        logits = anchors @ contrast.T / max(self.temperature, 1e-8)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.ones_like(logits, dtype=torch.bool)
        self_mask[:, : anchors.size(0)] = True
        self_mask[torch.arange(anchors.size(0), device=anchors.device), torch.arange(anchors.size(0), device=anchors.device)] = False

        positive_mask = anchor_labels[:, None].eq(contrast_labels[None, :]) & self_mask
        valid = positive_mask.sum(dim=1) > 0
        exp_logits = torch.exp(logits) * self_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        per_anchor = -((positive_mask.float() * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1))
        per_anchor = torch.where(valid, per_anchor, torch.zeros_like(per_anchor))

        per_sample = 0.5 * (per_anchor[:batch_size] + per_anchor[batch_size:])
        sample_valid = valid[:batch_size] | valid[batch_size:]
        if not sample_valid.any():
            return original.sum() * 0.0
        if self.use_class_balance:
            return _class_balanced_mean(per_sample[sample_valid], labels[sample_valid], self.num_classes)
        return per_sample[sample_valid].mean()

    def _relation_loss(
        self,
        original: torch.Tensor,
        weak: torch.Tensor,
        strong: torch.Tensor,
        labels: torch.Tensor,
        strong_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference = torch.cat([weak, original], dim=0).detach()
        weak_sim = weak @ reference.T / max(self.relation_temperature, 1e-8)
        strong_sim = strong @ reference.T / max(self.relation_temperature, 1e-8)
        weak_prob = F.softmax(weak_sim, dim=1).detach()
        strong_log_prob = F.log_softmax(strong_sim, dim=1)
        per_sample = F.kl_div(strong_log_prob, weak_prob, reduction="none").sum(dim=1)

        reliability = self._strong_reliability(strong_logits, labels)
        if self.use_class_balance:
            loss = _class_balanced_mean(per_sample, labels, self.num_classes, reliability)
        else:
            loss = (per_sample * reliability).sum() / reliability.sum().clamp_min(1e-6)
        return loss, reliability

    def _strong_reliability(self, strong_logits: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
        if strong_logits is None:
            return torch.ones_like(labels, dtype=torch.float32)
        logits = strong_logits.detach()
        true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, labels.view(-1, 1), -torch.inf)
        competitor = masked.max(dim=1).values
        margin = true_logits - competitor
        reliability = torch.sigmoid(margin / max(self.reliability_tau, 1e-8))
        return reliability.clamp(self.reliability_min, 1.0)
