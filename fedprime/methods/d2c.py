from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def entropy(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=dim)


class D2CServer:
    def __init__(
        self,
        temperature: float = 3.0,
        beta: float = 0.5,
        eta: float = 0.5,
        p_min: float = 1e-3,
        eps: float = 1e-6,
        adaptive_beta: bool = False,
        ema_alpha: float | None = None,
        use_prior_debias: bool = True,
        use_class_balanced: bool = True,
        use_sample_confidence: bool = True,
    ):
        self.temperature = temperature
        self.beta = beta
        self.eta = eta
        self.p_min = p_min
        self.eps = eps
        self.adaptive_beta = adaptive_beta
        self.ema_alpha = ema_alpha
        self.use_prior_debias = use_prior_debias
        self.use_class_balanced = use_class_balanced
        self.use_sample_confidence = use_sample_confidence
        self._prior_ema: torch.Tensor | None = None

    def build_teacher(
        self,
        logits: torch.Tensor,
        oracle_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a D2C global teacher.

        Args:
            logits: Tensor with shape [K, B, C].

        Returns:
            teacher: Tensor with shape [B, C].
            prior: Tensor with shape [K, C].
        """
        if logits.ndim != 3:
            raise ValueError(f"Expected logits [K, B, C], got {tuple(logits.shape)}")

        num_clients, _, num_classes = logits.shape
        probs = F.softmax(logits / self.temperature, dim=-1)
        prior = oracle_prior.to(logits.device) if oracle_prior is not None else probs.mean(dim=1)
        prior = prior.clamp(min=self.p_min, max=1.0)
        prior = prior / prior.sum(dim=-1, keepdim=True)

        if self.ema_alpha is not None:
            if self._prior_ema is None or self._prior_ema.shape != prior.shape:
                self._prior_ema = prior.detach()
            else:
                self._prior_ema = (
                    self.ema_alpha * self._prior_ema
                    + (1.0 - self.ema_alpha) * prior.detach()
                )
            prior = self._prior_ema

        beta = self._client_beta(prior, num_classes).view(num_clients, 1, 1)
        if self.use_prior_debias:
            debiased_logits = logits - beta * torch.log(prior[:, None, :] + self.eps)
        else:
            debiased_logits = logits
        debiased_probs = F.softmax(debiased_logits / self.temperature, dim=-1)

        if self.use_class_balanced:
            class_weight = (prior + self.eps).pow(self.eta)
            class_weight = class_weight / class_weight.sum(dim=0, keepdim=True).clamp_min(self.eps)
        else:
            class_weight = torch.full_like(prior, 1.0 / num_clients)

        if self.use_sample_confidence:
            sample_conf = 1.0 - entropy(debiased_probs, dim=-1) / math.log(num_classes)
            sample_conf = sample_conf.clamp(min=0.0, max=1.0)
        else:
            sample_conf = torch.ones_like(debiased_probs[..., 0])

        weight = class_weight[:, None, :] * sample_conf[:, :, None]
        scores = (weight * debiased_probs).sum(dim=0)
        teacher = scores / scores.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return teacher.detach(), prior.detach()

    def _client_beta(self, prior: torch.Tensor, num_classes: int) -> torch.Tensor:
        if not self.adaptive_beta:
            return torch.full((prior.shape[0],), self.beta, device=prior.device)

        normalized_entropy = entropy(prior, dim=-1) / math.log(num_classes)
        return self.beta * (1.0 - normalized_entropy).clamp(min=0.0, max=1.0)


def complementary_kd_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    client_prior: torch.Tensor,
    temperature: float = 3.0,
    rho: float = 1.0,
    eps: float = 1e-8,
    use_gate: bool = False,
    use_complementary: bool = True,
) -> torch.Tensor:
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    if use_complementary:
        comp = (1.0 - client_prior).clamp_min(eps).pow(rho)
        comp = comp / comp.mean().clamp_min(eps)
    else:
        comp = torch.ones_like(client_prior)

    per_class_kl = teacher_probs * (teacher_probs.clamp_min(eps).log() - student_log_probs)
    per_sample = (per_class_kl * comp.view(1, -1)).sum(dim=-1)

    if use_gate:
        num_classes = student_logits.shape[-1]
        gate = entropy(student_probs, dim=-1) / math.log(num_classes)
        per_sample = per_sample * gate.detach()

    return (temperature ** 2) * per_sample.mean()
