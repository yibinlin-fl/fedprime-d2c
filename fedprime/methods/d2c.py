from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class D2CTeacherDiagnostics:
    """Detached server-side values used to audit D2C without changing training."""

    predicted_prior: torch.Tensor
    used_prior: torch.Tensor
    client_beta: torch.Tensor
    class_weight: torch.Tensor
    sample_confidence: torch.Tensor


def entropy(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy H(p), used to measure prior skew and prediction certainty."""
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
        """Compatibility entry point returning only values used by training."""
        teacher, prior, _ = self.build_teacher_with_diagnostics(
            logits,
            oracle_prior=oracle_prior,
        )
        return teacher, prior

    def build_teacher_with_diagnostics(
        self,
        logits: torch.Tensor,
        oracle_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, D2CTeacherDiagnostics]:
        """Build a D2C global teacher.

        Args:
            logits: Tensor with shape [K, B, C].

        Returns:
            teacher: Tensor with shape [B, C].
            prior: Tensor with shape [K, C].
            diagnostics: Detached intermediate values for prior auditing.
        """
        if logits.ndim != 3:
            raise ValueError(f"Expected logits [K, B, C], got {tuple(logits.shape)}")

        num_clients, _, num_classes = logits.shape

        # Public predictions from all clients:
        #   p_k(y|x) = softmax(z_k(x) / T)
        # Shape: [K clients, B public samples, C classes].
        probs = F.softmax(logits / self.temperature, dim=-1)

        # Predictive prior estimates each client's class tendency on public data:
        #   pi_k(y) = mean_x p_k(y|x)
        # This is the core signal for detecting Non-IID label bias. For oracle
        # ablations, pi_k can be replaced by the true private label histogram.
        predicted_prior = probs.mean(dim=1)
        predicted_prior = predicted_prior.clamp(min=self.p_min, max=1.0)
        predicted_prior = predicted_prior / predicted_prior.sum(dim=-1, keepdim=True)
        if oracle_prior is None:
            # Keep the original predicted-prior operation order exactly intact.
            prior = predicted_prior
        else:
            prior = oracle_prior.to(logits.device)
            prior = prior.clamp(min=self.p_min, max=1.0)
            prior = prior / prior.sum(dim=-1, keepdim=True)

        # EMA prior smooths noisy batch-level prior estimates across rounds:
        #   pi_k <- alpha * pi_k_ema + (1 - alpha) * pi_k_batch
        # It should make D2C less sensitive to a single unlucky public batch.
        if self.ema_alpha is not None:
            if self._prior_ema is None or self._prior_ema.shape != prior.shape:
                self._prior_ema = prior.detach()
            else:
                self._prior_ema = (
                    self.ema_alpha * self._prior_ema
                    + (1.0 - self.ema_alpha) * prior.detach()
                )
            prior = self._prior_ema

        # Prior debias removes the client's estimated label-prior preference:
        #   z'_k(y|x) = z_k(y|x) - beta_k * log(pi_k(y) + eps)
        # Under label shift, logits often contain a prior term. Subtracting it
        # pushes public predictions closer to class-conditional evidence.
        beta = self._client_beta(prior, num_classes).view(num_clients, 1, 1)
        if self.use_prior_debias:
            debiased_logits = logits - beta * torch.log(prior[:, None, :] + self.eps)
        else:
            debiased_logits = logits
        debiased_probs = F.softmax(debiased_logits / self.temperature, dim=-1)

        # Class-balanced aggregation builds a per-class client weight:
        #   a_k,c = pi_k(c)^eta / sum_j pi_j(c)^eta
        # The aim is not to average clients uniformly for every class. Instead,
        # each class can borrow more from clients that appear informative for it.
        if self.use_class_balanced:
            class_weight = (prior + self.eps).pow(self.eta)
            class_weight = class_weight / class_weight.sum(dim=0, keepdim=True).clamp_min(self.eps)
        else:
            class_weight = torch.full_like(prior, 1.0 / num_clients)

        # Sample confidence down-weights high-entropy client predictions:
        #   conf_k(x) = 1 - H(p'_k(.|x)) / log(C)
        # A client contributes less on public samples where it is uncertain.
        if self.use_sample_confidence:
            sample_conf = 1.0 - entropy(debiased_probs, dim=-1) / math.log(num_classes)
            sample_conf = sample_conf.clamp(min=0.0, max=1.0)
        else:
            sample_conf = torch.ones_like(debiased_probs[..., 0])

        # Global D2C teacher:
        #   q(y|x) = normalize_y sum_k a_k,y * conf_k(x) * p'_k(y|x)
        # The teacher is detached because it is a server-side target, not a
        # differentiable ensemble optimized jointly with the client models.
        weight = class_weight[:, None, :] * sample_conf[:, :, None]
        scores = (weight * debiased_probs).sum(dim=0)
        teacher = scores / scores.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        diagnostics = D2CTeacherDiagnostics(
            predicted_prior=predicted_prior.detach(),
            used_prior=prior.detach(),
            client_beta=beta.view(num_clients).detach(),
            class_weight=class_weight.detach(),
            sample_confidence=sample_conf.detach(),
        )
        return teacher.detach(), prior.detach(), diagnostics

    def _client_beta(self, prior: torch.Tensor, num_classes: int) -> torch.Tensor:
        if not self.adaptive_beta:
            return torch.full((prior.shape[0],), self.beta, device=prior.device)

        # Adaptive beta strengthens debiasing for skewed clients:
        #   beta_k = beta * (1 - H(pi_k) / log(C))
        # If a client's prior is uniform, beta_k is small. If it is highly
        # concentrated, beta_k approaches beta.
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
    # Student distribution on public data, softened by the same KD temperature.
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # Complementary KD emphasizes classes missing from this client's private
    # label prior:
    #   m_k(c) = (1 - pi_k(c))^rho
    # A client receives stronger distillation on classes it has seen less often.
    if use_complementary:
        comp = (1.0 - client_prior).clamp_min(eps).pow(rho)
        comp = comp / comp.mean().clamp_min(eps)
    else:
        comp = torch.ones_like(client_prior)

    per_class_kl = teacher_probs * (teacher_probs.clamp_min(eps).log() - student_log_probs)
    per_sample = (per_class_kl * comp.view(1, -1)).sum(dim=-1)

    # Self-preserving gate lets uncertain students learn more from the teacher:
    #   gate(x) = H(p_k(.|x)) / log(C)
    # Confident local predictions are disturbed less by public KD.
    if use_gate:
        num_classes = student_logits.shape[-1]
        gate = entropy(student_probs, dim=-1) / math.log(num_classes)
        per_sample = per_sample * gate.detach()

    # Standard KD multiplies KL by T^2 to keep gradient scale comparable.
    return (temperature ** 2) * per_sample.mean()
