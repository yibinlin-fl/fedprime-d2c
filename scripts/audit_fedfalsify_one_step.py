from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.fedease import FedEASEEvaluationDataset  # noqa: E402
from fedprime.data.loaders import (  # noqa: E402
    CorruptionSkewClientDataset,
    _private_test_transform,
)
from fedprime.methods.fedfalsify.audit_runtime import (  # noqa: E402
    load_models_from_archive,
)
from fedprime.methods.fedfalsify.evidence import (  # noqa: E402
    compute_paired_advantage,
    planned_stratified_audit_counts,
)
from fedprime.methods.fedfalsify.transfer import (  # noqa: E402
    conservative_margin_transfer_loss,
    direct_peer_kd_loss,
    fixed_margin_loss,
    gradient_cosine_from_losses,
)
from fedprime.models.factory import forward_logits  # noqa: E402
from fedprime.utils.config import load_config  # noqa: E402
from fedprime.utils.env import resolve_device, seed_everything  # noqa: E402


GAMMA_VALUES = {
    "00": 0.0,
    "06": 0.6,
    "09": 0.9,
}


@dataclass(frozen=True)
class ClassAuditBatches:
    fit_images: torch.Tensor
    fit_labels: torch.Tensor
    audit_images: torch.Tensor
    audit_labels: torch.Tensor
    evidence_indices: np.ndarray
    projected_audit_count: int
    train_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact one-step FedFalsify audits from stored heterogeneous "
            "RAHFL checkpoints without starting federated training."
        )
    )
    parser.add_argument("--gamma", choices=sorted(GAMMA_VALUES), default="09")
    parser.add_argument(
        "--checkpoint-archive",
        type=Path,
        default=Path("outputs/cle_rahfl_diagnostic_outputs.tar.gz"),
    )
    parser.add_argument(
        "--config-template",
        default="configs/diagnostic_rahfl_cle_alpha05_gamma{token}.yaml",
    )
    parser.add_argument(
        "--tensor-dir",
        type=Path,
        default=Path("outputs/fedfalsify_audit/foreign_tensor"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fedfalsify_audit/one_step"),
    )
    parser.add_argument("--audit-ratio", type=float, default=0.15)
    parser.add_argument("--min-audit-per-class", type=int, default=10)
    parser.add_argument("--advantage-kappa", type=float, default=1.0)
    parser.add_argument("--shrinkage-nu", type=float, default=10.0)
    parser.add_argument("--fit-batch-size", type=int, default=32)
    parser.add_argument("--audit-batch-size", type=int, default=64)
    parser.add_argument("--virtual-lr", type=float, default=0.005)
    parser.add_argument("--lambda-aux", type=float, default=0.5)
    parser.add_argument("--margin-clip", type=float, default=2.0)
    parser.add_argument("--fixed-margin", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--max-triplets", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def stack_dataset_samples(dataset, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    labels = []
    for index in indices.tolist():
        sample = dataset[int(index)]
        images.append(sample[0])
        labels.append(int(sample[1]))
    if not images:
        raise ValueError("Cannot construct an empty audit batch")
    return torch.stack(images), torch.as_tensor(labels, dtype=torch.long)


def build_class_batch_cache(
    *,
    private_root: Path,
    receiver_id: int,
    num_classes: int,
    audit_ratio: float,
    fit_batch_size: int,
    audit_batch_size: int,
    seed: int,
) -> dict[int, ClassAuditBatches]:
    fit_dataset = CorruptionSkewClientDataset(
        root=private_root,
        client_id=receiver_id,
        train=True,
        transform=_private_test_transform(),
        return_corruption=False,
    )
    audit_dataset = FedEASEEvaluationDataset(
        private_root / "test_same" / f"client_{receiver_id}"
    )
    train_labels = np.asarray(fit_dataset.targets, dtype=np.int64)
    audit_labels = np.asarray(audit_dataset.labels, dtype=np.int64)
    projected_counts = planned_stratified_audit_counts(
        train_labels,
        num_classes=num_classes,
        audit_ratio=audit_ratio,
    )
    train_counts = np.bincount(train_labels, minlength=num_classes)
    rng = np.random.default_rng(seed + receiver_id * 1009)
    cache = {}

    for class_id in range(num_classes):
        fit_indices = np.flatnonzero(train_labels == class_id)
        independent_indices = np.flatnonzero(audit_labels == class_id)
        rng.shuffle(fit_indices)
        rng.shuffle(independent_indices)
        if fit_indices.size == 0 or independent_indices.size == 0:
            continue
        evidence_count = min(
            int(projected_counts[class_id]),
            max(int(independent_indices.size) - int(audit_batch_size), 0),
        )
        if evidence_count <= 0:
            continue
        selected_fit = fit_indices[: min(int(fit_batch_size), int(fit_indices.size))]
        evidence_indices = independent_indices[:evidence_count]
        selected_audit = independent_indices[
            evidence_count:evidence_count + int(audit_batch_size)
        ]
        if selected_audit.size == 0:
            continue
        fit_images, fit_targets = stack_dataset_samples(fit_dataset, selected_fit)
        audit_images, audit_targets = stack_dataset_samples(audit_dataset, selected_audit)
        cache[class_id] = ClassAuditBatches(
            fit_images=fit_images,
            fit_labels=fit_targets,
            audit_images=audit_images,
            audit_labels=audit_targets,
            evidence_indices=evidence_indices,
            projected_audit_count=int(projected_counts[class_id]),
            train_count=int(train_counts[class_id]),
        )
    return cache


def clone_trainable(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    clone = copy.deepcopy(model).to(device)
    for parameter in clone.parameters():
        parameter.requires_grad_(True)
    return clone


def audit_ce_loss(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        loss = F.cross_entropy(forward_logits(model, images), labels)
    model.train(was_training)
    return float(loss.item())


def one_virtual_step(
    *,
    base_model: torch.nn.Module,
    source_logits: torch.Tensor,
    fit_images: torch.Tensor,
    fit_labels: torch.Tensor,
    audit_images: torch.Tensor,
    audit_labels: torch.Tensor,
    method: str,
    virtual_lr: float,
    lambda_aux: float,
    margin_clip: float,
    fixed_margin: float,
    temperature: float,
    max_grad_norm: float,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model = clone_trainable(base_model, device)
    before = audit_ce_loss(model, audit_images, audit_labels)
    # This is a parameter-direction audit, not a simulation of BatchNorm
    # running-statistics adaptation. Eval mode keeps BN statistics fixed while
    # autograd still computes parameter gradients.
    model.eval()
    logits = forward_logits(model, fit_images)
    ce_loss = F.cross_entropy(logits, fit_labels)
    if method == "ce":
        auxiliary = logits.sum() * 0.0
        total = ce_loss
    elif method == "fixed_margin":
        auxiliary = fixed_margin_loss(
            logits,
            fit_labels,
            target_margin=fixed_margin,
        )
        total = ce_loss + float(lambda_aux) * auxiliary
    elif method == "direct_kd":
        auxiliary = direct_peer_kd_loss(
            logits,
            source_logits,
            temperature=temperature,
        )
        total = ce_loss + float(lambda_aux) * auxiliary
    elif method == "cmt":
        auxiliary = conservative_margin_transfer_loss(
            logits,
            source_logits,
            fit_labels,
            margin_clip=margin_clip,
            source_correct_only=True,
        )
        total = ce_loss + float(lambda_aux) * auxiliary
    elif method == "cmt_only":
        auxiliary = conservative_margin_transfer_loss(
            logits,
            source_logits,
            fit_labels,
            margin_clip=margin_clip,
            source_correct_only=True,
        )
        total = auxiliary
    else:
        raise ValueError(f"Unknown virtual-step method: {method}")

    model.zero_grad(set_to_none=True)
    total.backward()
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(max_grad_norm),
        ).item()
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-float(virtual_lr))
    after = audit_ce_loss(model, audit_images, audit_labels)
    del model
    return before - after, float(total.item()), float(auxiliary.item()), grad_norm


def action_utility(
    *,
    receiver_model: torch.nn.Module,
    source_logits: torch.Tensor,
    fit_images: torch.Tensor,
    fit_labels: torch.Tensor,
    audit_images: torch.Tensor,
    audit_labels: torch.Tensor,
    margin_clip: float,
    parameter_scope: str,
    device: torch.device,
) -> tuple[float, float, float]:
    model = clone_trainable(receiver_model, device)
    model.eval()
    transfer_loss = conservative_margin_transfer_loss(
        forward_logits(model, fit_images),
        source_logits,
        fit_labels,
        margin_clip=margin_clip,
        source_correct_only=True,
    )
    audit_loss = F.cross_entropy(
        forward_logits(model, audit_images),
        audit_labels,
    )
    if parameter_scope == "full":
        parameters = model.parameters()
    elif parameter_scope == "head":
        module = model.module if hasattr(model, "module") else model
        if not hasattr(module, "linear"):
            raise AttributeError("Head-only TAU requires a `.linear` classifier")
        parameters = module.linear.parameters()
    else:
        raise ValueError(f"Unknown TAU parameter scope: {parameter_scope}")
    utility, audit_norm, transfer_norm = gradient_cosine_from_losses(
        audit_loss,
        transfer_loss,
        parameters,
    )
    del model
    return utility, audit_norm, transfer_norm


def binary_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    positive = scores[targets == 1]
    negative = scores[targets == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0)))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    utilities = np.asarray([row["action_utility"] for row in rows], dtype=np.float64)
    cmt_only_delta = np.asarray([row["cmt_only_delta"] for row in rows], dtype=np.float64)
    ce_delta = np.asarray([row["ce_delta"] for row in rows], dtype=np.float64)
    cmt_delta = np.asarray([row["cmt_delta"] for row in rows], dtype=np.float64)
    cmt_increment = cmt_delta - ce_delta
    advantage_active = np.asarray([row["advantage_gate_active"] for row in rows], dtype=bool)
    utility_active = utilities > 0.0
    full_active = advantage_active & utility_active
    cmt_only_target = (cmt_only_delta > 0.0).astype(np.int64)
    incremental_target = (cmt_increment > 0.0).astype(np.int64)

    method_keys = {
        "ce": "ce_delta",
        "fixed_margin": "fixed_margin_delta",
        "direct_kd": "direct_kd_delta",
        "cmt_without_gate": "cmt_delta",
        "paired_advantage_gate": "fra_selected_delta",
        "utility_gate": "utility_selected_delta",
        "full_fedfalsify_gate": "full_selected_delta",
    }
    methods = {}
    for name, key in method_keys.items():
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        methods[name] = {
            "mean_audit_loss_reduction": float(values.mean()),
            "median_audit_loss_reduction": float(np.median(values)),
            "positive_transfer_fraction": float(np.mean(values > 0.0)),
            "large_negative_fraction": float(np.mean(values < -0.01)),
        }

    full_cmt_only_precision = (
        float(np.mean(cmt_only_target[full_active] == 1))
        if np.any(full_active)
        else float("nan")
    )
    full_increment_precision = (
        float(np.mean(incremental_target[full_active] == 1))
        if np.any(full_active)
        else float("nan")
    )
    utility_increment_precision = (
        float(np.mean(incremental_target[utility_active] == 1))
        if np.any(utility_active)
        else float("nan")
    )
    incremental_positive = incremental_target > 0
    utility_increment_recall = (
        float(np.mean(utility_active[incremental_positive]))
        if np.any(incremental_positive)
        else float("nan")
    )
    return {
        "num_triplets": len(rows),
        "advantage_gate_activation_rate": float(advantage_active.mean()),
        "utility_positive_rate": float(utility_active.mean()),
        "full_gate_activation_rate": float(full_active.mean()),
        "utility_auc_for_cmt_only_positive_gain": binary_auc(utilities, cmt_only_target),
        "utility_auc_for_cmt_increment_over_ce": binary_auc(utilities, incremental_target),
        "utility_gate_precision_for_cmt_increment_over_ce": utility_increment_precision,
        "utility_gate_recall_for_cmt_increment_over_ce": utility_increment_recall,
        "full_gate_precision_for_cmt_only_positive_gain": full_cmt_only_precision,
        "full_gate_precision_for_cmt_increment_over_ce": full_increment_precision,
        "incremental_effects_over_ce": {
            "fixed_margin_mean": float(np.mean(
                np.asarray([row["fixed_margin_delta"] for row in rows]) - ce_delta
            )),
            "direct_kd_mean": float(np.mean(
                np.asarray([row["direct_kd_delta"] for row in rows]) - ce_delta
            )),
            "cmt_mean": float(cmt_increment.mean()),
            "cmt_utility_selected_mean": float(
                np.mean(cmt_increment[utility_active])
                if np.any(utility_active)
                else 0.0
            ),
            "cmt_full_selected_mean": float(
                np.mean(cmt_increment[full_active])
                if np.any(full_active)
                else 0.0
            ),
        },
        "methods": methods,
    }


def main() -> None:
    args = parse_args()
    token = args.gamma
    gamma = GAMMA_VALUES[token]
    config = load_config(Path(args.config_template.format(token=token)))
    seed_everything(args.seed)
    device = resolve_device(args.device)
    private_root = Path(config["data"]["private_root"])
    model_names = list(config["models"]["names"])
    num_clients = len(model_names)
    num_classes = int(config["data"].get("num_classes", 10))
    tensor_path = args.tensor_dir / f"gamma{token}_foreign_predictions.npz"
    if not tensor_path.is_file():
        raise FileNotFoundError(
            f"Missing {tensor_path}; run audit_fedfalsify_foreign_tensor.py first."
        )
    prediction_payload = np.load(tensor_path, allow_pickle=False)
    foreign_predictions = prediction_payload["predictions"]
    foreign_labels = prediction_payload["labels"]

    print(
        f"[audit] FedFalsify exact one-step audit gamma={gamma:.1f} device={device}",
        flush=True,
    )
    print(
        "[audit] fit batches come from client train data; evidence and audit-loss "
        "batches are disjoint subsets of independent test_same data.",
        flush=True,
    )
    models = load_models_from_archive(
        checkpoint_archive=args.checkpoint_archive,
        experiment_name=str(config["experiment_name"]),
        model_names=model_names,
        num_classes=num_classes,
        device=device,
    )

    rows = []
    triplet_index = 0
    for receiver_id in range(num_clients):
        cache = build_class_batch_cache(
            private_root=private_root,
            receiver_id=receiver_id,
            num_classes=num_classes,
            audit_ratio=args.audit_ratio,
            fit_batch_size=args.fit_batch_size,
            audit_batch_size=args.audit_batch_size,
            seed=args.seed,
        )
        for source_id in range(num_clients):
            if source_id == receiver_id:
                continue
            source_model = models[source_id]
            receiver_model = models[receiver_id]
            for class_id, batches in cache.items():
                if args.max_triplets is not None and triplet_index >= args.max_triplets:
                    break
                evidence = compute_paired_advantage(
                    foreign_predictions[source_id, receiver_id, batches.evidence_indices],
                    foreign_predictions[receiver_id, receiver_id, batches.evidence_indices],
                    foreign_labels[receiver_id, batches.evidence_indices],
                    class_id=class_id,
                    kappa=args.advantage_kappa,
                    shrinkage_nu=args.shrinkage_nu,
                    min_count=args.min_audit_per_class,
                )
                fit_images = batches.fit_images.to(device)
                fit_labels = batches.fit_labels.to(device)
                audit_images = batches.audit_images.to(device)
                audit_labels = batches.audit_labels.to(device)
                with torch.inference_mode():
                    source_logits = forward_logits(source_model, fit_images).detach()

                utility, audit_grad_norm, transfer_grad_norm = action_utility(
                    receiver_model=receiver_model,
                    source_logits=source_logits,
                    fit_images=fit_images,
                    fit_labels=fit_labels,
                    audit_images=audit_images,
                    audit_labels=audit_labels,
                    margin_clip=args.margin_clip,
                    parameter_scope="full",
                    device=device,
                )
                head_utility, head_audit_grad_norm, head_transfer_grad_norm = action_utility(
                    receiver_model=receiver_model,
                    source_logits=source_logits,
                    fit_images=fit_images,
                    fit_labels=fit_labels,
                    audit_images=audit_images,
                    audit_labels=audit_labels,
                    margin_clip=args.margin_clip,
                    parameter_scope="head",
                    device=device,
                )
                method_results = {}
                for method in ("ce", "fixed_margin", "direct_kd", "cmt", "cmt_only"):
                    method_results[method] = one_virtual_step(
                        base_model=receiver_model,
                        source_logits=source_logits,
                        fit_images=fit_images,
                        fit_labels=fit_labels,
                        audit_images=audit_images,
                        audit_labels=audit_labels,
                        method=method,
                        virtual_lr=args.virtual_lr,
                        lambda_aux=args.lambda_aux,
                        margin_clip=args.margin_clip,
                        fixed_margin=args.fixed_margin,
                        temperature=args.temperature,
                        max_grad_norm=args.max_grad_norm,
                        device=device,
                    )

                ce_delta = method_results["ce"][0]
                cmt_delta = method_results["cmt"][0]
                advantage_active = evidence.is_active
                full_active = advantage_active and utility > 0.0
                row = {
                    "gamma": gamma,
                    "source_client": source_id,
                    "source_model": model_names[source_id],
                    "receiver_client": receiver_id,
                    "receiver_model": model_names[receiver_id],
                    "class_id": class_id,
                    "train_count": batches.train_count,
                    "projected_audit_count": batches.projected_audit_count,
                    "source_accuracy": evidence.source_accuracy,
                    "receiver_accuracy": evidence.receiver_accuracy,
                    "paired_advantage": evidence.paired_advantage,
                    "conservative_advantage": evidence.conservative_advantage,
                    "advantage_strength": evidence.advantage_strength,
                    "advantage_gate_active": int(advantage_active),
                    "action_utility": utility,
                    "head_action_utility": head_utility,
                    "audit_gradient_norm": audit_grad_norm,
                    "transfer_gradient_norm": transfer_grad_norm,
                    "head_audit_gradient_norm": head_audit_grad_norm,
                    "head_transfer_gradient_norm": head_transfer_grad_norm,
                    "utility_gate_active": int(utility > 0.0),
                    "full_gate_active": int(full_active),
                    "ce_delta": ce_delta,
                    "fixed_margin_delta": method_results["fixed_margin"][0],
                    "direct_kd_delta": method_results["direct_kd"][0],
                    "cmt_delta": cmt_delta,
                    "cmt_only_delta": method_results["cmt_only"][0],
                    "fra_selected_delta": cmt_delta if advantage_active else ce_delta,
                    "utility_selected_delta": cmt_delta if utility > 0.0 else ce_delta,
                    "full_selected_delta": cmt_delta if full_active else ce_delta,
                    "cmt_aux_loss": method_results["cmt"][2],
                    "direct_kd_aux_loss": method_results["direct_kd"][2],
                    "fixed_margin_aux_loss": method_results["fixed_margin"][2],
                    "cmt_grad_norm": method_results["cmt"][3],
                }
                rows.append(row)
                triplet_index += 1
                print(
                    f"[heartbeat] triplet={triplet_index} {source_id}->{receiver_id} "
                    f"class={class_id} adv={evidence.paired_advantage:+.3f} "
                    f"utility={utility:+.3f} head_utility={head_utility:+.3f} "
                    f"cmt_delta={cmt_delta:+.5f}",
                    flush=True,
                )
            if args.max_triplets is not None and triplet_index >= args.max_triplets:
                break
        if args.max_triplets is not None and triplet_index >= args.max_triplets:
            break

    if not rows:
        raise RuntimeError("No source-receiver-class triplets were auditable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"gamma{token}_one_step_triplets.csv", rows)
    summary = summarize(rows)
    summary.update({
        "audit_only": True,
        "gamma": gamma,
        "virtual_lr": float(args.virtual_lr),
        "lambda_aux": float(args.lambda_aux),
        "warning": (
            "Independent test_same labels are used only for offline Go/No-Go "
            "auditing and must not enter a formal training router."
        ),
    })
    (args.output_dir / f"gamma{token}_one_step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[audit] utility_positive_rate={summary['utility_positive_rate']:.3f} "
        f"utility_auc={summary['utility_auc_for_cmt_only_positive_gain']}",
        flush=True,
    )
    print(f"[audit] wrote one-step audit to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
