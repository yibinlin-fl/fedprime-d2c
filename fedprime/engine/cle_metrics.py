from __future__ import annotations

import json
from pathlib import Path

import torch

from fedprime.models.factory import forward_logits


def evaluate_cle_split(
    models: dict[int, torch.nn.Module],
    loaders,
    *,
    device: torch.device,
    num_classes: int,
    num_environments: int,
    max_batches: int | None = None,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    client_accuracies = []
    details = []
    aggregate_correct = torch.zeros(num_classes, num_environments, dtype=torch.float64)
    aggregate_total = torch.zeros(num_classes, num_environments, dtype=torch.float64)

    for client_id in sorted(models):
        loader = loaders[client_id] if isinstance(loaders, dict) else loaders
        model = models[client_id]
        model.eval()
        correct_total = 0
        sample_total = 0
        class_environment_correct = torch.zeros_like(aggregate_correct)
        class_environment_total = torch.zeros_like(aggregate_total)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                images, labels, environments = batch
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()
                predictions = forward_logits(model, images).argmax(dim=1)
                correct = predictions.eq(labels)
                correct_total += int(correct.sum().item())
                sample_total += int(labels.numel())
                environments = environments.long()
                labels_cpu = labels.cpu()
                correct_cpu = correct.cpu()
                for class_id in range(num_classes):
                    for environment_id in range(num_environments):
                        mask = (labels_cpu == class_id) & (environments == environment_id)
                        if bool(mask.any()):
                            class_environment_total[class_id, environment_id] += int(mask.sum().item())
                            class_environment_correct[class_id, environment_id] += int(correct_cpu[mask].sum().item())
        accuracy = 100.0 * correct_total / max(sample_total, 1)
        client_accuracies.append(accuracy)
        details.append({"client": client_id, "accuracy": accuracy, "total": sample_total})
        aggregate_correct += class_environment_correct
        aggregate_total += class_environment_total

    valid = aggregate_total > 0
    class_environment_accuracy = 100.0 * aggregate_correct / aggregate_total.clamp_min(1.0)
    wcca = float(class_environment_accuracy[valid].min().item()) if bool(valid.any()) else float("nan")
    class_gaps = []
    for class_id in range(num_classes):
        class_valid = valid[class_id]
        if int(class_valid.sum().item()) >= 2:
            values = class_environment_accuracy[class_id][class_valid]
            class_gaps.append(float((values.max() - values.min()).item()))
    cfg = sum(class_gaps) / len(class_gaps) if class_gaps else float("nan")
    metrics = {
        "avg_acc": sum(client_accuracies) / max(len(client_accuracies), 1),
        "worst_acc": min(client_accuracies) if client_accuracies else 0.0,
        "wcca": wcca,
        "cfg": cfg,
    }
    return metrics, details


def write_cle_evaluation(
    path: str | Path,
    split_metrics: dict[str, dict[str, float]],
    details: dict[str, list[dict[str, float | int]]],
) -> dict[str, object]:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    same = split_metrics.get("same", {}).get("avg_acc")
    swapped = split_metrics.get("swapped", {}).get("avg_acc")
    ers = float(same - swapped) if same is not None and swapped is not None else float("nan")
    summary = {"splits": split_metrics, "ers": ers, "per_client": details}
    (path / "fedease_evaluation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    import csv

    with (path / "fedease_evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "avg_acc", "worst_acc", "wcca", "cfg", "ers"])
        writer.writeheader()
        for split, metrics in split_metrics.items():
            writer.writerow({"split": split, **metrics, "ers": ers if split == "swapped" else ""})
    return summary
