from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


def load_operator_metadata(root: str | Path) -> dict:
    path = Path(root) / "metadata.json"
    if not path.is_file():
        return {}
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if int(metadata.get("protocol_version", 1)) < 2:
        return {}
    return metadata


def summarize_operator_splits(
    group_summary: dict[str, object],
    metadata: dict,
) -> dict[str, dict[str, float]]:
    """Derive seen/unseen metrics from operator-level evaluation output."""

    operator_splits = {
        str(operator): str(split)
        for operator, split in metadata.get("operator_splits", {}).items()
    }
    if not operator_splits:
        return {}

    groups = {
        str(name): float(value)
        for name, value in dict(group_summary.get("groups", {})).items()
    }
    clients = {
        int(client_id): {
            str(name): float(value)
            for name, value in values.items()
        }
        for client_id, values in dict(group_summary.get("clients", {})).items()
    }
    class_rows = list(group_summary.get("class_corruption_rows", []))
    summaries = {}
    for split_name in ("seen", "unseen", "all"):
        operators = [
            operator
            for operator in groups
            if split_name == "all" or operator_splits.get(operator) == split_name
        ]
        if not operators:
            continue
        operator_set = set(operators)
        client_accuracies = []
        client_operator_values = []
        for values in clients.values():
            selected = [value for operator, value in values.items() if operator in operator_set]
            if selected:
                client_accuracies.append(sum(selected) / len(selected))
                client_operator_values.extend(selected)

        correct_by_class_operator = defaultdict(float)
        total_by_class_operator = defaultdict(int)
        for row in class_rows:
            operator = str(row["group"])
            if operator not in operator_set:
                continue
            key = (int(row["class_id"]), operator)
            total = int(row["total"])
            correct_by_class_operator[key] += float(row["acc"]) * total / 100.0
            total_by_class_operator[key] += total
        class_operator_acc = {
            key: 100.0 * correct_by_class_operator[key] / total
            for key, total in total_by_class_operator.items()
            if total > 0
        }
        class_gaps = []
        class_ids = sorted({class_id for class_id, _ in class_operator_acc})
        for class_id in class_ids:
            values = [
                value
                for (row_class, _), value in class_operator_acc.items()
                if row_class == class_id
            ]
            if len(values) >= 2:
                class_gaps.append(max(values) - min(values))

        summaries[split_name] = {
            "avg_acc": sum(client_accuracies) / max(len(client_accuracies), 1),
            "worst_acc": min(client_accuracies) if client_accuracies else 0.0,
            "worst_operator_acc": min(
                (groups[operator] for operator in operators),
                default=0.0,
            ),
            "worst_client_operator_acc": min(client_operator_values, default=0.0),
            "wcca": min(class_operator_acc.values(), default=0.0),
            "cfg": sum(class_gaps) / max(len(class_gaps), 1),
        }
    return summaries


def operator_rows_for_round(
    *,
    round_idx: int,
    group_summary: dict[str, object],
    metadata: dict,
) -> tuple[list[dict], list[dict]]:
    operator_splits = {
        str(operator): str(split)
        for operator, split in metadata.get("operator_splits", {}).items()
    }
    client_rows = []
    for client_id, values in dict(group_summary.get("clients", {})).items():
        for operator, accuracy in values.items():
            client_rows.append(
                {
                    "round": int(round_idx),
                    "client": int(client_id),
                    "operator": str(operator),
                    "split": operator_splits.get(str(operator), "unknown"),
                    "accuracy": float(accuracy),
                }
            )

    class_rows = []
    for row in group_summary.get("class_corruption_rows", []):
        operator = str(row["group"])
        class_rows.append(
            {
                "round": int(round_idx),
                "client": int(row["client"]),
                "class_id": int(row["class_id"]),
                "operator": operator,
                "split": operator_splits.get(operator, "unknown"),
                "accuracy": float(row["acc"]),
                "total": int(row["total"]),
            }
        )
    return client_rows, class_rows
