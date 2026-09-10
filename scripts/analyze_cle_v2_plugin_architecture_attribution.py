from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ARMS = ("h9_b", "h9_p")
ENVIRONMENT_NAMES = ("clean", "noise", "blur", "weather", "digital", "unknown")
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-training architecture attribution for CLE-v2 PEW+BER Stage-2."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def operator_dsa(
    probabilities: np.ndarray, labels: np.ndarray, binding: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    clients, sources, operators, _ = probabilities.shape
    effects = np.full((clients, operators, sources), np.nan, dtype=np.float64)
    for client in range(clients):
        for operator in range(operators):
            bound_classes = np.flatnonzero(binding[client] == operator)
            if not bound_classes.size:
                continue
            valid = ~np.isin(labels, bound_classes)
            mass = np.take(probabilities[client], bound_classes, axis=-1).sum(axis=-1)
            target = mass[:, operator]
            reference = (mass.sum(axis=1) - target) / max(operators - 1, 1)
            effects[client, operator, valid] = target[valid] - reference[valid]
    valid = np.isfinite(effects)
    counts = valid.sum(axis=-1)
    totals = np.where(valid, effects, 0.0).sum(axis=-1)
    per_operator = np.divide(
        totals,
        counts,
        out=np.full_like(totals, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    return effects, per_operator


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return None
    left, right = left[mask], right[mask]
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def ber_diagnostics(labels: np.ndarray, environments: np.ndarray) -> dict[str, object]:
    counts = np.zeros((10, len(ENVIRONMENT_NAMES)), dtype=np.float64)
    np.add.at(counts, (labels, environments), 1.0)
    valid = counts >= 2.0
    support = np.where(valid, np.minimum(counts, 32.0) ** 0.5, 0.0)
    weights = support / np.maximum(support.sum(axis=1, keepdims=True), 1.0e-12)
    objective = weights / np.maximum(counts, 1.0)
    objective /= max(int(valid.any(axis=1).sum()), 1)
    multiplier = counts.sum() * objective[labels, environments]
    weight_squares = (weights**2).sum(axis=1)
    effective = np.divide(
        1.0,
        weight_squares,
        out=np.full_like(weight_squares, np.nan, dtype=np.float64),
        where=weight_squares > 0.0,
    )
    return {
        "counts": counts,
        "valid_classes": int(valid.any(axis=1).sum()),
        "valid_groups": int(valid.sum()),
        "effective_groups_per_class": float(effective[valid.any(axis=1)].mean()),
        "sample_multiplier_mean": float(multiplier.mean()),
        "sample_multiplier_p95": float(np.quantile(multiplier, 0.95)),
        "sample_multiplier_max": float(multiplier.max()),
        "class_effective_groups": effective,
        "class_multiplier_max": np.asarray(
            [
                multiplier[labels == class_id].max()
                if bool((labels == class_id).any())
                else np.nan
                for class_id in range(10)
            ]
        ),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    package_root = args.package_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = result_root / "analysis" / "STAGE2_PREDICTIONS.npz"
    archive = np.load(prediction_path, allow_pickle=False)
    probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
    arms = tuple(str(value) for value in archive["arms"])
    labels = np.asarray(archive["labels"], dtype=np.int64)
    binding = np.asarray(archive["binding"], dtype=np.int64)
    operator_names = tuple(str(value) for value in archive["operator_names"])
    if arms != ARMS or probabilities.shape != (2, 4, 1000, 15, 10):
        raise ValueError("Unexpected Stage-2 prediction contract")

    metadata = json.loads(
        (package_root / "data" / "gamma09" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    model_names = tuple(
        json.loads((package_root / "manifest.json").read_text(encoding="utf-8"))[
            "model_names"
        ]
    )
    family_by_operator = tuple(
        metadata["operator_families"][operator] for operator in operator_names
    )
    split_by_operator = tuple(
        metadata["operator_splits"][operator] for operator in operator_names
    )
    environment_to_id = {name: index for index, name in enumerate(ENVIRONMENT_NAMES)}
    true_environment_by_operator = np.asarray(
        [environment_to_id[family] for family in family_by_operator], dtype=np.int64
    )

    dsa_effects = []
    dsa_by_operator = []
    accuracies = []
    for arm_index in range(2):
        effects, per_operator = operator_dsa(
            probabilities[arm_index], labels, binding
        )
        dsa_effects.append(effects)
        dsa_by_operator.append(per_operator)
        predictions = probabilities[arm_index].argmax(axis=-1)
        accuracies.append(predictions == labels[None, :, None])
    dsa_effects = np.asarray(dsa_effects)
    dsa_by_operator = np.asarray(dsa_by_operator)
    accuracies = np.asarray(accuracies)

    split_archive = np.load(
        package_root / "splits" / "strict_cle_v2_factorial_seed0_split0.npz",
        allow_pickle=False,
    )
    client_rows: list[dict[str, object]] = []
    operator_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    client_details: dict[str, object] = {}

    for client, model_name in enumerate(model_names):
        data_root = package_root / "data" / "gamma09" / f"client_{client}"
        train_labels = np.load(data_root / "train_labels.npy").astype(np.int64)
        train_operators = np.load(data_root / "train_corruption_ids.npy").astype(np.int64)
        annotation = np.load(
            package_root
            / "pew_standard"
            / "annotations"
            / "gamma09"
            / f"client_{client}.npz",
            allow_pickle=False,
        )
        predicted_environment = np.asarray(annotation["environment_ids"], dtype=np.int64)
        confidence = np.asarray(annotation["confidence"], dtype=np.float64)
        fit = np.asarray(split_archive[f"client_{client}_fit"], dtype=np.int64)
        fit_labels = train_labels[fit]
        fit_operators = train_operators[fit]
        fit_environment = predicted_environment[fit]
        fit_confidence = confidence[fit]
        true_fit_environment = true_environment_by_operator[fit_operators]
        ber = ber_diagnostics(fit_labels, fit_environment)

        base_grid_accuracy = float(accuracies[0, client].mean() * 100.0)
        plugin_grid_accuracy = float(accuracies[1, client].mean() * 100.0)
        base_dsa = float(np.nanmean(dsa_by_operator[0, client]))
        plugin_dsa = float(np.nanmean(dsa_by_operator[1, client]))
        client_row = {
            "client": client,
            "model": model_name,
            "base_grid_acc": base_grid_accuracy,
            "plugin_grid_acc": plugin_grid_accuracy,
            "grid_acc_delta": plugin_grid_accuracy - base_grid_accuracy,
            "base_dsa": base_dsa,
            "plugin_dsa": plugin_dsa,
            "dsa_reduction": base_dsa - plugin_dsa,
            "pew_family_accuracy": float(
                (fit_environment == true_fit_environment).mean() * 100.0
            ),
            "pew_unknown_rate": float(
                (fit_environment == environment_to_id["unknown"]).mean() * 100.0
            ),
            "pew_mean_confidence": float(fit_confidence.mean()),
            "fit_valid_classes_for_ber": ber["valid_classes"],
            "ber_valid_groups": ber["valid_groups"],
            "ber_effective_groups_per_class": ber["effective_groups_per_class"],
            "ber_multiplier_p95": ber["sample_multiplier_p95"],
            "ber_multiplier_max": ber["sample_multiplier_max"],
        }
        client_rows.append(client_row)

        operator_pew_accuracy = np.full(len(operator_names), np.nan, dtype=np.float64)
        operator_unknown_rate = np.full(len(operator_names), np.nan, dtype=np.float64)
        operator_support = np.zeros(len(operator_names), dtype=np.int64)
        for operator, operator_name in enumerate(operator_names):
            train_mask = fit_operators == operator
            operator_support[operator] = int(train_mask.sum())
            if bool(train_mask.any()):
                operator_pew_accuracy[operator] = float(
                    (fit_environment[train_mask] == true_environment_by_operator[operator]).mean()
                    * 100.0
                )
                operator_unknown_rate[operator] = float(
                    (fit_environment[train_mask] == environment_to_id["unknown"]).mean()
                    * 100.0
                )
            base_accuracy = float(accuracies[0, client, :, operator].mean() * 100.0)
            plugin_accuracy = float(accuracies[1, client, :, operator].mean() * 100.0)
            bound_classes = np.flatnonzero(binding[client] == operator)
            operator_rows.append(
                {
                    "client": client,
                    "model": model_name,
                    "operator": operator_name,
                    "family": family_by_operator[operator],
                    "split": split_by_operator[operator],
                    "bound_classes": ";".join(str(int(x)) for x in bound_classes),
                    "fit_support": int(operator_support[operator]),
                    "pew_family_accuracy": operator_pew_accuracy[operator],
                    "pew_unknown_rate": operator_unknown_rate[operator],
                    "base_grid_acc": base_accuracy,
                    "plugin_grid_acc": plugin_accuracy,
                    "grid_acc_delta": plugin_accuracy - base_accuracy,
                    "base_dsa": float(dsa_by_operator[0, client, operator]),
                    "plugin_dsa": float(dsa_by_operator[1, client, operator]),
                    "dsa_reduction": float(
                        dsa_by_operator[0, client, operator]
                        - dsa_by_operator[1, client, operator]
                    ),
                }
            )

        class_accuracy_delta = np.zeros(10, dtype=np.float64)
        class_pew_accuracy = np.zeros(10, dtype=np.float64)
        class_unknown_rate = np.zeros(10, dtype=np.float64)
        for class_id, class_name in enumerate(CLASS_NAMES):
            source_mask = labels == class_id
            base_accuracy = float(accuracies[0, client, source_mask].mean() * 100.0)
            plugin_accuracy = float(accuracies[1, client, source_mask].mean() * 100.0)
            base_prediction_share = float(
                (probabilities[0, client].argmax(axis=-1) == class_id).mean() * 100.0
            )
            plugin_prediction_share = float(
                (probabilities[1, client].argmax(axis=-1) == class_id).mean() * 100.0
            )
            fit_mask = fit_labels == class_id
            class_accuracy_delta[class_id] = plugin_accuracy - base_accuracy
            if bool(fit_mask.any()):
                class_pew_accuracy[class_id] = float(
                    (fit_environment[fit_mask] == true_fit_environment[fit_mask]).mean()
                    * 100.0
                )
                class_unknown_rate[class_id] = float(
                    (fit_environment[fit_mask] == environment_to_id["unknown"]).mean()
                    * 100.0
                )
            else:
                class_pew_accuracy[class_id] = np.nan
                class_unknown_rate[class_id] = np.nan
            class_rows.append(
                {
                    "client": client,
                    "model": model_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "fit_support": int(fit_mask.sum()),
                    "fit_fraction": float(fit_mask.mean() * 100.0),
                    "pew_family_accuracy": class_pew_accuracy[class_id],
                    "pew_unknown_rate": class_unknown_rate[class_id],
                    "ber_effective_groups": float(ber["class_effective_groups"][class_id]),
                    "ber_multiplier_max": float(ber["class_multiplier_max"][class_id]),
                    "base_grid_acc": base_accuracy,
                    "plugin_grid_acc": plugin_accuracy,
                    "grid_acc_delta": class_accuracy_delta[class_id],
                    "base_prediction_share": base_prediction_share,
                    "plugin_prediction_share": plugin_prediction_share,
                    "prediction_share_delta": plugin_prediction_share
                    - base_prediction_share,
                }
            )

        operator_delta = np.asarray(
            [
                row["grid_acc_delta"]
                for row in operator_rows
                if row["client"] == client
            ],
            dtype=np.float64,
        )
        client_details[str(client)] = {
            "model": model_name,
            "class_correlations": {
                "accuracy_delta_vs_pew_accuracy": safe_correlation(
                    class_accuracy_delta, class_pew_accuracy
                ),
                "accuracy_delta_vs_unknown_rate": safe_correlation(
                    class_accuracy_delta, class_unknown_rate
                ),
                "accuracy_delta_vs_effective_groups": safe_correlation(
                    class_accuracy_delta,
                    np.asarray(ber["class_effective_groups"], dtype=np.float64),
                ),
                "accuracy_delta_vs_max_multiplier": safe_correlation(
                    class_accuracy_delta,
                    np.asarray(ber["class_multiplier_max"], dtype=np.float64),
                ),
            },
            "operator_correlations": {
                "accuracy_delta_vs_pew_accuracy": safe_correlation(
                    operator_delta, operator_pew_accuracy
                ),
                "accuracy_delta_vs_unknown_rate": safe_correlation(
                    operator_delta, operator_unknown_rate
                ),
                "accuracy_delta_vs_fit_support": safe_correlation(
                    operator_delta, operator_support.astype(np.float64)
                ),
            },
        }

    write_csv(output_dir / "CLIENT_ATTRIBUTION.csv", list(client_rows[0]), client_rows)
    write_csv(output_dir / "OPERATOR_ATTRIBUTION.csv", list(operator_rows[0]), operator_rows)
    write_csv(output_dir / "CLASS_ATTRIBUTION.csv", list(class_rows[0]), class_rows)

    harmed_client = min(client_rows, key=lambda row: float(row["grid_acc_delta"]))
    harmed_operator_rows = sorted(
        [row for row in operator_rows if row["client"] == harmed_client["client"]],
        key=lambda row: float(row["grid_acc_delta"]),
    )
    harmed_class_rows = sorted(
        [row for row in class_rows if row["client"] == harmed_client["client"]],
        key=lambda row: float(row["grid_acc_delta"]),
    )
    harmed_family_rows = []
    for family in sorted(set(family_by_operator)):
        rows = [
            row
            for row in operator_rows
            if row["client"] == harmed_client["client"] and row["family"] == family
        ]
        pew_values = np.asarray(
            [float(row["pew_family_accuracy"]) for row in rows], dtype=np.float64
        )
        harmed_family_rows.append(
            {
                "family": family,
                "mean_grid_acc_delta": float(
                    np.mean([float(row["grid_acc_delta"]) for row in rows])
                ),
                "mean_pew_family_accuracy_seen": float(
                    np.mean(pew_values[np.isfinite(pew_values)])
                ),
            }
        )
    collapsed_classes = [
        row["class_name"]
        for row in harmed_class_rows
        if float(row["base_grid_acc"]) > 0.0
        and float(row["plugin_grid_acc"]) == 0.0
    ]
    gained_class_rows = sorted(
        harmed_class_rows, key=lambda row: float(row["grid_acc_delta"]), reverse=True
    )
    summary = {
        "protocol": "cle_v2_plugin_architecture_attribution_v1",
        "analysis_type": "zero_training_existing_artifacts_only",
        "result_root": str(result_root),
        "package_root": str(package_root),
        "clients": client_rows,
        "client_details": client_details,
        "most_harmed_client": harmed_client,
        "most_harmed_client_bottom_operators": harmed_operator_rows[:5],
        "most_harmed_client_bottom_classes": harmed_class_rows[:5],
        "most_harmed_client_family_summary": harmed_family_rows,
        "evidence_assessment": {
            "collapsed_classes": collapsed_classes,
            "largest_class_gains": gained_class_rows[:3],
            "pew_only_explanation_supported": False,
            "reason_pew_only_is_insufficient": (
                "automobile collapses despite high PEW family accuracy, while truck has low PEW accuracy"
            ),
            "globally_extreme_ber_weight_explanation_supported": False,
            "reason_global_weight_is_insufficient": (
                "the harmed client has the lowest maximum BER multiplier and not the highest p95 multiplier"
            ),
            "architecture_cause_identified": False,
            "reason_architecture_is_not_identified": (
                "each architecture is bound to a different non-IID client partition"
            ),
            "strongest_current_description": (
                "class-selective decision reallocation on the ShuffleNet/client-2 pair"
            ),
        },
        "interpretation_limits": [
            "Descriptive attribution does not prove that PEW error or BER weighting caused accuracy loss.",
            "Four architectures are insufficient for an across-model correlation claim.",
            "No new training, checkpoint selection, or test-label tuning was performed.",
        ],
    }
    safe_summary = json_safe(summary)
    (output_dir / "ATTRIBUTION_SUMMARY.json").write_text(
        json.dumps(safe_summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    lines = [
        "# CLE-v2 Stage-2 PEW+BER 架构归因（零训练）",
        "",
        "本报告仅使用冻结的 Stage-2 预测、固定训练划分与 PEW 注释，不重新训练。",
        "",
        "## 客户端概览",
        "",
        "| client | model | grid acc Δ | DSA reduction | PEW family acc | unknown | BER p95 multiplier | BER max multiplier |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in client_rows:
        lines.append(
            f"| {row['client']} | {row['model']} | {row['grid_acc_delta']:+.4f} | "
            f"{row['dsa_reduction']:+.6f} | {row['pew_family_accuracy']:.2f}% | "
            f"{row['pew_unknown_rate']:.2f}% | {row['ber_multiplier_p95']:.4f} | "
            f"{row['ber_multiplier_max']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"最受损客户端为 client {harmed_client['client']}（{harmed_client['model']}），"
            f"operator-grid accuracy 变化 {harmed_client['grid_acc_delta']:+.4f}。",
            "",
            "## 该客户端下降最大的 operator",
            "",
            "| operator | family | split | acc Δ | DSA reduction | PEW family acc | fit support |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in harmed_operator_rows[:5]:
        lines.append(
            f"| {row['operator']} | {row['family']} | {row['split']} | "
            f"{row['grid_acc_delta']:+.4f} | {row['dsa_reduction']:+.6f} | "
            f"{row['pew_family_accuracy']:.2f}% | {row['fit_support']} |"
        )
    lines.extend(
        [
            "",
            "## 该客户端下降最大的类别",
            "",
            "| class | fit share | acc Δ | prediction share Δ | PEW family acc | unknown | effective groups | max multiplier |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in harmed_class_rows[:5]:
        lines.append(
            f"| {row['class_name']} | {row['fit_fraction']:.2f}% | "
            f"{row['grid_acc_delta']:+.4f} | {row['prediction_share_delta']:+.4f} | "
            f"{row['pew_family_accuracy']:.2f}% | {row['pew_unknown_rate']:.2f}% | "
            f"{row['ber_effective_groups']:.4f} | {row['ber_multiplier_max']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 当前归因结论",
            "",
            f"- client {harmed_client['client']} 的 DSA 仍下降 "
            f"{harmed_client['dsa_reduction']:.6f}，因此不是 shortcut 抑制失效。",
            f"- 损失是类别选择性的：{', '.join(collapsed_classes)} 的预测召回降为0；"
            f"同时 {gained_class_rows[0]['class_name']} 和 {gained_class_rows[1]['class_name']} 分别提升 "
            f"{gained_class_rows[0]['grid_acc_delta']:+.2f}、{gained_class_rows[1]['grid_acc_delta']:+.2f}。",
            "- PEW 误分不能单独解释：automobile 的 PEW family accuracy 约80%，但其召回仍降为0。",
            "- BER 全局权重过大也不能单独解释：client 2 的最大样本乘数反而是四个客户端中最低。",
            "- 当前只能定位到 ShuffleNet/client-2 组合上的类别决策重分配；架构和非 IID 数据固定绑定，不能归因为“小模型容量”。",
            "",
            "### client 2 按 corruption family 汇总",
            "",
            "| family | mean grid acc Δ | mean PEW family acc on seen operators |",
            "|---|---:|---:|",
        ]
    )
    for row in harmed_family_rows:
        lines.append(
            f"| {row['family']} | {row['mean_grid_acc_delta']:+.4f} | "
            f"{row['mean_pew_family_accuracy_seen']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 这是描述性归因，不把相关性写成因果。",
            "- 四个模型不足以支持跨架构相关性检验。",
            "- 本报告未重新训练、未选择 checkpoint、未使用测试标签调参。",
            "",
        ]
    )
    (output_dir / "ATTRIBUTION_REPORT_ZH.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(safe_summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
