from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.engine.ber_theory import (  # noqa: E402
    ber_sample_weights,
    class_balanced_sample_weights,
    class_environment_counts,
    environment_advantage_tv_bound_holds,
    environment_only_bayes_advantage,
    maximum_within_class_support_ratio,
    mutual_information,
    pseudo_to_true_dependence_bound,
    total_variation_dependence,
    weighted_joint,
)


SUPPORT_GAMMA = 0.5
COUNT_CAP = 32
MIN_GROUP_COUNT = 2
NUM_CLASSES = 10
NUM_PEW_ENVIRONMENTS = 6
FAMILY_TO_PEW = {"noise": 1, "blur": 2, "weather": 3, "digital": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only BER effective-distribution audit.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def metrics(joint: np.ndarray) -> dict[str, float | bool]:
    return {
        "tv_dependence": total_variation_dependence(joint),
        "mutual_information_nats": mutual_information(joint),
        "environment_only_bayes_advantage": environment_only_bayes_advantage(joint),
        "maximum_within_class_support_ratio": maximum_within_class_support_ratio(joint),
        "bayes_advantage_le_tv": environment_advantage_tv_bound_holds(joint),
    }


def true_family_ids(operator_ids: np.ndarray, metadata: dict) -> np.ndarray:
    id_to_name = {int(value): key for key, value in metadata["operator_to_id"].items()}
    return np.asarray(
        [FAMILY_TO_PEW[metadata["operator_families"][id_to_name[int(value)]]] for value in operator_ids],
        dtype=np.int64,
    )


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((package_root / "data/gamma09/metadata.json").read_text(encoding="utf-8"))
    split = np.load(package_root / "splits/strict_cle_v2_factorial_seed0_split0.npz")

    client_results = {}
    identity_errors = []
    pseudo_decreases = []
    true_decreases = []
    for client_id in range(4):
        data_root = package_root / "data/gamma09" / f"client_{client_id}"
        annotations = np.load(
            package_root / "pew_standard/annotations/gamma09" / f"client_{client_id}.npz"
        )
        fit = np.asarray(split[f"client_{client_id}_fit"], dtype=np.int64)
        labels = np.load(data_root / "train_labels.npy").astype(np.int64)[fit]
        pseudo = np.asarray(annotations["environment_ids"], dtype=np.int64)[fit]
        operators = np.load(data_root / "train_corruption_ids.npy").astype(np.int64)[fit]
        true_environment = true_family_ids(operators, metadata)

        counts = class_environment_counts(
            labels,
            pseudo,
            num_classes=NUM_CLASSES,
            num_environments=NUM_PEW_ENVIRONMENTS,
        )
        reference_weights = class_balanced_sample_weights(labels, num_classes=NUM_CLASSES)
        ber_weights, environment_weights, valid = ber_sample_weights(
            labels,
            pseudo,
            counts,
            support_gamma=SUPPORT_GAMMA,
            count_cap=COUNT_CAP,
            min_group_count=MIN_GROUP_COUNT,
        )

        reference_pseudo_joint = weighted_joint(
            labels, pseudo, reference_weights,
            num_classes=NUM_CLASSES, num_environments=NUM_PEW_ENVIRONMENTS,
        )
        ber_pseudo_joint = weighted_joint(
            labels, pseudo, ber_weights,
            num_classes=NUM_CLASSES, num_environments=NUM_PEW_ENVIRONMENTS,
        )
        reference_true_joint = weighted_joint(
            labels, true_environment, reference_weights,
            num_classes=NUM_CLASSES, num_environments=NUM_PEW_ENVIRONMENTS,
        )
        ber_true_joint = weighted_joint(
            labels, true_environment, ber_weights,
            num_classes=NUM_CLASSES, num_environments=NUM_PEW_ENVIRONMENTS,
        )

        expected_pseudo_joint = environment_weights / float(valid.any(axis=1).sum())
        identity_error = float(np.abs(ber_pseudo_joint - expected_pseudo_joint).max())
        mismatch = pseudo != true_environment
        unweighted_pew_error = float(np.mean(mismatch))
        reference_weighted_pew_error = float(np.sum(reference_weights * mismatch))
        weighted_pew_error = float(np.sum(ber_weights * mismatch))
        reference_pseudo = metrics(reference_pseudo_joint)
        effective_pseudo = metrics(ber_pseudo_joint)
        reference_true = metrics(reference_true_joint)
        effective_true = metrics(ber_true_joint)
        pseudo_drop = float(reference_pseudo["tv_dependence"] - effective_pseudo["tv_dependence"])
        true_drop = float(reference_true["tv_dependence"] - effective_true["tv_dependence"])
        identity_errors.append(identity_error)
        pseudo_decreases.append(pseudo_drop > 0)
        true_decreases.append(true_drop > 0)
        client_results[str(client_id)] = {
            "fit_samples": int(len(fit)),
            "valid_groups": int(valid.sum()),
            "valid_classes": int(valid.any(axis=1).sum()),
            "implementation_identity_max_error": identity_error,
            "reference_class_balanced": {
                "pseudo_environment": reference_pseudo,
                "true_family": reference_true,
            },
            "ber_effective_distribution": {
                "pseudo_environment": effective_pseudo,
                "true_family": effective_true,
            },
            "pseudo_tv_absolute_drop": pseudo_drop,
            "pseudo_tv_relative_drop": pseudo_drop / float(reference_pseudo["tv_dependence"]),
            "true_tv_absolute_drop": true_drop,
            "true_tv_relative_drop": true_drop / float(reference_true["tv_dependence"]),
            "pew_family_error": {
                "unweighted_fit": unweighted_pew_error,
                "class_balanced_reference": reference_weighted_pew_error,
                "ber_effective_distribution": weighted_pew_error,
            },
            "true_tv_conditional_upper_bound": pseudo_to_true_dependence_bound(
                float(effective_pseudo["tv_dependence"]), weighted_pew_error
            ),
        }

    pooled_reference_pseudo = float(np.mean([
        value["reference_class_balanced"]["pseudo_environment"]["tv_dependence"]
        for value in client_results.values()
    ]))
    pooled_ber_pseudo = float(np.mean([
        value["ber_effective_distribution"]["pseudo_environment"]["tv_dependence"]
        for value in client_results.values()
    ]))
    pooled_reference_true = float(np.mean([
        value["reference_class_balanced"]["true_family"]["tv_dependence"]
        for value in client_results.values()
    ]))
    pooled_ber_true = float(np.mean([
        value["ber_effective_distribution"]["true_family"]["tv_dependence"]
        for value in client_results.values()
    ]))
    pseudo_relative_drop = (pooled_reference_pseudo - pooled_ber_pseudo) / pooled_reference_pseudo
    gates = {
        "T0_code_theory_identity_error_le_1e_10": max(identity_errors) <= 1e-10,
        "T1_pseudo_dependence_decreases_4_of_4": all(pseudo_decreases),
        "T2_pooled_pseudo_tv_relative_drop_ge_0p50": pseudo_relative_drop >= 0.50,
        "T3_true_family_dependence_decreases_4_of_4": all(true_decreases),
    }
    result = {
        "protocol": "ber_effective_distribution_mechanism_audit_v1",
        "training_or_inference_run": False,
        "data_scope": "strict fit only; gamma09; fixed cle_hfl_v2 seed0_split0",
        "parameters": {
            "support_gamma": SUPPORT_GAMMA,
            "count_cap": COUNT_CAP,
            "min_group_count": MIN_GROUP_COUNT,
        },
        "reference": "class-balanced empirical distribution before environment reweighting",
        "clients": client_results,
        "pooled_equal_client": {
            "pseudo_tv_before": pooled_reference_pseudo,
            "pseudo_tv_after": pooled_ber_pseudo,
            "pseudo_tv_relative_drop": pseudo_relative_drop,
            "true_family_tv_before": pooled_reference_true,
            "true_family_tv_after": pooled_ber_true,
            "true_family_tv_relative_drop": (
                pooled_reference_true - pooled_ber_true
            ) / pooled_reference_true,
        },
        "frozen_gates": gates,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "limitations": [
            "This is a CPU distribution audit, not a training result.",
            "BER balances PEW pseudo-environments; the true-family conclusion is reporting-only.",
            "Dependence contraction does not by itself guarantee zero DSA or higher accuracy.",
            "The PEW-error upper bound is conditional and may be loose.",
        ],
    }
    (output_dir / "BER_MECHANISM_AUDIT.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = [
        "# BER 有效分布与 CLE 失衡压缩审计",
        "",
        "本审计只读取固定 strict-fit 标签、冻结PEW标注和报告专用真实operator metadata；不训练、",
        "不推理、不使用GPU。真实metadata不进入方法，只用于检验理论到真实family的连接。",
        "",
        "## Pooled结果",
        "",
        f"- pseudo-environment TV: `{pooled_reference_pseudo:.6f} -> {pooled_ber_pseudo:.6f}` "
        f"（相对下降 `{100.0 * pseudo_relative_drop:.2f}%`）。",
        f"- true-family TV: `{pooled_reference_true:.6f} -> {pooled_ber_true:.6f}` "
        f"（相对下降 `{100.0 * (pooled_reference_true - pooled_ber_true) / pooled_reference_true:.2f}%`）。",
        f"- code/theory identity最大误差：`{max(identity_errors):.3e}`。",
        f"- 冻结门槛：`{json.dumps(gates, ensure_ascii=False)}`。",
        f"- verdict: `{result['verdict']}`。",
        "",
        "## 边界",
        "",
        "该结果证明当前BER权重确实构造了理论定义的有效分布，并检验类别—环境依赖是否被压缩；",
        "它不证明训练后的DSA必然下降，也不替代现有Formal A/B。",
    ]
    (output_dir / "BER_MECHANISM_AUDIT_ZH.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
