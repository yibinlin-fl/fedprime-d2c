from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.engine.cle_probe_directional_promotion import (  # noqa: E402
    decide_zero_training_gate,
    probe_directional_promotion,
    score_binding_retrieval,
    shuffled_retrieval_nulls,
)


ARM_NAMES = ("h0", "h9", "l0", "l9")
DEFAULT_ROOT = (
    ROOT
    / "outputs/openi_downloads/cle_shortcut_amplification_phase_a1a_seed0/extracted"
    / "outputs/cle_shortcut_amplification_phase_a1a_seed0_analysis"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the zero-training CLE PIDR observability gate.")
    parser.add_argument("--round40", type=Path, default=DEFAULT_ROOT / "round_040_predictions.npz")
    parser.add_argument("--round12", type=Path, default=DEFAULT_ROOT / "round_012_predictions.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "deliverables/cle_pidr_zero_training_gate_20260830",
    )
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def analyze_cache(path: Path, *, permutations: int, seed: int) -> tuple[dict[str, object], np.ndarray, dict[str, np.ndarray]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    # Blind estimation phase: binding and family metadata are deliberately not read.
    with np.load(path, allow_pickle=False) as archive:
        probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        operator_names = np.asarray(archive["operator_names"])
    if probabilities.shape[:2] != (4, 4):
        raise ValueError(f"expected [4 arms,4 clients,...], got {probabilities.shape}")
    promotions = {
        arm: probe_directional_promotion(probabilities[index], labels)
        for index, arm in enumerate(ARM_NAMES)
    }

    # Scoring phase: hidden truth is opened only after all promotion matrices exist.
    with np.load(path, allow_pickle=False) as archive:
        binding = np.asarray(archive["binding"], dtype=np.int64)
        family_ids = np.asarray(archive["operator_family_ids"], dtype=np.int64)

    arm_results: dict[str, dict[str, object]] = {}
    for index, arm in enumerate(ARM_NAMES):
        retrieval = score_binding_retrieval(promotions[arm].matrix, binding, family_ids)
        nulls = shuffled_retrieval_nulls(
            promotions[arm].matrix,
            binding,
            family_ids,
            permutations=permutations,
            seed=seed + 1009 * index,
        )
        arm_results[arm] = {
            "pidr": promotions[arm].pooled,
            "pidr_client": promotions[arm].client.tolist(),
            "retrieval": retrieval,
            "nulls": nulls,
        }
    decision = decide_zero_training_gate(arm_results)
    return (
        {
            "prediction_cache": str(path),
            "shape": list(probabilities.shape),
            "estimator_inputs": ["probabilities", "labels", "operator identity by tensor position"],
            "estimator_forbidden_inputs": ["binding", "operator_family_ids"],
            "arms": arm_results,
            "decision": decision,
        },
        np.stack([promotions[arm].matrix for arm in ARM_NAMES], axis=0),
        {
            "operator_names": operator_names,
            "binding": binding,
            "operator_family_ids": family_ids,
        },
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(round40: dict[str, object], round12: dict[str, object]) -> str:
    lines = [
        "# CLE PIDR Zero-Training Gate Result",
        "",
        "Updated: 2026-08-30",
        "",
        "This gate reused cached softmax predictions only; no inference or training was run.",
        "Binding and operator-family metadata were hidden during promotion-matrix estimation and opened only for scoring.",
        "",
        "## Round 40 primary",
        "",
        "| arm | PIDR | mAP | AUC | positive precision | positive recall | class-to-family hit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_NAMES:
        result = round40["arms"][arm]
        retrieval = result["retrieval"]
        lines.append(
            f"| {arm} | {result['pidr']:.6f} | {retrieval['mean_average_precision']:.6f} | "
            f"{retrieval['roc_auc']:.6f} | {retrieval['positive_precision']:.6f} | "
            f"{retrieval['positive_recall']:.6f} | {retrieval['class_to_probe_family_hit_rate']:.6f} |"
        )
    lines.extend(["", "```json", json.dumps(round40["decision"], indent=2), "```", "", "## Round 12 diagnostic", ""])
    for arm in ARM_NAMES:
        result = round12["arms"][arm]
        lines.append(
            f"- `{arm}`: PIDR={result['pidr']:.6f}, "
            f"mAP={result['retrieval']['mean_average_precision']:.6f}, "
            f"hit={result['retrieval']['class_to_probe_family_hit_rate']:.6f}"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "A pass establishes only oracle-side directional observability with clean paired sources and distinguishable probes.",
            "It does not establish that ordinary i.i.d. AugMix views overwrite an already-present degradation, nor method novelty.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    round40, matrix40, truth40 = analyze_cache(
        args.round40,
        permutations=int(args.permutations),
        seed=int(args.seed),
    )
    round12, matrix12, truth12 = analyze_cache(
        args.round12,
        permutations=int(args.permutations),
        seed=int(args.seed) + 12,
    )
    if not np.array_equal(truth40["binding"], truth12["binding"]):
        raise ValueError("round12/round40 binding mismatch")
    if not np.array_equal(truth40["operator_family_ids"], truth12["operator_family_ids"]):
        raise ValueError("round12/round40 probe-family mismatch")

    summary = {
        "protocol": "cle_pidr_zero_training_gate_20260830",
        "no_training": True,
        "no_inference": True,
        "round40_primary": round40,
        "round12_diagnostic": round12,
        "formal_verdict": round40["decision"]["verdict"],
    }
    (output_dir / "cle_pidr_zero_training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "cle_pidr_promotion_matrices.npz",
        round40=matrix40,
        round12=matrix12,
        arm_names=np.asarray(ARM_NAMES),
        operator_names=truth40["operator_names"],
        binding=truth40["binding"],
        operator_family_ids=truth40["operator_family_ids"],
    )

    rows: list[dict[str, object]] = []
    for round_name, result in (("round40", round40), ("round12", round12)):
        for arm in ARM_NAMES:
            retrieval = result["arms"][arm]["retrieval"]
            rows.append(
                {
                    "round": round_name,
                    "arm": arm,
                    "pidr": result["arms"][arm]["pidr"],
                    "map": retrieval["mean_average_precision"],
                    "auc": retrieval["roc_auc"],
                    "positive_precision": retrieval["positive_precision"],
                    "positive_recall": retrieval["positive_recall"],
                    "class_to_probe_family_hit_rate": retrieval["class_to_probe_family_hit_rate"],
                    "class_map_p": result["arms"][arm]["nulls"]["class_map_null"]["mean_average_precision"]["one_sided_p"],
                    "probe_identity_p": result["arms"][arm]["nulls"]["probe_identity_null"]["mean_average_precision"]["one_sided_p"],
                }
            )
    write_csv(output_dir / "cle_pidr_per_arm.csv", rows)
    (output_dir / "RESULT_SUMMARY_ZH.md").write_text(
        markdown_summary(round40, round12),
        encoding="utf-8",
    )
    print(json.dumps(round40["decision"], indent=2), flush=True)
    print(f"[complete] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
