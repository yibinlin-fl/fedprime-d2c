from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.engine.cle_v2_factorial import compute_operator_dsa  # noqa: E402
from fedprime.engine.dsa_theory import (  # noqa: E402
    categorical_jsd,
    operator_exchangeable_projection,
    probability_mixture,
)


ARMS = ("h0_b", "h9_b", "l0_b", "l9_b")
MIXTURE_WEIGHTS = np.linspace(0.0, 1.0, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate three frozen DSA propositions from cache.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = np.load(args.predictions.resolve(), allow_pickle=False)
    probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
    arms = tuple(str(value) for value in archive["arms"])
    labels = np.asarray(archive["labels"], dtype=np.int64)
    binding = np.asarray(archive["binding"], dtype=np.int64)
    if arms != ARMS or probabilities.shape != (4, 4, 1000, 15, 10):
        raise ValueError("Unexpected Stage-1 prediction cache contract")

    dsa = {
        arm: float(compute_operator_dsa(probabilities[index], labels, binding).pooled)
        for index, arm in enumerate(arms)
    }

    exchangeable = operator_exchangeable_projection(probabilities[1])
    exchangeable_dsa = float(compute_operator_dsa(exchangeable, labels, binding).pooled)
    empirical_gamma0_max_abs = max(abs(dsa["h0_b"]), abs(dsa["l0_b"]))
    proposition_1 = {
        "statement": "operator-conditional exchangeability implies DSA=0",
        "constructed_exchangeable_dsa": exchangeable_dsa,
        "empirical_gamma0": {"h0_b": dsa["h0_b"], "l0_b": dsa["l0_b"]},
        "empirical_gamma0_max_abs": empirical_gamma0_max_abs,
        "gates": {
            "constructed_abs_dsa_le_1e-12": abs(exchangeable_dsa) <= 1.0e-12,
            "empirical_gamma0_abs_le_0p01": empirical_gamma0_max_abs <= 0.01,
        },
    }

    linearity = {}
    maximum_linearity_error = 0.0
    monotone_pairs = True
    for system, clean_arm, shortcut_arm in (
        ("hfl", "h0_b", "h9_b"),
        ("local", "l0_b", "l9_b"),
    ):
        clean = probabilities[arms.index(clean_arm)]
        shortcut = probabilities[arms.index(shortcut_arm)]
        rows = []
        values = []
        for weight in MIXTURE_WEIGHTS:
            mixture = probability_mixture(clean, shortcut, float(weight))
            observed = float(compute_operator_dsa(mixture, labels, binding).pooled)
            expected = (1.0 - float(weight)) * dsa[clean_arm] + float(weight) * dsa[shortcut_arm]
            error = abs(observed - expected)
            maximum_linearity_error = max(maximum_linearity_error, error)
            values.append(observed)
            rows.append(
                {
                    "lambda": float(weight),
                    "observed_dsa": observed,
                    "linear_prediction": expected,
                    "absolute_error": error,
                }
            )
        monotone = bool(np.all(np.diff(values) > 0.0))
        monotone_pairs = monotone_pairs and monotone
        linearity[system] = {"monotone_increasing": monotone, "curve": rows}
    proposition_2 = {
        "statement": "DSA is affine under semantic-shortcut probability mixtures",
        "systems": linearity,
        "maximum_linearity_error": maximum_linearity_error,
        "gates": {
            "maximum_error_le_1e-12": maximum_linearity_error <= 1.0e-12,
            "both_cached_pairs_monotone": monotone_pairs,
        },
    }

    h9 = probabilities[arms.index("h9_b")]
    identical_views = np.stack([h9, h9, h9], axis=0)
    jsd = categorical_jsd(identical_views)
    counterexample_jsd = float(jsd.max())
    counterexample_dsa = dsa["h9_b"]
    proposition_3 = {
        "statement": "zero within-operator view JSD does not imply zero cross-operator DSA",
        "constructed_max_jsd": counterexample_jsd,
        "cached_h9_dsa": counterexample_dsa,
        "gates": {
            "max_jsd_le_1e-12": counterexample_jsd <= 1.0e-12,
            "dsa_ge_0p10": counterexample_dsa >= 0.10,
        },
    }

    all_gates = [
        *proposition_1["gates"].values(),
        *proposition_2["gates"].values(),
        *proposition_3["gates"].values(),
    ]
    summary = {
        "protocol": "cle_dsa_theory_cache_validation_v1",
        "input": str(args.predictions.resolve()),
        "training_or_inference_run": False,
        "cached_dsa": dsa,
        "proposition_1": proposition_1,
        "proposition_2": proposition_2,
        "proposition_3": proposition_3,
        "all_frozen_gates_pass": bool(all(all_gates)),
        "interpretation": (
            "Numerical validation supports the algebra and provides an empirical counterexample; "
            "it does not replace the stated assumptions or establish cross-scenario generality."
        ),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "DSA_THEORY_VALIDATION.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# DSA 三个理论命题与缓存验证",
        "",
        "本验证只读取冻结的 Stage-1 概率缓存，不训练、不推理、不使用 GPU。",
        "",
        "## 结论",
        "",
        f"- 命题1：exchangeable projection DSA = `{exchangeable_dsa:.3e}`；"
        f"gamma0最大绝对DSA = `{empirical_gamma0_max_abs:.6f}`。",
        f"- 命题2：11点混合曲线最大线性误差 = `{maximum_linearity_error:.3e}`；"
        f"HFL/Local均严格单调 = `{monotone_pairs}`。",
        f"- 命题3：三个完全一致视图的最大JSD = `{counterexample_jsd:.3e}`，"
        f"同时跨operator DSA = `{counterexample_dsa:.6f}`。",
        f"- 冻结门槛总判定：`{'PASS' if all(all_gates) else 'FAIL'}`。",
        "",
        "## 证据边界",
        "",
        "- 命题1和命题2来自DSA作为预测概率线性泛函的代数性质；缓存验证是数值审计。",
        "- 命题3是反例：同一operator附近增强一致，不约束不同operator之间的binding方向。",
        "- gamma0接近0是固定场景经验支持，不是任意数据分布下的无条件保证。",
        "",
    ]
    (output_dir / "DSA_THEORY_VALIDATION_ZH.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
