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
    add_operator_invariant_probability_shift,
    binding_aligned_distribution,
    categorical_jsd,
    operator_exchangeable_projection,
    probability_mixture,
    source_level_hoeffding_radius,
)
from fedprime.engine.cle_v2_factorial import shuffled_binding_null  # noqa: E402


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

    clients, sources, operators, classes = probabilities.shape[1:]
    uniform_semantic = np.full(
        (clients, sources, operators, classes),
        1.0 / float(classes),
        dtype=np.float64,
    )
    aligned = binding_aligned_distribution(
        binding,
        sources=sources,
        operators=operators,
        classes=classes,
    )
    aligned_dsa = float(compute_operator_dsa(aligned, labels, binding).pooled)
    injection_rows = []
    maximum_injection_error = 0.0
    for strength in MIXTURE_WEIGHTS:
        injected = probability_mixture(uniform_semantic, aligned, float(strength))
        observed = float(compute_operator_dsa(injected, labels, binding).pooled)
        expected = float(strength) * aligned_dsa
        error = abs(observed - expected)
        maximum_injection_error = max(maximum_injection_error, error)
        injection_rows.append(
            {
                "injected_strength": float(strength),
                "observed_dsa": observed,
                "known_linear_target": expected,
                "absolute_error": error,
            }
        )
    injected_quarter = probability_mixture(uniform_semantic, aligned, 0.25)
    true_dsa_before_shift = float(
        compute_operator_dsa(injected_quarter, labels, binding).pooled
    )
    class_pattern = np.linspace(-0.004, 0.004, classes, dtype=np.float64)
    class_pattern -= class_pattern.mean()
    common_shift = np.broadcast_to(
        class_pattern[None, None, :], (clients, sources, classes)
    ).copy()
    shifted = add_operator_invariant_probability_shift(injected_quarter, common_shift)
    true_dsa_after_shift = float(compute_operator_dsa(shifted, labels, binding).pooled)
    shift_invariance_error = abs(true_dsa_after_shift - true_dsa_before_shift)
    injected_null = shuffled_binding_null(
        injected_quarter,
        labels,
        binding,
        permutations=1000,
        seed=20260911,
    )
    proposition_4 = {
        "statement": (
            "paired DSA cancels operator-invariant semantic probability shifts and "
            "recovers controlled binding-aligned response strength"
        ),
        "aligned_endpoint_dsa": aligned_dsa,
        "injection_curve": injection_rows,
        "maximum_known_strength_recovery_error": maximum_injection_error,
        "operator_invariant_shift": {
            "before": true_dsa_before_shift,
            "after": true_dsa_after_shift,
            "absolute_error": shift_invariance_error,
        },
        "binding_specificity": {
            "observed": float(injected_null["observed"]),
            "null_p95": float(injected_null["null_p95"]),
            "p_value": float(injected_null["p_value"]),
        },
        "gates": {
            "known_strength_max_error_le_1e-12": maximum_injection_error <= 1.0e-12,
            "operator_invariant_shift_error_le_1e-12": shift_invariance_error <= 1.0e-12,
            "aligned_response_exceeds_shuffled_null": bool(
                float(injected_null["observed"]) > float(injected_null["null_p95"])
                and float(injected_null["p_value"]) <= 0.01
            ),
        },
    }

    hoeffding_radius = source_level_hoeffding_radius(sources, delta=0.05)
    proposition_5 = {
        "statement": (
            "independent source-level effects in [-1,1] admit a conservative "
            "two-sided Hoeffding confidence radius"
        ),
        "source_count": int(sources),
        "delta": 0.05,
        "bounded_interval": [-1.0, 1.0],
        "hoeffding_radius": hoeffding_radius,
        "resampling_unit": "source",
        "warning": (
            "operator images from the same source are paired and must not be treated as "
            "independent bootstrap observations"
        ),
        "gates": {
            "finite_positive_radius": bool(np.isfinite(hoeffding_radius) and hoeffding_radius > 0.0),
            "source_contract_is_1000": int(sources) == 1000,
        },
    }

    all_gates = [
        *proposition_1["gates"].values(),
        *proposition_2["gates"].values(),
        *proposition_3["gates"].values(),
        *proposition_4["gates"].values(),
        *proposition_5["gates"].values(),
    ]
    summary = {
        "protocol": "cle_dsa_identification_cache_validation_v2",
        "input": str(args.predictions.resolve()),
        "training_or_inference_run": False,
        "cached_dsa": dsa,
        "proposition_1": proposition_1,
        "proposition_2": proposition_2,
        "proposition_3": proposition_3,
        "proposition_4": proposition_4,
        "proposition_5": proposition_5,
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
        "# DSA 识别理论五项命题与缓存验证",
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
        f"- 命题4：已知shortcut强度恢复最大误差 = `{maximum_injection_error:.3e}`；"
        f"operator-invariant偏移不变性误差 = `{shift_invariance_error:.3e}`；"
        f"shuffled-binding p = `{float(injected_null['p_value']):.6f}`。",
        f"- 命题5：n={sources}、delta=0.05、source effect位于[-1,1]时，"
        f"保守Hoeffding半径 = `{hoeffding_radius:.6f}`。",
        f"- 冻结门槛总判定：`{'PASS' if all(all_gates) else 'FAIL'}`。",
        "",
        "## 证据边界",
        "",
        "- 命题1和命题2来自DSA作为预测概率线性泛函的代数性质；缓存验证是数值审计。",
        "- 命题3是反例：同一operator附近增强一致，不约束不同operator之间的binding方向。",
        "- gamma0接近0是固定场景经验支持，不是任意数据分布下的无条件保证。",
        "- 命题4使用人工注入的已知binding-aligned概率响应，验证识别方向而非现实生成充分性。",
        "- 命题5要求独立source；同一source的operator图像必须作为配对簇共同重采样。",
        "",
    ]
    (output_dir / "DSA_THEORY_VALIDATION_ZH.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
