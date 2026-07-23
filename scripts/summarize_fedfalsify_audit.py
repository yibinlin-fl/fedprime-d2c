from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TOKENS = ("00", "06", "09")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the FedFalsify offline audit.")
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("outputs/fedfalsify_audit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("deliverables/fedfalsify_offline_audit"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    foreign = read_json(args.audit_root / "foreign_tensor" / "foreign_tensor_summary.json")
    coverage10 = read_json(args.audit_root / "gate_coverage" / "gate_coverage_summary.json")
    coverage5 = read_json(
        args.audit_root / "gate_coverage_min5" / "gate_coverage_summary.json"
    )
    ranking = read_json(
        args.audit_root / "source_ranking" / "source_ranking_summary.json"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for token in TOKENS:
        one_step = read_json(
            args.audit_root / "one_step" / f"gamma{token}_one_step_summary.json"
        )
        foreign_row = foreign["gammas"][token]
        coverage10_row = coverage10["gammas"][token]
        coverage5_row = coverage5["gammas"][token]
        incremental = one_step["incremental_effects_over_ce"]
        rows.append({
            "gamma": float(one_step["gamma"]),
            "foreign_survival_gap": float(foreign_row["mean_foreign_survival_gap"]),
            "positive_survival_gap_fraction": float(
                foreign_row["positive_gap_fraction"]
            ),
            "count_auditable_fraction_min10": float(
                coverage10_row["count_auditable_fraction"]
            ),
            "projected_sample_activation_min10": float(
                coverage10_row["mean_projected_sample_activation_rate"]
            ),
            "count_auditable_fraction_min5": float(
                coverage5_row["count_auditable_fraction"]
            ),
            "projected_sample_activation_min5": float(
                coverage5_row["mean_projected_sample_activation_rate"]
            ),
            "fra_activation": float(one_step["advantage_gate_activation_rate"]),
            "tau_activation": float(one_step["utility_positive_rate"]),
            "full_gate_activation": float(one_step["full_gate_activation_rate"]),
            "direct_kd_increment_over_ce": float(incremental["direct_kd_mean"]),
            "cmt_increment_over_ce": float(incremental["cmt_mean"]),
            "tau_selected_cmt_increment": float(
                incremental["cmt_utility_selected_mean"]
            ),
            "full_selected_cmt_increment": float(
                incremental["cmt_full_selected_mean"]
            ),
            "tau_increment_precision": float(
                one_step["utility_gate_precision_for_cmt_increment_over_ce"]
            ),
            "tau_increment_recall": float(
                one_step["utility_gate_recall_for_cmt_increment_over_ce"]
            ),
            "full_increment_precision": float(
                one_step["full_gate_precision_for_cmt_increment_over_ce"]
            ),
        })

    csv_path = args.output_dir / "fedfalsify_offline_audit_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# FedFalsify v0.1 离线审计摘要",
        "",
        "> 这里使用独立 `test_same` 标签做离线 Go/No-Go 审计，绝不能把这些标签接入正式训练路由。",
        "",
        "## 核心结果",
        "",
        "| gamma | 外来环境生存差距 | min=5 预计样本激活率 | FRA+TAU 组合激活率 | 直接 KD 相对 CE 净变化 | CMT 相对 CE 净变化 | TAU 精度/召回 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                fmt(row["gamma"], 1),
                fmt(row["foreign_survival_gap"]),
                fmt(row["projected_sample_activation_min5"]),
                fmt(row["full_gate_activation"]),
                fmt(row["direct_kd_increment_over_ce"]),
                fmt(row["cmt_increment_over_ce"]),
                (
                    f"{fmt(row['tau_increment_precision'])}/"
                    f"{fmt(row['tau_increment_recall'])}"
                ),
            ])
            + " |"
        )

    lines.extend([
        "",
        "## 冻结判断",
        "",
        "1. **问题信号成立。** 随着纠缠增强，模型对自己环境的相对优势显著扩大，外来客户端知识更容易失效。",
        "2. **直接 peer KD 明显不安全。** 它相对本地 CE 的净影响为负，且随 gamma 增大进一步恶化。",
        "3. **CMT 方向正确但效应很小。** 三档 gamma 都只有约 0.0017-0.0019 的平均审计损失净改善。",
        "4. **TAU 有筛选价值，但计算代价高。** 一步审计中精度约 0.86-0.90、召回率为 1.00。",
        "5. **FRA 是当前瓶颈。** 强纠缠时完整组合只覆盖约 14% 的 source-receiver-class，投影到训练样本约 4.5%。",
        "",
        "因此，FedFalsify v0.1 **不应直接进入昂贵的 40 轮训练**。",
        "",
        "## v0.2 Source Ranking Audit",
        "",
        "| gamma | policy | coverage | positive precision | selected mean increment | policy mean including abstention |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    ranking_rows = ranking["policies"]
    for row in ranking_rows:
        if row["policy"] not in {
            "foreign_accuracy_top1",
            "fra_top1",
            "tau_top1",
            "head_tau_top1",
            "oracle_increment_top1",
        }:
            continue
        agreement = row.get("source_agreement_with_full_tau")
        policy_name = row["policy"]
        if agreement is not None:
            policy_name += f" (agree={fmt(agreement)})"
        lines.append(
            "| "
            + " | ".join([
                fmt(row["gamma"], 1),
                policy_name,
                fmt(row["coverage"]),
                fmt(row["positive_precision"]),
                fmt(row["mean_selected_increment"], 6),
                fmt(row["mean_policy_increment_including_abstention"], 6),
            ])
            + " |"
        )

    lines.extend([
        "",
        "`TAU Top-1` 在三档 gamma 都覆盖全部可审计 receiver-class，并且接近离线 oracle。FRA Top-1 的单次增益较高，但覆盖率随纠缠增强降至 31.4%。因此 v0.2 应使用 TAU 作为安全门和来源排序依据，FRA 只作软先验或平局处理。",
        "",
        "Head-only TAU 的来源一致率为 80.0% / 62.9% / 74.3%，正净增益精度为 91.4% / 91.4% / 85.7%。它略逊于 full TAU，但分类头参数只占各模型约 0.067%-0.224%，适合作为 12 轮 probe 的默认低成本版本。",
        "",
        "这仍然只是一步更新审计，不等价于 12 轮收益证明。",
        "",
        "## 复现实验",
        "",
        "```powershell",
        "conda run -n pytorch python -u scripts/audit_fedfalsify_foreign_tensor.py --batch-size 512",
        "conda run -n pytorch python -u scripts/audit_fedfalsify_gate_coverage.py --bootstrap-repeats 200 --min-audit-per-class 5 --output-dir outputs/fedfalsify_audit/gate_coverage_min5",
        "conda run -n pytorch python -u scripts/audit_fedfalsify_one_step.py --gamma 00",
        "conda run -n pytorch python -u scripts/audit_fedfalsify_one_step.py --gamma 06",
        "conda run -n pytorch python -u scripts/audit_fedfalsify_one_step.py --gamma 09",
        "conda run -n pytorch python -u scripts/audit_fedfalsify_source_ranking.py",
        "conda run -n pytorch python -u scripts/summarize_fedfalsify_audit.py",
        "```",
        "",
    ])
    report_path = args.output_dir / "FEDFALSIFY_OFFLINE_AUDIT_ZH.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
