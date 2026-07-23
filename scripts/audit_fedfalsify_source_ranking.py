from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


TOKENS = ("00", "06", "09")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FedFalsify v0.2 source-ranking policies offline."
    )
    parser.add_argument(
        "--one-step-dir",
        type=Path,
        default=Path("outputs/fedfalsify_audit/one_step"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fedfalsify_audit/source_ranking"),
    )
    parser.add_argument("--tokens", nargs="+", choices=TOKENS, default=list(TOKENS))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric_fields = (
        "gamma",
        "source_client",
        "receiver_client",
        "class_id",
        "source_accuracy",
        "receiver_accuracy",
        "advantage_strength",
        "action_utility",
        "head_action_utility",
        "ce_delta",
        "cmt_delta",
    )
    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])
        row["advantage_gate_active"] = bool(int(row["advantage_gate_active"]))
        row["cmt_increment"] = row["cmt_delta"] - row["ce_delta"]
    return rows


def select_candidate(policy: str, candidates: list[dict]) -> dict | None:
    if policy == "tau_top1":
        eligible = [row for row in candidates if row["action_utility"] > 0.0]
        key = lambda row: row["action_utility"]
    elif policy == "head_tau_top1":
        eligible = [row for row in candidates if row["head_action_utility"] > 0.0]
        key = lambda row: row["head_action_utility"]
    elif policy == "fra_top1":
        eligible = [row for row in candidates if row["advantage_gate_active"]]
        key = lambda row: row["advantage_strength"]
    elif policy == "fra_tau_top1":
        eligible = [
            row
            for row in candidates
            if row["advantage_gate_active"] and row["action_utility"] > 0.0
        ]
        key = lambda row: row["advantage_strength"] * row["action_utility"]
    elif policy == "foreign_accuracy_top1":
        eligible = candidates
        key = lambda row: row["source_accuracy"]
    elif policy == "oracle_increment_top1":
        eligible = candidates
        key = lambda row: row["cmt_increment"]
    else:
        raise ValueError(f"Unknown ranking policy: {policy}")
    if not eligible:
        return None
    return max(eligible, key=key)


def summarize_policy(
    policy: str,
    grouped: dict[tuple[int, int], list[dict]],
) -> tuple[dict, list[dict]]:
    selections = []
    increments = []
    for (receiver_id, class_id), candidates in sorted(grouped.items()):
        selected = select_candidate(policy, candidates)
        increment = float(selected["cmt_increment"]) if selected is not None else 0.0
        increments.append(increment)
        selections.append({
            "policy": policy,
            "receiver_client": receiver_id,
            "class_id": class_id,
            "selected": int(selected is not None),
            "source_client": (
                int(selected["source_client"]) if selected is not None else -1
            ),
            "advantage_strength": (
                float(selected["advantage_strength"]) if selected is not None else 0.0
            ),
            "action_utility": (
                float(selected["action_utility"]) if selected is not None else 0.0
            ),
            "head_action_utility": (
                float(selected.get("head_action_utility", 0.0))
                if selected is not None
                else 0.0
            ),
            "cmt_increment_over_ce": increment,
        })

    selected_increments = [
        row["cmt_increment_over_ce"] for row in selections if row["selected"]
    ]
    selected_count = len(selected_increments)
    total_count = len(selections)
    summary = {
        "policy": policy,
        "receiver_class_groups": total_count,
        "selected_groups": selected_count,
        "coverage": selected_count / max(total_count, 1),
        "positive_precision": (
            sum(value > 0.0 for value in selected_increments) / selected_count
            if selected_count
            else 0.0
        ),
        "negative_fraction": (
            sum(value < 0.0 for value in selected_increments) / selected_count
            if selected_count
            else 0.0
        ),
        "mean_selected_increment": (
            sum(selected_increments) / selected_count if selected_count else 0.0
        ),
        "mean_policy_increment_including_abstention": (
            sum(increments) / max(total_count, 1)
        ),
    }
    return summary, selections


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    policies = (
        "foreign_accuracy_top1",
        "fra_top1",
        "tau_top1",
        "head_tau_top1",
        "fra_tau_top1",
        "oracle_increment_top1",
    )
    summaries = []
    selections = []
    for token in args.tokens:
        rows = read_rows(args.one_step_dir / f"gamma{token}_one_step_triplets.csv")
        grouped = defaultdict(list)
        for row in rows:
            grouped[(int(row["receiver_client"]), int(row["class_id"]))].append(row)
        gamma = float(rows[0]["gamma"])
        gamma_summaries = {}
        for policy in policies:
            summary, policy_selections = summarize_policy(policy, grouped)
            summary["gamma"] = gamma
            summaries.append(summary)
            gamma_summaries[policy] = summary
            for row in policy_selections:
                row["gamma"] = gamma
                selections.append(row)
        tau = gamma_summaries["tau_top1"]
        head_tau = gamma_summaries["head_tau_top1"]
        full_selections = {
            (row["receiver_client"], row["class_id"]): row["source_client"]
            for row in selections
            if row["gamma"] == gamma and row["policy"] == "tau_top1"
        }
        head_selections = {
            (row["receiver_client"], row["class_id"]): row["source_client"]
            for row in selections
            if row["gamma"] == gamma and row["policy"] == "head_tau_top1"
        }
        comparable = sorted(set(full_selections) & set(head_selections))
        head_tau["source_agreement_with_full_tau"] = (
            sum(full_selections[key] == head_selections[key] for key in comparable)
            / max(len(comparable), 1)
        )
        print(
            f"[audit] gamma={gamma:.1f} TAU-top1 coverage={tau['coverage']:.3f} "
            f"precision={tau['positive_precision']:.3f} "
            f"mean_increment={tau['mean_selected_increment']:.6f} "
            f"head_agreement={head_tau['source_agreement_with_full_tau']:.3f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "source_ranking_summary.csv", summaries)
    write_csv(args.output_dir / "source_ranking_selections.csv", selections)
    (args.output_dir / "source_ranking_summary.json").write_text(
        json.dumps({"policies": summaries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[audit] wrote source-ranking audit to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
