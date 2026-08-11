#!/usr/bin/env python3
"""Audit environment-mixture contrast rank without training a model."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = (
    ROOT
    / "RAHFL-master"
    / "Dataset"
    / "cifar_10_cle_v2"
    / "alpha05_gamma09_seed0_split0"
    / "metadata.json"
)


def matrix_rank(matrix: Iterable[Iterable[Fraction]]) -> int:
    rows = [list(row) for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Matrix rows must have equal width")

    rank = 0
    for column in range(width):
        pivot = next(
            (index for index in range(rank, len(rows)) if rows[index][column]),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for index, row in enumerate(rows):
            if index == rank or not row[column]:
                continue
            factor = row[column]
            rows[index] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(row, rows[rank])
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def centered_rank(matrix: list[list[Fraction]]) -> int:
    if not matrix:
        return 0
    row_count = len(matrix)
    means = [sum(column, Fraction()) / row_count for column in zip(*matrix)]
    centered = [
        [value - mean for value, mean in zip(row, means)] for row in matrix
    ]
    return matrix_rank(centered)


def audit(metadata: dict, level: str) -> dict:
    seen_operators = list(metadata["seen_operators"])
    if level == "family":
        operator_families = metadata["operator_families"]
        group_of = {operator: operator_families[operator] for operator in seen_operators}
    elif level == "operator":
        group_of = {operator: operator for operator in seen_operators}
    else:
        raise ValueError(f"Unsupported level: {level}")

    groups = sorted(set(group_of.values()))
    clients = sorted(metadata["class_operator_map"], key=int)
    class_ids = sorted(metadata["class_operator_map"][clients[0]], key=int)
    gamma = Fraction(str(metadata["gamma"]))
    remainder = (Fraction(1) - gamma) / (len(seen_operators) - 1)

    classes = []
    for class_id in class_ids:
        dominant_operators = [
            metadata["class_operator_map"][client][class_id] for client in clients
        ]
        mixture = []
        for dominant in dominant_operators:
            row = []
            for group in groups:
                probability = sum(
                    gamma if operator == dominant else remainder
                    for operator in seen_operators
                    if group_of[operator] == group
                )
                row.append(probability)
            mixture.append(row)

        rank = centered_rank(mixture)
        classes.append(
            {
                "class_id": int(class_id),
                "dominant_operators": dominant_operators,
                "dominant_groups": [group_of[operator] for operator in dominant_operators],
                "centered_rank": rank,
                "required_full_rank": len(groups) - 1,
                "full_contrast_coverage": rank == len(groups) - 1,
            }
        )

    return {
        "dataset_name": metadata.get("dataset_name"),
        "level": level,
        "num_clients": len(clients),
        "groups": groups,
        "maximum_client_contrast_rank": len(clients) - 1,
        "required_full_rank": len(groups) - 1,
        "classes": classes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit client-mixture identifiability from CLE metadata only."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--level", choices=("family", "operator"), default="family")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.metadata.open("r", encoding="utf-8") as handle:
        result = audit(json.load(handle), args.level)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
