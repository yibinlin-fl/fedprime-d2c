from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_strict_pew_asymhfl_40round import METRICS, build_payload, summarize


def analyze(pairs: dict[int, tuple[Path, Path]]) -> dict:
    per_scenario = {}
    for scenario_seed, (control_path, candidate_path) in sorted(pairs.items()):
        per_scenario[str(scenario_seed)] = build_payload(
            summarize(control_path), summarize(candidate_path)
        )
    aggregate = {}
    for metric in METRICS:
        values = np.asarray([
            payload["candidate_minus_control"]["last_ten"][metric]
            for payload in per_scenario.values()
        ], dtype=np.float64)
        aggregate[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "randomness_unit": "independently regenerated CLE scenario and matched training seed",
        "per_scenario": per_scenario,
        "aggregate_last_ten_delta": aggregate,
        "all_scenarios_go": all(item["verdict"] == "GO" for item in per_scenario.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate 40-round cross-scenario CLE results.")
    parser.add_argument(
        "--pair", action="append", required=True,
        help="SCENARIO_SEED=CONTROL_DIRECTORY,CANDIDATE_DIRECTORY",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = {}
    for raw in args.pair:
        seed_raw, separator, directories = raw.partition("=")
        if not separator or "," not in directories:
            raise ValueError(f"Expected SEED=CONTROL,CANDIDATE, got: {raw}")
        control, candidate = directories.split(",", maxsplit=1)
        pairs[int(seed_raw)] = (Path(control), Path(candidate))
    payload = analyze(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
