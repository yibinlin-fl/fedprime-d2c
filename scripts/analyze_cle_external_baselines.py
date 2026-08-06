from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_strict_pew_asymhfl_probe import METRICS, subtract, summarize


def analyze(arms: dict[str, Path], reference: str = "rahfl") -> dict:
    summaries = {name: summarize(path) for name, path in arms.items()}
    if reference not in summaries:
        raise ValueError(f"Missing reference baseline: {reference}")
    return {
        "metrics": list(METRICS),
        "reference": reference,
        "arms": summaries,
        "arm_minus_reference": {
            name: {
                scope: subtract(summary[scope], summaries[reference][scope])
                for scope in ("final", "last_five")
            }
            for name, summary in summaries.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CLE external baselines.")
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--reference", default="rahfl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    for item in args.arm:
        name, separator, path = item.partition("=")
        if not separator:
            raise ValueError(f"Expected NAME=PATH, got {item}")
        arms[name] = Path(path)
    payload = analyze(arms, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
