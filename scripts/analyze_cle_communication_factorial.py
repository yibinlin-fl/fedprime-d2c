from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analyze_strict_pew_asymhfl_probe import METRICS, subtract, summarize


COMMUNICATIONS = ("none", "hfl", "asymhfl_val")


def analyze(arms: dict[str, Path]) -> dict:
    summaries = {name: summarize(path) for name, path in arms.items()}
    required = {f"{local}_{communication}" for local in ("l0", "l1") for communication in COMMUNICATIONS}
    missing = sorted(required - set(summaries))
    if missing:
        raise ValueError(f"Missing factorial arms: {missing}")
    local_effect = {
        communication: {
            scope: subtract(
                summaries[f"l1_{communication}"][scope],
                summaries[f"l0_{communication}"][scope],
            )
            for scope in ("final", "last_five")
        }
        for communication in COMMUNICATIONS
    }
    communication_effect = {
        local: {
            communication: {
                scope: subtract(
                    summaries[f"{local}_{communication}"][scope],
                    summaries[f"{local}_none"][scope],
                )
                for scope in ("final", "last_five")
            }
            for communication in ("hfl", "asymhfl_val")
        }
        for local in ("l0", "l1")
    }
    return {
        "metrics": list(METRICS),
        "arms": summaries,
        "local_l1_minus_l0_by_communication": local_effect,
        "communication_minus_local_only": communication_effect,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the 2x3 CLE communication factorial.")
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {
        f"{local}_{communication}": args.root / f"cle_comm_factorial_{local}_{communication}_seed0_12round"
        for local in ("l0", "l1")
        for communication in COMMUNICATIONS
    }
    payload = analyze(arms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
