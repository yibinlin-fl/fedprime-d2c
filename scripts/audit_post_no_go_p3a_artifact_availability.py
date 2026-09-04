from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("h0", "h9", "l0", "l9")
CLIENTS = ((0, "ResNet10"), (1, "ResNet12"), (2, "ShuffleNet"), (3, "Mobilenetv2"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only P3-A artifact availability audit.")
    parser.add_argument(
        "--a1a-predictions",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_shortcut_amplification_phase_a1a_seed0/extracted/outputs/cle_shortcut_amplification_phase_a1a_seed0_analysis/round_040_predictions.npz",
    )
    parser.add_argument(
        "--k1-minimal-oracle-root",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_k1_c_minimal_formal_seed0/extracted/outputs/cle_k1_c_minimal_seed0_formal/oracle_predictions",
    )
    parser.add_argument(
        "--k0a-root",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_public_carrier_k0a_seed0/formal_extracted/outputs/cle_public_carrier_k0a_seed0_formal",
    )
    parser.add_argument(
        "--k0b-root",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_generic_probe_k0b_seed0/formal_extracted/outputs/cle_generic_probe_k0b_seed0_formal",
    )
    parser.add_argument(
        "--phase-b0-predictions",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_public_canonicalization_phase_b0_seed0/formal_extracted/outputs/cle_public_canonicalization_phase_b0_seed0_formal/analysis/cle_public_canonicalization_phase_b0_predictions.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "deliverables/post_no_go_p3a_routing_identity_causal_audit_20260904",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def npz_schema(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            key: {"shape": list(payload[key].shape), "dtype": str(payload[key].dtype)}
            for key in payload.files
        }


def artifact_record(path: Path, role: str, compatibility: str, reason: str) -> dict[str, object]:
    return {
        "role": role,
        "path": path.as_posix(),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else "",
        "schema": json.dumps(npz_schema(path), sort_keys=True) if path.is_file() else "{}",
        "p3a_compatibility": compatibility,
        "reason": reason,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    a1a = args.a1a_predictions.resolve()
    k1_root = args.k1_minimal_oracle_root.resolve()
    k0a_root = args.k0a_root.resolve()
    k0b_root = args.k0b_root.resolve()
    phase_b0 = args.phase_b0_predictions.resolve()

    inventory: list[dict[str, object]] = []
    inventory.append(
        artifact_record(
            a1a,
            "existing round-40 real-corruption DSA tensor",
            "CORRUPT_ONLY",
            "Contains four arms x four clients and task labels/binding, but no paired clean/base output.",
        )
    )
    inventory.append(
        artifact_record(
            phase_b0,
            "existing PNCB real-corruption tensor",
            "CORRUPT_ONLY",
            "Contains original/overlay/canonical corruption probabilities for all arms, but no paired clean/base output.",
        )
    )
    for arm in ("h9", "l9"):
        path = k1_root / f"{arm}_ab_frozen.npz"
        inventory.append(
            artifact_record(
                path,
                "K1-C-Minimal paired real-corruption output",
                "PARTIAL_COMPATIBLE",
                "Contains internally paired clean/corrupt probabilities only for clients 0 and 3 in the strong-CLE arm.",
            )
        )
    for arm in ARMS:
        for client, _ in CLIENTS:
            k0a_path = k0a_root / "responses" / f"{arm}_client{client}.npz"
            inventory.append(
                artifact_record(
                    k0a_path,
                    "K0-A real-operator public-carrier response",
                    "INCOMPATIBLE_ENDPOINT",
                    "Has base/operator logits, but carriers are unlabeled CIFAR-100 public images rather than the CIFAR-10 CLE DSA sources.",
                )
            )
            k0b_path = k0b_root / "responses" / f"{arm}_client{client}.npz"
            inventory.append(
                artifact_record(
                    k0b_path,
                    "K0-B generic-probe public-carrier response",
                    "INCOMPATIBLE_INTERVENTION_DOMAIN",
                    "Has base/PRIME logits for routing design, but not real-corruption logits for DSA evaluation.",
                )
            )

    if not all(bool(row["exists"]) for row in inventory):
        raise FileNotFoundError("one or more expected retrospective artifacts are absent")

    with np.load(a1a, allow_pickle=False) as payload:
        probabilities = np.asarray(payload["probabilities"])
        if probabilities.shape != (4, 4, 1000, 16, 10):
            raise AssertionError("unexpected Phase-A1a probability shape")
        if np.count_nonzero(probabilities == 0) != 0 or not np.isfinite(probabilities).all():
            raise AssertionError("Phase-A1a probabilities cannot safely recover centered logits")
        a1a_min_probability = float(probabilities.min())

    coverage: list[dict[str, object]] = []
    partial_details: dict[tuple[str, int], dict[str, object]] = {}
    max_corrupt_difference = 0.0
    min_clean_probability = 1.0
    for arm, arm_index in (("h9", 1), ("l9", 3)):
        path = k1_root / f"{arm}_ab_frozen.npz"
        with np.load(path, allow_pickle=False) as payload:
            selected = np.asarray(payload["selected_client_ids"], dtype=np.int64)
            clean = np.asarray(payload["clean_probabilities"])
            corrupt = np.asarray(payload["probabilities"])
            if selected.tolist() != [0, 3] or clean.shape != (2, 1000, 10) or corrupt.shape != (2, 1000, 16, 10):
                raise AssertionError(f"unexpected K1-C-Minimal oracle schema: {path}")
            if np.count_nonzero(clean == 0) != 0 or not np.isfinite(clean).all():
                raise AssertionError(f"clean probabilities cannot recover centered logits: {path}")
            min_clean_probability = min(min_clean_probability, float(clean.min()))
            max_corrupt_difference = max(
                max_corrupt_difference,
                float(np.max(np.abs(corrupt - probabilities[arm_index, selected]))),
            )
            for position, client in enumerate(selected.tolist()):
                partial_details[(arm, client)] = {
                    "paired_source": path.as_posix(),
                    "paired_source_sha256": sha256_file(path),
                    "position": position,
                }

    for arm in ARMS:
        system = "hfl" if arm.startswith("h") else "local"
        gamma = 9 if arm.endswith("9") else 0
        for client, model in CLIENTS:
            pair = partial_details.get((arm, client))
            coverage.append(
                {
                    "arm": arm,
                    "system": system,
                    "gamma": gamma,
                    "client": client,
                    "model": model,
                    "real_corruption_probabilities": True,
                    "paired_clean_probabilities": pair is not None,
                    "centered_logit_response_legally_recoverable": pair is not None,
                    "source": pair["paired_source"] if pair else a1a.as_posix(),
                    "reason": (
                        "Internally paired clean and corrupted probabilities are strictly positive; centered logits are recoverable up to common shift."
                        if pair
                        else "No clean/base output exists for the same model and CIFAR-10 source identities."
                    ),
                }
            )

    paired_count = sum(bool(row["paired_clean_probabilities"]) for row in coverage)
    control_count = sum(bool(row["paired_clean_probabilities"]) and int(row["gamma"]) == 0 for row in coverage)
    all_clients_covered = paired_count == len(coverage)
    full_gate_possible = all_clients_covered and control_count == 8
    if full_gate_possible:
        raise AssertionError("audit assumptions changed: P3-A would no longer be artifact-blocked")

    write_csv(output_dir / "artifact_inventory.csv", inventory)
    write_csv(output_dir / "per_client_results.csv", coverage)
    blocked_row = [{
        "status": "NOT_RUN_INSUFFICIENT_EXISTING_ARTIFACTS",
        "reason": "Full matched clean-corrupt logit responses are unavailable for H0/H9/L0/L9 x four clients.",
    }]
    for name in ("invariance_audit.csv", "targeted_results.csv", "random_derangement_null.csv"):
        write_csv(output_dir / name, blocked_row)

    permutation_manifest = {
        "protocol": "p3a_matched_routing_identity_causal_audit",
        "generated": False,
        "status": "NOT_GENERATED_INSUFFICIENT_EXISTING_ARTIFACTS",
        "reason": "Stage 0 failed before taxonomy-free intervention design; no targeted or random permutation was generated.",
        "k0b_routing_source_available": True,
        "oracle_or_binding_used_for_permutation": False,
    }
    (output_dir / "permutation_manifest.json").write_text(
        json.dumps(permutation_manifest, indent=2), encoding="utf-8"
    )

    summary = {
        "protocol": "p3a_matched_routing_identity_causal_audit",
        "verdict": "INSUFFICIENT_EXISTING_ARTIFACTS",
        "status": "STOP_BEFORE_PERMUTATION_OR_ORACLE_EVALUATION",
        "execution": {
            "gpu": False,
            "openi": False,
            "model_inference": False,
            "training": False,
            "checkpoint_loaded": False,
            "permutation_generated": False,
            "oracle_dsa_recomputed": False,
        },
        "coverage": {
            "required_arm_client_contexts": 16,
            "paired_clean_corrupt_contexts": paired_count,
            "paired_fraction": paired_count / 16.0,
            "no_cle_control_contexts": control_count,
            "functionally_unique_paired_contexts_upper_bound": 3,
            "reason_for_unique_bound": "H9/L9 client0 frozen outputs inherit the documented bit-identical ResNet10 model/output duplication.",
        },
        "numerical_facts": {
            "a1a_min_probability": a1a_min_probability,
            "k1_minimal_clean_min_probability": min_clean_probability,
            "max_k1_vs_a1a_corrupt_probability_difference": max_corrupt_difference,
            "probability_to_centered_logit_recovery": "valid only where paired clean and corrupt probabilities are both available",
        },
        "missing_requirements": [
            "paired clean/base outputs for all H0 clients",
            "paired clean/base outputs for all L0 clients",
            "paired clean/base outputs for H9 clients 1 and 2",
            "paired clean/base outputs for L9 clients 1 and 2",
            "at least 3/4 per-arm client coverage required by the frozen direction gate",
            "H0/L0 controls required by the frozen no-new-DSA gate",
        ],
        "scientific_boundary": (
            "Centered logits can be recovered from strictly positive softmax probabilities, but a clean-conditioned response "
            "cannot be recovered when the paired clean probabilities were never saved. Using the mean corrupted view as a "
            "surrogate base or joining K0-A CIFAR-100 base logits to Phase-A1a CIFAR-10 corruptions would change the frozen estimand."
        ),
    }
    (output_dir / "oracle_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    availability = f"""# P3-A Artifact Availability

Date: 2026-09-04

## Frozen requirement

P3-A requires the same model and CIFAR-10 source to have both clean/base output and all 16
real-corruption outputs, for H0/H9/L0/L9 and four clients. Only then is
`d = P_C(z_corrupt - z_clean)` identified without new inference.

## What exists

| Asset | Coverage | Why it is or is not sufficient |
| --- | --- | --- |
| Phase-A1a round-40 predictions | H0/H9/L0/L9 x 4 clients, corruptions only | Existing DSA source, but no clean/base output |
| K1-C-Minimal frozen oracle files | H9/L9, clients 0 and 3 | Valid paired clean/corrupt probabilities, but only 4/16 contexts and no H0/L0 controls |
| K0-A response files | Four arms x four clients, base + 16 real operators | CIFAR-100 public carriers, not the CIFAR-10 sources/labels used by DSA |
| K0-B response files | Four arms x four clients, base + 128 PRIME views | Valid routing-design source, but generic probes are not the real-corruption DSA grid |
| Phase-B0 PNCB predictions | Four arms x four clients, real corruptions | Original/overlay/canonical probabilities, but no clean/base output |

All saved probabilities inspected here are finite and strictly positive. Therefore pre-softmax
centered logits are recoverable as `log(p) - mean_c log(p)` where both clean and corrupted views
exist. This does not create a missing clean view. The maximum difference between K1-C-Minimal's
saved frozen corrupted probabilities and the aligned Phase-A1a entries is `{max_corrupt_difference:.3e}`.

## Coverage decision

- Required matched contexts: 16.
- Available matched contexts: {paired_count}/16.
- Available no-CLE controls: {control_count}/8.
- Functionally unique matched contexts: at most 3, because H9/L9 client0 is duplicated.

Verdict: `INSUFFICIENT_EXISTING_ARTIFACTS`.

No permutation, counterfactual output, DSA null, model inference or OpenI job was produced.
"""
    (output_dir / "artifact_availability.md").write_text(availability, encoding="utf-8")

    report = """# P3-A Matched Routing-Identity Causal Audit

Date: 2026-09-04

## Verdict

```text
INSUFFICIENT_EXISTING_ARTIFACTS
status: STOP_BEFORE_PERMUTATION_OR_ORACLE_EVALUATION
```

The frozen Stage-0 availability gate failed. Existing artifacts do not contain a complete matched
clean-to-real-corruption response grid for H0/H9/L0/L9 and all four clients. Consequently P3-A did
not generate the targeted rank-reversal permutation, the 1,000 random derangements, counterfactual
logits, invariance tables or DSA nulls.

## Why the missing base matters

Phase-A1a stores strictly positive softmax probabilities for all real-corruption views, so each
view's centered logits can be recovered. It does not store the clean output for the same source.
Without it, `P_C(z_corrupt-z_clean)` is not identified. Permuting the full corrupted logit would also
permute semantic class evidence; using the mean corruption view as a surrogate base would define a
different intervention. Neither substitution is allowed after the P3-A contract was frozen.

K0-A cannot fill the gap: its base logits belong to different CIFAR-100 public carriers. K1-C-Minimal
does contain paired clean/corrupt probabilities, but only for H9/L9 clients 0 and 3, with no H0/L0
control and at most three functionally unique contexts. That cannot satisfy the 3/4 direction gate
or the no-CLE safety gate.

## Scientific consequence

P2 remains valid as observational evidence for CLE-specific stable class-visible routing. P3-A has
neither passed nor failed scientifically; its causal estimand is untestable from the complete current
artifact set. The frozen verdict explicitly forbids automatically launching new inference.

If the user later considers filling the gap, that must be a separately costed, inference-only data
export protocol that saves clean logits for the exact 1,000 CIFAR-10 sources and 16 existing
round-40 checkpoints. It is not authorized by this audit.
"""
    (output_dir / "P3A_ROUTING_IDENTITY_CAUSAL_AUDIT.md").write_text(report, encoding="utf-8")

    final_files = (
        "P3A_ROUTING_IDENTITY_CAUSAL_AUDIT.md",
        "artifact_availability.md",
        "artifact_inventory.csv",
        "permutation_manifest.json",
        "invariance_audit.csv",
        "targeted_results.csv",
        "random_derangement_null.csv",
        "per_client_results.csv",
        "oracle_gate_summary.json",
    )
    manifest = {
        "protocol": "post_no_go_p3a_routing_identity_causal_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": summary["verdict"],
        "status": summary["status"],
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "files": {
            name: {"bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)}
            for name in final_files
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": summary["verdict"], "coverage": summary["coverage"], "output_dir": output_dir.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
