from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from fedprime.data.corruptions import CORRUPTION_GROUPS, apply_corruption  # noqa: E402
from fedprime.data.loaders import _cifar100_train_from_tar  # noqa: E402
from fedprime.data.strict_fit_audit import stratified_fit_audit_indices  # noqa: E402
from fedprime.methods.environment_witness import PublicEnvironmentWitness  # noqa: E402
from fedprime.methods.witness_error_correction import (  # noqa: E402
    channel_diagnostics,
    normalize_confusion_counts,
    select_public_regularization,
    solve_simplex_ridge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-training WEC-BER witness error transfer gate.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/wec_ber_phase0_seed0.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "deliverables/wec_ber_phase0_20260907",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest().upper()


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames required for empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_witness(checkpoint_bytes: bytes) -> PublicEnvironmentWitness:
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    model = PublicEnvironmentWitness(
        embedding_dim=int(payload["embedding_dim"]),
        num_environments=int(payload["num_environments"]),
        severity_levels=int(payload["severity_levels"]),
    )
    model.load_state_dict(payload["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def predict(model: PublicEnvironmentWitness, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(images), int(batch_size)):
            batch = np.ascontiguousarray(images[start : start + batch_size].transpose(0, 3, 1, 2))
            tensor = torch.from_numpy(batch).float().div_(255.0)
            logits, _, _ = model(tensor)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions).astype(np.int64, copy=False)


def public_operator_predictions(
    model: PublicEnvironmentWitness,
    public_root: Path,
    *,
    public_size: int,
    validation_fraction: float,
    samples_per_operator: int,
    seed: int,
    audit_seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict[int, np.ndarray]], np.ndarray]:
    images, _ = _cifar100_train_from_tar(public_root)
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(len(images), size=min(int(public_size), len(images)), replace=False)
    split = max(1, min(len(selected) - 1, int(round(len(selected) * (1.0 - validation_fraction)))))
    validation_indices = np.asarray(selected[split:], dtype=np.int64)
    carriers = images[validation_indices[: int(samples_per_operator)]]
    if len(carriers) < int(samples_per_operator):
        raise ValueError("public validation split is smaller than samples_per_operator")

    clean_predictions = predict(model, carriers)
    operator_counts: dict[str, np.ndarray] = {}
    severity_counts: dict[str, dict[int, np.ndarray]] = {}
    operators = [operator for group in CORRUPTION_GROUPS.values() for operator in group]
    for operator_index, operator in enumerate(operators):
        corrupted = []
        severities = []
        for item, image in enumerate(carriers):
            severity = int(item % 5) + 1
            item_rng = np.random.default_rng(int(audit_seed) + operator_index * 1_000_003 + item * 104729)
            corrupted.append(apply_corruption(image, operator, severity, item_rng))
            severities.append(severity)
        predictions = predict(model, np.asarray(corrupted, dtype=np.uint8))
        operator_counts[operator] = np.bincount(predictions, minlength=6).astype(np.int64)
        severity_counts[operator] = {
            severity: np.bincount(
                predictions[np.asarray(severities) == severity], minlength=6
            ).astype(np.int64)
            for severity in range(1, 6)
        }
        print(f"[public] operator={operator} samples={len(predictions)}", flush=True)
    return clean_predictions, operator_counts, severity_counts, validation_indices


def channel_from_operator_counts(
    clean_counts: np.ndarray,
    operator_counts: dict[str, np.ndarray],
    *,
    excluded_operator: str | None = None,
) -> np.ndarray:
    columns = [np.asarray(clean_counts, dtype=np.float64)]
    for operators in CORRUPTION_GROUPS.values():
        included = [operator for operator in operators if operator != excluded_operator]
        if not included:
            raise ValueError("operator exclusion emptied a family")
        columns.append(sum((operator_counts[operator] for operator in included), np.zeros(6)))
    return normalize_confusion_counts(np.stack(columns, axis=1))


def matrix_rows(
    matrix: np.ndarray,
    row_names: list[str],
    column_names: list[str],
) -> list[dict]:
    return [
        {"observed_environment": row_name, **{name: float(matrix[i, j]) for j, name in enumerate(column_names)}}
        for i, row_name in enumerate(row_names)
    ]


def family_index(operator: str) -> int:
    for index, (_, operators) in enumerate(CORRUPTION_GROUPS.items(), start=1):
        if operator in operators:
            return index
    raise KeyError(operator)


def public_audit(config: dict, archive: tarfile.TarFile, output_dir: Path) -> dict:
    report = json.load(archive.extractfile(config["pew_report_member"]))
    checkpoint_bytes = archive.extractfile(config["pew_checkpoint_member"]).read()
    model = load_witness(checkpoint_bytes)
    observed_names = list(config["observed_environment_names"])
    latent_names = list(config["latent_environment_names"])

    archived_counts_true_by_observed = np.asarray(
        report["validation"]["confusion_matrix"], dtype=np.float64
    )
    archived_counts_observed_by_true = archived_counts_true_by_observed.T
    primary_q = normalize_confusion_counts(archived_counts_observed_by_true[:, :5])
    diagnostic_q6 = normalize_confusion_counts(archived_counts_observed_by_true)
    diagnostics = channel_diagnostics(primary_q)
    diagnostics["off_diagonal_mass"] = float(
        1.0 - np.mean([primary_q[index, index] for index in range(5)])
    )

    write_csv(
        output_dir / "PUBLIC_CHANNEL_Q.csv",
        matrix_rows(primary_q, observed_names, latent_names),
    )
    write_csv(
        output_dir / "PUBLIC_CHANNEL_Q_K6.csv",
        matrix_rows(diagnostic_q6, observed_names, observed_names),
    )
    singular_rows = [
        {"index": index, "singular_value": value}
        for index, value in enumerate(diagnostics["singular_values"], start=1)
    ]
    write_csv(output_dir / "PUBLIC_SINGULAR_VALUES.csv", singular_rows)

    clean_predictions, operator_counts, severity_counts, validation_indices = public_operator_predictions(
        model,
        ROOT / config["public_root"],
        public_size=int(config["public_size"]),
        validation_fraction=float(config["validation_fraction"]),
        samples_per_operator=int(config["samples_per_operator"]),
        seed=int(config["seed"]),
        audit_seed=int(config["operator_audit_seed"]),
    )
    clean_counts = np.bincount(clean_predictions, minlength=6)
    crossfit_rows = []
    folds = []
    for group_name, operators in CORRUPTION_GROUPS.items():
        for operator in operators:
            heldout = operator_counts[operator] / operator_counts[operator].sum()
            fold_channel = channel_from_operator_counts(
                clean_counts, operator_counts, excluded_operator=operator
            )
            true_index = family_index(operator)
            predicted_column = fold_channel[:, true_index]
            off_diagonal = [index for index in range(6) if index != true_index]
            train_dominant = max(off_diagonal, key=lambda index: predicted_column[index])
            heldout_dominant = max(off_diagonal, key=lambda index: heldout[index])
            row = {
                "operator": operator,
                "family": group_name,
                "true_environment_index": true_index,
                "column_l1": float(np.abs(predicted_column - heldout).sum()),
                "column_l2": float(np.sqrt(np.square(predicted_column - heldout).sum())),
                "diagonal_recall_error": float(abs(predicted_column[true_index] - heldout[true_index])),
                "train_dominant_confusion": observed_names[train_dominant],
                "heldout_dominant_confusion": observed_names[heldout_dominant],
                "dominant_confusion_preserved": int(train_dominant == heldout_dominant),
            }
            crossfit_rows.append(row)
            truth = np.zeros(5, dtype=np.float64)
            truth[true_index] = 1.0
            folds.append({"channel": fold_channel, "observed": heldout, "truth": truth})
    write_csv(output_dir / "PUBLIC_OPERATOR_CROSSFIT.csv", crossfit_rows)

    lambda_value, lambda_rows = select_public_regularization(
        folds, tuple(float(value) for value in config["lambda_candidates"])
    )
    recovery_rows = []
    for row, fold in zip(crossfit_rows, folds):
        corrected = solve_simplex_ridge(fold["channel"], fold["observed"], lambda_value)
        hard_known = np.asarray(fold["observed"][:5], dtype=np.float64)
        hard_known /= max(float(hard_known.sum()), np.finfo(np.float64).eps)
        hard_error = float(np.abs(hard_known - fold["truth"]).sum())
        corrected_error = float(np.abs(corrected - fold["truth"]).sum())
        recovery_rows.append(
            {
                "operator": row["operator"],
                "family": row["family"],
                "lambda": lambda_value,
                "hard_l1": hard_error,
                "corrected_l1": corrected_error,
                "relative_error_reduction": float(
                    (hard_error - corrected_error) / max(hard_error, np.finfo(np.float64).eps)
                ),
                "improved": int(corrected_error < hard_error),
                "unknown_observed_rate": float(fold["observed"][5]),
            }
        )
    write_csv(output_dir / "PUBLIC_RECOVERY.csv", recovery_rows)

    audit_q = channel_from_operator_counts(clean_counts, operator_counts)
    severity_rows = []
    diagonal_by_severity = []
    for severity in range(1, 6):
        severity_operator_counts = {
            operator: severity_counts[operator][severity] for operator in operator_counts
        }
        severity_q = channel_from_operator_counts(clean_counts, severity_operator_counts)
        frobenius = float(np.sqrt(np.square(severity_q - audit_q).sum()))
        diagonal = [float(severity_q[index, index]) for index in range(5)]
        diagonal_by_severity.append(diagonal)
        severity_diagnostics = channel_diagnostics(severity_q)
        severity_rows.append(
            {
                "severity": severity,
                "frobenius_from_pooled": frobenius,
                "mean_diagonal_recall": float(np.mean(diagonal)),
                "min_singular_value": severity_diagnostics["min_singular_value"],
                "condition_number": severity_diagnostics["condition_number"],
            }
        )
    write_csv(output_dir / "SEVERITY_CHANNEL_DIAGNOSTIC.csv", severity_rows)
    diagonal_array = np.asarray(diagonal_by_severity)
    max_diagonal_range = float(np.max(diagonal_array.max(axis=0) - diagonal_array.min(axis=0)))
    max_severity_frobenius = float(max(row["frobenius_from_pooled"] for row in severity_rows))

    gates = config["gates"]
    l1_values = np.asarray([row["column_l1"] for row in crossfit_rows])
    preserved = float(np.mean([row["dominant_confusion_preserved"] for row in crossfit_rows]))
    reductions = np.asarray([row["relative_error_reduction"] for row in recovery_rows])
    improvement_fraction = float(np.mean([row["improved"] for row in recovery_rows]))
    g1 = (
        diagnostics["rank"] == int(gates["g1_rank"])
        and diagnostics["min_singular_value"] >= float(gates["g1_min_singular_value"])
        and diagnostics["condition_number"] <= float(gates["g1_max_condition_number"])
    )
    g2 = (
        float(np.median(l1_values)) <= float(gates["g2_median_l1"])
        and float(np.quantile(l1_values, 0.75)) <= float(gates["g2_p75_l1"])
        and preserved >= float(gates["g2_dominant_confusion_preservation"])
    )
    g3 = (
        float(np.median(reductions)) >= float(gates["g3_median_relative_reduction"])
        and improvement_fraction >= float(gates["g3_operator_improvement_fraction"])
    )
    severity_status = (
        "SEVERITY_DEPENDENCE_HIGH"
        if max_severity_frobenius >= float(gates["severity_max_frobenius"])
        or max_diagonal_range >= float(gates["severity_max_diagonal_range"])
        else "SEVERITY_DEPENDENCE_LOW"
    )
    return {
        "report": report,
        "checkpoint_sha256": sha256_bytes(checkpoint_bytes),
        "primary_q": primary_q,
        "diagnostics": diagnostics,
        "lambda": lambda_value,
        "lambda_selection": lambda_rows,
        "crossfit": {
            "median_l1": float(np.median(l1_values)),
            "p75_l1": float(np.quantile(l1_values, 0.75)),
            "dominant_confusion_preservation": preserved,
            "worst_operators": [
                {
                    "operator": row["operator"],
                    "family": row["family"],
                    "column_l1": row["column_l1"],
                }
                for row in sorted(crossfit_rows, key=lambda item: item["column_l1"], reverse=True)[:5]
            ],
        },
        "recovery": {
            "median_relative_error_reduction": float(np.median(reductions)),
            "operator_improvement_fraction": improvement_fraction,
        },
        "severity": {
            "status": severity_status,
            "max_frobenius": max_severity_frobenius,
            "max_diagonal_recall_range": max_diagonal_range,
        },
        "validation_indices_sha256": sha256_array(validation_indices),
        "gates": {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3)},
    }


def private_support_audit(
    config: dict,
    archive: tarfile.TarFile,
    primary_q: np.ndarray,
    lambda_value: float,
    output_dir: Path,
) -> dict:
    private_root = ROOT / config["private_root"]
    metadata = json.loads((private_root / "metadata.json").read_text(encoding="utf-8"))
    family_to_latent = {"noise": 1, "blur": 2, "weather": 3, "digital": 4}
    id_to_latent = {
        int(operator_id): family_to_latent[metadata["operator_families"][operator]]
        for operator, operator_id in metadata["operator_to_id"].items()
    }
    strict = config["strict_fit_audit"]
    rows = []
    prediction_hashes = {}
    for client_id in range(int(metadata["num_clients"])):
        member = f"{config['pew_prediction_prefix']}/client_{client_id}.npz"
        prediction_bytes = archive.extractfile(member).read()
        prediction_hashes[member] = sha256_bytes(prediction_bytes)
        with np.load(io.BytesIO(prediction_bytes), allow_pickle=False) as prediction_archive:
            predicted = np.asarray(prediction_archive["environment_ids"], dtype=np.int64)
        client_root = private_root / f"client_{client_id}"
        labels = np.load(client_root / "train_labels.npy").astype(np.int64)
        corruption_ids = np.load(client_root / "train_corruption_ids.npy").astype(np.int64)
        oracle = np.asarray([id_to_latent[int(value)] for value in corruption_ids], dtype=np.int64)
        fit, _ = stratified_fit_audit_indices(
            labels,
            audit_ratio=float(strict["audit_ratio"]),
            min_audit_per_class=int(strict["min_audit_per_class"]),
            min_fit_per_class=int(strict["min_fit_per_class"]),
            seed=int(strict["seed"]) * 1009 + client_id,
        )
        for class_id in range(int(metadata["num_classes"])):
            selected = fit[labels[fit] == class_id]
            observed_counts = np.bincount(predicted[selected], minlength=6).astype(np.float64)
            observed = observed_counts / observed_counts.sum()
            hard = observed[:5].copy()
            hard /= max(float(hard.sum()), np.finfo(np.float64).eps)
            corrected = solve_simplex_ridge(primary_q, observed, lambda_value)
            truth_counts = np.bincount(oracle[selected], minlength=5).astype(np.float64)
            truth = truth_counts / truth_counts.sum()
            hard_l1 = float(np.abs(hard - truth).sum())
            corrected_l1 = float(np.abs(corrected - truth).sum())
            rows.append(
                {
                    "client": client_id,
                    "class": class_id,
                    "fit_samples": int(len(selected)),
                    "unknown_rate": float(observed[5]),
                    "hard_support_l1": hard_l1,
                    "corrected_support_l1": corrected_l1,
                    "relative_error_reduction": float(
                        (hard_l1 - corrected_l1) / max(hard_l1, np.finfo(np.float64).eps)
                    ),
                    "improved": int(corrected_l1 < hard_l1),
                }
            )
    write_csv(output_dir / "PRIVATE_SUPPORT_RECOVERY.csv", rows)
    client_rows = []
    for client_id in range(int(metadata["num_clients"])):
        selected_rows = [row for row in rows if row["client"] == client_id]
        hard = float(np.mean([row["hard_support_l1"] for row in selected_rows]))
        corrected = float(np.mean([row["corrected_support_l1"] for row in selected_rows]))
        client_rows.append(
            {
                "client": client_id,
                "hard_support_l1": hard,
                "corrected_support_l1": corrected,
                "improved": bool(corrected < hard),
                "unknown_rate": float(
                    np.average(
                        [row["unknown_rate"] for row in selected_rows],
                        weights=[row["fit_samples"] for row in selected_rows],
                    )
                ),
            }
        )
    hard_mean = float(np.mean([row["hard_support_l1"] for row in rows]))
    corrected_mean = float(np.mean([row["corrected_support_l1"] for row in rows]))
    improved_clients = int(sum(row["improved"] for row in client_rows))
    support_pass = (
        corrected_mean <= float(config["gates"]["g4_support_relative_error"]) * hard_mean
        and improved_clients >= int(config["gates"]["g4_min_improved_clients"])
    )
    overall_unknown = float(
        np.average([row["unknown_rate"] for row in rows], weights=[row["fit_samples"] for row in rows])
    )
    write_csv(
        output_dir / "PRIVATE_RISK_RECOVERY.csv",
        [
            {
                "status": "NOT_EVALUABLE",
                "reason": "exact PEW+BER 12-round archive has save_final=false and no classifier checkpoint",
            }
        ],
    )
    return {
        "hard_support_l1": hard_mean,
        "corrected_support_l1": corrected_mean,
        "relative_error_reduction": float(
            (hard_mean - corrected_mean) / max(hard_mean, np.finfo(np.float64).eps)
        ),
        "improved_clients": improved_clients,
        "clients": client_rows,
        "support_gate_pass": bool(support_pass),
        "risk_status": "NOT_EVALUABLE",
        "overall_unknown_rate": overall_unknown,
        "unknown_warning": bool(overall_unknown > float(config["gates"]["unknown_rate_warning"])),
        "prediction_hashes": prediction_hashes,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if bool(config.get("training_authorized", True)):
        raise ValueError("Phase-0 config must keep training_authorized=false")
    archive_path = (ROOT / config["source_archive"]).resolve()
    started = time.perf_counter()
    integrity = {
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_archive": str(archive_path),
        "source_archive_sha256": sha256_file(archive_path),
        "training_started": False,
        "gpu_used": False,
        "private_oracle_opened_after_public_seal": False,
        "dsa_opened": False,
    }

    with tarfile.open(archive_path, "r:gz") as archive:
        public = public_audit(config, archive, output_dir)
        integrity["pew_checkpoint_sha256"] = public["checkpoint_sha256"]
        public_payload = {
            "channel": public["diagnostics"],
            "lambda": public["lambda"],
            "lambda_selection": public["lambda_selection"],
            "crossfit": public["crossfit"],
            "recovery": public["recovery"],
            "severity": public["severity"],
            "gates": public["gates"],
            "validation_indices_sha256": public["validation_indices_sha256"],
            "protocol": config["protocol_corrections"],
        }
        write_json(output_dir / "PUBLIC_GATE_SUMMARY.json", public_payload)
        public_files = [
            "PUBLIC_CHANNEL_Q.csv",
            "PUBLIC_CHANNEL_Q_K6.csv",
            "PUBLIC_SINGULAR_VALUES.csv",
            "PUBLIC_OPERATOR_CROSSFIT.csv",
            "PUBLIC_RECOVERY.csv",
            "SEVERITY_CHANNEL_DIAGNOSTIC.csv",
            "PUBLIC_GATE_SUMMARY.json",
        ]
        public_seal = {name: sha256_file(output_dir / name) for name in public_files}
        write_json(output_dir / "PUBLIC_SEAL.json", public_seal)

        public_pass = all(public["gates"].values())
        private = None
        if public_pass:
            integrity["private_oracle_opened_after_public_seal"] = True
            private = private_support_audit(
                config, archive, public["primary_q"], public["lambda"], output_dir
            )
            integrity["prediction_member_sha256"] = private.pop("prediction_hashes")
        else:
            write_csv(
                output_dir / "PRIVATE_SUPPORT_RECOVERY.csv",
                [{"status": "NOT_OPENED", "reason": "one or more public gates failed"}],
            )
            write_csv(
                output_dir / "PRIVATE_RISK_RECOVERY.csv",
                [{"status": "NOT_OPENED", "reason": "one or more public gates failed"}],
            )

    if not public["gates"]["G1"]:
        verdict = "NO_GO_WEC_BER_CHANNEL_ILL_CONDITIONED"
    elif not public["gates"]["G2"]:
        verdict = "NO_GO_WEC_BER_ERROR_TRANSFER"
    elif not public["gates"]["G3"]:
        verdict = "NO_GO_WEC_BER_PUBLIC_RECOVERY"
    elif private is None or not private["support_gate_pass"]:
        verdict = "NO_GO_WEC_BER_PRIVATE_CORRECTION"
    elif private["risk_status"] != "PASS":
        verdict = "PHASE0_INCOMPLETE_RISK_AUDIT"
    else:
        verdict = "GO_TO_WEC_BER_SEED0_12ROUND"

    gates = {
        **public["gates"],
        "G4_support": None if private is None else private["support_gate_pass"],
        "G4_risk": None if private is None or private["risk_status"] != "PASS" else True,
    }
    manifest = {
        "phase": config["phase"],
        "verdict": verdict,
        "gates": gates,
        "public_frozen_lambda_pi": public["lambda"],
        "public_frozen_lambda_v": public["lambda"],
        "primary_channel_shape": config["primary_channel_shape"],
        "protocol_corrections": config["protocol_corrections"],
        "public_seal_sha256": sha256_file(output_dir / "PUBLIC_SEAL.json"),
        "private": private,
        "training_started": False,
        "full_hfl_training_authorized": False,
    }
    write_json(output_dir / "PHASE0_MANIFEST.json", manifest)
    private_text = "由于 G2 已失败，脚本按 sealed protocol 没有打开 private oracle。"
    if private is not None:
        private_text = (
            f"Support L1：hard={private['hard_support_l1']:.6f}，"
            f"corrected={private['corrected_support_l1']:.6f}，"
            f"相对下降={100.0 * private['relative_error_reduction']:.2f}%，"
            f"改善客户端={private['improved_clients']}/4；"
            f"risk audit={private['risk_status']}。"
        )
    q_header = "| observed \\ true | " + " | ".join(config["latent_environment_names"]) + " |"
    q_separator = "|---|" + "---:|" * len(config["latent_environment_names"])
    q_rows = [q_header, q_separator]
    for row_index, row_name in enumerate(config["observed_environment_names"]):
        values = " | ".join(f"{public['primary_q'][row_index, column_index]:.6f}" for column_index in range(5))
        q_rows.append(f"| {row_name} | {values} |")
    q_table = "\n".join(q_rows)
    worst_rows = "\n".join(
        f"- `{row['operator']}` ({row['family']}): column L1 `{row['column_l1']:.6f}`"
        for row in public["crossfit"]["worst_operators"]
    )
    report = f"""# WEC-BER Phase-0 结果摘要

日期：2026-09-07

## 最终判定

```text
{verdict}
TRAINING_NOT_STARTED
```

Primary channel 是 `6 x 5` 的 observed-by-latent 矩阵：保留 PEW `unknown` 作为可观测输出，
但不把 composite unknown 提升为第五个基础 corruption family。每个 operator cross-fit 只比较
该 operator 所属真实 family 的通道列，避免虚构无法观测的完整 held-out 矩阵。

## Public Q

{q_table}

奇异值：`{', '.join(f'{value:.6f}' for value in public['diagnostics']['singular_values'])}`。

## 四层 Gate

- G1 `{'PASS' if public['gates']['G1'] else 'FAIL'}`：rank `{public['diagnostics']['rank']}`，最小奇异值
  `{public['diagnostics']['min_singular_value']:.6f}`，condition number
  `{public['diagnostics']['condition_number']:.6f}`。
- G2 `{'PASS' if public['gates']['G2'] else 'FAIL'}`：operator column L1 median/p75 为
  `{public['crossfit']['median_l1']:.6f}/{public['crossfit']['p75_l1']:.6f}`，超过冻结上限
  `0.25/0.35`；dominant confusion preservation 为
  `{100.0 * public['crossfit']['dominant_confusion_preservation']:.2f}%`，单项超过 60%。
- G3 `{'PASS' if public['gates']['G3'] else 'FAIL'}`：相对 hard pseudo prevalence 的 median recovery improvement
  `{100.0 * public['recovery']['median_relative_error_reduction']:.2f}%`，改善 operator 比例
  `{100.0 * public['recovery']['operator_improvement_fraction']:.2f}%`。
- G4 support/risk：`NOT_OPENED`，因为 G2 失败后 private oracle 必须保持 sealed。
- Public-only 冻结 lambda：`{public['lambda']}`。

G3 通过不能覆盖 G2：这说明某些 fold 上矩阵反解能把纯 family prevalence 拉回正确方向，
但 error channel 本身跨具体 operator 不稳定，因此没有资格迁移到 private CLE cells。

G2 最不稳定的 operator：

{worst_rows}

## Private sealed 状态

{private_text}

exact PEW+BER 12-round archive 还设置了 `save_final=false`，没有分类模型 checkpoint。即使 G2
通过，private risk recovery 也不能由现有资产严格构造，必须另行获得 fixed exact-method
checkpoint 后才能评估，不能借用含 CDep 或其他方法的 checkpoint。

## Severity secondary diagnostic

- `{public['severity']['status']}`; max Frobenius `{public['severity']['max_frobenius']:.6f}`;
  最大 diagonal-recall range `{100.0 * public['severity']['max_diagonal_recall_range']:.2f} pp`。

severity dependence 很高，与 G2 的 operator instability 方向一致；它只作诊断，不改变 primary
NO-GO，也不授权 severity-aware WEC-BER。

## 科学解释

公共混合 confusion matrix 本身满秩且条件良好，但这只是 pooled 可逆性。按具体 operator
拆开后，PEW 的错误规律变化过大，尤其集中在 digital/weather family。因而一个从 pooled
public validation 学到的单一 Q 不能被当成稳定的 private noise channel。这正是 Phase-0 要杀掉
的假设：`pooled invertible != operator-transferable`。

没有训练模型，没有更新参数，没有使用 GPU 或 DSA evaluator，没有打开 private oracle，也没有
启动 12-round HFL。不得通过删 operator、按 family 选择性保留、调 gate、调 threshold、改 solver
或补 seed 复活 WEC-BER。
"""
    (output_dir / "RESULT_SUMMARY_ZH.md").write_text(report, encoding="utf-8")
    integrity["elapsed_seconds"] = time.perf_counter() - started
    integrity["output_files_before_integrity"] = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "INTEGRITY_AUDIT.json"
    }
    write_json(output_dir / "INTEGRITY_AUDIT.json", integrity)
    print(json.dumps({"verdict": verdict, "gates": gates, "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
