from __future__ import annotations

import tarfile

import numpy as np

from fedprime.engine.cle_response_spectrum import (
    bootstrap_clean_concentration,
    bootstrap_count_matrix,
    bootstrap_response_concentration,
    clean_spectrum,
    response_spectrum,
    spectrum_from_gram,
)
from scripts.analyze_cle_k1_c0_response_spectrum import (
    decide_gates,
    saved_artifact_recomputation,
)
from scripts.openi_cle_k1_c0_response_spectrum_entry import safe_extract


def test_response_concentration_distinguishes_shared_from_orthogonal_directions() -> None:
    carriers = 20
    probes = 4
    shared = np.zeros((carriers, probes, probes), dtype=np.float64)
    orthogonal = np.zeros_like(shared)
    shared[:, :, 0] = 1.0
    for probe in range(probes):
        orthogonal[:, probe, probe] = 1.0
    shared_chi = response_spectrum(shared).statistics.concentration
    orthogonal_chi = response_spectrum(orthogonal).statistics.concentration
    np.testing.assert_allclose(shared_chi, 1.0, atol=1.0e-10)
    np.testing.assert_allclose(orthogonal_chi, 1.0 / probes, atol=1.0e-10)


def test_response_normalization_removes_probe_scale() -> None:
    rng = np.random.default_rng(8)
    delta = rng.normal(size=(30, 3, 5)) + np.asarray([1.0, 0.0, 0.0])[None, :, None]
    scaled = delta * np.asarray([0.1, 5.0, 20.0])[None, :, None]
    np.testing.assert_allclose(
        response_spectrum(delta).gram,
        response_spectrum(scaled).gram,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_bootstrap_response_recomputes_full_statistic() -> None:
    rng = np.random.default_rng(9)
    delta = rng.normal(size=(12, 4, 6))
    indices = np.asarray(
        [np.arange(12), np.repeat(np.arange(6), 2)],
        dtype=np.int64,
    )
    counts = bootstrap_count_matrix(indices, carriers=12)
    actual = bootstrap_response_concentration(delta, counts, chunk_size=1)
    expected = []
    for draw in indices:
        expected.append(response_spectrum(delta[draw]).statistics.concentration)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-10)


def test_bootstrap_clean_matches_direct_resampling() -> None:
    rng = np.random.default_rng(10)
    features = rng.normal(size=(14, 7))
    indices = np.asarray(
        [np.arange(14), rng.integers(0, 14, size=14), np.repeat(np.arange(7), 2)],
        dtype=np.int64,
    )
    counts = bootstrap_count_matrix(indices, carriers=14)
    actual = bootstrap_clean_concentration(features, counts, chunk_size=2)
    expected = [clean_spectrum(features[draw]).concentration for draw in indices]
    np.testing.assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-10)


def test_spectrum_diagnostics_are_consistent() -> None:
    gram = np.diag([4.0, 2.0, 1.0])
    result = spectrum_from_gram(gram)
    np.testing.assert_allclose(result.concentration, 21.0 / 49.0)
    np.testing.assert_allclose(result.effective_rank, 49.0 / 21.0, rtol=1.0e-10)
    np.testing.assert_allclose(result.top1_share, 4.0 / 7.0)
    np.testing.assert_allclose(result.top3_share, 1.0)


def test_frozen_ten_gate_decision_uses_response_vs_clean_contrast() -> None:
    response_rows = []
    clean_rows = []
    bootstrap = {}
    for arm in ("h9", "h0", "l9", "l0"):
        strong = arm.endswith("9")
        for client in range(4):
            clean_rows.append({"arm": arm, "client": client, "chi_clean": 0.1})
            bootstrap[f"clean_{arm}_c{client}"] = np.full(20, 0.1)
            for bank in ("a", "b"):
                value = 0.4 if strong else 0.2
                response_rows.append({"arm": arm, "client": client, "bank": bank, "chi_resp": value})
                bootstrap[f"response_{arm}_c{client}_{bank}"] = np.full(20, value)
    decision = decide_gates(response_rows, clean_rows, bootstrap)
    assert decision["verdict"] == "GO_TO_K1_C_CRSF_SURGERY"
    assert decision["passed_gates"] == decision["total_gates"] == 10

    for row in clean_rows:
        if str(row["arm"]).endswith("9"):
            row["chi_clean"] = 0.3
            bootstrap[f"clean_{row['arm']}_c{row['client']}"] = np.full(20, 0.3)
    decision = decide_gates(response_rows, clean_rows, bootstrap)
    assert decision["verdict"] == "NO_GO_RESPONSE_SPECTRAL_MECHANISM"
    assert not decision["gates"]["H5_response_specificity"]
    assert not decision["gates"]["L5_response_specificity"]


def test_saved_artifact_recomputation_rebuilds_all_gates(tmp_path) -> None:
    response_grams = {}
    clean_eigenvalues = {}
    bootstrap = {}
    response_rows = []
    clean_rows = []
    for arm in ("h9", "h0", "l9", "l0"):
        strong = arm.endswith("9")
        gram = np.diag([1.0, 0.0] if strong else [1.0, 1.0])
        chi = 1.0 if strong else 0.5
        for client in range(4):
            clean_eigenvalues[f"{arm}_c{client}_u1"] = np.asarray([1.0, 1.0])
            clean_eigenvalues[f"{arm}_c{client}_u2"] = np.asarray([1.0, 1.0])
            clean_rows.append({"arm": arm, "client": client, "chi_clean": 0.5})
            bootstrap[f"clean_{arm}_c{client}"] = np.full(20, 0.5)
            for bank in ("a", "b"):
                response_grams[f"{arm}_c{client}_{bank}_u1"] = gram
                response_grams[f"{arm}_c{client}_{bank}_u2"] = gram
                response_rows.append({"arm": arm, "client": client, "bank": bank, "chi_resp": chi})
                bootstrap[f"response_{arm}_c{client}_{bank}"] = np.full(20, chi)
    decision = decide_gates(response_rows, clean_rows, bootstrap)
    np.savez_compressed(tmp_path / "response_gram_matrices.npz", **response_grams)
    np.savez_compressed(tmp_path / "clean_eigenvalues.npz", **clean_eigenvalues)
    np.savez_compressed(tmp_path / "bootstrap_metrics.npz", **bootstrap)
    audit = saved_artifact_recomputation(tmp_path, response_rows, clean_rows, decision)
    assert audit["audit_pass"]
    assert audit["gate_inconsistencies"] == 0


def test_openi_extraction_does_not_materialize_forbidden_evaluation_assets(tmp_path) -> None:
    source = tmp_path / "source"
    (source / "payload/public").mkdir(parents=True)
    (source / "payload/evaluation").mkdir(parents=True)
    (source / "payload/public/images.bin").write_bytes(b"public")
    (source / "payload/evaluation/test_labels.npy").write_bytes(b"forbidden")
    archive = tmp_path / "input.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source / "payload", arcname="payload")
    destination = tmp_path / "extracted"
    safe_extract(archive, destination)
    assert (destination / "payload/public/images.bin").read_bytes() == b"public"
    assert not (destination / "payload/evaluation").exists()
