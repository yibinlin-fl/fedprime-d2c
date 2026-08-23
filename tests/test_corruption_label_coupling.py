from __future__ import annotations

import csv
import json

import numpy as np

from fedprime.data.corruption_label_coupling import (
    _largest_remainder_quotas,
    prepare_coupling_artifacts,
)


def _write_synthetic_source(root, num_samples: int = 800) -> None:
    train = root / "train"
    train.mkdir(parents=True)
    images = np.arange(num_samples * 3, dtype=np.uint8).reshape(num_samples, 1, 1, 3)
    labels = np.arange(num_samples, dtype=np.int64) % 10
    corruption_types = (np.arange(num_samples, dtype=np.int64) // 4) % 2
    severities = (np.arange(num_samples, dtype=np.int64) % 4) + 1
    np.save(train / "random_corrupt_1.npy", images)
    np.save(train / "labels.npy", labels)
    np.save(train / "corruption_type.npy", corruption_types)
    np.save(train / "corruption_severity.npy", severities)
    np.save(train / "corruption_mask.npy", np.ones(num_samples, dtype=np.bool_))
    (train / "corruption_manifest.json").write_text(
        json.dumps({"corruption_names": ["type0", "type1"]}), encoding="utf-8"
    )


def test_largest_remainder_is_exact() -> None:
    quotas = _largest_remainder_quotas(np.asarray([13, 12, 7, 9]), total=8)
    assert int(quotas.sum()) == 8
    assert np.all(quotas >= 0)
    assert np.all(quotas <= np.asarray([13, 12, 7, 9]))


def test_paired_coupling_artifacts_preserve_margins(tmp_path) -> None:
    data_root = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"
    _write_synthetic_source(data_root)

    manifest = prepare_coupling_artifacts(
        data_root=data_root,
        artifact_root=artifact_root,
        num_clients=4,
        samples_per_client=160,
        audit_ratio=0.10,
        noise_rate=0.20,
        betas=(0.0, 4.0),
        partition_seed=0,
        split_seed=0,
        noise_seed=0,
    )

    with np.load(artifact_root / "partition_disjoint_iid.npz") as partition:
        client_sets = [set(partition[f"client_{client_id}_global"].tolist()) for client_id in range(4)]
    for left in range(4):
        for right in range(left + 1, 4):
            assert client_sets[left].isdisjoint(client_sets[right])

    with np.load(artifact_root / "fit_audit_split.npz") as splits:
        for client_id in range(4):
            fit = np.asarray(splits[f"client_{client_id}_fit"], dtype=np.int64)
            audit = np.asarray(splits[f"client_{client_id}_audit"], dtype=np.int64)
            assert fit.size == 144
            assert audit.size == 16
            masks = {}
            transitions = {}
            for token in ("beta0", "beta4"):
                masks[token] = np.load(
                    artifact_root / token / f"client_{client_id}_noisy_mask.npy"
                )
                transitions[token] = np.load(
                    artifact_root / token / f"client_{client_id}_transition.npy"
                )
                assert int(masks[token][fit].sum()) == round(0.20 * fit.size)
                assert not bool(masks[token][audit].any())
            assert np.array_equal(transitions["beta0"], transitions["beta4"])

    beta0 = manifest["regimes"]["beta0"]
    beta4 = manifest["regimes"]["beta4"]
    assert beta0["num_noisy"] == beta4["num_noisy"]
    assert beta0["transition_matrix"] == beta4["transition_matrix"]
    assert beta4["mean_severity_noisy"] > beta0["mean_severity_noisy"]

    with (artifact_root / "beta0" / "stratum_statistics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        beta0_rows = list(csv.DictReader(handle))
    with (artifact_root / "beta4" / "stratum_statistics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        beta4_rows = list(csv.DictReader(handle))
    assert [row["noise_quota"] for row in beta0_rows] == [
        row["noise_quota"] for row in beta4_rows
    ]
