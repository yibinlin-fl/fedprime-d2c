from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.engine.cle_generic_probe_gate import generic_probe_statistics  # noqa: E402
from scripts.run_cle_k1_sdmn_headonly import sha256_array  # noqa: E402


EXPECTED_ARCHIVE_SHA256 = "1E02A16C765D8AB976A692D444FA9DAEBE38C30F8279CD6DCCFC49D1BFF88608"
EXPECTED_D_SELECT_SHA256 = "731B8CFFDCBD241474D33B261E323F9EC11C2EA59BC7705261140A3B8572F6CA"
EXPECTED_BANK_HASHES = {
    "a": "6CAE529D4240715162B19B3968D47FA037A940B4D52D688FF52B859C5523DC01",
    "b": "4A53497EC5DB6EC05C312E6166109FA4B52A5CC402CCE74E6EDB1253D913BF4E",
}
ARMS = ("h9", "l9")
BANK_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze K1-B0 probe selections from audited K0-B outputs.")
    parser.add_argument("--k0b-archive", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fedprime/augmentations/assets/cle_k1_b0/selection_manifest.json",
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


def member_bytes(handle: tarfile.TarFile, suffix: str) -> tuple[str, bytes]:
    matches = [member for member in handle.getmembers() if member.isfile() and member.name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one archive member ending in {suffix!r}, found {len(matches)}")
    extracted = handle.extractfile(matches[0])
    if extracted is None:
        raise FileNotFoundError(matches[0].name)
    return matches[0].name, extracted.read()


def selected_payload(response: np.ndarray) -> dict[str, object]:
    statistics = generic_probe_statistics(np.asarray(response, dtype=np.float64)[None])
    active = statistics.active[0]
    active_ids = np.flatnonzero(active)
    count = max(1, int(np.ceil(0.20 * active_ids.size)))
    order = np.argsort(statistics.rho[0, active_ids], kind="stable")
    selected = active_ids[order[-count:]]
    selected = selected[np.argsort(statistics.rho[0, selected], kind="stable")[::-1]]
    selected_rho = statistics.rho[0, selected]
    if float(selected_rho.sum()) <= 0.0:
        raise ValueError("frozen K0-B selected rho has zero total")
    return {
        "active_probe_ids": active_ids.astype(int).tolist(),
        "selected_probe_ids": selected.astype(int).tolist(),
        "selected_rho": selected_rho.tolist(),
        "weights": (selected_rho / selected_rho.sum()).tolist(),
        "rho_all": statistics.rho[0].tolist(),
        "logit_response_energy_all": statistics.energy[0].tolist(),
    }


def main() -> None:
    args = parse_args()
    archive = args.k0b_archive.resolve()
    archive_sha256 = sha256_file(archive)
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"K0-B archive hash mismatch: {archive_sha256}")
    selections: dict[str, object] = {}
    response_sources = []
    with tarfile.open(archive, "r:gz") as handle:
        index_name, index_payload = member_bytes(handle, "/selected_public_indices.npy")
        indices = np.load(io.BytesIO(index_payload), allow_pickle=False)
        if sha256_array(indices) != EXPECTED_D_SELECT_SHA256:
            raise ValueError("K0-B D_select index hash mismatch")
        for arm in ARMS:
            arm_payload: dict[str, object] = {}
            for client_id in range(4):
                suffix = f"/responses/{arm}_client{client_id}.npz"
                response_name, payload = member_bytes(handle, suffix)
                with np.load(io.BytesIO(payload), allow_pickle=False) as values:
                    response = np.asarray(values["centered_response"], dtype=np.float64)
                if response.shape != (1000, 128, 10):
                    raise ValueError(f"unexpected K0-B response shape: {response.shape}")
                client_payload = {}
                for bank_name, start in (("a", 0), ("b", BANK_SIZE)):
                    client_payload[bank_name] = selected_payload(
                        response[:, start : start + BANK_SIZE]
                    )
                arm_payload[str(client_id)] = client_payload
                response_sources.append(
                    {
                        "arm": arm,
                        "client": client_id,
                        "archive_member": response_name,
                        "bytes": len(payload),
                        "sha256": sha256_bytes(payload),
                    }
                )
            selections[arm] = arm_payload
    manifest = {
        "protocol": "cle_k1_b0_cdr_snr_selection_v1",
        "source_k0b_archive": {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
        },
        "d_select": {
            "archive_member": index_name,
            "count": int(indices.size),
            "sha256": sha256_array(indices),
        },
        "bank_sha256": EXPECTED_BANK_HASHES,
        "selection_rule": "top 20% rho among active probes; stable descending rho order",
        "selection_arms": {"hfl": "h9", "local": "l9"},
        "response_sources": response_sources,
        "selections": selections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
