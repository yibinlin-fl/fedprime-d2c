from __future__ import annotations

import copy
import json
import random
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import fedprime.engine.round_recovery as recovery
import scripts.publish_cle_round_recovery as publisher
from scripts.openi_cle_v2_moderate_entry import extract_round_resume


def seed():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)


@pytest.mark.parametrize("consumed", [0, 2, 5, 6])
def test_public_iterator_replay_preserves_next_batch_and_global_rng(consumed):
    seed()
    loader = DataLoader(TensorDataset(torch.arange(20)), batch_size=4, shuffle=True)
    original = recovery.ReplayablePublicIterator(loader)
    # Model construction and local training consume RNG between iterator creation and use.
    torch.rand(7)
    for _ in range(consumed):
        try:
            next(original)
        except StopIteration:
            pass
        torch.rand(3)
    state = original.state_dict()
    current = recovery.rng_state()
    restored = recovery.ReplayablePublicIterator.from_state(loader, state)
    assert torch.equal(torch.get_rng_state(), current["torch"])
    for _ in range(3):
        try:
            expected = next(original)
        except StopIteration:
            with pytest.raises(StopIteration):
                next(restored)
            break
        after_expected = recovery.rng_state()
        if consumed == 0:
            recovery.restore_rng(current)
        actual = next(restored)
        assert torch.equal(actual[0], expected[0])
        recovery.restore_rng(after_expected)
        current = after_expected


def make_experiment(root, arm):
    root.mkdir(parents=True)
    config = {
        "experiment_name": arm, "output_root": str(root.parent), "seed": 0, "num_workers": 0,
        "train": {"rounds": 3, "pretrain_epochs": 0},
        "method": {"communication": "aughfl_fidelity" if arm == "aughfl_fidelity" else "asymhfl_val"},
    }
    return SimpleNamespace(
        config=config, output_dir=root,
        _communication_strategy=SimpleNamespace(last_metrics={"previous": 3.0}),
        _last_fedease_metrics={}, _last_spurious_metrics={}, _last_baseline_metrics={},
    )


@pytest.mark.parametrize("arm", ["aughfl_fidelity", "rahfl_fidelity"])
def test_round_snapshot_continuation_matches_uninterrupted_adam(tmp_path, monkeypatch, arm):
    seed()
    exp = make_experiment(tmp_path / "original", arm)
    options = {
        "shard": arm, "save_rounds": [1, 2, 3], "input_sha256": "frozen", "code_sha256": "source",
        "configs_root": str(tmp_path), "archive": str(tmp_path / "progress.tar.gz"),
    }
    hook = recovery.RoundRecovery(exp.config, options)
    model = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.Dropout(.2), torch.nn.Linear(3, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    loader = DataLoader(TensorDataset(torch.arange(12).float()[:, None]), batch_size=3, shuffle=True)
    stream = recovery.ReplayablePublicIterator(loader)
    generator = torch.Generator().manual_seed(32)
    generators = [(generator, generator.get_state())]

    def step(mod, opt, iterator):
        values = next(iterator)[0]
        jitter = float(np.random.rand() + random.random())
        noise = torch.rand(values.shape)
        loss = (mod(values + noise) + jitter).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.detach())

    step(model, optimizer, stream)
    (exp.output_dir / "metrics.csv").write_text("round,loss\n0,1\n")
    (exp.output_dir / "local_batch_trace.jsonl").write_text('{"round":1}\n')
    uploads = []
    monkeypatch.setattr(recovery.subprocess, "run", lambda *args, **kwargs: uploads.append(args))
    hook.save(exp, {0: model}, {0: optimizer}, stream, generators, 1)
    assert len(uploads) == 1
    state_root = exp.output_dir / "checkpoints/round_recovery"
    (state_root / "history").mkdir()
    for name in ("metrics.csv", "local_batch_trace.jsonl"):
        (state_root / "history" / name).write_bytes((exp.output_dir / name).read_bytes())
    expected_loss = step(model, optimizer, stream)
    expected_state = copy.deepcopy(model.state_dict())
    expected_rng = recovery.rng_state()
    new_exp = make_experiment(tmp_path / "resumed", arm)
    new_model = copy.deepcopy(model)
    new_opt = torch.optim.Adam(new_model.parameters(), lr=.01)
    new_hook = recovery.RoundRecovery(new_exp.config, {**options, "resume_root": str(state_root)})
    completed, new_stream = new_hook.restore(
        new_exp, {0: new_model}, {0: new_opt}, loader, None, generators
    )
    assert completed == 1
    assert (new_exp.output_dir / "metrics.csv").read_text() == "round,loss\n0,1\n"
    assert step(new_model, new_opt, new_stream) == expected_loss
    assert all(torch.equal(value, expected_state[key]) for key, value in new_model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), expected_rng["torch"])
    wrong = recovery.RoundRecovery(new_exp.config, {**options, "input_sha256": "other", "resume_root": str(state_root)})
    with pytest.raises(ValueError, match="identity mismatch"):
        wrong.restore(new_exp, {0: new_model}, {0: new_opt}, loader, None, generators)
    wrong_code = recovery.RoundRecovery(new_exp.config, {**options, "code_sha256": "changed", "resume_root": str(state_root)})
    with pytest.raises(ValueError, match="identity mismatch: code_sha256"):
        wrong_code.restore(new_exp, {0: new_model}, {0: new_opt}, loader, None, generators)


def test_progress_upload_retains_previous_round_on_next_failure(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    state_dir = root / "checkpoints/round_recovery"
    state_dir.mkdir(parents=True)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "CONTRACT.json").write_text("{}")
    archive = tmp_path / "cle_aughfl_fidelity_formal_progress_outputs.tar.gz"
    context = SimpleNamespace(output_path=str(tmp_path / "remote"))
    uploaded = []
    monkeypatch.setattr(publisher, "upload", lambda ctx, paths: uploaded.append(paths[0]))
    for completed in (1, 2):
        (state_dir / "round_state.pt").write_bytes(b"state")
        (state_dir / "PROGRESS.json").write_text(json.dumps({"completed_rounds": completed}))
        publisher.publish(root, configs, archive, context)
    first_hash = uploaded[0].read_bytes()
    monkeypatch.setattr(publisher, "upload", lambda ctx, paths: (_ for _ in ()).throw(RuntimeError("upload failed")))
    (state_dir / "PROGRESS.json").write_text(json.dumps({"completed_rounds": 4}))
    with pytest.raises(RuntimeError, match="upload failed"):
        publisher.publish(root, configs, archive, context)
    assert uploaded[0].read_bytes() == first_hash
    assert uploaded[0] != uploaded[1]
    with tarfile.open(uploaded[0]) as handle:
        assert "round_state.pt" in handle.getnames()
    with pytest.raises(RuntimeError, match="context unavailable"):
        publisher.publish(root, configs, archive, None)


def test_progress_extract_rejects_links_and_wrong_arm(tmp_path):
    archive = tmp_path / "cle_aughfl_fidelity_formal_progress_round_002_outputs.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../outside"
        handle.addfile(info)
    with pytest.raises(ValueError, match="Unsafe"):
        extract_round_resume(archive, tmp_path / "restore", "aughfl_fidelity")
    with pytest.raises(ValueError, match="selected arm"):
        extract_round_resume(archive, tmp_path / "wrong", "rahfl_fidelity")


def test_robust_entry_publishes_identity_from_export_without_git(tmp_path, monkeypatch):
    import scripts.openi_cle_v2_moderate_entry as entry
    import sys
    package = tmp_path / "input"
    package.mkdir()
    source = tmp_path / "input.tar.gz"
    source.write_bytes(b"input")
    for name in entry.RECOVERY_SOURCE_DIRS:
        (tmp_path / name).mkdir()
    (tmp_path / "requirements.txt").write_text("torch\n")
    (tmp_path / "fedprime/runner.py").write_text("enabled = True\n")
    assert not (tmp_path / ".git").exists()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    fingerprint = entry.recovery_source_sha256(tmp_path)
    events = []
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry.os, "chdir", lambda path: None)
    monkeypatch.setattr(entry, "prepare_c2net", lambda: SimpleNamespace(output_path=str(tmp_path / "remote")))
    monkeypatch.setattr(entry, "find_archive", lambda roots: source)
    monkeypatch.setattr(entry, "extract_archive", lambda *args: package)

    def run(command, environment):
        if "--prepare-only" in command:
            root = Path(command[command.index("--config-root") + 1])
            root.mkdir(parents=True)
            (root / "CONTRACT_aughfl_fidelity.json").write_text("{}")
            events.append("prepared")
        elif "scripts/run_cle_v2_moderate.py" in command:
            options = json.loads(environment["CLE_ROUND_RECOVERY"])
            assert options["save_rounds"] == [20, 30, 40]
            assert options["code_sha256"] == fingerprint
            events.append("training")
        else:
            events.append("analysis")

    monkeypatch.setattr(entry, "run", run)
    monkeypatch.setattr(entry, "upload", lambda context, paths: events.append([p.name for p in paths]))
    monkeypatch.setattr(sys, "argv", ["entry", "--mode=formal", "--shard=aughfl_fidelity",
                                    "--confirm_formal=true", "--skip_install=true"])
    entry.main()
    assert events[0] == "prepared"
    assert events[1] == ["RUN_STARTED.json", "CONTRACT_aughfl_fidelity.json"]
    assert events[2] == "training"
    assert events[3][0].endswith("_recovery_outputs.tar.gz")
    assert events[4] == "analysis"
    assert json.loads(next(tmp_path.glob("outputs/*/RUN_STARTED.json")).read_text())["identity"]["code_sha256"] == fingerprint


def test_recovery_fingerprint_checks_runner_and_vendor_source(tmp_path):
    import scripts.openi_cle_v2_moderate_entry as entry

    for name in entry.RECOVERY_SOURCE_DIRS:
        (tmp_path / name).mkdir()
    (tmp_path / "requirements.txt").write_text("torch\n")
    vendor = tmp_path / "RAHFL-master/loss.py"
    vendor.write_text("weight = 1\n")
    first = entry.recovery_source_sha256(tmp_path)
    (tmp_path / "README.md").write_text("Documentation update")
    assert entry.recovery_source_sha256(tmp_path) == first
    vendor.write_text("weight = 2\n")
    assert entry.recovery_source_sha256(tmp_path) != first


class SmallViewDataset(torch.utils.data.Dataset):
    def __init__(self, public=False):
        self.public = public

    def __len__(self):
        return 12

    def __getitem__(self, index):
        base = torch.full((3, 4, 4), index / 12)
        views = [base + torch.rand_like(base) * .1 + (random.random() + np.random.rand()) * .01
                 for _ in range(4)]
        return ([views[:3], views[:3]] if self.public else views), index % 3


class SmallClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 8), torch.nn.Tanh())
        self.classifier = torch.nn.Linear(8, 10)

    def forward(self, values):
        return self.classifier(self.backbone(values))


@pytest.mark.parametrize("arm", ["aughfl_fidelity", "rahfl_fidelity"])
def test_full_runner_resume_matches_control_with_real_communication(tmp_path, monkeypatch, arm):
    import fedprime.methods.rahfl_asymhfl as runner
    from scripts.run_cle_v2_moderate import moderate_arm_config

    def strict_loaders(**kwargs):
        private = [
            DataLoader(SmallViewDataset(), batch_size=4, shuffle=True,
                               generator=torch.Generator().manual_seed(client + 30))
            for client in range(2)
        ]
        test = DataLoader(TensorDataset(torch.ones(8, 3, 4, 4), torch.arange(8) % 3), batch_size=4)
        return private, test, {}, {client: torch.ones(10) for client in range(2)}

    audit = DataLoader(TensorDataset(torch.ones(8, 3, 4, 4), torch.arange(8) % 3), batch_size=4)
    monkeypatch.setattr(runner, "build_strict_fit_audit_loaders", strict_loaders)
    monkeypatch.setattr(runner, "build_client_audit_loaders", lambda *a, **k: {0: audit, 1: audit})
    monkeypatch.setattr(runner, "build_public_loader", lambda **k: DataLoader(
        SmallViewDataset(public=True) if arm == "aughfl_fidelity" else
        TensorDataset(torch.arange(12 * 48).reshape(12, 3, 4, 4).float() / 576, torch.arange(12) % 3),
        batch_size=4, shuffle=True, num_workers=0,
    ))
    monkeypatch.setattr(runner, "build_models", lambda *a: {i: SmallClassifier() for i in range(2)})
    monkeypatch.setattr(runner, "load_operator_metadata", lambda *a: None)
    monkeypatch.setattr(runner, "corruption_group_names_from_test_loader", lambda *a: [])
    remote = tmp_path / "remote"
    remote.mkdir()

    def upload(ctx, paths):
        import shutil
        for path in paths:
            shutil.copy2(path, remote / path.name)

    monkeypatch.setattr(publisher, "upload", upload)
    interrupt = {"enabled": False}

    def publish_subprocess(command, **kwargs):
        values = dict(zip(command[3::2], command[4::2]))
        saved = publisher.publish(
            Path(values["--experiment-root"]), Path(values["--configs-root"]), Path(values["--archive"]),
            SimpleNamespace(output_path=str(remote)),
        )
        if interrupt["enabled"] and "round_002" in saved.name:
            raise RuntimeError("simulated platform stop after published round 2")

    monkeypatch.setattr(recovery.subprocess, "run", publish_subprocess)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "CONTRACT.json").write_text("{}")

    def execute(name, with_hook, resume=None):
        cfg = moderate_arm_config(arm, package_root=tmp_path / "input", mode="formal",
                                  output_root=tmp_path / name, device="cpu")
        cfg["models"]["names"] = ["small", "small"]
        cfg["train"].update(rounds=4, max_local_batches=1, public_batches_per_round=2,
                            max_test_batches=1, batch_size=4)
        cfg["checkpoints"]["load_dir"] = ""
        exp = runner.AsymHFLExperiment(cfg)
        if with_hook:
            exp.round_recovery = recovery.RoundRecovery(cfg, {
                "shard": arm, "save_rounds": [2, 4], "configs_root": str(configs),
                "archive": str(tmp_path / name / f"cle_{arm}_formal_progress_outputs.tar.gz"),
                "input_sha256": "same-data", "code_sha256": "same-code",
                "resume_root": str(resume) if resume else None,
            })
        exp.run()
        return exp.output_dir

    control = execute("control", False)
    uninterrupted = execute("snapshots", True)
    interrupt["enabled"] = True
    with pytest.raises(RuntimeError, match="simulated platform stop"):
        execute("stopped", True)
    assert (remote / f"cle_{arm}_formal_progress_round_002_outputs.tar.gz").is_file()
    restore = extract_round_resume(
        remote / f"cle_{arm}_formal_progress_round_002_outputs.tar.gz", tmp_path / "resume", arm
    )
    interrupt["enabled"] = False
    resumed = execute("resumed", True, restore)
    for client in range(2):
        expected = torch.load(control / f"checkpoints/client_{client}.pt", weights_only=True)
        for root in (uninterrupted, resumed):
            actual = torch.load(root / f"checkpoints/client_{client}.pt", weights_only=True)
            assert all(torch.equal(value, actual[key]) for key, value in expected.items())
    import csv
    for root in (uninterrupted, resumed):
        with (control / "metrics.csv").open() as f:
            expected = list(csv.DictReader(f))
        with (root / "metrics.csv").open() as f:
            actual = list(csv.DictReader(f))
        assert [row["round"] for row in actual] == ["0", "1", "2", "3"]
        for before, after in zip(expected, actual):
            assert all(before[k] == after[k] for k in before if k != "round_seconds")
        assert (root / "local_batch_trace.jsonl").read_bytes() == (control / "local_batch_trace.jsonl").read_bytes()
