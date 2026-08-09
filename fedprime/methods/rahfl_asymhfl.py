from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from fedprime.augmentations.prime_adapter import build_prime_module
from fedprime.data.fedease import (
    build_fedease_evaluation_loaders,
    build_fedease_fit_augmix_loaders,
    build_fedease_oracle_augmix_loaders,
    load_client_class_environment_counts,
)
from fedprime.data.fedfalsify import (
    build_client_audit_loaders,
    build_fedfalsify_loaders,
)
from fedprime.data.loaders import (
    build_augmix_private_loaders,
    build_corruption_skew_augmix_loaders,
    build_corruption_skew_pretrain_loaders,
    build_fedclear_private_loaders,
    build_prime_dcl_private_loaders,
    build_private_loaders,
    build_public_loader,
    corruption_group_names_from_test_loader,
    dataset_stats,
    load_corruption_skew_client_labels,
    load_private_labels,
    normalize_batch,
    partition_private_data,
)
from fedprime.methods.local_prime import train_local_prime_epoch
from fedprime.methods.local_prime import train_local_prime_dcl_epoch
from fedprime.methods.local_fedease import train_local_fedease_epoch
from fedprime.methods.local_fedclear import train_local_fedclear_epoch
from fedprime.methods.local_rahfl import train_local_augmix_dcl_epoch
from fedprime.methods.conditional_dependence import (
    BufferedConditionalMomentAlignment,
    FrozenRandomProjector,
)
from fedprime.methods.environment_structural_transfer import (
    aggregate_environment_balanced_relations,
    aggregate_leave_one_out_pair_relations,
    finalize_client_relations,
    finalize_pair_qualified_client_relations,
    new_relation_accumulator,
)
from fedprime.engine.cle_metrics import evaluate_cle_split, write_cle_evaluation
from fedprime.engine.operator_metrics import (
    load_operator_metadata,
    summarize_operator_splits,
)
from fedprime.methods.nir_dcl import NIRDCLFeatureQueue
from fedprime.methods.ird import (
    anchor_disagreement,
    invariant_anchor,
    leave_one_out_median,
    smooth_worst_view_distillation,
)
from fedprime.methods.pccd import (
    leave_one_out_consensus_teacher,
    log_opinion_consensus,
    normalized_entropy_confidence,
    paired_counterfactual_distillation,
    probability_view_disagreement,
    teacher_margin,
)
from fedprime.augmentations.counterfactual import build_counterfactual_views
from fedprime.models.factory import build_models, forward_logits
from fedprime.communication.public_logits import (
    CommunicationContext,
    build_core_communication_strategy,
)
from fedprime.communication.baselines import build_baseline_communication_strategy
from fedprime.utils.config import save_config
from fedprime.utils.env import resolve_device, seed_everything


class AsymHFLExperiment:
    """Unified runner for RAHFL-style AsymHFL, RAHFL+PRIME, and FedCARA."""

    def __init__(self, config: dict):
        self.config = config
        self.device = resolve_device(config.get("device", "auto"))
        self.output_dir = Path(config["output_root"]) / config["experiment_name"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, self.output_dir / "config.resolved.json")
        seed_everything(int(config.get("seed", 0)))
        self._nir_dcl_queues: dict[int, NIRDCLFeatureQueue] = {}
        self._last_ccre_metrics: dict[str, float] = {}
        self._last_ird_metrics: dict[str, float] = {}
        self._last_pccd_metrics: dict[str, float] = {}
        self._last_fedease_metrics: dict[str, float] = {}
        self._last_baseline_metrics: dict[str, float] = {}
        self._fedease_projectors: dict[int, FrozenRandomProjector] = {}
        self._fedease_cdep_v2_memories: dict[int, BufferedConditionalMomentAlignment] = {}
        self._fedease_client_relations: dict[int, dict[str, torch.Tensor]] = {}
        self._fedease_global_relation: dict[str, torch.Tensor | float] | None = None
        self._fedease_recipient_relations: dict[int, dict[str, torch.Tensor | float]] = {}
        self._fedease_evaluation_loaders = {}
        communication_name = str(config.get("method", {}).get("communication", "asymhfl"))
        self._communication_strategy = build_core_communication_strategy(communication_name)
        if self._communication_strategy is None:
            self._communication_strategy = build_baseline_communication_strategy(
                communication_name, config.get("method", {})
            )
        self._communication_private_loaders = None

    def run(self) -> None:
        data_cfg = self.config["data"]
        train_cfg = self.config["train"]
        method_cfg = self.config["method"]
        model_cfg = self.config["models"]

        num_clients = len(model_cfg["names"])
        num_classes = int(data_cfg.get("num_classes", 10))
        stats = dataset_stats(data_cfg.get("private_dataset", "cifar10"))

        use_prime = bool(method_cfg.get("use_prime", False))
        use_prime_dcl = use_prime and bool(method_cfg.get("use_dcl", True))
        scenario = str(data_cfg.get("scenario", "rahfl_cifar10c")).lower()
        prepared_corruption_scenarios = {
            "corruption_skew",
            "cle_hfl",
            "cle_hfl_v2",
        }
        strict_cfg = method_cfg.get("strict_fit_audit", {})
        strict_fit_audit = bool(strict_cfg.get("enabled", False))
        self._routing_audit_loaders = {}
        pretrain_loaders = None
        if scenario in prepared_corruption_scenarios:
            if use_prime:
                raise ValueError(f"{scenario} currently supports AugMix/SARA-style local training, not PRIME.")
            cl_module = str(method_cfg.get("cl_module", "dcl")).lower()
            client_splits = None
            if strict_fit_audit:
                if cl_module == "ccre":
                    raise ValueError("strict_fit_audit currently supports DCL or FedEASE local training")
                split_path = strict_cfg.get(
                    "split_path",
                    self.output_dir / "strict_fit_audit_split.npz",
                )
                print(
                    f"[setup] building strict fit/audit loaders; split={split_path}",
                    flush=True,
                )
                private_loaders, test_loader, client_splits, self._client_class_counts = (
                    build_fedfalsify_loaders(
                        root=data_cfg["private_root"],
                        num_clients=num_clients,
                        train_batch_size=train_cfg["batch_size"],
                        test_batch_size=train_cfg.get("test_batch_size", 512),
                        num_workers=int(self.config.get("num_workers", 2)),
                        split_path=split_path,
                        audit_ratio=float(strict_cfg.get("audit_ratio", 0.15)),
                        min_audit_per_class=int(strict_cfg.get("min_audit_per_class", 5)),
                        min_fit_per_class=int(strict_cfg.get("min_fit_per_class", 2)),
                        seed=int(strict_cfg.get("seed", self.config.get("seed", 0))),
                        num_classes=num_classes,
                        augmix_module=method_cfg.get("augmix_module", "jsd"),
                    )
                )
                self._routing_audit_loaders = build_client_audit_loaders(
                    client_splits,
                    batch_size=int(strict_cfg.get("audit_batch_size", 256)),
                    num_workers=int(self.config.get("num_workers", 2)),
                )
                if cl_module == "fedease":
                    private_loaders = build_fedease_fit_augmix_loaders(
                        root=data_cfg["private_root"],
                        client_splits=client_splits,
                        train_batch_size=train_cfg["batch_size"],
                        num_workers=int(self.config.get("num_workers", 2)),
                        augmix_module=method_cfg.get("augmix_module", "jsd"),
                        environment_annotations=getattr(
                            self,
                            "_fedease_environment_annotations",
                            None,
                        ),
                    )
                pretrain_loaders = private_loaders
                print(
                    "[setup] strict routing enabled: fit-only gradients, "
                    "audit-only AsymHFL routing, final test for reporting only",
                    flush=True,
                )
            elif cl_module == "fedease":
                print(f"[setup] building {scenario} FedEASE oracle-environment loaders", flush=True)
                private_loaders, test_loader, _, _ = build_fedease_oracle_augmix_loaders(
                    root=data_cfg["private_root"],
                    num_clients=num_clients,
                    train_batch_size=train_cfg["batch_size"],
                    test_batch_size=train_cfg.get("test_batch_size", 512),
                    num_workers=int(self.config.get("num_workers", 2)),
                    augmix_module=method_cfg.get("augmix_module", "jsd"),
                    environment_annotations=getattr(self, "_fedease_environment_annotations", None),
                )
            elif cl_module == "ccre":
                print(f"[setup] building {scenario} FedCLEAR private loaders", flush=True)
                private_loaders, test_loader, _, _ = build_fedclear_private_loaders(
                    root=data_cfg["private_root"],
                    num_clients=num_clients,
                    train_batch_size=train_cfg["batch_size"],
                    test_batch_size=train_cfg.get("test_batch_size", 512),
                    num_workers=int(self.config.get("num_workers", 2)),
                )
            else:
                print(f"[setup] building {scenario} AugMix private loaders", flush=True)
                private_loaders, test_loader, _, _ = build_corruption_skew_augmix_loaders(
                    root=data_cfg["private_root"],
                    num_clients=num_clients,
                    train_batch_size=train_cfg["batch_size"],
                    test_batch_size=train_cfg.get("test_batch_size", 512),
                    num_workers=int(self.config.get("num_workers", 2)),
                    augmix_module=method_cfg.get("augmix_module", "jsd"),
                )
            if not strict_fit_audit:
                pretrain_loaders = build_corruption_skew_pretrain_loaders(
                    root=data_cfg["private_root"],
                    num_clients=num_clients,
                    train_batch_size=train_cfg["batch_size"],
                    num_workers=int(self.config.get("num_workers", 2)),
                )
                client_labels = load_corruption_skew_client_labels(data_cfg["private_root"], num_clients)
                self._client_class_counts = {
                    client_id: torch.bincount(
                        torch.as_tensor(labels, dtype=torch.long),
                        minlength=num_classes,
                    ).float()
                    for client_id, labels in client_labels.items()
                }
            if cl_module == "fedease":
                num_environments = int(method_cfg.get("fedease", {}).get("num_environments", 4))
                self._client_class_environment_counts = load_client_class_environment_counts(
                    root=data_cfg["private_root"],
                    num_clients=num_clients,
                    num_classes=num_classes,
                    num_environments=num_environments,
                    environment_annotations=getattr(self, "_fedease_environment_annotations", None),
                    fit_indices=(
                        {
                            client_id: split.fit_indices
                            for client_id, split in client_splits.items()
                        }
                        if client_splits is not None
                        else None
                    ),
                )
                self._fedease_evaluation_loaders = build_fedease_evaluation_loaders(
                    root=data_cfg["private_root"],
                    num_clients=num_clients,
                    batch_size=train_cfg.get("test_batch_size", 512),
                    num_workers=int(self.config.get("num_workers", 2)),
                )
                print(
                    "[setup] FedEASE evaluation splits: "
                    + ",".join(self._fedease_evaluation_loaders)
                    if self._fedease_evaluation_loaders
                    else "[warning] no extended FedEASE evaluation splits found",
                    flush=True,
                )
            else:
                self._client_class_environment_counts = {}
            self._corruption_group_names = corruption_group_names_from_test_loader(test_loader)
            prime_aug = None
        else:
            print("[setup] AsymHFL/FedCARA loading private labels", flush=True)
            labels = load_private_labels(data_cfg["private_root"], data_cfg["private_corrupt_rate"])
            print("[setup] AsymHFL/FedCARA loading/creating private partition", flush=True)
            dataidx_map = partition_private_data(
                labels=labels,
                num_clients=num_clients,
                num_classes=num_classes,
                partition=data_cfg.get("partition", "dirichlet"),
                dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.5)),
                max_samples_per_client=data_cfg.get("private_samples_per_client"),
                partition_indices_path=data_cfg.get("partition_indices_path"),
                partition_seed=int(self.config.get("seed", 0)),
            )
            self._client_class_counts = self._build_client_class_counts(labels, dataidx_map, num_classes)
            self._corruption_group_names = []

        if scenario not in prepared_corruption_scenarios and use_prime:
            print("[setup] building PRIME private loaders", flush=True)
            if use_prime_dcl:
                private_loaders, test_loader = build_prime_dcl_private_loaders(
                    cifar10c_root=data_cfg["private_root"],
                    dataidx_map=dataidx_map,
                    train_batch_size=train_cfg["batch_size"],
                    test_batch_size=train_cfg.get("test_batch_size", 512),
                    corrupt_rate=data_cfg["private_corrupt_rate"],
                    test_corrupt_rate=data_cfg["test_corrupt_rate"],
                    num_workers=int(self.config.get("num_workers", 2)),
                )
            else:
                private_loaders, test_loader = build_private_loaders(
                    cifar10c_root=data_cfg["private_root"],
                    dataidx_map=dataidx_map,
                    train_batch_size=train_cfg["batch_size"],
                    test_batch_size=train_cfg.get("test_batch_size", 512),
                    corrupt_rate=data_cfg["private_corrupt_rate"],
                    test_corrupt_rate=data_cfg["test_corrupt_rate"],
                    num_workers=int(self.config.get("num_workers", 2)),
                    raw_for_prime=True,
                )
            prime_aug = build_prime_module(stats, method_cfg.get("prime", {})).to(self.device)
        elif scenario not in prepared_corruption_scenarios:
            print("[setup] building AugMix private loaders", flush=True)
            private_loaders, test_loader, _, _ = build_augmix_private_loaders(
                cifar10c_root=data_cfg["private_root"],
                dataidx_map=dataidx_map,
                train_batch_size=train_cfg["batch_size"],
                test_batch_size=train_cfg.get("test_batch_size", 512),
                corrupt_rate=data_cfg["private_corrupt_rate"],
                test_corrupt_rate=data_cfg["test_corrupt_rate"],
                num_workers=int(self.config.get("num_workers", 2)),
                augmix_module=method_cfg.get("augmix_module", "jsd"),
            )
            prime_aug = None

        self._communication_private_loaders = private_loaders
        strategy_requires_public = (
            self._communication_strategy is None
            or bool(getattr(self._communication_strategy, "requires_public_data", True))
        )
        if (
            not strategy_requires_public
            or self._use_ebst_communication(method_cfg)
        ):
            print("[setup] per-round public loader is not required", flush=True)
            public_loader = None
            public_iter = None
        else:
            print("[setup] building public loader", flush=True)
            public_loader = build_public_loader(
                cifar100_root=data_cfg["public_root"],
                public_size=int(data_cfg.get("public_size", 5000)),
                batch_size=train_cfg["public_batch_size"],
                num_workers=int(self.config.get("num_workers", 2)),
                seed=int(self.config.get("seed", 0)),
                download=bool(data_cfg.get("download_public", False)),
                public_dataset=str(data_cfg.get("public_dataset", "cifar100")),
                public_views=(
                    "aughfl_fidelity"
                    if str(method_cfg.get("communication", "")).lower() == "aughfl_fidelity"
                    else (
                        "augmix"
                        if str(method_cfg.get("communication", "")).lower() == "aughfl"
                        else "tensor"
                    )
                ),
                public_view_clients=num_clients,
            )
            public_iter = iter(public_loader)

        print("[setup] building heterogeneous client models", flush=True)
        models = build_models(model_cfg["names"], num_classes)
        models = {idx: model.to(self.device) for idx, model in models.items()}
        self._load_models_if_configured(models)
        optimizers = {idx: self._build_optimizer(model) for idx, model in models.items()}

        pretrain_epochs = int(train_cfg.get("pretrain_epochs", 0))
        if pretrain_epochs > 0:
            print(f"[setup] running local CE pretraining for {pretrain_epochs} epochs", flush=True)
            self._pretrain_phase(
                models,
                optimizers,
                pretrain_loaders if pretrain_loaders is not None else private_loaders,
                pretrain_epochs,
                train_cfg,
            )

        metrics_path = self.output_dir / "metrics.csv"
        group_names = getattr(self, "_corruption_group_names", [])
        operator_metadata = load_operator_metadata(data_cfg["private_root"])
        if operator_metadata:
            print(
                "[setup] CLE-HFL v2 operator evaluation enabled: "
                f"seen={len(operator_metadata.get('seen_operators', []))} "
                f"unseen={len(operator_metadata.get('unseen_operators', []))}; "
                "metadata is not visible to training",
                flush=True,
            )
        group_metrics_path = self.output_dir / "corruption_group_acc.csv"
        client_group_metrics_path = self.output_dir / "client_group_acc.csv"
        class_corruption_metrics_path = self.output_dir / "class_corruption_acc.csv"
        operator_split_metrics_path = self.output_dir / "operator_split_metrics.csv"
        group_file = group_metrics_path.open("w", newline="", encoding="utf-8") if group_names else None
        client_group_file = client_group_metrics_path.open("w", newline="", encoding="utf-8") if group_names else None
        class_corruption_file = (
            class_corruption_metrics_path.open("w", newline="", encoding="utf-8")
            if group_names
            else None
        )
        operator_split_file = (
            operator_split_metrics_path.open("w", newline="", encoding="utf-8")
            if operator_metadata
            else None
        )
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "round",
                    "round_seconds",
                    "peak_cuda_memory_mb",
                    "avg_acc",
                    "worst_acc",
                    "worst_group_acc",
                    "worst_client_group_acc",
                    "wcca",
                    "cfg",
                    "seen_avg_acc",
                    "seen_worst_acc",
                    "seen_wcca",
                    "seen_cfg",
                    "unseen_avg_acc",
                    "unseen_worst_acc",
                    "unseen_wcca",
                    "unseen_cfg",
                    "local_loss",
                    "col_loss",
                    "ccre_loss",
                    "ccre_worst_view_risk",
                    "ird_loss",
                    "ird_anchor_disagreement",
                    "ird_worst_view_kl",
                    "pccd_loss",
                    "pccd_teacher_entropy",
                    "pccd_teacher_confidence",
                    "pccd_view_disagreement",
                    "pccd_teacher_margin",
                    "pccd_worst_view_kl",
                    "fedease_clean_ce",
                    "fedease_classification_loss",
                    "fedease_ber_loss",
                    "fedease_jsd_loss",
                    "fedease_dcl_loss",
                    "fedease_cdep_loss",
                    "fedease_cdep_valid_classes",
                    "fedease_cdep_mean_abs_covariance",
                    "fedease_cdep_v2_valid_groups",
                    "fedease_cdep_v2_buffer_samples",
                    "fedease_cdep_v2_ramp",
                    "fedease_ber_valid_groups",
                    "fedease_ebst_loss",
                    "fedease_ebst_active_samples",
                    "fedease_scp_gradient_dot",
                    "fedease_scp_gradient_cosine",
                    "fedease_scp_conflict_rate",
                    "fedease_scp_projection_norm_ratio",
                    "fedease_ebst_valid_environment_fraction",
                    "fedease_ebst_valid_pair_fraction",
                    "fedease_ebst_mean_source_count",
                    "fedease_ebst_mean_gate",
                    "baseline_teacher_entropy",
                    "baseline_teacher_disagreement",
                    "baseline_teacher_weight_min",
                    "baseline_teacher_weight_max",
                    "baseline_view_consistency",
                    "baseline_coefficient_loss",
                    "baseline_coefficient_entropy",
                    "baseline_coefficient_diagonal",
                    "baseline_coefficient_offdiagonal",
                    "baseline_coefficient_drift",
                    "baseline_server_updates",
                ],
            )
            writer.writeheader()
            group_writer = None
            client_group_writer = None
            class_corruption_writer = None
            operator_split_writer = None
            if group_file is not None:
                group_writer = csv.DictWriter(
                    group_file,
                    fieldnames=["round", *group_names, "worst_group_acc"],
                )
                group_writer.writeheader()
            if client_group_file is not None:
                client_group_writer = csv.DictWriter(
                    client_group_file,
                    fieldnames=["round", "client", *group_names, "worst_client_group_acc"],
                )
                client_group_writer.writeheader()
            if class_corruption_file is not None:
                class_corruption_writer = csv.DictWriter(
                    class_corruption_file,
                    fieldnames=["round", "client", "class_id", "group", "acc", "total"],
                )
                class_corruption_writer.writeheader()
            if operator_split_file is not None:
                operator_split_writer = csv.DictWriter(
                    operator_split_file,
                    fieldnames=[
                        "round",
                        "split",
                        "avg_acc",
                        "worst_acc",
                        "worst_operator_acc",
                        "worst_client_operator_acc",
                        "wcca",
                        "cfg",
                    ],
                )
                operator_split_writer.writeheader()

            for round_idx in range(int(train_cfg["rounds"])):
                round_started = time.perf_counter()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                print(f"[heartbeat] round {round_idx:03d} start", flush=True)
                if self._communication_strategy is not None and not bool(
                    getattr(self._communication_strategy, "uses_accuracy_routing", False)
                ):
                    accs_before = [0.0 for _ in range(num_clients)]
                    class_accs_before = None
                elif self._use_cara_communication(method_cfg):
                    accs_before, class_accs_before = self._evaluate_detailed(models, test_loader, num_classes)
                elif (
                    self._use_ird_communication(method_cfg)
                    or self._use_pccd_communication(method_cfg)
                    or self._use_ebst_communication(method_cfg)
                ):
                    accs_before = [0.0 for _ in range(num_clients)]
                    class_accs_before = None
                elif self._use_ccad_communication(method_cfg) and not self._ccad_uses_asymhfl_route(method_cfg):
                    accs_before = [0.0 for _ in range(num_clients)]
                    class_accs_before = None
                else:
                    routing_loaders = getattr(self, "_routing_audit_loaders", {})
                    if routing_loaders:
                        accs_before = self._evaluate_private_loaders(models, routing_loaders)
                        print(
                            "[routing] audit accuracies="
                            + ",".join(f"c{idx}:{acc:.2f}" for idx, acc in enumerate(accs_before)),
                            flush=True,
                        )
                    else:
                        accs_before = self._evaluate(models, test_loader)
                    class_accs_before = None
                communication_phase = str(
                    getattr(self._communication_strategy, "phase", "pre_local")
                )
                if communication_phase == "post_local":
                    print(
                        f"[heartbeat] round {round_idx:03d} collaborative phase deferred until after local update",
                        flush=True,
                    )
                    col_loss = 0.0
                else:
                    print(f"[heartbeat] round {round_idx:03d} collaborative phase", flush=True)
                    col_loss = self._collaborative_phase(
                        models=models,
                        optimizers=optimizers,
                        public_loader=public_loader,
                        public_iter=public_iter,
                        accs=accs_before,
                        class_accs=class_accs_before,
                        stats=stats,
                        round_idx=round_idx,
                    )
                print(f"[heartbeat] round {round_idx:03d} local phase", flush=True)
                local_loss = self._local_phase(
                    models=models,
                    optimizers=optimizers,
                    private_loaders=private_loaders,
                    prime_aug=prime_aug,
                    use_prime=use_prime,
                    use_prime_dcl=use_prime_dcl,
                    train_cfg=train_cfg,
                    method_cfg=method_cfg,
                    stats=stats,
                    num_classes=num_classes,
                    round_idx=round_idx,
                )
                if communication_phase == "post_local":
                    print(f"[heartbeat] round {round_idx:03d} post-local collaborative phase", flush=True)
                    col_loss = self._collaborative_phase(
                        models=models,
                        optimizers=optimizers,
                        public_loader=public_loader,
                        public_iter=public_iter,
                        accs=accs_before,
                        class_accs=class_accs_before,
                        stats=stats,
                        round_idx=round_idx,
                    )
                if self._use_ebst_communication(method_cfg):
                    col_loss = float(self._last_fedease_metrics.get("ebst_loss", 0.0))
                if group_names:
                    group_summary = self._evaluate_corruption_groups(models, test_loader, group_names, num_classes)
                    accs = group_summary.get("client_accs")
                    if accs is None:
                        accs = self._evaluate(models, test_loader)
                else:
                    accs = self._evaluate(models, test_loader)
                    group_summary = {}
                split_summaries = (
                    summarize_operator_splits(group_summary, operator_metadata)
                    if operator_metadata
                    else {}
                )
                row = {
                    "round": round_idx,
                    "round_seconds": time.perf_counter() - round_started,
                    "peak_cuda_memory_mb": (
                        torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
                        if self.device.type == "cuda"
                        else 0.0
                    ),
                    "avg_acc": sum(accs) / len(accs),
                    "worst_acc": min(accs),
                    "worst_group_acc": group_summary.get("worst_group_acc", ""),
                    "worst_client_group_acc": group_summary.get("worst_client_group_acc", ""),
                    "wcca": group_summary.get("wcca", ""),
                    "cfg": group_summary.get("cfg", ""),
                    "seen_avg_acc": split_summaries.get("seen", {}).get("avg_acc", ""),
                    "seen_worst_acc": split_summaries.get("seen", {}).get("worst_acc", ""),
                    "seen_wcca": split_summaries.get("seen", {}).get("wcca", ""),
                    "seen_cfg": split_summaries.get("seen", {}).get("cfg", ""),
                    "unseen_avg_acc": split_summaries.get("unseen", {}).get("avg_acc", ""),
                    "unseen_worst_acc": split_summaries.get("unseen", {}).get("worst_acc", ""),
                    "unseen_wcca": split_summaries.get("unseen", {}).get("wcca", ""),
                    "unseen_cfg": split_summaries.get("unseen", {}).get("cfg", ""),
                    "local_loss": local_loss,
                    "col_loss": col_loss,
                    "ccre_loss": self._last_ccre_metrics.get("ccre_loss", ""),
                    "ccre_worst_view_risk": self._last_ccre_metrics.get("ccre_worst_view_risk", ""),
                    "ird_loss": self._last_ird_metrics.get("ird_loss", ""),
                    "ird_anchor_disagreement": self._last_ird_metrics.get("ird_anchor_disagreement", ""),
                    "ird_worst_view_kl": self._last_ird_metrics.get("ird_worst_view_kl", ""),
                    "pccd_loss": self._last_pccd_metrics.get("pccd_loss", ""),
                    "pccd_teacher_entropy": self._last_pccd_metrics.get("pccd_teacher_entropy", ""),
                    "pccd_teacher_confidence": self._last_pccd_metrics.get("pccd_teacher_confidence", ""),
                    "pccd_view_disagreement": self._last_pccd_metrics.get("pccd_view_disagreement", ""),
                    "pccd_teacher_margin": self._last_pccd_metrics.get("pccd_teacher_margin", ""),
                    "pccd_worst_view_kl": self._last_pccd_metrics.get("pccd_worst_view_kl", ""),
                    "fedease_clean_ce": self._last_fedease_metrics.get("clean_ce", ""),
                    "fedease_classification_loss": self._last_fedease_metrics.get("classification_loss", ""),
                    "fedease_ber_loss": self._last_fedease_metrics.get("ber_loss", ""),
                    "fedease_jsd_loss": self._last_fedease_metrics.get("jsd_loss", ""),
                    "fedease_dcl_loss": self._last_fedease_metrics.get("dcl_loss", ""),
                    "fedease_cdep_loss": self._last_fedease_metrics.get("cdep_loss", ""),
                    "fedease_cdep_valid_classes": self._last_fedease_metrics.get("cdep_valid_classes", ""),
                    "fedease_cdep_mean_abs_covariance": self._last_fedease_metrics.get("cdep_mean_abs_covariance", ""),
                    "fedease_cdep_v2_valid_groups": self._last_fedease_metrics.get("cdep_v2_valid_groups", ""),
                    "fedease_cdep_v2_buffer_samples": self._last_fedease_metrics.get("cdep_v2_buffer_samples", ""),
                    "fedease_cdep_v2_ramp": self._last_fedease_metrics.get("cdep_v2_ramp", ""),
                    "fedease_ber_valid_groups": self._last_fedease_metrics.get("ber_valid_groups", ""),
                    "fedease_ebst_loss": self._last_fedease_metrics.get("ebst_loss", ""),
                    "fedease_ebst_active_samples": self._last_fedease_metrics.get("ebst_active_samples", ""),
                    "fedease_scp_gradient_dot": self._last_fedease_metrics.get("scp_gradient_dot", ""),
                    "fedease_scp_gradient_cosine": self._last_fedease_metrics.get("scp_gradient_cosine", ""),
                    "fedease_scp_conflict_rate": self._last_fedease_metrics.get("scp_conflict", ""),
                    "fedease_scp_projection_norm_ratio": self._last_fedease_metrics.get(
                        "scp_projection_norm_ratio", ""
                    ),
                    "fedease_ebst_valid_environment_fraction": (
                        self._fedease_global_relation.get("valid_environment_fraction", "")
                        if self._fedease_global_relation is not None
                        else ""
                    ),
                    "fedease_ebst_valid_pair_fraction": (
                        self._fedease_global_relation.get("valid_pair_fraction", "")
                        if self._fedease_global_relation is not None
                        else ""
                    ),
                    "fedease_ebst_mean_source_count": (
                        self._fedease_global_relation.get("mean_source_count", "")
                        if self._fedease_global_relation is not None
                        else ""
                    ),
                    "fedease_ebst_mean_gate": (
                        self._fedease_global_relation.get("mean_gate", "")
                        if self._fedease_global_relation is not None
                        else ""
                    ),
                    "baseline_teacher_entropy": self._last_baseline_metrics.get("teacher_entropy", self._last_baseline_metrics.get("teacher_weight_entropy", "")),
                    "baseline_teacher_disagreement": self._last_baseline_metrics.get("teacher_disagreement", ""),
                    "baseline_teacher_weight_min": self._last_baseline_metrics.get("teacher_weight_min", ""),
                    "baseline_teacher_weight_max": self._last_baseline_metrics.get("teacher_weight_max", ""),
                    "baseline_view_consistency": self._last_baseline_metrics.get("view_consistency", ""),
                    "baseline_coefficient_loss": self._last_baseline_metrics.get("coefficient_loss", ""),
                    "baseline_coefficient_entropy": self._last_baseline_metrics.get("coefficient_entropy", ""),
                    "baseline_coefficient_diagonal": self._last_baseline_metrics.get("coefficient_diagonal", ""),
                    "baseline_coefficient_offdiagonal": self._last_baseline_metrics.get("coefficient_offdiagonal", ""),
                    "baseline_coefficient_drift": self._last_baseline_metrics.get("coefficient_drift", ""),
                    "baseline_server_updates": self._last_baseline_metrics.get("server_updates", ""),
                }
                writer.writerow(row)
                f.flush()
                if group_writer is not None:
                    group_writer.writerow({"round": round_idx, **group_summary.get("groups", {}), "worst_group_acc": group_summary.get("worst_group_acc", "")})
                    group_file.flush()
                if client_group_writer is not None:
                    for client_id, values in group_summary.get("clients", {}).items():
                        client_group_writer.writerow({
                            "round": round_idx,
                            "client": client_id,
                            **values,
                            "worst_client_group_acc": min(values.values()) if values else "",
                        })
                    client_group_file.flush()
                if class_corruption_writer is not None:
                    for cc_row in group_summary.get("class_corruption_rows", []):
                        class_corruption_writer.writerow({"round": round_idx, **cc_row})
                    class_corruption_file.flush()
                if operator_split_writer is not None:
                    for split_name, values in split_summaries.items():
                        operator_split_writer.writerow(
                            {"round": round_idx, "split": split_name, **values}
                        )
                    operator_split_file.flush()
                method_metrics = ""
                if row["ccre_loss"] != "":
                    method_metrics += (
                        f"ccre_loss={float(row['ccre_loss']):.4f} "
                        f"ccre_worst={float(row['ccre_worst_view_risk']):.4f} "
                    )
                if row["ird_loss"] != "":
                    method_metrics += (
                        f"ird_loss={float(row['ird_loss']):.4f} "
                        f"anchor_dis={float(row['ird_anchor_disagreement']):.4f} "
                        f"ird_worst_kl={float(row['ird_worst_view_kl']):.4f} "
                    )
                if row["pccd_loss"] != "":
                    method_metrics += (
                        f"pccd_loss={float(row['pccd_loss']):.4f} "
                        f"teacher_conf={float(row['pccd_teacher_confidence']):.4f} "
                        f"view_dis={float(row['pccd_view_disagreement']):.4f} "
                        f"teacher_margin={float(row['pccd_teacher_margin']):.4f} "
                        f"pccd_worst_kl={float(row['pccd_worst_view_kl']):.4f} "
                    )
                if row["fedease_classification_loss"] != "":
                    method_metrics += (
                        f"cls={float(row['fedease_classification_loss']):.4f} "
                        f"ber={float(row['fedease_ber_loss']):.4f} "
                        f"cdep={float(row['fedease_cdep_loss']):.4f} "
                        f"cdep_classes={float(row['fedease_cdep_valid_classes']):.2f} "
                    )
                    if row["fedease_cdep_v2_buffer_samples"] != "":
                        method_metrics += (
                            f"cdep_v2_groups={float(row['fedease_cdep_v2_valid_groups']):.2f} "
                            f"cdep_v2_buffer={float(row['fedease_cdep_v2_buffer_samples']):.0f} "
                            f"cdep_v2_ramp={float(row['fedease_cdep_v2_ramp']):.2f} "
                        )
                    if row["fedease_ebst_loss"] != "":
                        method_metrics += (
                            f"ebst={float(row['fedease_ebst_loss']):.4f} "
                            f"scp_conflict={float(row['fedease_scp_conflict_rate']):.2f} "
                            f"gate={float(row['fedease_ebst_mean_gate']):.3f} "
                            + (
                                f"valid_pairs={float(row['fedease_ebst_valid_pair_fraction']):.3f} "
                                f"sources={float(row['fedease_ebst_mean_source_count']):.2f} "
                                if row["fedease_ebst_valid_pair_fraction"] != ""
                                else ""
                            )
                            if row["fedease_ebst_mean_gate"] != ""
                            else f"ebst={float(row['fedease_ebst_loss']):.4f} "
                        )
                print(
                    f"[round {round_idx:03d}] "
                    f"avg_acc={row['avg_acc']:.2f} "
                    f"worst_acc={row['worst_acc']:.2f} "
                    + (
                        f"worst_group_acc={float(row['worst_group_acc']):.2f} "
                        f"worst_client_group_acc={float(row['worst_client_group_acc']):.2f} "
                        f"wcca={float(row['wcca']):.2f} "
                        f"cfg={float(row['cfg']):.2f} "
                        if row["worst_group_acc"] != "" else ""
                    )
                    +
                    f"local_loss={local_loss:.4f} "
                    f"col_loss={col_loss:.4f} "
                    f"{method_metrics}".rstrip(),
                    flush=True,
                )
                if split_summaries:
                    seen = split_summaries["seen"]
                    unseen = split_summaries["unseen"]
                    print(
                        f"[operator-eval {round_idx:03d}] "
                        f"seen={seen['avg_acc']:.2f}/{seen['wcca']:.2f}/"
                        f"CFG {seen['cfg']:.2f} "
                        f"unseen={unseen['avg_acc']:.2f}/{unseen['wcca']:.2f}/"
                        f"CFG {unseen['cfg']:.2f}",
                        flush=True,
                    )

        if group_file is not None:
            group_file.close()
        if client_group_file is not None:
            client_group_file.close()
        if class_corruption_file is not None:
            class_corruption_file.close()
        if operator_split_file is not None:
            operator_split_file.close()
        if self._fedease_evaluation_loaders:
            self._run_fedease_extended_evaluation(models, num_classes)
        if bool(self.config.get("checkpoints", {}).get("save_final", True)):
            self._save_models(models)
        else:
            print("[checkpoint] final model saving disabled by checkpoints.save_final=false", flush=True)

    def _build_optimizer(self, model):
        opt_cfg = self.config["train"].get("optimizer", {})
        name = opt_cfg.get("name", "adam").lower()
        lr = float(opt_cfg.get("lr", 1e-3))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))
        if name == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=float(opt_cfg.get("momentum", 0.9)),
                weight_decay=weight_decay,
            )
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _use_no_communication(self, method_cfg: dict) -> bool:
        return str(method_cfg.get("communication", "asymhfl")).lower() in {"none", "local_only"}

    def _use_cara_communication(self, method_cfg: dict) -> bool:
        return method_cfg.get("communication", "asymhfl").lower() in {"cara", "cara_c", "fedcara"}

    def _use_ccad_communication(self, method_cfg: dict) -> bool:
        return method_cfg.get("communication", "asymhfl").lower() in {"ccad", "ccad_hybrid"}

    def _use_cs_communication(self, method_cfg: dict) -> bool:
        return method_cfg.get("communication", "asymhfl").lower() in {"cs_asymhfl", "fedsara_cs"}

    def _use_ird_communication(self, method_cfg: dict) -> bool:
        return method_cfg.get("communication", "asymhfl").lower() in {"ird", "fedclear"}

    def _use_pccd_communication(self, method_cfg: dict) -> bool:
        return method_cfg.get("communication", "asymhfl").lower() in {"pccd", "fedclear_pccd"}

    def _use_ebst_communication(self, method_cfg: dict) -> bool:
        return method_cfg.get("communication", "asymhfl").lower() in {
            "ebst",
            "ebst_v2",
            "fedease",
        }

    def _use_ebst_v2(self, method_cfg: dict) -> bool:
        communication = method_cfg.get("communication", "asymhfl").lower()
        ebst_cfg = method_cfg.get("fedease", {}).get("ebst", {})
        return communication == "ebst_v2" or int(ebst_cfg.get("version", 1)) >= 2

    def _ccad_uses_asymhfl_route(self, method_cfg: dict) -> bool:
        if not self._use_ccad_communication(method_cfg):
            return False
        ccad_cfg = method_cfg.get("ccad", {})
        return float(ccad_cfg.get("base_asymhfl_weight", 0.0)) > 0

    def _collaborative_phase(
        self,
        models,
        optimizers,
        public_loader,
        public_iter,
        accs,
        class_accs,
        stats,
        round_idx: int = 0,
    ) -> float:
        method_cfg = self.config.get("method", {})
        core_strategy = self._communication_strategy
        if core_strategy is not None:
            if self._use_no_communication(method_cfg):
                print("[heartbeat] communication disabled for local-only probe", flush=True)
            value = core_strategy.step(
                CommunicationContext(
                    models=models,
                    optimizers=optimizers,
                    public_loader=public_loader,
                    public_iter=public_iter,
                    accuracies=accs,
                    stats=stats,
                    device=self.device,
                    public_batches_per_round=int(
                        self.config["train"].get("public_batches_per_round", 1)
                    ),
                    private_loaders=self._communication_private_loaders,
                    num_classes=int(self.config.get("data", {}).get("num_classes", 10)),
                    round_idx=int(round_idx),
                )
            )
            self._last_baseline_metrics = dict(getattr(core_strategy, "last_metrics", {}))
            return value

        losses = []
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        if self._use_ebst_communication(method_cfg):
            return self._ebst_collaborative_phase(round_idx)
        if self._use_pccd_communication(method_cfg):
            return self._pccd_collaborative_phase(
                models=models,
                optimizers=optimizers,
                public_loader=public_loader,
                public_iter=public_iter,
                stats=stats,
                round_idx=round_idx,
            )
        if self._use_ird_communication(method_cfg):
            return self._ird_collaborative_phase(
                models=models,
                optimizers=optimizers,
                public_loader=public_loader,
                public_iter=public_iter,
                stats=stats,
                round_idx=round_idx,
            )
        use_cara = self._use_cara_communication(method_cfg)
        use_ccad = self._use_ccad_communication(method_cfg)
        use_cs = self._use_cs_communication(method_cfg)
        use_class_residual = self._use_class_residual(method_cfg)
        class_residual_cfg = method_cfg.get("class_residual", {})
        class_residual_lambda = float(class_residual_cfg.get("lambda_residual", 0.0))
        ccad_cfg = method_cfg.get("ccad", {})
        ccad_base_asym_weight = float(ccad_cfg.get("base_asymhfl_weight", 0.0))
        ccad_lambda = float(ccad_cfg.get("lambda_ccad", 1.0))
        cs_cfg = method_cfg.get("cs_asymhfl", {})
        cs_base_asym_weight = float(cs_cfg.get("base_asymhfl_weight", 1.0))
        cs_lambda = float(cs_cfg.get("lambda_cs", 0.2))
        num_batches = int(self.config["train"].get("public_batches_per_round", 1))
        for _ in range(num_batches):
            try:
                images, _ = next(public_iter)
            except StopIteration:
                public_iter = iter(public_loader)
                images, _ = next(public_iter)
            images = images.to(self.device, non_blocking=True)

            if use_cs:
                cs_views = self._cs_public_views(images, stats, cs_cfg)
                cs_state = self._cs_collect_state(models, cs_views, cs_cfg)
                target_probs = cs_state["target_probs"]
                student_log_probs = cs_state["student_log_probs"]
            elif use_ccad:
                views = self._ccad_public_views(images, stats, ccad_cfg)
                ccad_state = self._ccad_collect_state(models, views, ccad_cfg)
                target_probs = ccad_state["target_probs"]
                student_log_probs = ccad_state["student_log_probs"]
            else:
                images = normalize_batch(images, stats)
                target_probs = {}
                student_log_probs = {}
                for client_id in sorted(models):
                    models[client_id].eval()
                    with torch.no_grad():
                        logits = forward_logits(models[client_id], images)
                        target_probs[client_id] = F.softmax(logits, dim=1).detach()
                    models[client_id].train()
                    logits = forward_logits(models[client_id], images)
                    student_log_probs[client_id] = F.log_softmax(logits, dim=1)


            for client_id in sorted(models):
                learn_losses = []
                for other_id in sorted(models):
                    if other_id == client_id:
                        continue
                    if use_cara:
                        if class_accs is None:
                            raise RuntimeError("CARA-C communication requires class-wise accuracies.")
                        class_weights = self._cara_class_weights(
                            student_class_acc=class_accs[client_id],
                            teacher_class_acc=class_accs[other_id],
                            method_cfg=method_cfg,
                        )
                        if class_weights is None:
                            continue
                        learn_losses.append(self._weighted_kd_loss(
                            student_log_probs=student_log_probs[client_id],
                            teacher_probs=target_probs[other_id],
                            class_weights=class_weights,
                        ))
                    elif use_ccad:
                        if ccad_base_asym_weight > 0 and accs[client_id] <= accs[other_id]:
                            learn_losses.append(
                                ccad_base_asym_weight
                                * criterion(student_log_probs[client_id], target_probs[other_id])
                            )
                        ccad_loss = self._ccad_pair_loss(
                            student_id=client_id,
                            teacher_id=other_id,
                            student_log_probs=student_log_probs[client_id],
                            teacher_probs=target_probs[other_id],
                            ccad_state=ccad_state,
                            ccad_cfg=ccad_cfg,
                        )
                        if ccad_loss is not None:
                            learn_losses.append(ccad_lambda * ccad_loss)
                    elif use_cs:
                        if cs_base_asym_weight > 0 and accs[client_id] <= accs[other_id]:
                            learn_losses.append(
                                cs_base_asym_weight
                                * criterion(student_log_probs[client_id], target_probs[other_id])
                            )
                        cs_loss = self._cs_pair_loss(
                            student_id=client_id,
                            teacher_id=other_id,
                            cs_state=cs_state,
                            cs_cfg=cs_cfg,
                        )
                        if cs_loss is not None:
                            learn_losses.append(cs_lambda * cs_loss)
                    elif accs[client_id] <= accs[other_id]:
                        kd_loss = criterion(student_log_probs[client_id], target_probs[other_id])
                        if use_class_residual and class_residual_lambda > 0:
                            class_weights = self._private_class_need_weights(client_id, method_cfg)
                            if class_weights is not None:
                                kd_loss = kd_loss + class_residual_lambda * self._weighted_kd_loss(
                                    student_log_probs=student_log_probs[client_id],
                                    teacher_probs=target_probs[other_id],
                                    class_weights=class_weights,
                                )
                        learn_losses.append(kd_loss)
                if not learn_losses:
                    continue
                loss = sum(learn_losses) / len(learn_losses)
                optimizers[client_id].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[client_id].step()
                losses.append(float(loss.detach().cpu()))
        return sum(losses) / max(len(losses), 1)

    def _ebst_collaborative_phase(self, round_idx: int) -> float:
        method_cfg = self.config.get("method", {})
        fedease_cfg = method_cfg.get("fedease", {})
        ebst_cfg = fedease_cfg.get("ebst", fedease_cfg.get("structural_transfer", {}))
        warmup_rounds = int(ebst_cfg.get("warmup_rounds", 1))
        if round_idx < warmup_rounds or not self._fedease_client_relations:
            self._fedease_global_relation = None
            self._fedease_recipient_relations = {}
            print(
                f"[heartbeat] EBST warmup round={round_idx:03d}/{warmup_rounds:03d}; "
                "waiting for client relation statistics",
                flush=True,
            )
            return 0.0
        if self._use_ebst_v2(method_cfg):
            result = aggregate_leave_one_out_pair_relations(
                self._fedease_client_relations,
                min_source_clients=int(ebst_cfg.get("min_source_clients", 2)),
                use_stability_gate=bool(ebst_cfg.get("stability_gate", {}).get("enabled", False)),
                variance_temperature=float(
                    ebst_cfg.get("stability_gate", {}).get(
                        "variance_temperature",
                        ebst_cfg.get("variance_temperature", 0.5),
                    )
                ),
                eps=float(ebst_cfg.get("eps", 1.0e-6)),
            )
            self._fedease_recipient_relations = result.pop("recipients")
            self._fedease_global_relation = result
            print(
                f"[heartbeat] EBST-v2 LOO aggregated round={round_idx:03d} "
                f"valid_env={result['valid_environment_fraction']:.3f} "
                f"valid_pairs={result['valid_pair_fraction']:.3f} "
                f"sources={result['mean_source_count']:.2f} "
                f"mean_gate={result['mean_gate']:.3f}",
                flush=True,
            )
            return 0.0
        self._fedease_recipient_relations = {}
        self._fedease_global_relation = aggregate_environment_balanced_relations(
            self._fedease_client_relations,
            use_stability_gate=bool(ebst_cfg.get("stability_gate", {}).get("enabled", False)),
            variance_temperature=float(
                ebst_cfg.get("stability_gate", {}).get(
                    "variance_temperature",
                    ebst_cfg.get("variance_temperature", 0.5),
                )
            ),
            eps=float(ebst_cfg.get("eps", 1.0e-6)),
        )
        print(
            f"[heartbeat] EBST aggregated round={round_idx:03d} "
            f"valid_env={self._fedease_global_relation['valid_environment_fraction']:.3f} "
            f"mean_gate={self._fedease_global_relation['mean_gate']:.3f}",
            flush=True,
        )
        return 0.0

    def _pccd_collaborative_phase(
        self,
        models,
        optimizers,
        public_loader,
        public_iter,
        stats,
        round_idx: int,
    ) -> float:
        method_cfg = self.config.get("method", {})
        train_cfg = self.config.get("train", {})
        pccd_cfg = method_cfg.get("pccd", {})
        warmup_rounds = int(pccd_cfg.get("warmup_rounds", 3))
        empty_metrics = {
            "pccd_loss": 0.0,
            "pccd_teacher_entropy": 0.0,
            "pccd_teacher_confidence": 0.0,
            "pccd_view_disagreement": 0.0,
            "pccd_teacher_margin": 0.0,
            "pccd_worst_view_kl": 0.0,
        }
        if round_idx < warmup_rounds:
            self._last_pccd_metrics = empty_metrics
            print(
                f"[heartbeat] PCCD warmup round={round_idx:03d}/{warmup_rounds:03d}; communication skipped",
                flush=True,
            )
            return 0.0

        losses = []
        teacher_entropies = []
        teacher_confidences = []
        view_disagreements = []
        teacher_margins = []
        worst_view_kls = []
        num_batches = int(train_cfg.get("public_batches_per_round", 1))
        lambda_pccd = float(pccd_cfg.get("lambda_pccd", 1.0))
        skip_nonfinite = bool(train_cfg.get("skip_nonfinite", False))
        max_grad_norm = pccd_cfg.get("max_grad_norm", train_cfg.get("max_grad_norm"))
        base_seed = int(self.config.get("seed", 0))
        eps = float(pccd_cfg.get("eps", 1e-7))

        for batch_idx in range(num_batches):
            try:
                images, _ = next(public_iter)
            except StopIteration:
                public_iter = iter(public_loader)
                images, _ = next(public_iter)
            images = images.to(self.device, non_blocking=True)
            view_seed = base_seed + round_idx * 1_000_003 + batch_idx
            raw_views, operator_names = build_counterfactual_views(images, pccd_cfg, view_seed)
            views = [normalize_batch(view, stats) for view in raw_views]

            consensuses: dict[int, torch.Tensor] = {}
            confidences: dict[int, torch.Tensor] = {}
            for client_id in sorted(models):
                model = models[client_id]
                model.eval()
                with torch.no_grad():
                    probability_views = [F.softmax(forward_logits(model, view), dim=1) for view in views]
                    consensus = log_opinion_consensus(probability_views, eps=eps)
                    consensuses[client_id] = consensus.detach()
                    confidences[client_id] = normalized_entropy_confidence(consensus, eps=eps).detach()
                    view_disagreements.append(
                        float(probability_view_disagreement(probability_views, eps=eps).detach().cpu())
                    )

            for client_id in sorted(models):
                teacher, teacher_confidence = leave_one_out_consensus_teacher(
                    consensuses,
                    confidences,
                    receiver_id=client_id,
                    eps=eps,
                )
                teacher = teacher.detach()
                teacher_confidence = teacher_confidence.detach()
                safe_teacher = teacher.clamp_min(eps)
                teacher_entropies.append(
                    float((-(safe_teacher * safe_teacher.log()).sum(dim=1).mean()).detach().cpu())
                )
                teacher_confidences.append(float(teacher_confidence.mean().detach().cpu()))
                teacher_margins.append(float(teacher_margin(teacher).mean().detach().cpu()))

                model = models[client_id]
                model.train()
                student_logits_views = [forward_logits(model, view) for view in views]
                result = paired_counterfactual_distillation(
                    student_logits_views,
                    teacher_probabilities=teacher,
                    sample_weights=teacher_confidence,
                    eps=eps,
                )
                loss = lambda_pccd * result.loss
                if not torch.isfinite(loss):
                    message = (
                        f"PCCD communication, round={round_idx}, client={client_id}: "
                        f"non-finite loss at public batch {batch_idx}"
                    )
                    if skip_nonfinite:
                        print(f"[warning] {message}; skipping update", flush=True)
                        continue
                    raise FloatingPointError(message)

                optimizers[client_id].zero_grad(set_to_none=True)
                loss.backward()
                grads_finite = all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                )
                if not grads_finite:
                    optimizers[client_id].zero_grad(set_to_none=True)
                    message = (
                        f"PCCD communication, round={round_idx}, client={client_id}: "
                        f"non-finite gradient at public batch {batch_idx}"
                    )
                    if skip_nonfinite:
                        print(f"[warning] {message}; skipping update", flush=True)
                        continue
                    raise FloatingPointError(message)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                optimizers[client_id].step()
                losses.append(float(loss.detach().cpu()))
                worst_view_kls.append(float(result.worst_view_kl.detach().cpu()))

            print(
                f"[heartbeat] PCCD round={round_idx:03d} public_batch={batch_idx + 1}/{num_batches} "
                f"operators={','.join(operator_names)} "
                f"teacher_conf={sum(teacher_confidences) / max(len(teacher_confidences), 1):.4f} "
                f"view_dis={sum(view_disagreements) / max(len(view_disagreements), 1):.4f}",
                flush=True,
            )

        self._last_pccd_metrics = {
            "pccd_loss": sum(losses) / max(len(losses), 1),
            "pccd_teacher_entropy": sum(teacher_entropies) / max(len(teacher_entropies), 1),
            "pccd_teacher_confidence": sum(teacher_confidences) / max(len(teacher_confidences), 1),
            "pccd_view_disagreement": sum(view_disagreements) / max(len(view_disagreements), 1),
            "pccd_teacher_margin": sum(teacher_margins) / max(len(teacher_margins), 1),
            "pccd_worst_view_kl": sum(worst_view_kls) / max(len(worst_view_kls), 1),
        }
        return self._last_pccd_metrics["pccd_loss"]

    def _ird_collaborative_phase(
        self,
        models,
        optimizers,
        public_loader,
        public_iter,
        stats,
        round_idx: int,
    ) -> float:
        method_cfg = self.config.get("method", {})
        train_cfg = self.config.get("train", {})
        ird_cfg = method_cfg.get("ird", {})
        warmup_rounds = int(ird_cfg.get("warmup_rounds", 3))
        if round_idx < warmup_rounds:
            self._last_ird_metrics = {
                "ird_loss": 0.0,
                "ird_anchor_disagreement": 0.0,
                "ird_worst_view_kl": 0.0,
            }
            print(
                f"[heartbeat] IRD warmup round={round_idx:03d}/{warmup_rounds:03d}; communication skipped",
                flush=True,
            )
            return 0.0

        losses = []
        disagreements = []
        worst_view_kls = []
        num_batches = int(train_cfg.get("public_batches_per_round", 1))
        distill_temperature = float(ird_cfg.get("temperature", 2.0))
        smooth_temperature = float(ird_cfg.get("smooth_temperature", 0.5))
        lambda_ird = float(ird_cfg.get("lambda_ird", 1.0))
        skip_nonfinite = bool(train_cfg.get("skip_nonfinite", False))
        max_grad_norm = ird_cfg.get("max_grad_norm", train_cfg.get("max_grad_norm"))
        base_seed = int(self.config.get("seed", 0))

        for batch_idx in range(num_batches):
            try:
                images, _ = next(public_iter)
            except StopIteration:
                public_iter = iter(public_loader)
                images, _ = next(public_iter)
            images = images.to(self.device, non_blocking=True)
            view_seed = base_seed + round_idx * 1_000_003 + batch_idx
            raw_views, operator_names = build_counterfactual_views(images, ird_cfg, view_seed)
            views = [normalize_batch(view, stats) for view in raw_views]

            anchors: dict[int, torch.Tensor] = {}
            for client_id in sorted(models):
                model = models[client_id]
                model.eval()
                with torch.no_grad():
                    logits_views = [forward_logits(model, view) for view in views]
                    anchors[client_id] = invariant_anchor(logits_views).detach()
            disagreement = anchor_disagreement(anchors)
            disagreements.append(float(disagreement.detach().cpu()))

            for client_id in sorted(models):
                teacher_anchor = leave_one_out_median(anchors, client_id).detach()
                model = models[client_id]
                model.train()
                student_logits_views = [forward_logits(model, view) for view in views]
                result = smooth_worst_view_distillation(
                    student_logits_views,
                    teacher_anchor,
                    distill_temperature=distill_temperature,
                    smooth_temperature=smooth_temperature,
                )
                loss = lambda_ird * result.loss
                if not torch.isfinite(loss):
                    message = (
                        f"IRD communication, round={round_idx}, client={client_id}: "
                        f"non-finite loss at public batch {batch_idx}"
                    )
                    if skip_nonfinite:
                        print(f"[warning] {message}; skipping update", flush=True)
                        continue
                    raise FloatingPointError(message)

                optimizers[client_id].zero_grad(set_to_none=True)
                loss.backward()
                grads_finite = all(
                    param.grad is None or bool(torch.isfinite(param.grad).all())
                    for param in model.parameters()
                )
                if not grads_finite:
                    optimizers[client_id].zero_grad(set_to_none=True)
                    message = (
                        f"IRD communication, round={round_idx}, client={client_id}: "
                        f"non-finite gradient at public batch {batch_idx}"
                    )
                    if skip_nonfinite:
                        print(f"[warning] {message}; skipping update", flush=True)
                        continue
                    raise FloatingPointError(message)
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                optimizers[client_id].step()
                losses.append(float(loss.detach().cpu()))
                worst_view_kls.append(float(result.worst_view_kl.detach().cpu()))

            print(
                f"[heartbeat] IRD round={round_idx:03d} public_batch={batch_idx + 1}/{num_batches} "
                f"operators={','.join(operator_names)} "
                f"anchor_dis={disagreements[-1]:.4f}",
                flush=True,
            )

        self._last_ird_metrics = {
            "ird_loss": sum(losses) / max(len(losses), 1),
            "ird_anchor_disagreement": sum(disagreements) / max(len(disagreements), 1),
            "ird_worst_view_kl": sum(worst_view_kls) / max(len(worst_view_kls), 1),
        }
        return self._last_ird_metrics["ird_loss"]

    def _cs_public_views(self, images: torch.Tensor, stats, cs_cfg: dict) -> dict[str, torch.Tensor]:
        probe_groups = list(cs_cfg.get("probe_groups", ["clean", "noise", "blur", "weather", "digital"]))
        views = {}
        for group in probe_groups:
            group = str(group).lower()
            if group == "clean":
                view = images
            elif group == "noise":
                view = self._cs_noise(images, cs_cfg)
            elif group == "blur":
                view = self._cs_blur(images, cs_cfg)
            elif group == "weather":
                view = self._cs_weather(images, cs_cfg)
            elif group == "digital":
                view = self._cs_digital(images, cs_cfg)
            else:
                raise ValueError(f"Unknown CS public probe group: {group}")
            views[group] = normalize_batch(view.clamp(0.0, 1.0), stats)
        if "clean" not in views:
            views["clean"] = normalize_batch(images, stats)
        return views

    def _cs_noise(self, images: torch.Tensor, cs_cfg: dict) -> torch.Tensor:
        std = float(cs_cfg.get("noise_std", 0.08))
        return images.detach() + torch.randn_like(images) * std

    def _cs_blur(self, images: torch.Tensor, cs_cfg: dict) -> torch.Tensor:
        kernel = int(cs_cfg.get("blur_kernel", 3))
        if kernel % 2 == 0:
            kernel += 1
        padding = kernel // 2
        return F.avg_pool2d(images.detach(), kernel_size=kernel, stride=1, padding=padding)

    def _cs_weather(self, images: torch.Tensor, cs_cfg: dict) -> torch.Tensor:
        x = images.detach()
        strength = float(cs_cfg.get("weather_strength", 0.25))
        haze = torch.empty_like(x).uniform_(0.70, 1.0)
        return (1.0 - strength) * x + strength * haze

    def _cs_digital(self, images: torch.Tensor, cs_cfg: dict) -> torch.Tensor:
        x = images.detach()
        size = int(cs_cfg.get("pixelate_size", 16))
        small = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        pixelated = F.interpolate(small, size=x.shape[-2:], mode="nearest")
        contrast = float(cs_cfg.get("digital_contrast", 0.7))
        mean = pixelated.mean(dim=(2, 3), keepdim=True)
        return (pixelated - mean) * contrast + mean

    def _cs_collect_state(self, models, views: dict[str, torch.Tensor], cs_cfg: dict) -> dict[str, object]:
        eps = float(cs_cfg.get("eps", 1e-8))
        groups = [group for group in views if group != "clean"]
        target_probs = {}
        student_log_probs = {}
        group_teacher_probs: dict[str, dict[int, torch.Tensor]] = {group: {} for group in groups}
        group_student_log_probs: dict[str, dict[int, torch.Tensor]] = {group: {} for group in groups}
        reliability: dict[str, dict[int, torch.Tensor]] = {group: {} for group in groups}
        need: dict[str, dict[int, torch.Tensor]] = {group: {} for group in groups}

        for client_id in sorted(models):
            model = models[client_id]
            model.eval()
            with torch.no_grad():
                clean_prob = F.softmax(forward_logits(model, views["clean"]), dim=1).detach()
                target_probs[client_id] = clean_prob
                for group in groups:
                    group_prob = F.softmax(forward_logits(model, views[group]), dim=1).detach()
                    group_teacher_probs[group][client_id] = group_prob
                    jsd = self._pair_jsd(clean_prob, group_prob, eps=eps)
                    entropy = self._ccad_entropy(group_prob)
                    confidence = (1.0 - entropy).clamp(0.0, 1.0)
                    consistency = torch.exp(-jsd / max(float(cs_cfg.get("consistency_tau", 0.05)), eps))
                    reliability[group][client_id] = (confidence * consistency).detach()
                    need[group][client_id] = (
                        entropy
                        + float(cs_cfg.get("need_instability_weight", 0.5))
                        * (jsd / max(float(cs_cfg.get("consistency_tau", 0.05)), eps)).clamp(0.0, float(cs_cfg.get("max_instability", 4.0)))
                    ).detach()

            model.train()
            student_log_probs[client_id] = F.log_softmax(forward_logits(model, views["clean"]), dim=1)
            for group in groups:
                group_student_log_probs[group][client_id] = F.log_softmax(forward_logits(model, views[group]), dim=1)

        return {
            "groups": groups,
            "target_probs": target_probs,
            "student_log_probs": student_log_probs,
            "group_teacher_probs": group_teacher_probs,
            "group_student_log_probs": group_student_log_probs,
            "reliability": reliability,
            "need": need,
        }

    def _pair_jsd(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        m = ((p + q) * 0.5).clamp_min(eps)
        return 0.5 * (p * (p.log() - m.log())).sum(dim=1) + 0.5 * (q * (q.log() - m.log())).sum(dim=1)

    def _cs_pair_loss(
        self,
        student_id: int,
        teacher_id: int,
        cs_state: dict[str, object],
        cs_cfg: dict,
    ) -> torch.Tensor | None:
        eps = float(cs_cfg.get("eps", 1e-8))
        losses = []
        weights_all = []
        groups = cs_state["groups"]
        for group in groups:
            teacher_reliability = cs_state["reliability"][group][teacher_id]
            student_reliability = cs_state["reliability"][group][student_id]
            student_need = cs_state["need"][group][student_id]
            weights = teacher_reliability.pow(float(cs_cfg.get("teacher_power", 1.0)))
            weights = weights * student_need.clamp_min(0.0).pow(float(cs_cfg.get("need_power", 1.0)))
            if bool(cs_cfg.get("better_only", True)):
                margin = float(cs_cfg.get("reliability_margin", 0.0))
                weights = weights * (teacher_reliability > student_reliability + margin).float()
            max_weight = cs_cfg.get("max_pair_weight")
            if max_weight is not None:
                weights = weights.clamp_max(float(max_weight))
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            active = weights > float(cs_cfg.get("min_pair_weight", 1e-6))
            if not bool(active.any()):
                continue
            if bool(cs_cfg.get("normalize_pair_weights", True)):
                weights = weights / weights[active].mean().clamp_min(eps)

            teacher_probs = cs_state["group_teacher_probs"][group][teacher_id].clamp_min(eps)
            student_log_probs = cs_state["group_student_log_probs"][group][student_id]
            per_sample_kl = (teacher_probs * (teacher_probs.log() - student_log_probs)).sum(dim=1)
            losses.append(per_sample_kl)
            weights_all.append(weights)

        if not losses:
            return None
        loss_vec = torch.cat(losses, dim=0)
        weight_vec = torch.cat(weights_all, dim=0)
        return (loss_vec * weight_vec).sum() / weight_vec.sum().clamp_min(eps)

    def _ccad_public_views(self, images: torch.Tensor, stats, ccad_cfg: dict) -> list[torch.Tensor]:
        num_aug_views = int(ccad_cfg.get("num_aug_views", 2))
        views = [normalize_batch(images, stats)]
        for _ in range(num_aug_views):
            views.append(normalize_batch(self._ccad_tensor_augment(images, ccad_cfg), stats))
        return views

    def _ccad_tensor_augment(self, images: torch.Tensor, ccad_cfg: dict) -> torch.Tensor:
        x = images.detach()
        batch_size, _, height, width = x.shape
        padding = int(ccad_cfg.get("crop_padding", 4))
        if padding > 0:
            padded = F.pad(x, (padding, padding, padding, padding), mode="reflect")
            max_offset = 2 * padding + 1
            tops = torch.randint(0, max_offset, (batch_size,), device=x.device)
            lefts = torch.randint(0, max_offset, (batch_size,), device=x.device)
            x = torch.stack([
                padded[idx, :, tops[idx]:tops[idx] + height, lefts[idx]:lefts[idx] + width]
                for idx in range(batch_size)
            ], dim=0)

        flip_p = float(ccad_cfg.get("hflip_p", 0.5))
        if flip_p > 0:
            mask = torch.rand(batch_size, device=x.device) < flip_p
            if mask.any():
                x = x.clone()
                x[mask] = torch.flip(x[mask], dims=[3])

        brightness = float(ccad_cfg.get("brightness", 0.2))
        if brightness > 0:
            factor = 1.0 + (torch.rand(batch_size, 1, 1, 1, device=x.device) * 2.0 - 1.0) * brightness
            x = x * factor

        contrast = float(ccad_cfg.get("contrast", 0.2))
        if contrast > 0:
            mean = x.mean(dim=(2, 3), keepdim=True)
            factor = 1.0 + (torch.rand(batch_size, 1, 1, 1, device=x.device) * 2.0 - 1.0) * contrast
            x = (x - mean) * factor + mean

        noise_std = float(ccad_cfg.get("noise_std", 0.03))
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        return x.clamp(0.0, 1.0)

    def _ccad_collect_state(self, models, views: list[torch.Tensor], ccad_cfg: dict) -> dict[str, dict[int, torch.Tensor]]:
        target_probs = {}
        student_log_probs = {}
        reliability = {}
        need = {}
        jsd = {}
        entropy = {}
        for client_id in sorted(models):
            model = models[client_id]
            model.eval()
            with torch.no_grad():
                probs = [F.softmax(forward_logits(model, view), dim=1).detach() for view in views]
            clean_probs = probs[0]
            client_jsd = self._ccad_jsd(probs)
            client_entropy = self._ccad_entropy(clean_probs)
            target_probs[client_id] = clean_probs
            jsd[client_id] = client_jsd
            entropy[client_id] = client_entropy
            reliability[client_id] = self._ccad_reliability(clean_probs, client_jsd, client_entropy, ccad_cfg)
            need[client_id] = self._ccad_need(client_jsd, client_entropy, ccad_cfg)

            model.train()
            logits = forward_logits(model, views[0])
            student_log_probs[client_id] = F.log_softmax(logits, dim=1)

        return {
            "target_probs": target_probs,
            "student_log_probs": student_log_probs,
            "reliability": reliability,
            "need": need,
            "jsd": jsd,
            "entropy": entropy,
        }

    def _ccad_jsd(self, probs: list[torch.Tensor]) -> torch.Tensor:
        eps = 1e-8
        safe_probs = [prob.clamp_min(eps) for prob in probs]
        mixture = torch.stack(safe_probs, dim=0).mean(dim=0).clamp_min(eps)
        divergences = [
            (prob * (prob.log() - mixture.log())).sum(dim=1)
            for prob in safe_probs
        ]
        return torch.stack(divergences, dim=0).mean(dim=0)

    def _ccad_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        num_classes = max(int(probs.shape[1]), 2)
        entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=1)
        return (entropy / math.log(num_classes)).clamp(0.0, 1.0)

    def _ccad_reliability(
        self,
        clean_probs: torch.Tensor,
        jsd: torch.Tensor,
        entropy: torch.Tensor,
        ccad_cfg: dict,
    ) -> torch.Tensor:
        tau = float(ccad_cfg.get("consistency_tau", 0.05))
        confidence_mode = str(ccad_cfg.get("confidence_mode", "max_prob")).lower()
        if confidence_mode == "inverse_entropy":
            confidence = (1.0 - entropy).clamp(0.0, 1.0)
        else:
            confidence = clean_probs.max(dim=1).values.clamp(0.0, 1.0)
        confidence = confidence.pow(float(ccad_cfg.get("confidence_power", 1.0)))
        consistency = torch.exp(-jsd / max(tau, 1e-8))
        return (confidence * consistency).detach()

    def _ccad_need(self, jsd: torch.Tensor, entropy: torch.Tensor, ccad_cfg: dict) -> torch.Tensor:
        entropy_need = entropy.clamp(0.0, 1.0).pow(float(ccad_cfg.get("need_entropy_power", 1.0)))
        tau = float(ccad_cfg.get("consistency_tau", 0.05))
        instability = (jsd / max(tau, 1e-8)).clamp(0.0, float(ccad_cfg.get("max_instability", 4.0)))
        need = entropy_need + float(ccad_cfg.get("need_instability_weight", 0.5)) * instability
        return need.detach()

    def _ccad_pair_loss(
        self,
        student_id: int,
        teacher_id: int,
        student_log_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        ccad_state: dict[str, dict[int, torch.Tensor]],
        ccad_cfg: dict,
    ) -> torch.Tensor | None:
        eps = float(ccad_cfg.get("eps", 1e-8))
        teacher_reliability = ccad_state["reliability"][teacher_id]
        student_reliability = ccad_state["reliability"][student_id]
        student_need = ccad_state["need"][student_id]

        weights = teacher_reliability.pow(float(ccad_cfg.get("teacher_power", 1.0)))
        weights = weights * student_need.clamp_min(0.0).pow(float(ccad_cfg.get("need_power", 1.0)))

        if bool(ccad_cfg.get("better_only", True)):
            margin = float(ccad_cfg.get("reliability_margin", 0.0))
            weights = weights * (teacher_reliability > student_reliability + margin).float()

        max_weight = ccad_cfg.get("max_pair_weight")
        if max_weight is not None:
            weights = weights.clamp_max(float(max_weight))
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        min_weight = float(ccad_cfg.get("min_pair_weight", 1e-6))
        active = weights > min_weight
        if not bool(active.any()):
            return None
        if bool(ccad_cfg.get("normalize_pair_weights", True)):
            weights = weights / weights[active].mean().clamp_min(eps)

        safe_teacher = teacher_probs.clamp_min(eps)
        per_sample_kl = (safe_teacher * (safe_teacher.log() - student_log_probs)).sum(dim=1)
        return (per_sample_kl * weights).sum() / weights.sum().clamp_min(eps)

    def _cara_class_weights(
        self,
        student_class_acc: torch.Tensor,
        teacher_class_acc: torch.Tensor,
        method_cfg: dict,
    ) -> torch.Tensor | None:
        cara_cfg = method_cfg.get("cara", {})
        student = student_class_acc.to(self.device).float().clamp(0.0, 1.0)
        teacher = teacher_class_acc.to(self.device).float().clamp(0.0, 1.0)
        margin = float(cara_cfg.get("better_margin", 0.0))
        teacher_power = float(cara_cfg.get("teacher_power", 1.0))
        need_power = float(cara_cfg.get("need_power", 1.0))
        min_weight = float(cara_cfg.get("min_weight", 1e-6))

        teacher_reliability = teacher.clamp_min(0.0).pow(teacher_power)
        receiver_need = (1.0 - student).clamp_min(0.0).pow(need_power)
        weights = teacher_reliability * receiver_need
        if bool(cara_cfg.get("better_only", True)):
            weights = weights * (teacher > student + margin).float()
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if float(weights.sum().detach().cpu()) <= min_weight:
            return None
        if bool(cara_cfg.get("normalize", True)):
            active = weights > min_weight
            weights = weights / weights[active].mean().clamp_min(min_weight)
        return weights.detach()

    def _use_class_residual(self, method_cfg: dict) -> bool:
        residual_cfg = method_cfg.get("class_residual", {})
        return bool(residual_cfg.get("enabled", False))

    def _private_class_need_weights(self, client_id: int, method_cfg: dict) -> torch.Tensor | None:
        """Receiver-side class-need weights computed from private local counts.

        The vector is used only inside the receiving client loss. In a real FL
        deployment the server can still send ordinary public logits; the receiver
        privately reweights its own KD objective without uploading counts.
        """
        counts = getattr(self, "_client_class_counts", {}).get(client_id)
        if counts is None:
            return None
        cfg = method_cfg.get("class_residual", {})
        counts = counts.detach().float().to(self.device)
        if counts.numel() == 0:
            return None

        mode = str(cfg.get("need_mode", "inverse_count")).lower()
        power = float(cfg.get("need_power", 0.5))
        eps = float(cfg.get("eps", 1e-8))
        if mode == "inverse_count":
            smoothing = float(cfg.get("smoothing", 10.0))
            reference = counts.sum().clamp_min(1.0) / float(counts.numel())
            weights = (reference / (counts + smoothing).clamp_min(eps)).pow(power)
        elif mode == "complement_prior":
            prior = counts / counts.sum().clamp_min(1.0)
            weights = (1.0 - prior).clamp_min(0.0).pow(power)
        else:
            raise ValueError(f"Unknown class_residual.need_mode: {mode}")

        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if bool(cfg.get("normalize", True)):
            weights = weights / weights.mean().clamp_min(eps)
        min_weight = cfg.get("min_weight")
        max_weight = cfg.get("max_weight")
        if min_weight is not None:
            weights = weights.clamp_min(float(min_weight))
        if max_weight is not None:
            weights = weights.clamp_max(float(max_weight))
        return weights.detach()

    def _weighted_kd_loss(
        self,
        student_log_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> torch.Tensor:
        safe_teacher = teacher_probs.clamp_min(1e-8)
        per_class_kl = safe_teacher * (safe_teacher.log() - student_log_probs)
        return (per_class_kl * class_weights.view(1, -1)).sum(dim=1).mean()

    def _get_nir_dcl_queue(self, client_id: int, num_classes: int, method_cfg: dict) -> NIRDCLFeatureQueue:
        if client_id not in self._nir_dcl_queues:
            nir_cfg = method_cfg.get("cara_l", method_cfg.get("nir_dcl", {}))
            self._nir_dcl_queues[client_id] = NIRDCLFeatureQueue(
                num_classes=num_classes,
                max_size_per_class=int(nir_cfg.get("queue_size", 64)),
            )
        return self._nir_dcl_queues[client_id]

    def _build_client_class_counts(self, labels, dataidx_map, num_classes: int) -> dict[int, torch.Tensor]:
        label_tensor = torch.as_tensor(labels, dtype=torch.long)
        counts = {}
        for client_id, indices in dataidx_map.items():
            idx_tensor = torch.as_tensor(indices, dtype=torch.long)
            client_labels = label_tensor[idx_tensor]
            counts[int(client_id)] = torch.bincount(client_labels, minlength=num_classes).float()
        return counts

    def _local_phase(
        self,
        models,
        optimizers,
        private_loaders,
        prime_aug,
        use_prime,
        use_prime_dcl,
        train_cfg,
        method_cfg,
        stats,
        num_classes: int,
        round_idx: int = 0,
    ) -> float:
        losses = []
        ccre_diagnostics = []
        fedease_diagnostics = []
        round_relation_states: dict[int, dict[str, torch.Tensor]] = {}
        for client_id, loader in enumerate(private_loaders):
            relation_accumulator = None
            for _ in range(int(train_cfg.get("local_epochs", 1))):
                cl_module = str(method_cfg.get("cl_module", "dcl")).lower()
                if cl_module == "fedease":
                    fedease_cfg = method_cfg.get("fedease", {})
                    cdep_cfg = fedease_cfg.get("cdep", {})
                    ebst_cfg = fedease_cfg.get("ebst", fedease_cfg.get("structural_transfer", {}))
                    if client_id not in self._fedease_projectors:
                        self._fedease_projectors[client_id] = FrozenRandomProjector(
                            output_dim=int(cdep_cfg.get("projection_dim", 64)),
                            seed=int(self.config.get("seed", 0)) * 1009 + client_id,
                        )
                    if (
                        bool(cdep_cfg.get("enabled", True))
                        and str(cdep_cfg.get("version", "v1")).lower() == "v2"
                        and client_id not in self._fedease_cdep_v2_memories
                    ):
                        self._fedease_cdep_v2_memories[client_id] = BufferedConditionalMomentAlignment(
                            num_classes=num_classes,
                            num_environments=int(fedease_cfg.get("num_environments", 6)),
                            max_size_per_group=int(cdep_cfg.get("buffer_size_per_group", 64)),
                        )
                    if bool(ebst_cfg.get("enabled", False)) and relation_accumulator is None:
                        relation_accumulator = new_relation_accumulator(
                            num_classes=num_classes,
                            num_environments=int(fedease_cfg.get("num_environments", 4)),
                        )
                    epoch_diagnostics = {}
                    loss = train_local_fedease_epoch(
                        model=models[client_id],
                        loader=loader,
                        optimizer=optimizers[client_id],
                        device=self.device,
                        projector=self._fedease_projectors[client_id],
                        fedease_cfg=fedease_cfg,
                        class_environment_counts=getattr(
                            self,
                            "_client_class_environment_counts",
                            {},
                        ).get(client_id),
                        lambda_jsd=float(method_cfg.get("lambda_jsd", 12.0)),
                        max_batches=train_cfg.get("max_local_batches"),
                        max_grad_norm=train_cfg.get("max_grad_norm"),
                        skip_nonfinite=bool(train_cfg.get("skip_nonfinite", False)),
                        log_interval=train_cfg.get("local_log_interval"),
                        context=f"FedEASE local phase, round={round_idx}, client={client_id}",
                        diagnostics=epoch_diagnostics,
                        cdep_v2_memory=self._fedease_cdep_v2_memories.get(client_id),
                        round_idx=round_idx,
                        relation_accumulator=relation_accumulator,
                        global_relation_state=(
                            self._fedease_recipient_relations.get(client_id)
                            if self._use_ebst_v2(method_cfg)
                            else self._fedease_global_relation
                        ),
                        client_supported_classes=(
                            getattr(self, "_client_class_environment_counts", {})
                            .get(client_id, torch.empty(0))
                            .sum(dim=1)
                            .gt(0)
                            if client_id in getattr(self, "_client_class_environment_counts", {})
                            else None
                        ),
                    )
                    fedease_diagnostics.append(epoch_diagnostics)
                elif cl_module == "ccre":
                    epoch_diagnostics: dict[str, float] = {}
                    loss = train_local_fedclear_epoch(
                        model=models[client_id],
                        loader=loader,
                        optimizer=optimizers[client_id],
                        normalizer=lambda x: normalize_batch(x, stats),
                        device=self.device,
                        lambda_jsd=float(method_cfg.get("lambda_jsd", 12.0)),
                        ccre_cfg=method_cfg.get("ccre", {}),
                        view_cfg=method_cfg.get("counterfactual_views", {}),
                        round_idx=round_idx,
                        client_id=client_id,
                        seed=int(self.config.get("seed", 0)),
                        client_class_counts=getattr(self, "_client_class_counts", {}).get(client_id),
                        max_batches=train_cfg.get("max_local_batches"),
                        max_grad_norm=train_cfg.get("max_grad_norm"),
                        skip_nonfinite=bool(train_cfg.get("skip_nonfinite", False)),
                        log_interval=train_cfg.get("local_log_interval"),
                        diagnostics=epoch_diagnostics,
                    )
                    ccre_diagnostics.append(epoch_diagnostics)
                elif use_prime:
                    if use_prime_dcl:
                        loss = train_local_prime_dcl_epoch(
                            model=models[client_id],
                            loader=loader,
                            optimizer=optimizers[client_id],
                            prime_aug=prime_aug,
                            normalizer=lambda x: normalize_batch(x, stats),
                            device=self.device,
                            lambda_jsd=float(method_cfg.get("lambda_jsd", 12.0)),
                            cl_module=method_cfg.get("cl_module", "dcl"),
                            max_batches=train_cfg.get("max_local_batches"),
                        )
                    else:
                        loss = train_local_prime_epoch(
                            model=models[client_id],
                            loader=loader,
                            optimizer=optimizers[client_id],
                            prime_aug=prime_aug,
                            device=self.device,
                            lambda_jsd=float(method_cfg.get("lambda_jsd", 12.0)),
                            max_batches=train_cfg.get("max_local_batches"),
                        )
                else:
                    feature_queue = None
                    if method_cfg.get("cl_module", "dcl") in {"nir_dcl", "cara_l"}:
                        feature_queue = self._get_nir_dcl_queue(client_id, num_classes, method_cfg)
                    loss = train_local_augmix_dcl_epoch(
                        model=models[client_id],
                        loader=loader,
                        optimizer=optimizers[client_id],
                        device=self.device,
                        lambda_jsd=float(method_cfg.get("lambda_jsd", 12.0)),
                        cl_module=method_cfg.get("cl_module", "dcl"),
                        num_classes=num_classes,
                        nir_dcl_cfg=method_cfg.get("cara_l", method_cfg.get("nir_dcl", {})),
                        sara_cfg=method_cfg.get("sara", {}),
                        feature_queue=feature_queue,
                        client_class_counts=getattr(self, "_client_class_counts", {}).get(client_id),
                        max_batches=train_cfg.get("max_local_batches"),
                        max_grad_norm=train_cfg.get("max_grad_norm"),
                        skip_nonfinite=bool(train_cfg.get("skip_nonfinite", False)),
                        log_interval=train_cfg.get("local_log_interval"),
                        communication_loss_fn=(
                            (
                                lambda *, receiver_logits, clean_images, labels,
                                _model=models[client_id], _strategy=self._communication_strategy:
                                _strategy.local_loss(
                                    model=_model,
                                    clean_images=clean_images,
                                    labels=labels,
                                )
                            )
                            if hasattr(self._communication_strategy, "local_loss")
                            else None
                        ),
                    )
                losses.append(loss)
            if relation_accumulator is not None:
                if self._use_ebst_v2(method_cfg):
                    round_relation_states[client_id] = finalize_pair_qualified_client_relations(
                        relation_accumulator,
                        min_group_support=int(ebst_cfg.get("min_group_support", 4)),
                        min_competing_class_support=int(
                            ebst_cfg.get("min_pair_class_support", 16)
                        ),
                        eps=float(ebst_cfg.get("eps", 1.0e-6)),
                    )
                else:
                    round_relation_states[client_id] = finalize_client_relations(
                        relation_accumulator,
                        min_group_support=int(ebst_cfg.get("min_group_support", 4)),
                        eps=float(ebst_cfg.get("eps", 1.0e-6)),
                    )
        if round_relation_states:
            self._fedease_client_relations = round_relation_states
        if ccre_diagnostics:
            self._last_ccre_metrics = {
                key: sum(item.get(key, 0.0) for item in ccre_diagnostics) / len(ccre_diagnostics)
                for key in ("ccre_loss", "ccre_worst_view_risk")
            }
        else:
            self._last_ccre_metrics = {}
        if fedease_diagnostics:
            metric_names = set().union(*(item.keys() for item in fedease_diagnostics))
            self._last_fedease_metrics = {
                key: sum(item.get(key, 0.0) for item in fedease_diagnostics) / len(fedease_diagnostics)
                for key in metric_names
            }
        else:
            self._last_fedease_metrics = {}
        return sum(losses) / max(len(losses), 1)

    def _pretrain_phase(self, models, optimizers, private_loaders, pretrain_epochs: int, train_cfg: dict) -> None:
        criterion = torch.nn.CrossEntropyLoss()
        max_batches = train_cfg.get("max_pretrain_batches")
        log_interval = train_cfg.get("pretrain_log_interval", train_cfg.get("local_log_interval", 50))
        for epoch in range(pretrain_epochs):
            for client_id, loader in enumerate(private_loaders):
                model = models[client_id]
                model.train()
                running = []
                for batch_idx, batch in enumerate(loader):
                    images, labels, _ = self._unpack_batch(batch)
                    if isinstance(images, (tuple, list)):
                        images = images[0]
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True).long()
                    loss = criterion(forward_logits(model, images), labels)
                    optimizers[client_id].zero_grad(set_to_none=True)
                    loss.backward()
                    max_grad_norm = train_cfg.get("max_grad_norm")
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                    optimizers[client_id].step()
                    running.append(float(loss.detach().cpu()))
                    if log_interval and (batch_idx + 1) % int(log_interval) == 0:
                        mean_loss = sum(running[-int(log_interval):]) / min(len(running), int(log_interval))
                        print(
                            f"[heartbeat] pretrain epoch={epoch:03d} client={client_id} "
                            f"batch={batch_idx + 1} loss={mean_loss:.4f}",
                            flush=True,
                        )
                    if max_batches is not None and batch_idx + 1 >= int(max_batches):
                        break
                if running:
                    print(
                        f"[heartbeat] pretrain epoch={epoch:03d} client={client_id} "
                        f"done loss={sum(running) / len(running):.4f}",
                        flush=True,
                    )

    def _evaluate(self, models, test_loader) -> list[float]:
        accs = []
        max_test_batches = self.config.get("train", {}).get("max_test_batches")
        for client_id in sorted(models):
            model = models[client_id]
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if max_test_batches is not None and batch_idx >= int(max_test_batches):
                        break
                    images, labels, _ = self._unpack_batch(batch)
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True).long()
                    logits = forward_logits(model, images)
                    pred = logits.argmax(dim=1)
                    total += labels.numel()
                    correct += (pred == labels).sum().item()
            accs.append(100.0 * correct / max(total, 1))
        return accs

    def _evaluate_private_loaders(self, models, loaders) -> list[float]:
        """Evaluate each client only on its own held-out routing audit split."""

        accs = []
        max_audit_batches = self.config.get("method", {}).get(
            "strict_fit_audit",
            {},
        ).get("max_audit_batches")
        for client_id in sorted(models):
            if client_id not in loaders:
                raise KeyError(f"Missing strict routing audit loader for client {client_id}")
            model = models[client_id]
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(loaders[client_id]):
                    if max_audit_batches is not None and batch_idx >= int(max_audit_batches):
                        break
                    images, labels, _ = self._unpack_batch(batch)
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True).long()
                    predictions = forward_logits(model, images).argmax(dim=1)
                    total += int(labels.numel())
                    correct += int((predictions == labels).sum().item())
            if total == 0:
                raise ValueError(f"Client {client_id} strict routing audit loader is empty")
            accs.append(100.0 * correct / total)
        return accs

    def _evaluate_detailed(self, models, test_loader, num_classes: int) -> tuple[list[float], dict[int, torch.Tensor]]:
        accs = []
        class_accs = {}
        max_test_batches = self.config.get("train", {}).get("max_test_batches")
        for client_id in sorted(models):
            model = models[client_id]
            model.eval()
            correct = 0
            total = 0
            class_correct = torch.zeros(num_classes, dtype=torch.float64)
            class_total = torch.zeros(num_classes, dtype=torch.float64)
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if max_test_batches is not None and batch_idx >= int(max_test_batches):
                        break
                    images, labels, _ = self._unpack_batch(batch)
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True).long()
                    logits = forward_logits(model, images)
                    pred = logits.argmax(dim=1)
                    total += labels.numel()
                    correct_mask = pred == labels
                    correct += correct_mask.sum().item()
                    for class_id in range(num_classes):
                        mask = labels == class_id
                        if mask.any():
                            class_total[class_id] += mask.sum().item()
                            class_correct[class_id] += correct_mask[mask].sum().item()
            accs.append(100.0 * correct / max(total, 1))
            class_accs[client_id] = (class_correct / class_total.clamp_min(1.0)).float()
        return accs, class_accs

    def _evaluate_corruption_groups(
        self,
        models,
        test_loader,
        group_names: list[str],
        num_classes: int,
    ) -> dict[str, object]:
        if not group_names:
            return {}
        num_groups = len(group_names)
        client_results = {}
        client_accs = []
        class_corruption_rows = []
        group_correct_total = torch.zeros(num_groups, dtype=torch.float64)
        group_total_total = torch.zeros(num_groups, dtype=torch.float64)
        class_group_correct_total = torch.zeros(num_classes, num_groups, dtype=torch.float64)
        class_group_total_total = torch.zeros(num_classes, num_groups, dtype=torch.float64)
        max_test_batches = self.config.get("train", {}).get("max_test_batches")

        for client_id in sorted(models):
            model = models[client_id]
            model.eval()
            group_correct = torch.zeros(num_groups, dtype=torch.float64)
            group_total = torch.zeros(num_groups, dtype=torch.float64)
            class_group_correct = torch.zeros(num_classes, num_groups, dtype=torch.float64)
            class_group_total = torch.zeros(num_classes, num_groups, dtype=torch.float64)
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if max_test_batches is not None and batch_idx >= int(max_test_batches):
                        break
                    images, labels, corruption_ids = self._unpack_batch(batch)
                    if corruption_ids is None:
                        return {}
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True).long()
                    corruption_ids = corruption_ids.long()
                    pred = forward_logits(model, images).argmax(dim=1).cpu()
                    correct = pred.eq(labels.cpu())
                    for group_id in range(num_groups):
                        mask = corruption_ids == group_id
                        if mask.any():
                            group_total[group_id] += mask.sum().item()
                            group_correct[group_id] += correct[mask].sum().item()
                    labels_cpu = labels.cpu()
                    for class_id in range(num_classes):
                        class_mask = labels_cpu == class_id
                        if not class_mask.any():
                            continue
                        for group_id in range(num_groups):
                            mask = class_mask & (corruption_ids == group_id)
                            if mask.any():
                                class_group_total[class_id, group_id] += mask.sum().item()
                                class_group_correct[class_id, group_id] += correct[mask].sum().item()
            acc = 100.0 * group_correct / group_total.clamp_min(1.0)
            client_results[client_id] = {
                group_names[group_id]: float(acc[group_id])
                for group_id in range(num_groups)
            }
            client_accs.append(
                100.0 * float(group_correct.sum().item()) / max(float(group_total.sum().item()), 1.0)
            )
            class_group_acc = 100.0 * class_group_correct / class_group_total.clamp_min(1.0)
            for class_id in range(num_classes):
                for group_id in range(num_groups):
                    class_corruption_rows.append({
                        "client": client_id,
                        "class_id": class_id,
                        "group": group_names[group_id],
                        "acc": float(class_group_acc[class_id, group_id]),
                        "total": int(class_group_total[class_id, group_id].item()),
                    })
            group_correct_total += group_correct
            group_total_total += group_total
            class_group_correct_total += class_group_correct
            class_group_total_total += class_group_total

        group_acc = 100.0 * group_correct_total / group_total_total.clamp_min(1.0)
        group_values = {
            group_names[group_id]: float(group_acc[group_id])
            for group_id in range(num_groups)
        }
        global_class_group_acc = 100.0 * class_group_correct_total / class_group_total_total.clamp_min(1.0)
        valid = class_group_total_total > 0
        wcca = float(global_class_group_acc[valid].min().item()) if bool(valid.any()) else 0.0
        class_gaps = []
        for class_id in range(num_classes):
            class_valid = class_group_total_total[class_id] > 0
            if bool(class_valid.any()):
                values = global_class_group_acc[class_id][class_valid]
                class_gaps.append(float((values.max() - values.min()).item()))
        cfg = sum(class_gaps) / max(len(class_gaps), 1)
        worst_client_group = min(
            (value for values in client_results.values() for value in values.values()),
            default=0.0,
        )
        return {
            "groups": group_values,
            "clients": client_results,
            "client_accs": client_accs,
            "worst_group_acc": min(group_values.values()) if group_values else 0.0,
            "worst_client_group_acc": worst_client_group,
            "wcca": wcca,
            "cfg": cfg,
            "class_corruption_rows": class_corruption_rows,
        }

    def _run_fedease_extended_evaluation(self, models, num_classes: int) -> None:
        evaluation_cfg = self.config.get("method", {}).get("fedease", {}).get("evaluation", {})
        configured_splits = evaluation_cfg.get(
            "splits", ["clean", "same", "random", "swapped", "unseen"]
        )
        max_batches = evaluation_cfg.get("max_batches")
        split_metrics = {}
        split_details = {}
        print("[heartbeat] FedEASE extended evaluation start", flush=True)
        for split in configured_splits:
            loaders = self._fedease_evaluation_loaders.get(split)
            if loaders is None:
                print(f"[warning] FedEASE evaluation split missing: {split}", flush=True)
                continue
            metrics, details = evaluate_cle_split(
                models,
                loaders,
                device=self.device,
                num_classes=num_classes,
                num_environments=len(getattr(self, "_corruption_group_names", [])) or 4,
                max_batches=max_batches,
            )
            split_metrics[split] = metrics
            split_details[split] = details
            print(
                f"[evaluation] split={split} avg={metrics['avg_acc']:.2f} "
                f"worst={metrics['worst_acc']:.2f} wcca={metrics['wcca']:.2f} "
                f"cfg={metrics['cfg']:.2f}",
                flush=True,
            )
        summary = write_cle_evaluation(self.output_dir, split_metrics, split_details)
        print(f"[evaluation] ERS(same-swapped)={summary['ers']:.2f}", flush=True)

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            images, labels, corruption_ids = batch
            return images, labels, corruption_ids
        images, labels = batch
        return images, labels, None

    def _save_models(self, models) -> None:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for client_id, model in models.items():
            torch.save(model.state_dict(), ckpt_dir / f"client_{client_id}.pt")

    def _load_models_if_configured(self, models) -> None:
        ckpt_cfg = self.config.get("checkpoints", {})
        load_dir = ckpt_cfg.get("resume_dir") if ckpt_cfg.get("resume", False) else ckpt_cfg.get("load_dir")
        if not load_dir:
            return
        load_dir = Path(load_dir)
        for client_id, model in models.items():
            path = load_dir / f"client_{client_id}.pt"
            if not path.exists():
                continue
            state = torch.load(path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            cleaned = {
                (key[7:] if key.startswith("module.") else key): value
                for key, value in state.items()
            }
            model.load_state_dict(cleaned, strict=bool(ckpt_cfg.get("strict", True)))
