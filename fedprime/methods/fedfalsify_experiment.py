from __future__ import annotations

import csv
import gc
import time
from pathlib import Path

import torch

from fedprime.data.fedfalsify import build_fedfalsify_loaders
from fedprime.data.loaders import corruption_group_names_from_test_loader
from fedprime.engine.operator_metrics import (
    load_operator_metadata,
    operator_rows_for_round,
    summarize_operator_splits,
)
from fedprime.methods.fedfalsify.router import (
    CandidateAudit,
    FedFalsifyRouter,
    FedFalsifyTransferPlan,
)
from fedprime.methods.local_rahfl import train_local_augmix_dcl_epoch
from fedprime.methods.rahfl_asymhfl import AsymHFLExperiment
from fedprime.models.factory import build_models


class FedFalsifyExperiment(AsymHFLExperiment):
    """Strict fit/audit FedFalsify probe without public data or test routing."""

    def run(self) -> None:
        data_cfg = self.config["data"]
        train_cfg = self.config["train"]
        method_cfg = self.config["method"]
        falsify_cfg = method_cfg.get("fedfalsify", {})
        split_cfg = falsify_cfg.get("split", {})
        router_cfg = falsify_cfg.get("router", {})
        transfer_cfg = falsify_cfg.get("transfer", {})
        model_cfg = self.config["models"]

        if str(data_cfg.get("scenario", "")).lower() not in {"cle_hfl", "cle_hfl_v2"}:
            raise ValueError("FedFalsify currently requires the CLE-HFL prepared dataset")
        num_clients = len(model_cfg["names"])
        num_classes = int(data_cfg.get("num_classes", 10))
        split_path = Path(
            split_cfg.get(
                "path",
                self.output_dir / "fedfalsify_fit_audit_split.npz",
            )
        )
        fit_loaders, test_loader, client_splits, class_counts = build_fedfalsify_loaders(
            root=data_cfg["private_root"],
            num_clients=num_clients,
            train_batch_size=int(train_cfg["batch_size"]),
            test_batch_size=int(train_cfg.get("test_batch_size", 512)),
            num_workers=int(self.config.get("num_workers", 2)),
            split_path=split_path,
            audit_ratio=float(split_cfg.get("audit_ratio", 0.15)),
            min_audit_per_class=int(split_cfg.get("min_audit_per_class", 5)),
            min_fit_per_class=int(split_cfg.get("min_fit_per_class", 2)),
            seed=int(self.config.get("seed", 0)),
            num_classes=num_classes,
            augmix_module=str(method_cfg.get("augmix_module", "jsd")),
        )
        self._client_class_counts = class_counts
        group_names = corruption_group_names_from_test_loader(test_loader)
        operator_metadata = load_operator_metadata(data_cfg["private_root"])
        if operator_metadata:
            print(
                "[setup] CLE-HFL v2 operator evaluation enabled: "
                f"seen={len(operator_metadata.get('seen_operators', []))} "
                f"unseen={len(operator_metadata.get('unseen_operators', []))}; "
                "operator metadata is evaluation-only",
                flush=True,
            )

        print("[setup] building heterogeneous FedFalsify client models", flush=True)
        models = {
            client_id: model.to(self.device)
            for client_id, model in build_models(model_cfg["names"], num_classes).items()
        }
        self._load_models_if_configured(models)
        optimizers = {
            client_id: self._build_optimizer(model)
            for client_id, model in models.items()
        }
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for model in models.values()
            for parameter in model.parameters()
        )
        print(
            f"[setup] FedFalsify peer snapshot payload per round="
            f"{parameter_bytes / (1024 ** 2):.2f} MiB total",
            flush=True,
        )

        pretrain_epochs = int(train_cfg.get("pretrain_epochs", 0))
        if pretrain_epochs > 0:
            print(
                f"[setup] FedFalsify fit-only pretraining epochs={pretrain_epochs}",
                flush=True,
            )
            self._pretrain_phase(
                models,
                optimizers,
                fit_loaders,
                pretrain_epochs,
                train_cfg,
            )

        router = FedFalsifyRouter(
            num_classes=num_classes,
            fit_samples_per_class=int(router_cfg.get("fit_samples_per_class", 16)),
            audit_samples_per_class=int(router_cfg.get("audit_samples_per_class", 16)),
            min_fit_count=int(router_cfg.get("min_fit_count", 2)),
            min_audit_count=int(router_cfg.get("min_audit_count", 5)),
            min_tau=float(router_cfg.get("min_tau", 0.0)),
            fra_weight=float(router_cfg.get("fra_weight", 0.0)),
            fra_kappa=float(router_cfg.get("fra_kappa", 1.0)),
            fra_shrinkage_nu=float(router_cfg.get("fra_shrinkage_nu", 10.0)),
            noninferiority_veto=bool(
                router_cfg.get("noninferiority_veto", False)
            ),
            noninferiority_margin=float(
                router_cfg.get("noninferiority_margin", 0.0)
            ),
            margin_clip=float(transfer_cfg.get("margin_clip", 2.0)),
            source_correct_only=bool(transfer_cfg.get("source_correct_only", True)),
        )
        warmup_rounds = int(falsify_cfg.get("warmup_rounds", 3))
        route_interval = max(int(falsify_cfg.get("route_interval", 1)), 1)
        communication_enabled = bool(falsify_cfg.get("enabled", True))

        metrics_fields = [
            "round",
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
            "cmt_loss",
            "cmt_active_samples",
            "active_route_count",
            "route_coverage",
            "mean_selected_tau",
            "mean_selected_fra",
            "mean_selected_fra_upper_bound",
            "noninferiority_eligible_count",
            "statistically_inferior_count",
            "route_seconds",
            "round_seconds",
        ]
        route_fields = [
            "round",
            "receiver_id",
            "class_id",
            "source_id",
            "tau",
            "fra_strength",
            "fra_advantage",
            "fra_standard_error",
            "fra_upper_bound",
            "noninferiority_eligible",
            "rejection_reason",
            "score",
            "selected",
        ]
        metrics_path = self.output_dir / "metrics.csv"
        route_path = self.output_dir / "route_candidates.csv"
        operator_metrics_path = self.output_dir / "operator_split_metrics.csv"
        client_operator_path = self.output_dir / "client_operator_accuracy.csv"
        class_operator_path = self.output_dir / "client_class_operator_accuracy.csv"
        current_plan: FedFalsifyTransferPlan | None = None

        with (
            metrics_path.open("w", newline="", encoding="utf-8") as metrics_file,
            route_path.open("w", newline="", encoding="utf-8") as route_file,
            operator_metrics_path.open("w", newline="", encoding="utf-8") as operator_file,
            client_operator_path.open("w", newline="", encoding="utf-8") as client_operator_file,
            class_operator_path.open("w", newline="", encoding="utf-8") as class_operator_file,
        ):
            metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_fields)
            route_writer = csv.DictWriter(route_file, fieldnames=route_fields)
            operator_writer = csv.DictWriter(
                operator_file,
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
            client_operator_writer = csv.DictWriter(
                client_operator_file,
                fieldnames=["round", "client", "operator", "split", "accuracy"],
            )
            class_operator_writer = csv.DictWriter(
                class_operator_file,
                fieldnames=[
                    "round",
                    "client",
                    "class_id",
                    "operator",
                    "split",
                    "accuracy",
                    "total",
                ],
            )
            metrics_writer.writeheader()
            route_writer.writeheader()
            operator_writer.writeheader()
            client_operator_writer.writeheader()
            class_operator_writer.writeheader()

            for round_idx in range(int(train_cfg["rounds"])):
                round_start = time.perf_counter()
                print(f"[heartbeat] FedFalsify round {round_idx:03d} start", flush=True)

                candidates: list[CandidateAudit] = []
                route_seconds = 0.0
                if not communication_enabled:
                    current_plan = None
                    print(
                        "[heartbeat] FedFalsify strict fit-only control; "
                        "routing and CMT disabled",
                        flush=True,
                    )
                elif round_idx < warmup_rounds:
                    current_plan = None
                    print(
                        f"[heartbeat] FedFalsify warmup {round_idx + 1}/{warmup_rounds}; "
                        "CMT disabled",
                        flush=True,
                    )
                elif (
                    current_plan is None
                    or (round_idx - warmup_rounds) % route_interval == 0
                ):
                    route_start = time.perf_counter()
                    print(
                        f"[heartbeat] FedFalsify round {round_idx:03d} "
                        "building receiver-side head-TAU routes",
                        flush=True,
                    )
                    current_plan, candidates = router.build_plan(
                        models=models,
                        client_splits=client_splits,
                        device=self.device,
                        lambda_cmt=float(transfer_cfg.get("lambda_cmt", 0.5)),
                    )
                    route_seconds = time.perf_counter() - route_start
                    for candidate in candidates:
                        route_writer.writerow({"round": round_idx, **candidate.to_dict()})
                    route_file.flush()
                    selected = [
                        route
                        for routes in current_plan.routes.values()
                        for route in routes.values()
                    ]
                    mean_tau = (
                        sum(route.tau for route in selected) / len(selected)
                        if selected
                        else 0.0
                    )
                    eligible_count = sum(
                        int(candidate.noninferiority_eligible)
                        for candidate in candidates
                    )
                    inferior_count = sum(
                        int(candidate.rejection_reason == "statistically_inferior")
                        for candidate in candidates
                    )
                    print(
                        f"[heartbeat] FedFalsify routes={len(selected)}/"
                        f"{num_clients * num_classes} mean_tau={mean_tau:.4f} "
                        f"eligible={eligible_count}/{len(candidates)} "
                        f"inferior_rejected={inferior_count} "
                        f"elapsed={route_seconds:.1f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[heartbeat] FedFalsify round {round_idx:03d} "
                        "reusing frozen route map with refreshed peer snapshots disabled",
                        flush=True,
                    )

                if current_plan is not None:
                    current_plan.reset_diagnostics()
                local_losses = []
                for client_id, loader in enumerate(fit_loaders):
                    print(
                        f"[heartbeat] FedFalsify round {round_idx:03d} "
                        f"client={client_id} local start",
                        flush=True,
                    )
                    for _ in range(int(train_cfg.get("local_epochs", 1))):
                        communication_loss_fn = None
                        if current_plan is not None:
                            communication_loss_fn = (
                                lambda *, receiver_logits, clean_images, labels, cid=client_id:
                                current_plan.loss_for_batch(
                                    receiver_id=cid,
                                    receiver_logits=receiver_logits,
                                    clean_images=clean_images,
                                    labels=labels,
                                )
                            )
                        loss = train_local_augmix_dcl_epoch(
                            model=models[client_id],
                            loader=loader,
                            optimizer=optimizers[client_id],
                            device=self.device,
                            lambda_jsd=float(method_cfg.get("lambda_jsd", 12.0)),
                            cl_module=str(method_cfg.get("cl_module", "dcl")),
                            num_classes=num_classes,
                            client_class_counts=class_counts[client_id],
                            max_batches=train_cfg.get("max_local_batches"),
                            max_grad_norm=train_cfg.get("max_grad_norm"),
                            skip_nonfinite=bool(train_cfg.get("skip_nonfinite", True)),
                            log_interval=train_cfg.get("local_log_interval"),
                            context=(
                                f"FedFalsify local round={round_idx} client={client_id}"
                            ),
                            communication_loss_fn=communication_loss_fn,
                        )
                        local_losses.append(loss)

                print(
                    f"[heartbeat] FedFalsify round {round_idx:03d} evaluating final test",
                    flush=True,
                )
                group_summary = self._evaluate_corruption_groups(
                    models,
                    test_loader,
                    group_names,
                    num_classes,
                )
                split_summaries = (
                    summarize_operator_splits(group_summary, operator_metadata)
                    if operator_metadata
                    else {}
                )
                if split_summaries:
                    for split_name, split_values in split_summaries.items():
                        operator_writer.writerow(
                            {
                                "round": round_idx,
                                "split": split_name,
                                **split_values,
                            }
                        )
                    client_rows, class_rows = operator_rows_for_round(
                        round_idx=round_idx,
                        group_summary=group_summary,
                        metadata=operator_metadata,
                    )
                    client_operator_writer.writerows(client_rows)
                    class_operator_writer.writerows(class_rows)
                    operator_file.flush()
                    client_operator_file.flush()
                    class_operator_file.flush()
                accs = group_summary.get("client_accs") or self._evaluate(models, test_loader)
                transfer_metrics = (
                    current_plan.diagnostics()
                    if current_plan is not None
                    else {
                        "cmt_loss": 0.0,
                        "cmt_active_samples": 0.0,
                        "active_route_count": 0.0,
                    }
                )
                selected_routes = (
                    [
                        route
                        for routes in current_plan.routes.values()
                        for route in routes.values()
                    ]
                    if current_plan is not None
                    else []
                )
                row = {
                    "round": round_idx,
                    "avg_acc": sum(accs) / max(len(accs), 1),
                    "worst_acc": min(accs),
                    "worst_group_acc": group_summary.get("worst_group_acc", ""),
                    "worst_client_group_acc": group_summary.get(
                        "worst_client_group_acc", ""
                    ),
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
                    "local_loss": sum(local_losses) / max(len(local_losses), 1),
                    "cmt_loss": transfer_metrics["cmt_loss"],
                    "cmt_active_samples": transfer_metrics["cmt_active_samples"],
                    "active_route_count": len(selected_routes),
                    "route_coverage": len(selected_routes)
                    / max(num_clients * num_classes, 1),
                    "mean_selected_tau": (
                        sum(route.tau for route in selected_routes)
                        / max(len(selected_routes), 1)
                    ),
                    "mean_selected_fra": (
                        sum(route.fra_strength for route in selected_routes)
                        / max(len(selected_routes), 1)
                    ),
                    "mean_selected_fra_upper_bound": (
                        sum(route.fra_upper_bound for route in selected_routes)
                        / max(len(selected_routes), 1)
                    ),
                    "noninferiority_eligible_count": sum(
                        int(candidate.noninferiority_eligible)
                        for candidate in candidates
                    ),
                    "statistically_inferior_count": sum(
                        int(candidate.rejection_reason == "statistically_inferior")
                        for candidate in candidates
                    ),
                    "route_seconds": route_seconds,
                    "round_seconds": time.perf_counter() - round_start,
                }
                metrics_writer.writerow(row)
                metrics_file.flush()
                print(
                    f"[round {round_idx:03d}] avg_acc={row['avg_acc']:.2f} "
                    f"worst_acc={row['worst_acc']:.2f} "
                    f"wcca={float(row['wcca']):.2f} cfg={float(row['cfg']):.2f} "
                    f"local_loss={row['local_loss']:.4f} "
                    f"cmt_loss={row['cmt_loss']:.4f} "
                    f"routes={row['active_route_count']} "
                    f"tau={row['mean_selected_tau']:.4f} "
                    f"fra_ucb={row['mean_selected_fra_upper_bound']:.4f} "
                    f"elapsed={row['round_seconds']:.1f}s",
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

                if current_plan is not None and route_interval == 1:
                    del current_plan
                    current_plan = None
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

        if bool(self.config.get("checkpoints", {}).get("save_final", False)):
            self._save_models(models)
