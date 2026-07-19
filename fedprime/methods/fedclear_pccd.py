from __future__ import annotations

from fedprime.methods.rahfl_asymhfl import AsymHFLExperiment


class FedClearPCCDExperiment(AsymHFLExperiment):
    """RAHFL local robust learning with PCCD heterogeneous communication."""

    def __init__(self, config: dict):
        method_cfg = config.get("method", {})
        data_cfg = config.get("data", {})
        if str(data_cfg.get("scenario", "")).lower() != "cle_hfl":
            raise ValueError("FedCLEAR-PCCD requires data.scenario=cle_hfl.")
        if str(data_cfg.get("public_dataset", "")).lower() != "cifar10_npy":
            raise ValueError("FedCLEAR-PCCD requires data.public_dataset=cifar10_npy.")
        if str(method_cfg.get("cl_module", "")).lower() != "dcl":
            raise ValueError("FedCLEAR-PCCD requires the fixed RAHFL local cl_module=dcl.")
        if str(method_cfg.get("communication", "")).lower() != "pccd":
            raise ValueError("FedCLEAR-PCCD requires method.communication=pccd.")
        super().__init__(config)
