from __future__ import annotations

from fedprime.methods.rahfl_asymhfl import AsymHFLExperiment


class FedClearExperiment(AsymHFLExperiment):
    """CCRE local counterfactual learning plus IRD heterogeneous communication."""

    def __init__(self, config: dict):
        method_cfg = config.get("method", {})
        data_cfg = config.get("data", {})
        if str(data_cfg.get("scenario", "")).lower() != "cle_hfl":
            raise ValueError("FedCLEAR currently requires data.scenario=cle_hfl.")
        if str(method_cfg.get("cl_module", "")).lower() != "ccre":
            raise ValueError("FedCLEAR requires method.cl_module=ccre.")
        if str(method_cfg.get("communication", "")).lower() != "ird":
            raise ValueError("FedCLEAR requires method.communication=ird.")
        super().__init__(config)
