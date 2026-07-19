from __future__ import annotations

import unittest

from fedprime.utils.config import load_config


class PCCDConfigTests(unittest.TestCase):
    def test_matching_probe_changes_only_method_identity_and_communication(self):
        rahfl = load_config("configs/openi_v100_rahfl_cle_indomain_probe.yaml")
        pccd = load_config("configs/openi_v100_fedclear_pccd_probe.yaml")

        self.assertEqual(rahfl["seed"], pccd["seed"])
        self.assertEqual(rahfl["data"], pccd["data"])
        self.assertEqual(rahfl["models"], pccd["models"])
        self.assertEqual(rahfl["train"], pccd["train"])
        for key in ["use_prime", "augmix_module", "cl_module", "lambda_jsd"]:
            self.assertEqual(rahfl["method"][key], pccd["method"][key])
        self.assertEqual(rahfl["method"]["communication"], "asymhfl")
        self.assertEqual(pccd["method"]["communication"], "pccd")
        self.assertEqual(pccd["data"]["public_dataset"], "cifar10_npy")


if __name__ == "__main__":
    unittest.main()
