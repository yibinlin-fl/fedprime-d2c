from __future__ import annotations

import unittest

from fedprime.utils.config import load_config


class RahflPrimeConfigTest(unittest.TestCase):
    def test_t4_prime_control_only_changes_local_augmentation(self):
        rahfl = load_config("configs/kaggle_t4_rahfl.yaml")
        rahfl_prime = load_config("configs/kaggle_t4_rahfl_prime.yaml")

        self.assertEqual(rahfl_prime["method_name"], "rahfl_prime")
        self.assertEqual(rahfl["data"], rahfl_prime["data"])
        self.assertEqual(rahfl["models"], rahfl_prime["models"])
        self.assertEqual(rahfl["train"], rahfl_prime["train"])

        self.assertTrue(rahfl_prime["method"]["use_prime"])
        self.assertTrue(rahfl_prime["method"]["use_dcl"])
        self.assertEqual(rahfl_prime["method"]["cl_module"], "dcl")
        self.assertNotIn("communication", rahfl_prime["method"])

    def test_debug_prime_control_uses_the_same_execution_path(self):
        config = load_config("configs/debug_rahfl_prime_cifar10c.yaml")

        self.assertEqual(config["method_name"], "rahfl_prime")
        self.assertEqual(config["train"]["rounds"], 1)
        self.assertEqual(config["train"]["max_local_batches"], 1)
        self.assertTrue(config["method"]["use_prime"])
        self.assertTrue(config["method"]["use_dcl"])


if __name__ == "__main__":
    unittest.main()
