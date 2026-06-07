from __future__ import annotations

import copy
import unittest

from fedprime.utils.config import load_config


class OracleConfigTests(unittest.TestCase):
    def test_oracle_config_only_changes_experiment_and_prior_diagnostics(self):
        predicted = load_config("configs/kaggle_t4_fedprime_d2c_warmup3.yaml")
        oracle = load_config("configs/kaggle_t4_fedprime_d2c_oracle_warmup3.yaml")

        predicted = copy.deepcopy(predicted)
        oracle = copy.deepcopy(oracle)
        predicted.pop("experiment_name")
        oracle.pop("experiment_name")
        oracle["method"].pop("prior_diagnostics")
        oracle["method"]["prior_source"] = "predicted"
        self.assertEqual(predicted, oracle)


if __name__ == "__main__":
    unittest.main()
