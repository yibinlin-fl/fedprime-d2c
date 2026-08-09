from __future__ import annotations

from scripts.openi_cle_multilabel_softber_entry import ARM_ORDER, build_configs, compare


def test_paired_configs_change_only_pew_and_ber_assignment() -> None:
    configs = build_configs()
    assert tuple(configs) == ARM_ORDER
    hard = configs["hard_pew_ber"]
    soft = configs["multilabel_softber"]
    assert hard["seed"] == soft["seed"] == 0
    assert hard["train"]["rounds"] == soft["train"]["rounds"] == 12
    assert hard["method"]["communication"] == soft["method"]["communication"] == "asymhfl_val"
    assert hard["method"]["strict_fit_audit"] == soft["method"]["strict_fit_audit"]
    assert hard["method"]["fedease"]["pew"]["label_mode"] == "hard"
    assert soft["method"]["fedease"]["pew"]["label_mode"] == "multi_label"
    assert hard["method"]["fedease"]["ber"]["assignment"] == "hard"
    assert soft["method"]["fedease"]["ber"]["assignment"] == "soft"


def test_decision_requires_all_four_pre_registered_gates() -> None:
    report = {
        "runs": {
            "hard_pew_ber": {"last_five": {"avg_acc": 30, "worst_acc": 20, "wcca": 3, "cfg": 25}},
            "multilabel_softber": {"last_five": {"avg_acc": 31, "worst_acc": 21, "wcca": 4, "cfg": 24}},
        }
    }
    result = compare(report)
    assert result["pass"] is True
