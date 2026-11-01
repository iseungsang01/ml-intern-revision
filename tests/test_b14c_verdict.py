"""B.14 sec. 10 verdict rule: arithmetic on hand-written cells (no data)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ces_prediction" / "experiments" / "b14_loss"))
from verdict_b14c import joint, passes  # noqa: E402


def test_pass_rule_and_joint_verdict():
    assert passes({"mse_sig_deficits": 0, "m1_splits_pos": 3})
    assert not passes({"mse_sig_deficits": 1, "m1_splits_pos": 4})
    assert not passes({"mse_sig_deficits": 0, "m1_splits_pos": 2})
    assert joint(True, True).startswith("unconditional")
    assert joint(True, False).startswith("cut-conditional: loss")
    assert joint(False, True).startswith("population-specific")
    assert joint(False, False).startswith("cut-conditional under MSE")
