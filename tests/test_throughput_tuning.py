from copy import deepcopy

import pytest
import yaml

from scripts.run_minsnr_ablation import ARMS, PROJECT_ROOT, assert_matched_configs, make_arm_configs
from scripts.tune_minsnr_throughput import choose_candidate, with_batch


def test_pro6000_protocol_is_matched_and_budgeted():
    protocol = yaml.safe_load((PROJECT_ROOT / "configs/fold03_minsnr_pro6000.yaml").read_text("utf-8"))
    _, arms = make_arm_configs(protocol, PROJECT_ROOT)
    assert_matched_configs(arms)
    config = arms["A_mse"]["train_diffusion"]
    assert config["mixed_precision"] == "bf16"
    assert config["gradient_checkpointing"] is False
    assert config["batch_size"] * config["max_steps"] == 160000
    assert config["fused_adamw"] is True
    for batch in (32, 64, 128, 160, 256, 320, 640, 1000, 1280):
        _, candidates = make_arm_configs(with_batch(protocol, batch), PROJECT_ROOT)
        assert candidates["A_mse"]["train_diffusion"]["max_steps"] * batch == 160000
        assert_matched_configs(candidates)
    legacy = yaml.safe_load((PROJECT_ROOT / "configs/fold03_minsnr_pilot.yaml").read_text("utf-8"))
    _, legacy_arms = make_arm_configs(legacy, PROJECT_ROOT)
    assert legacy_arms["A_mse"]["train_diffusion"]["batch_size"] == 32
    assert legacy_arms["A_mse"]["train_diffusion"]["gradient_checkpointing"] is True


@pytest.mark.parametrize("overrides", [{"typo": 1}, {"batch_size": 0}, {"num_workers": -1},
                                        {"mixed_precision": "fp8"}, {"fused_adamw": "false"}])
def test_performance_overrides_fail_closed(overrides):
    protocol = yaml.safe_load((PROJECT_ROOT / "configs/fold03_minsnr_pro6000.yaml").read_text("utf-8"))
    protocol["training"]["performance"] = overrides
    with pytest.raises(ValueError):
        make_arm_configs(protocol, PROJECT_ROOT)


def candidate(batch, rate, peak, status="ok"):
    return {"batch_size": batch, "status": status, "arms": {
        arm: {"images_per_second": rate, "peak_reserved_bytes": peak,
              "initial_free_bytes": 95, "total_bytes": 100} for arm in ARMS}}


def test_selection_uses_speed_not_size_and_reserves_headroom():
    choices = [candidate(32, 10, 20), candidate(64, 30, 40), candidate(128, 20, 70),
               candidate(160, 40, 90), candidate(256, 100, 100, "oom")]
    assert choose_candidate(choices, .08)["batch_size"] == 64
    slower_b = deepcopy(choices[1])
    slower_b["arms"]["B_minsnr5"]["images_per_second"] = 5
    assert choose_candidate([choices[0], slower_b], .08)["batch_size"] == 32
    with pytest.raises(RuntimeError, match="No safe batch"):
        choose_candidate([choices[-1]], .08)
    with pytest.raises(ValueError, match="divide"):
        with_batch({}, 192)
