import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase4_common import load_phase4_config
from run_phase4_ablations import ablation_env_config, action_transform_for_ablation, zero_z_action


def test_a1_uses_isotropic_antenna():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    env_config = ablation_env_config(config["ablations"]["A1_isotropic_antenna"])

    assert env_config.channel.antenna_model == "isotropic"


def test_a2_sets_balance_penalty_to_zero():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    env_config = ablation_env_config(config["ablations"]["A2_no_balance_penalty"])

    assert env_config.reward.omega_B == 0.0


def test_a3_sets_energy_penalty_to_zero():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    env_config = ablation_env_config(config["ablations"]["A3_no_energy_penalty"])

    assert env_config.reward.omega_E == 0.0


def test_a4_z_direction_action_is_zero():
    action = zero_z_action(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    assert action.tolist() == [1.0, 2.0, 0.0]


def test_a4_transform_is_registered():
    config = load_phase4_config("configs/phase4_experiments.yaml")

    assert action_transform_for_ablation(config["ablations"]["A4_fixed_relay_height"]) is zero_z_action


def test_ablation_groups_keep_non_target_protocol_same():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    full_env = ablation_env_config(config["ablations"]["A0_full_td3"])
    a2_env = ablation_env_config(config["ablations"]["A2_no_balance_penalty"])
    a3_env = ablation_env_config(config["ablations"]["A3_no_energy_penalty"])
    a4_env = ablation_env_config(config["ablations"]["A4_fixed_relay_height"])

    assert a2_env.channel == full_env.channel
    assert a2_env.rate == full_env.rate
    assert a3_env.channel == full_env.channel
    assert a3_env.rate == full_env.rate
    assert a4_env == full_env
    assert config["experiment"]["training_seeds"] == [0, 1, 2]
    assert int(config["experiment"]["episodes"]) == 300
    assert int(config["experiment"]["max_steps"]) == 100
