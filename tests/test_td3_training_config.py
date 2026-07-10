import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_td3 import build_agent, load_td3_training_config
from uav_relay_env import UAVRelayCommEnv
from uav_relay_env.drl.utils import make_output_dir


def test_td3_default_config_can_be_read():
    config = load_td3_training_config("configs/td3_default.yaml")

    assert "training" in config
    assert "td3" in config
    assert "network" in config
    assert "normalizer" in config
    assert "output" in config


def test_td3_default_training_parameters_are_legal():
    config = load_td3_training_config("configs/td3_default.yaml")
    training = config["training"]

    assert training["episodes"] > 0
    assert training["batch_size"] > 0
    assert training["replay_size"] >= training["batch_size"]
    assert training["eval_interval"] > 0


def test_hidden_sizes_are_passed_to_network():
    config = load_td3_training_config("configs/td3_default.yaml", overrides={"network": {"hidden_sizes": [32, 16]}})
    env = UAVRelayCommEnv()
    agent = build_agent(config, env)

    assert agent.actor.net[0].out_features == 32
    assert agent.actor.net[2].out_features == 16


def test_normalizer_parameters_can_be_read():
    config = load_td3_training_config("configs/td3_default.yaml")

    assert config["normalizer"]["enabled"] is True
    assert float(config["normalizer"]["clip_value"]) > 0.0


def test_output_path_can_be_created(tmp_path):
    output = make_output_dir(tmp_path / "phase3")

    assert output.exists()


def test_best_model_path_logic(tmp_path):
    best_actor = tmp_path / "best_td3_actor.pt"
    final_actor = tmp_path / "td3_actor.pt"

    assert not best_actor.exists()
    assert final_actor.name == "td3_actor.pt"
    assert best_actor.name == "best_td3_actor.pt"
