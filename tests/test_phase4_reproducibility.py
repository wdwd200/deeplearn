import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase4_common import (
    build_agent,
    env_config_from_overrides,
    evaluate_saved_agent,
    load_phase4_config,
    phase4_eval_scenarios,
    phase4_training_metadata,
    phase4_training_overrides,
    validate_phase4_reuse,
)
from run_phase4_ablations import parse_args as parse_ablation_args
from train_phase4_algorithms import parse_args as parse_training_args
from uav_relay_env import UAVRelayCommEnv
from uav_relay_env.drl.utils import save_json, set_seed


def _minimal_td3_model(tmp_path: Path) -> tuple[Path, dict, list[dict]]:
    config = load_phase4_config("configs/phase4_experiments.yaml")
    scenarios = phase4_eval_scenarios(config)
    env_config = env_config_from_overrides()
    model_dir = tmp_path / "td3_seed_0"
    overrides = phase4_training_overrides(config, 0, "td3", model_dir)
    overrides["training"]["max_steps"] = 3
    overrides["training"]["episodes"] = 1
    overrides["training"]["batch_size"] = 4
    metadata = phase4_training_metadata("td3", overrides, env_config, scenarios)
    params = dict(overrides)
    params.update(metadata)

    set_seed(123)
    env = UAVRelayCommEnv(config=env_config)
    agent = build_agent("td3", overrides, env, device="cpu")
    agent.save(model_dir, prefix="best")
    save_json(model_dir / "training_params.json", params)
    return model_dir, overrides, scenarios


def test_same_model_and_same_scenario_are_reproducible(tmp_path):
    model_dir, _, scenarios = _minimal_td3_model(tmp_path)
    env_config = env_config_from_overrides()

    first = evaluate_saved_agent("td3", model_dir, 0, 0, max_steps=3, env_config=env_config, scenario=scenarios[0])
    second = evaluate_saved_agent("td3", model_dir, 0, 0, max_steps=3, env_config=env_config, scenario=scenarios[0])

    assert first == second


def test_different_scenario_ids_have_different_positions():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    scenarios = phase4_eval_scenarios(config)
    signatures = {(tuple(item["q_H"]), tuple(item["q_R"]), tuple(item["q_L"])) for item in scenarios}

    assert len({item["id"] for item in scenarios}) == 5
    assert len(signatures) == 5


def test_matching_config_hash_allows_explicit_reuse(tmp_path):
    _, overrides, scenarios = _minimal_td3_model(tmp_path)
    env_config = env_config_from_overrides()
    model_dir = tmp_path / "reuse_ok"
    expected = phase4_training_metadata("td3", overrides, env_config, scenarios)

    model_dir.mkdir()
    for filename in ["eval_log.csv", "best_actor.pt", "best_critic.pt"]:
        (model_dir / filename).touch()
    save_json(model_dir / "training_params.json", dict(expected))

    validate_phase4_reuse(model_dir, expected)


def test_different_config_hash_rejects_reuse(tmp_path):
    _, overrides, scenarios = _minimal_td3_model(tmp_path)
    env_config = env_config_from_overrides()
    model_dir = tmp_path / "reuse_reject"
    saved = phase4_training_metadata("td3", overrides, env_config, scenarios)
    expected = dict(saved)
    expected["config_hash"] = "0" * 64

    model_dir.mkdir()
    for filename in ["eval_log.csv", "best_actor.pt", "best_critic.pt"]:
        (model_dir / filename).touch()
    save_json(model_dir / "training_params.json", dict(saved))

    with pytest.raises(RuntimeError, match="use --force"):
        validate_phase4_reuse(model_dir, expected)


def test_incomplete_existing_result_requires_force(tmp_path):
    model_dir = tmp_path / "incomplete"
    model_dir.mkdir()

    with pytest.raises(RuntimeError, match="use --force"):
        validate_phase4_reuse(model_dir, {"config_hash": "unused"})


def test_force_flag_is_available_for_training_and_ablations():
    assert parse_training_args(["--force"]).force is True
    assert parse_ablation_args(["--force"]).force is True


def test_phase4_eval_result_paths_are_relative():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    scenarios = phase4_eval_scenarios(config)
    overrides = phase4_training_overrides(config, 0, "td3", Path("results/phase4/algorithms/td3/seed_0"))
    metadata = phase4_training_metadata("td3", overrides, env_config_from_overrides(), scenarios)

    assert not Path(overrides["output"]["root_dir"]).is_absolute()
    assert not Path(config["output"]["root_dir"]).is_absolute()
    assert not Path(config["experiment"]["eval_scenarios_path"]).is_absolute()
    assert not Path(metadata["eval_scenarios_path"]).is_absolute()
