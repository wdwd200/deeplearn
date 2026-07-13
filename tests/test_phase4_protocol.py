import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from phase4_common import (
    ALGORITHMS,
    EPISODE_RESULT_FIELDS,
    baseline_episode_rows,
    build_agent,
    env_config_from_overrides,
    load_phase4_config,
    phase4_eval_scenarios,
    phase4_training_overrides,
    select_agent_action,
)
from uav_relay_env import UAVRelayCommEnv


def test_phase4_agents_share_observation_and_action_dimensions():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    env = UAVRelayCommEnv()
    dims = []

    for algorithm in ALGORITHMS:
        overrides = phase4_training_overrides(config, 0, algorithm, Path("results/phase4") / algorithm / "seed_0")
        agent = build_agent(algorithm, overrides, env, device="cpu")
        dims.append((agent.obs_dim, agent.action_dim))

    assert len(set(dims)) == 1
    assert dims[0] == (env.observation_dim, env.action_dim)


def test_phase4_training_budget_is_identical():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    budgets = []

    for algorithm in ALGORITHMS:
        overrides = phase4_training_overrides(config, 1, algorithm, Path("results/phase4") / algorithm / "seed_1")
        budgets.append(
            (
                overrides["training"]["episodes"],
                overrides["training"]["max_steps"],
                overrides["training"]["batch_size"],
                overrides["training"]["replay_size"],
                overrides["network"]["hidden_sizes"],
            )
        )

    assert len(set(str(item) for item in budgets)) == 1


def test_phase4_algorithms_use_same_seeds():
    config = load_phase4_config("configs/phase4_experiments.yaml")

    assert [int(seed) for seed in config["experiment"]["training_seeds"]] == [0, 1, 2]


def test_phase4_fixed_eval_scenarios_are_shared_and_unique():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    scenarios = phase4_eval_scenarios(config)
    scenario_ids = [scenario["id"] for scenario in scenarios]
    signatures = [(tuple(scenario["q_H"]), tuple(scenario["q_R"]), tuple(scenario["q_L"])) for scenario in scenarios]
    env_config = env_config_from_overrides()

    assert scenario_ids == ["eval_0", "eval_1", "eval_2", "eval_3", "eval_4"]
    assert len(set(scenario_ids)) == 5
    assert len(set(signatures)) == 5
    for scenario in scenarios:
        assert env_config.mobility.bounds_m.contains(tuple(scenario["q_R"]))

    for algorithm in ALGORITHMS:
        overrides = phase4_training_overrides(config, 0, algorithm, Path("results/phase4") / algorithm / "seed_0")
        assert overrides["eval_scenarios_path"] == config["experiment"]["eval_scenarios_path"]


def test_phase4_training_seed_does_not_change_eval_scenarios():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    reference = phase4_eval_scenarios(config)

    for seed in config["experiment"]["training_seeds"]:
        overrides = phase4_training_overrides(config, int(seed), "td3", Path("results/phase4") / "td3" / f"seed_{seed}")
        assert phase4_eval_scenarios(overrides) == reference


def test_phase4_results_record_scenario_id_for_drl_and_baselines():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    tiny_config = dict(config)
    tiny_config["experiment"] = dict(config["experiment"])
    tiny_config["experiment"]["max_steps"] = 1
    scenarios = phase4_eval_scenarios(tiny_config)

    rows = baseline_episode_rows(
        tiny_config,
        env_config=env_config_from_overrides(),
        eval_scenarios=scenarios,
    )

    assert "scenario_id" in EPISODE_RESULT_FIELDS
    assert {row["scenario_id"] for row in rows} == {scenario["id"] for scenario in scenarios}


def test_phase4_eval_action_disables_exploration_noise():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    env = UAVRelayCommEnv()
    obs, _ = env.reset()

    for algorithm in ALGORITHMS:
        overrides = phase4_training_overrides(config, 0, algorithm, Path("results/phase4") / algorithm / "seed_0")
        agent = build_agent(algorithm, overrides, env, device="cpu")
        action_a = select_agent_action(agent, obs, algorithm, explore=False)
        action_b = select_agent_action(agent, obs, algorithm, explore=False)
        assert np.allclose(action_a, action_b)


def test_phase4_common_no_longer_uses_seed_generated_eval_scenes():
    source = (ROOT / "scripts" / "phase4_common.py").read_text(encoding="utf-8")

    assert "30_000" not in source
    assert "40_000" not in source


def test_phase4_output_dirs_do_not_overlap_and_are_relative():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    output_dirs = []

    for algorithm in ALGORITHMS:
        for seed in config["experiment"]["training_seeds"]:
            output = Path(config["output"]["root_dir"]) / "algorithms" / algorithm / f"seed_{seed}"
            overrides = phase4_training_overrides(config, int(seed), algorithm, output)
            output_dirs.append(overrides["output"]["root_dir"])
            assert not Path(overrides["output"]["root_dir"]).is_absolute()

    assert len(output_dirs) == len(set(output_dirs))
