import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import phase4_common
from phase4_common import (
    build_agent,
    compute_source_code_hash,
    env_config_from_overrides,
    evaluate_saved_agent,
    filter_generated_result_status,
    git_dirty_from_status,
    is_official_result_dir,
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
    metadata = phase4_training_metadata(
        "td3",
        overrides,
        env_config,
        scenarios,
        git_dirty=False,
        source_code_hash="source-hash",
    )
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
    expected = phase4_training_metadata(
        "td3",
        overrides,
        env_config,
        scenarios,
        git_dirty=False,
        source_code_hash="source-hash",
    )

    model_dir.mkdir()
    for filename in ["eval_log.csv", "best_actor.pt", "best_critic.pt"]:
        (model_dir / filename).touch()
    save_json(model_dir / "training_params.json", dict(expected))

    validate_phase4_reuse(model_dir, expected)


def test_different_config_hash_rejects_reuse(tmp_path):
    _, overrides, scenarios = _minimal_td3_model(tmp_path)
    env_config = env_config_from_overrides()
    model_dir = tmp_path / "reuse_reject"
    saved = phase4_training_metadata(
        "td3",
        overrides,
        env_config,
        scenarios,
        git_dirty=False,
        source_code_hash="source-hash",
    )
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
    assert parse_training_args(["--allow-dirty"]).allow_dirty is True
    assert parse_ablation_args(["--allow-dirty"]).allow_dirty is True


def test_phase4_eval_result_paths_are_relative():
    config = load_phase4_config("configs/phase4_experiments.yaml")
    scenarios = phase4_eval_scenarios(config)
    overrides = phase4_training_overrides(config, 0, "td3", Path("results/phase4/algorithms/td3/seed_0"))
    metadata = phase4_training_metadata(
        "td3",
        overrides,
        env_config_from_overrides(),
        scenarios,
        git_dirty=False,
        source_code_hash="source-hash",
    )

    assert not Path(overrides["output"]["root_dir"]).is_absolute()
    assert not Path(config["output"]["root_dir"]).is_absolute()
    assert not Path(config["experiment"]["eval_scenarios_path"]).is_absolute()
    assert not Path(metadata["eval_scenarios_path"]).is_absolute()


def test_git_dirty_detection_from_status_output():
    assert git_dirty_from_status("") is False
    assert git_dirty_from_status(" M scripts/phase4_common.py\n") is True
    assert git_dirty_from_status("?? AGENTS.md\n") is True


def test_generated_phase4_results_do_not_block_formal_pipeline_status_filter():
    status = "M results/phase4/algorithm_training_summary.csv\n M results/phase4/algorithm_summary.csv\n M scripts/phase4_common.py\n"

    filtered = filter_generated_result_status(status)

    assert "results/phase4" not in filtered
    assert "scripts/phase4_common.py" in filtered


def test_assert_clean_git_worktree_rejects_dirty(monkeypatch):
    monkeypatch.setattr(phase4_common, "current_git_dirty", lambda: True)

    with pytest.raises(RuntimeError, match="clean Git worktree"):
        phase4_common.assert_clean_git_worktree(allow_dirty=False)

    assert phase4_common.assert_clean_git_worktree(allow_dirty=True) is True


def test_source_code_hash_is_stable_and_changes_with_file_content(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    first = source_dir / "a.py"
    second = source_dir / "b.py"
    first.write_text("x = 1\n", encoding="utf-8")
    second.write_text("y = 2\n", encoding="utf-8")

    original = compute_source_code_hash(["src"], root=tmp_path)
    assert compute_source_code_hash(["src"], root=tmp_path) == original

    second.write_text("y = 3\n", encoding="utf-8")
    assert compute_source_code_hash(["src"], root=tmp_path) != original


def test_git_commit_mismatch_rejects_reuse(tmp_path):
    _, overrides, scenarios = _minimal_td3_model(tmp_path)
    env_config = env_config_from_overrides()
    model_dir = tmp_path / "reuse_commit_reject"
    saved = phase4_training_metadata(
        "td3",
        overrides,
        env_config,
        scenarios,
        git_dirty=False,
        source_code_hash="source-hash",
    )
    expected = dict(saved)
    expected["git_commit"] = "different"

    model_dir.mkdir()
    for filename in ["eval_log.csv", "best_actor.pt", "best_critic.pt"]:
        (model_dir / filename).touch()
    save_json(model_dir / "training_params.json", dict(saved))

    with pytest.raises(RuntimeError, match="use --force"):
        validate_phase4_reuse(model_dir, expected)


def test_source_code_hash_mismatch_rejects_reuse(tmp_path):
    _, overrides, scenarios = _minimal_td3_model(tmp_path)
    env_config = env_config_from_overrides()
    model_dir = tmp_path / "reuse_source_reject"
    saved = phase4_training_metadata(
        "td3",
        overrides,
        env_config,
        scenarios,
        git_dirty=False,
        source_code_hash="source-hash",
    )
    expected = dict(saved)
    expected["source_code_hash"] = "different"

    model_dir.mkdir()
    for filename in ["eval_log.csv", "best_actor.pt", "best_critic.pt"]:
        (model_dir / filename).touch()
    save_json(model_dir / "training_params.json", dict(saved))

    with pytest.raises(RuntimeError, match="use --force"):
        validate_phase4_reuse(model_dir, expected)


def test_dirty_result_cannot_be_reused_as_official(tmp_path):
    _, overrides, scenarios = _minimal_td3_model(tmp_path)
    env_config = env_config_from_overrides()
    model_dir = tmp_path / "dirty_result"
    saved = phase4_training_metadata(
        "td3",
        overrides,
        env_config,
        scenarios,
        git_dirty=True,
        source_code_hash="source-hash",
    )
    expected = dict(saved)
    expected["git_dirty"] = False
    expected["official_result"] = True

    model_dir.mkdir()
    for filename in ["eval_log.csv", "best_actor.pt", "best_critic.pt"]:
        (model_dir / filename).touch()
    save_json(model_dir / "training_params.json", dict(saved))

    assert is_official_result_dir(model_dir) is False
    with pytest.raises(RuntimeError, match="use --force"):
        validate_phase4_reuse(model_dir, expected)
