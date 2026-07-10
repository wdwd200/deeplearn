import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_phase4_experiments import (
    build_ablation_reference_comparison,
    build_method_comparison,
    load_phase4_config,
    summarize_eval_rows,
)


def test_phase4_config_can_be_read():
    config = load_phase4_config("configs/phase4_ablation.yaml")

    assert config["phase4"]["output_dir"] == "results/phase4"
    assert "td3_short_default" in config["ablations"]
    assert "no_observation_normalizer" in config["ablations"]


def test_phase4_ablation_config_uses_td3_only():
    config = load_phase4_config("configs/phase4_ablation.yaml")
    forbidden_names = {"sac", "ddpg", "ppo"}

    for name in config["ablations"]:
        lowered = name.lower()
        assert all(forbidden not in lowered for forbidden in forbidden_names)


def test_summarize_eval_rows_uses_best_and_final_values(tmp_path):
    eval_rows = [
        {
            "episode": 10,
            "eval_reward": 1.0,
            "eval_average_rate_e2e": 100.0,
            "eval_outage_count": 0,
            "eval_constraint_violation_count": 1,
        },
        {
            "episode": 20,
            "eval_reward": 2.0,
            "eval_average_rate_e2e": 90.0,
            "eval_outage_count": 1,
            "eval_constraint_violation_count": 3,
        },
    ]

    row = summarize_eval_rows(
        "variant_a",
        {"seed": 4, "label": "Variant A", "changed_setting": "test"},
        episodes=20,
        eval_rows=eval_rows,
        model_dir=tmp_path,
    )

    assert row["best_eval_average_rate_e2e"] == 100.0
    assert row["final_eval_average_rate_e2e"] == 90.0
    assert row["best_episode"] == 10
    assert row["final_episode"] == 20
    assert row["seed"] == 4


def test_ablation_reference_comparison_computes_delta():
    rows = [
        {
            "variant": "small_network",
            "label": "Small",
            "changed_setting": "network.hidden_sizes=[128, 128]",
            "best_eval_average_rate_e2e": 120.0,
            "eval_constraint_violation_count": 7.0,
        }
    ]

    comparison = build_ablation_reference_comparison(rows, {"rate_mean": 100.0, "constraint_mean": 5.0})

    assert comparison[0]["rate_delta_vs_phase3_td3_mean"] == 20.0
    assert comparison[0]["rate_ratio_vs_phase3_td3_mean"] == 1.2
    assert comparison[0]["constraint_delta_vs_phase3_td3_mean"] == 2.0


def test_method_comparison_includes_td3_and_delta():
    rows = build_method_comparison(
        Path("results/phase1/baseline_results.csv"),
        Path("results/phase3"),
    )
    methods = {row["method"] for row in rows}

    assert "td3_multiseed_phase3" in methods
    assert "td3_minus_best_baseline" in methods
