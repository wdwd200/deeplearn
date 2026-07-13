from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase4_common import (
    ALGORITHMS,
    EPISODE_RESULT_FIELDS,
    PHASE4_CONFIG_PATH,
    SUMMARY_FIELDS,
    baseline_episode_rows,
    env_config_from_overrides,
    evaluate_saved_agent,
    is_official_result_dir,
    load_phase4_config,
    phase4_eval_scenarios,
    resolve_path,
    summarize_episode_results,
    write_csv,
)


def evaluate_phase4_algorithms(config_path: str | Path = PHASE4_CONFIG_PATH) -> tuple[list[dict], list[dict]]:
    config = load_phase4_config(config_path)
    output_root = resolve_path(config["output"]["root_dir"])
    experiment = config["experiment"]
    seeds = [int(seed) for seed in experiment["training_seeds"]]
    eval_scenarios = phase4_eval_scenarios(config)
    if int(experiment["evaluation_episodes"]) != len(eval_scenarios):
        raise ValueError("experiment.evaluation_episodes must match the fixed eval scenario count")
    env_config = env_config_from_overrides()
    rows: list[dict] = []

    for algorithm in ALGORITHMS:
        for seed in seeds:
            model_dir = output_root / "algorithms" / algorithm / f"seed_{seed}"
            if not (model_dir / "best_actor.pt").exists():
                raise FileNotFoundError(f"missing best model for {algorithm} seed={seed}: {model_dir}")
            if not is_official_result_dir(model_dir):
                print(f"skip unofficial result: {model_dir}")
                continue
            for eval_episode, scenario in enumerate(eval_scenarios):
                rows.append(
                    evaluate_saved_agent(
                        algorithm,
                        model_dir,
                        training_seed=seed,
                        evaluation_episode=eval_episode,
                        max_steps=int(experiment["max_steps"]),
                        env_config=env_config,
                        scenario=scenario,
                    )
                )

    rows.extend(baseline_episode_rows(config, env_config=env_config, eval_scenarios=eval_scenarios))
    summary_rows = summarize_episode_results(rows)
    episode_path = output_root / "algorithm_episode_results.csv"
    summary_path = output_root / "algorithm_summary.csv"
    write_csv(episode_path, rows, EPISODE_RESULT_FIELDS)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    print(f"saved Phase 4 episode results: {episode_path}")
    print(f"saved Phase 4 algorithm summary: {summary_path}")
    return rows, summary_rows


if __name__ == "__main__":
    evaluate_phase4_algorithms()
