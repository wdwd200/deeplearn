from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_td3 import load_td3_training_config, train_td3
from uav_relay_env.drl.utils import deep_update, load_yaml

PHASE4_CONFIG_PATH = ROOT / "configs" / "phase4_ablation.yaml"
TD3_CONFIG_PATH = ROOT / "configs" / "td3_default.yaml"

COMPARISON_FIELDS = [
    "method",
    "average_rate_e2e_mean",
    "average_rate_e2e_std",
    "outage_count_mean",
    "constraint_violation_count_mean",
    "source",
]

ABLATION_FIELDS = [
    "variant",
    "label",
    "changed_setting",
    "seed",
    "episodes",
    "best_eval_average_rate_e2e",
    "final_eval_average_rate_e2e",
    "best_eval_reward",
    "final_eval_reward",
    "best_episode",
    "final_episode",
    "eval_outage_count",
    "eval_constraint_violation_count",
    "model_dir",
]

ABLATION_COMPARE_FIELDS = [
    "variant",
    "label",
    "changed_setting",
    "best_eval_average_rate_e2e",
    "rate_delta_vs_phase3_td3_mean",
    "rate_ratio_vs_phase3_td3_mean",
    "constraint_delta_vs_phase3_td3_mean",
]


DEFAULT_PHASE4_CONFIG: dict[str, Any] = {
    "phase4": {
        "output_dir": "results/phase4",
        "reference_phase3_dir": "results/phase3",
        "baseline_results_path": "results/phase1/baseline_results.csv",
    },
    "ablation_training": {
        "training": {
            "episodes": 80,
            "max_steps": 100,
            "start_steps": 500,
            "batch_size": 128,
            "replay_size": 100_000,
            "eval_interval": 10,
            "save_interval": 40,
            "train_every": 100,
            "updates_per_train": 1,
            "reward_scale": 1.0,
        }
    },
    "ablations": {
        "td3_short_default": {
            "seed": 0,
            "label": "TD3 short default",
            "changed_setting": "short TD3 control using Phase 3 defaults",
            "overrides": {},
        }
    },
}


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric value: {value!r}")
    return result


def load_phase4_config(config_path: str | Path = PHASE4_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    file_config = load_yaml(path) if path.exists() else {}
    config = deep_update(DEFAULT_PHASE4_CONFIG, file_config)
    validate_phase4_config(config)
    return config


def validate_phase4_config(config: Mapping[str, Any]) -> None:
    phase4 = config.get("phase4", {})
    if not str(phase4.get("output_dir", "")):
        raise ValueError("phase4.output_dir must not be empty")
    if not str(phase4.get("reference_phase3_dir", "")):
        raise ValueError("phase4.reference_phase3_dir must not be empty")
    if not str(phase4.get("baseline_results_path", "")):
        raise ValueError("phase4.baseline_results_path must not be empty")
    ablations = config.get("ablations", {})
    if not isinstance(ablations, Mapping) or not ablations:
        raise ValueError("ablations must be a non-empty mapping")
    for name, data in ablations.items():
        if not isinstance(data, Mapping):
            raise ValueError(f"ablations.{name} must be a mapping")
        int(data.get("seed", 0))
        if not str(data.get("label", name)):
            raise ValueError(f"ablations.{name}.label must not be empty")
        overrides = data.get("overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, Mapping):
            raise ValueError(f"ablations.{name}.overrides must be a mapping")


def phase3_reference_stats(reference_dir: Path) -> dict[str, float]:
    rows = _read_csv(reference_dir / "td3_multiseed_mean_std.csv")
    stats = {row["metric"]: row for row in rows}
    rate = stats["eval_average_rate_e2e"]
    constraints = stats["eval_constraint_violation_count"]
    outage = stats["eval_outage_count"]
    return {
        "rate_mean": _as_float(rate["mean"]),
        "rate_std": _as_float(rate["std"]),
        "constraint_mean": _as_float(constraints["mean"]),
        "outage_mean": _as_float(outage["mean"]),
    }


def build_method_comparison(
    baseline_path: Path,
    reference_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_rows = _read_csv(baseline_path)
    for row in baseline_rows:
        rows.append(
            {
                "method": row["policy"],
                "average_rate_e2e_mean": row["average_rate_e2e"],
                "average_rate_e2e_std": 0.0,
                "outage_count_mean": row["outage_count"],
                "constraint_violation_count_mean": row["constraint_violation_count"],
                "source": "phase1_baseline",
            }
        )

    stats = phase3_reference_stats(reference_dir)
    rows.append(
        {
            "method": "td3_multiseed_phase3",
            "average_rate_e2e_mean": stats["rate_mean"],
            "average_rate_e2e_std": stats["rate_std"],
            "outage_count_mean": stats["outage_mean"],
            "constraint_violation_count_mean": stats["constraint_mean"],
            "source": "phase3_reference",
        }
    )
    best_baseline = max(
        [row for row in rows if row["source"] == "phase1_baseline"],
        key=lambda row: _as_float(row["average_rate_e2e_mean"]),
    )
    rows.append(
        {
            "method": "td3_minus_best_baseline",
            "average_rate_e2e_mean": stats["rate_mean"] - _as_float(best_baseline["average_rate_e2e_mean"]),
            "average_rate_e2e_std": stats["rate_std"],
            "outage_count_mean": "",
            "constraint_violation_count_mean": "",
            "source": f"best_baseline={best_baseline['method']}",
        }
    )
    return rows


def summarize_eval_rows(
    variant: str,
    variant_config: Mapping[str, Any],
    episodes: int,
    eval_rows: list[dict[str, Any]],
    model_dir: Path,
) -> dict[str, Any]:
    if not eval_rows:
        raise ValueError(f"{variant} has no eval rows")
    best = max(eval_rows, key=lambda row: _as_float(row["eval_average_rate_e2e"]))
    final = eval_rows[-1]
    return {
        "variant": variant,
        "label": variant_config.get("label", variant),
        "changed_setting": variant_config.get("changed_setting", ""),
        "seed": int(variant_config.get("seed", 0)),
        "episodes": episodes,
        "best_eval_average_rate_e2e": best["eval_average_rate_e2e"],
        "final_eval_average_rate_e2e": final["eval_average_rate_e2e"],
        "best_eval_reward": best["eval_reward"],
        "final_eval_reward": final["eval_reward"],
        "best_episode": best["episode"],
        "final_episode": final["episode"],
        "eval_outage_count": final["eval_outage_count"],
        "eval_constraint_violation_count": final["eval_constraint_violation_count"],
        "model_dir": str(model_dir.relative_to(ROOT) if model_dir.is_relative_to(ROOT) else model_dir),
    }


def _load_eval_rows(model_dir: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in _read_csv(model_dir / "eval_log.csv")]


def run_ablation_variant(
    variant: str,
    variant_config: Mapping[str, Any],
    common_overrides: Mapping[str, Any],
    output_dir: Path,
    force: bool,
) -> dict[str, Any]:
    variant_output_dir = output_dir / "ablations" / variant
    overrides = deep_update(dict(common_overrides), {"seed": int(variant_config.get("seed", 0))})
    overrides = deep_update(overrides, variant_config.get("overrides", {}) or {})
    config = load_td3_training_config(TD3_CONFIG_PATH, overrides)
    episodes = int(config["training"]["episodes"])

    if force or not (variant_output_dir / "eval_log.csv").exists():
        _, eval_rows, model_dir = train_td3(
            config_path=TD3_CONFIG_PATH,
            overrides=overrides,
            output_dir=variant_output_dir,
        )
    else:
        model_dir = variant_output_dir
        eval_rows = _load_eval_rows(model_dir)
        print(f"reuse existing ablation result: {variant_output_dir}")
    return summarize_eval_rows(variant, variant_config, episodes, eval_rows, model_dir)


def build_ablation_reference_comparison(
    ablation_rows: list[dict[str, Any]],
    reference_stats: Mapping[str, float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rate_mean = float(reference_stats["rate_mean"])
    constraint_mean = float(reference_stats["constraint_mean"])
    for row in ablation_rows:
        rate = _as_float(row["best_eval_average_rate_e2e"])
        constraints = _as_float(row["eval_constraint_violation_count"])
        output.append(
            {
                "variant": row["variant"],
                "label": row["label"],
                "changed_setting": row["changed_setting"],
                "best_eval_average_rate_e2e": rate,
                "rate_delta_vs_phase3_td3_mean": rate - rate_mean,
                "rate_ratio_vs_phase3_td3_mean": rate / rate_mean if rate_mean else "",
                "constraint_delta_vs_phase3_td3_mean": constraints - constraint_mean,
            }
        )
    return output


def run_phase4_experiments(
    config_path: str | Path = PHASE4_CONFIG_PATH,
    force: bool = False,
) -> dict[str, Path]:
    config = load_phase4_config(config_path)
    phase4 = config["phase4"]
    output_dir = _resolve(phase4["output_dir"])
    reference_dir = _resolve(phase4["reference_phase3_dir"])
    baseline_path = _resolve(phase4["baseline_results_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    method_rows = build_method_comparison(baseline_path, reference_dir)
    method_path = output_dir / "phase4_method_comparison.csv"
    _write_csv(method_path, method_rows, COMPARISON_FIELDS)

    common_overrides = config.get("ablation_training", {})
    ablation_rows = [
        run_ablation_variant(name, variant_config, common_overrides, output_dir, force)
        for name, variant_config in config["ablations"].items()
    ]
    ablation_path = output_dir / "phase4_ablation_summary.csv"
    _write_csv(ablation_path, ablation_rows, ABLATION_FIELDS)

    reference_stats = phase3_reference_stats(reference_dir)
    ablation_compare_rows = build_ablation_reference_comparison(ablation_rows, reference_stats)
    ablation_compare_path = output_dir / "phase4_ablation_vs_reference.csv"
    _write_csv(ablation_compare_path, ablation_compare_rows, ABLATION_COMPARE_FIELDS)

    print(f"saved Phase 4 method comparison: {method_path}")
    print(f"saved Phase 4 ablation summary: {ablation_path}")
    print(f"saved Phase 4 ablation comparison: {ablation_compare_path}")
    return {
        "method_comparison": method_path,
        "ablation_summary": ablation_path,
        "ablation_vs_reference": ablation_compare_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight Phase 4 comparison and TD3 ablation experiments.")
    parser.add_argument("--config", default=str(PHASE4_CONFIG_PATH), help="Phase 4 YAML config path.")
    parser.add_argument("--force", action="store_true", help="Retrain ablation variants even if logs already exist.")
    args = parser.parse_args()
    run_phase4_experiments(args.config, force=args.force)


if __name__ == "__main__":
    main()
