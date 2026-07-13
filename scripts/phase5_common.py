from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase4_common import (  # noqa: E402
    ALGORITHMS,
    COMM_CONFIG_PATH,
    EVAL_SCENARIOS_PATH,
    PHASE4_CONFIG_PATH,
    current_git_commit,
    env_config_for_scenario,
    load_agent_from_dir,
    load_phase4_config,
    load_training_metadata,
    load_yaml,
    phase4_eval_scenarios,
    relative_path,
    select_agent_action,
)
from uav_relay_env import EpisodeMetrics, UAVRelayCommEnv, make_default_policies  # noqa: E402

PHASE1_ROOT = ROOT / "results" / "phase1"
PHASE3_ROOT = ROOT / "results" / "phase3"
PHASE4_ROOT = ROOT / "results" / "phase4"
PHASE5_ROOT = ROOT / "results" / "phase5"

PHASE5_TABLES = PHASE5_ROOT / "tables"
PHASE5_FIGURES = PHASE5_ROOT / "figures"
PHASE5_ANALYSIS = PHASE5_ROOT / "analysis"
PHASE5_MANIFESTS = PHASE5_ROOT / "manifests"
PHASE5_SOURCE_DATA = PHASE5_ROOT / "source_data"

ALGORITHM_EPISODE_PATH = PHASE4_ROOT / "algorithm_episode_results.csv"
ALGORITHM_SUMMARY_PATH = PHASE4_ROOT / "algorithm_summary.csv"
ALGORITHM_TRAINING_SUMMARY_PATH = PHASE4_ROOT / "algorithm_training_summary.csv"
ABLATION_RESULTS_PATH = PHASE4_ROOT / "ablations" / "ablation_results.csv"
ABLATION_SUMMARY_PATH = PHASE4_ROOT / "ablations" / "ablation_summary.csv"
BASELINE_RESULTS_PATH = PHASE1_ROOT / "baseline_results.csv"
BASELINE_STEP_RESULTS_PATH = PHASE1_ROOT / "baseline_step_results.csv"

DRL_METHODS = ("TD3", "DDPG", "SAC")
BASELINE_METHODS = (
    "RandomPolicy",
    "StaticRelayPolicy",
    "MidpointPolicy",
    "HorizontalMidpointPolicy",
    "GreedyRatePolicy",
    "BalancedLinkPolicy",
)
PHASE1_POLICY_NAME_MAP = {
    "random": "RandomPolicy",
    "static": "StaticRelayPolicy",
    "midpoint": "MidpointPolicy",
    "horizontal_midpoint": "HorizontalMidpointPolicy",
    "greedy_rate": "GreedyRatePolicy",
    "balanced_link": "BalancedLinkPolicy",
}
ABLATION_GROUPS = (
    "A0_full_td3",
    "A1_isotropic_antenna",
    "A2_no_balance_penalty",
    "A3_no_energy_penalty",
    "A4_fixed_relay_height",
)
EVAL_SCENARIO_IDS = tuple(f"eval_{index}" for index in range(5))

EPISODE_REQUIRED_FIELDS = (
    "algorithm",
    "training_seed",
    "evaluation_episode",
    "scenario_id",
    "average_rate_e2e",
    "average_rate_HR",
    "average_rate_RL",
    "average_snr_HR",
    "average_snr_RL",
    "total_reward",
    "outage_count",
    "constraint_violation_count",
    "trajectory_length",
)
ALGORITHM_SUMMARY_REQUIRED_FIELDS = (
    "algorithm",
    "average_rate_e2e_mean",
    "average_rate_e2e_std",
    "average_reward_mean",
    "average_reward_std",
    "outage_count_mean",
    "constraint_violation_count_mean",
    "trajectory_length_mean",
)
ABLATION_SUMMARY_REQUIRED_FIELDS = (
    "ablation",
    "average_rate_e2e_mean",
    "average_rate_e2e_std",
    "average_reward_mean",
    "constraint_violation_count_mean",
    "trajectory_length_mean",
    "difference_from_full_td3",
)

PHASE5_SOURCE_HASH_PATHS = (
    "scripts/phase5_common.py",
    "scripts/validate_final_results.py",
    "scripts/build_paper_tables.py",
    "scripts/plot_paper_figures.py",
    "scripts/analyze_final_results.py",
    "scripts/build_reproducibility_manifest.py",
    "configs/comm_env_default.yaml",
    "configs/phase4_experiments.yaml",
    "configs/phase4_eval_scenarios.yaml",
    "configs/td3_default.yaml",
    "configs/ddpg_default.yaml",
    "configs/sac_default.yaml",
)
PHASE5_REQUIRED_SCRIPT_PATHS = (
    "scripts/validate_final_results.py",
    "scripts/build_paper_tables.py",
    "scripts/plot_paper_figures.py",
    "scripts/analyze_final_results.py",
    "scripts/build_reproducibility_manifest.py",
)
PHASE5_IGNORED_GENERATED_PREFIXES = ("results/phase5/",)


def ensure_phase5_dirs() -> None:
    for path in (PHASE5_TABLES, PHASE5_FIGURES, PHASE5_ANALYSIS, PHASE5_MANIFESTS, PHASE5_SOURCE_DATA):
        path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(relative_path(path))
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or infer_fieldnames(rows))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key, "")) for key in fields})


def infer_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def serialize_cell(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_table(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or infer_fieldnames(rows))
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(markdown_cell(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_cell(value: Any) -> str:
    text = str(serialize_cell(value))
    return text.replace("|", "\\|").replace("\n", " ")


def to_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {value!r}")
    return result


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def pstdev(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def group_rows(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(grouped)


def required_fields_present(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[str]:
    errors: list[str] = []
    if not rows:
        errors.append("file has no rows")
        return errors
    for field in fields:
        if field not in rows[0]:
            errors.append(f"missing field: {field}")
    return errors


def rel(path: Path | str) -> str:
    return relative_path(path).replace("\\", "/")


def run_git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_status_porcelain() -> str:
    try:
        return run_git(["status", "--porcelain"])
    except (OSError, subprocess.CalledProcessError):
        return ""


def _status_line_path(line: str) -> str:
    normalized = line.strip().strip('"').replace("\\", "/")
    for marker in ("results/", "scripts/", "configs/", "tests/", "docs/", "src/"):
        index = normalized.find(marker)
        if index >= 0:
            return normalized[index:]
    return normalized[3:].strip().strip('"') if len(normalized) >= 3 else normalized


def filter_phase5_generated_status(status_output: str) -> str:
    lines: list[str] = []
    for line in status_output.splitlines():
        path = _status_line_path(line)
        if any(path.startswith(prefix) for prefix in PHASE5_IGNORED_GENERATED_PREFIXES):
            continue
        lines.append(line)
    return "\n".join(lines)


def current_phase5_git_dirty() -> bool:
    return bool(filter_phase5_generated_status(git_status_porcelain()).strip())


def assert_clean_phase5_generation() -> bool:
    dirty = current_phase5_git_dirty()
    if dirty:
        raise RuntimeError(
            "Phase 5 formal results must be generated from a committed and clean code version. "
            "Please commit code first, then rerun."
        )
    return dirty


def iter_phase5_source_hash_files(
    paths: Sequence[str | Path] = PHASE5_SOURCE_HASH_PATHS,
    root: Path = ROOT,
) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = root / raw_path
        if path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and "__pycache__" not in candidate.parts
                and candidate.suffix in {".py", ".yaml", ".yml", ".json"}
            )
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(
        set(files),
        key=lambda candidate: str(candidate.resolve().relative_to(root.resolve())).replace("\\", "/"),
    )


def compute_phase5_source_code_hash(
    paths: Sequence[str | Path] = PHASE5_SOURCE_HASH_PATHS,
    root: Path = ROOT,
) -> str:
    digest = hashlib.sha256()
    for path in iter_phase5_source_hash_files(paths, root=root):
        rel_path = str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit_exists(commit: str) -> bool:
    try:
        run_git(["rev-parse", "--verify", f"{commit}^{{commit}}"])
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def git_tree_files(commit: str) -> set[str]:
    output = run_git(["ls-tree", "-r", "--name-only", commit])
    return {line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()}


def validate_phase5_commit_contains_scripts(commit: str) -> list[str]:
    if not git_commit_exists(commit):
        return [f"commit does not exist: {commit}"]
    files = git_tree_files(commit)
    missing = [path for path in PHASE5_REQUIRED_SCRIPT_PATHS if path not in files]
    return [f"missing from {commit}: {path}" for path in missing]


def is_relative_recorded_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return True
    normalized = value.replace("\\", "/")
    if ":\\" in value or normalized.startswith("/"):
        return False
    return True


def all_phase4_training_param_paths() -> list[Path]:
    return sorted(PHASE4_ROOT.rglob("training_params.json"))


def official_metadata_values() -> dict[str, Any]:
    commits: set[str] = set()
    source_hashes: set[str] = set()
    config_hashes: set[str] = set()
    for path in all_phase4_training_param_paths():
        metadata = load_yaml(path)
        if not isinstance(metadata, Mapping):
            continue
        if metadata.get("official_result") is True and metadata.get("git_dirty") is False:
            commits.add(str(metadata.get("git_commit", "")))
            source_hashes.add(str(metadata.get("source_code_hash", "")))
            config_hash = str(metadata.get("config_hash", ""))
            if config_hash:
                config_hashes.add(config_hash)
    return {
        "git_commit": sorted(commits)[0] if len(commits) == 1 else "",
        "source_code_hash": sorted(source_hashes)[0] if len(source_hashes) == 1 else "",
        "config_hashes": sorted(config_hashes),
        "git_commits": sorted(commits),
        "source_code_hashes": sorted(source_hashes),
    }


def summarize_by_method(rows: Sequence[Mapping[str, Any]], method_field: str) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method, method_rows in sorted(group_rows(rows, method_field).items()):
        rates = [to_float(row["average_rate_e2e"]) for row in method_rows]
        rewards = [to_float(row["total_reward"]) for row in method_rows]
        outages = [to_float(row["outage_count"]) for row in method_rows]
        constraints = [to_float(row["constraint_violation_count"]) for row in method_rows]
        lengths = [to_float(row["trajectory_length"]) for row in method_rows]
        summaries.append(
            {
                method_field: method,
                "average_rate_e2e_mean": mean(rates),
                "average_rate_e2e_std": pstdev(rates),
                "average_reward_mean": mean(rewards),
                "average_reward_std": pstdev(rewards),
                "outage_count_mean": mean(outages),
                "constraint_violation_count_mean": mean(constraints),
                "trajectory_length_mean": mean(lengths),
            }
        )
    return summaries


def rates_to_mbps(row: Mapping[str, Any], source_key: str) -> float:
    return to_float(row[source_key]) / 1_000_000.0


def with_source_metadata(
    rows: Sequence[Mapping[str, Any]],
    source_file: Path,
    source_stage: str,
    rate_key: str,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    metadata = metadata or official_metadata_values()
    output: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        if rate_key in row:
            rate_bps = to_float(row[rate_key])
            new_row["rate_bps"] = rate_bps
            new_row["rate_mbps"] = rate_bps / 1_000_000.0
        new_row["source_file"] = rel(source_file)
        new_row["source_stage"] = source_stage
        new_row["git_commit"] = metadata.get("git_commit", "")
        new_row["source_code_hash"] = metadata.get("source_code_hash", "")
        output.append(new_row)
    return output


def write_source_data() -> list[Path]:
    ensure_phase5_dirs()
    metadata = official_metadata_values()
    written: list[Path] = []

    source_specs = [
        (BASELINE_RESULTS_PATH, "phase1", "average_rate_e2e", "phase1_baseline_results.csv"),
        (ALGORITHM_EPISODE_PATH, "phase4", "average_rate_e2e", "phase4_algorithm_episode_results.csv"),
        (ALGORITHM_SUMMARY_PATH, "phase4", "average_rate_e2e_mean", "phase4_algorithm_summary.csv"),
        (ABLATION_RESULTS_PATH, "phase4", "average_rate_e2e", "phase4_ablation_results.csv"),
        (ABLATION_SUMMARY_PATH, "phase4", "average_rate_e2e_mean", "phase4_ablation_summary.csv"),
    ]
    for source_file, stage, rate_key, filename in source_specs:
        rows = with_source_metadata(read_csv_rows(source_file), source_file, stage, rate_key, metadata)
        target = PHASE5_SOURCE_DATA / filename
        write_csv_rows(target, rows)
        written.append(target)

    training_rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        for seed_dir in sorted((PHASE4_ROOT / "algorithms" / algorithm).glob("seed_*")):
            seed = int(seed_dir.name.split("_")[-1])
            source_file = seed_dir / "training_log.csv"
            for row in read_csv_rows(source_file):
                new_row = dict(row)
                new_row["algorithm"] = algorithm.upper()
                new_row["training_seed"] = seed
                training_rows.append(new_row)
    training_rows = with_source_metadata(
        training_rows,
        PHASE4_ROOT / "algorithms" / "<algorithm>" / "seed_<seed>" / "training_log.csv",
        "phase4",
        "average_rate_e2e",
        metadata,
    )
    target = PHASE5_SOURCE_DATA / "phase4_training_curves.csv"
    write_csv_rows(target, training_rows)
    written.append(target)

    constraint_rows: list[dict[str, Any]] = []
    for row in read_csv_rows(ALGORITHM_SUMMARY_PATH):
        constraint_rows.append(
            {
                "group": "algorithm",
                "name": row["algorithm"],
                "constraint_violation_count_mean": row["constraint_violation_count_mean"],
                "trajectory_length_mean": row["trajectory_length_mean"],
                "rate_bps": row["average_rate_e2e_mean"],
                "rate_mbps": to_float(row["average_rate_e2e_mean"]) / 1_000_000.0,
            }
        )
    for row in read_csv_rows(ABLATION_SUMMARY_PATH):
        constraint_rows.append(
            {
                "group": "ablation",
                "name": row["ablation"],
                "constraint_violation_count_mean": row["constraint_violation_count_mean"],
                "trajectory_length_mean": row["trajectory_length_mean"],
                "rate_bps": row["average_rate_e2e_mean"],
                "rate_mbps": to_float(row["average_rate_e2e_mean"]) / 1_000_000.0,
            }
        )
    constraint_rows = with_source_metadata(
        constraint_rows,
        ALGORITHM_SUMMARY_PATH,
        "phase4",
        "rate_bps",
        metadata,
    )
    target = PHASE5_SOURCE_DATA / "phase4_constraint_summary.csv"
    write_csv_rows(target, constraint_rows)
    written.append(target)
    return written


def finite_csv_report(paths: Sequence[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        for row_index, row in enumerate(read_csv_rows(path), start=2):
            for field, value in row.items():
                if value == "":
                    continue
                lowered = str(value).lower()
                if lowered in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                    errors.append(f"{rel(path)}:{row_index} {field}={value}")
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    errors.append(f"{rel(path)}:{row_index} {field}={value}")
    return errors


def finite_json_report(paths: Sequence[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{rel(path)} invalid json: {exc}")
            continue
        errors.extend(_finite_json_errors(data, rel(path)))
    return errors


def _finite_json_errors(data: Any, prefix: str) -> list[str]:
    errors: list[str] = []
    if isinstance(data, Mapping):
        for key, value in data.items():
            errors.extend(_finite_json_errors(value, f"{prefix}.{key}"))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            errors.extend(_finite_json_errors(value, f"{prefix}[{index}]"))
    elif isinstance(data, float) and not math.isfinite(data):
        errors.append(f"{prefix}={data}")
    return errors


def assert_close(actual: float, expected: float, tolerance: float = 1e-6) -> bool:
    scale = max(1.0, abs(actual), abs(expected))
    return abs(actual - expected) <= tolerance * scale


def recompute_algorithm_summary() -> list[dict[str, Any]]:
    return summarize_by_method(read_csv_rows(ALGORITHM_EPISODE_PATH), "algorithm")


def recompute_ablation_summary() -> list[dict[str, Any]]:
    rows = summarize_by_method(read_csv_rows(ABLATION_RESULTS_PATH), "algorithm")
    full_rate = next(row["average_rate_e2e_mean"] for row in rows if row["algorithm"] == "A0_full_td3")
    output = []
    for row in rows:
        output.append(
            {
                "ablation": row["algorithm"],
                "average_rate_e2e_mean": row["average_rate_e2e_mean"],
                "average_rate_e2e_std": row["average_rate_e2e_std"],
                "average_reward_mean": row["average_reward_mean"],
                "constraint_violation_count_mean": row["constraint_violation_count_mean"],
                "trajectory_length_mean": row["trajectory_length_mean"],
                "difference_from_full_td3": row["average_rate_e2e_mean"] - full_rate,
            }
        )
    return output


def compare_summary_rows(
    actual_rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
    key_field: str,
    numeric_fields: Sequence[str],
) -> list[str]:
    errors: list[str] = []
    actual = {str(row[key_field]): row for row in actual_rows}
    expected = {str(row[key_field]): row for row in expected_rows}
    if set(actual) != set(expected):
        errors.append(f"{key_field} mismatch: actual={sorted(actual)} expected={sorted(expected)}")
        return errors
    for key in sorted(actual):
        for field in numeric_fields:
            if not assert_close(to_float(actual[key][field]), to_float(expected[key][field])):
                errors.append(
                    f"{key}.{field} actual={actual[key][field]} expected={expected[key][field]}"
                )
    return errors


def run_episode_trace(
    method: str,
    scenario_id: str = "eval_0",
    seed: int = 0,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    config = load_phase4_config(PHASE4_CONFIG_PATH)
    scenarios = {str(scenario["id"]): scenario for scenario in phase4_eval_scenarios(config)}
    scenario = scenarios[scenario_id]
    env_config = env_config_for_scenario(scenario)
    env = UAVRelayCommEnv(config=env_config)
    observation, info = env.reset(seed=10_000 + seed)
    max_steps = max_steps or int(config["experiment"]["max_steps"])
    method_lower = method.lower()

    agent = None
    policy = None
    if method.upper() in DRL_METHODS:
        model_dir = PHASE4_ROOT / "algorithms" / method_lower / f"seed_{seed}"
        agent = load_agent_from_dir(method_lower, model_dir, env_config=env_config, prefer_best=True)
    else:
        policies = {policy.__class__.__name__: policy for policy in make_default_policies(env_config, seed=0)}
        policy = policies[method]

    rows: list[dict[str, Any]] = []
    for step_index in range(max_steps):
        if agent is not None:
            action = select_agent_action(agent, observation, method_lower, explore=False)
        elif policy is not None:
            action = policy.select_action(observation, info)
        else:
            raise ValueError(f"unsupported trace method: {method}")
        observation, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "method": method,
                "scenario_id": scenario_id,
                "step": step_index + 1,
                "q_R_x": info["q_R"][0],
                "q_R_y": info["q_R"][1],
                "q_R_z": info["q_R"][2],
                "rate_HR": info["rate_HR"],
                "rate_RL": info["rate_RL"],
                "rate_e2e": info["rate_e2e"],
                "reward": reward,
                "constraint_violation": int(bool(info["constraint_violation"])),
            }
        )
        if terminated or truncated:
            break
    return rows


def package_versions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for name in ("numpy", "matplotlib", "torch", "yaml"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        packages["pyyaml" if name == "yaml" else name] = str(getattr(module, "__version__", "unknown"))
    return packages


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_paths() -> list[Path]:
    paths: list[Path] = []
    for root in (ROOT / "configs", PHASE5_TABLES, PHASE5_FIGURES, PHASE5_ANALYSIS):
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    if PHASE4_ROOT.exists():
        paths.extend(path for path in PHASE4_ROOT.glob("*.csv") if path.is_file())
        paths.extend(path for path in PHASE4_ROOT.rglob("*.json") if path.is_file())
    return sorted(set(paths), key=rel)


def model_file_paths() -> list[str]:
    suffixes = {"*.pt"}
    paths: list[Path] = []
    for pattern in suffixes:
        paths.extend(PHASE4_ROOT.rglob(pattern))
    return [rel(path) for path in sorted(paths, key=rel)]


def git_tracked_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return current_git_commit()
    return result.stdout.strip()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def python_environment() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "package_versions": package_versions(),
    }
