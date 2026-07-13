# Phase 5 Reproducibility

## Environment

- Python version: 3.14.0
- Operating system used for this run: Windows-10-10.0.19045-SP0
- Required packages used in the generated manifest: `numpy`, `matplotlib`, `torch`
- CPU/GPU requirement: CPU execution is sufficient for reproducing the saved analysis and figures. GPU is optional for retraining, but Phase 5 does not retrain models.

Install dependencies with:

```bash
python -m pip install numpy matplotlib torch pyyaml pytest
```

## Repository and Metadata

- Formal Phase 4 experiment commit: `221c50db0924c5cd02ad69e0e440bbb39da79d6c`
- Formal Phase 4 source code hash: `013686511bc9830caa578e6aa7d4ea21962f1c845348e8e7a6db66acce6e9f97`
- `git_dirty` definition: `git status --porcelain` has output when the worktree is dirty. Formal Phase 4 result metadata requires `git_dirty = false`.

## Configuration Files

- Communication environment: `configs/comm_env_default.yaml`
- Phase 4 experiment protocol: `configs/phase4_experiments.yaml`
- Fixed evaluation scenarios: `configs/phase4_eval_scenarios.yaml`

All paths in metadata and Phase 5 outputs are repository-relative paths. On Windows,
commands can be run from PowerShell. On Linux, use the same relative paths with `/` path
separators.

## Reproduction Commands

Phase 1 baseline:

```bash
python scripts/run_baselines.py
```

Phase 3 TD3 multi-seed and plots:

```bash
python scripts/train_td3_multiseed.py
python scripts/evaluate_td3_multiseed.py
python scripts/plot_phase3_results.py
```

Phase 4 formal training:

```bash
python scripts/train_phase4_algorithms.py --force
```

Phase 4 ablations:

```bash
python scripts/run_phase4_ablations.py --force
```

Phase 4 evaluation and plots:

```bash
python scripts/evaluate_phase4_algorithms.py
python scripts/plot_phase4_results.py
```

Phase 5 validation, tables, figures, analysis, and manifest:

```bash
python scripts/validate_final_results.py
python scripts/build_paper_tables.py
python scripts/plot_paper_figures.py
python scripts/analyze_final_results.py
python scripts/build_reproducibility_manifest.py
```

## Result Files

- `results/phase5/analysis/final_validation_report.json`: machine-readable final validation report.
- `results/phase5/analysis/final_validation_report.md`: readable validation summary.
- `results/phase5/source_data/`: normalized copies or derived source data for paper tables and figures.
- `results/phase5/tables/`: CSV and Markdown paper tables.
- `results/phase5/figures/`: PNG and PDF paper figures.
- `results/phase5/analysis/algorithm_statistics.csv`: algorithm descriptive statistics and baseline differences.
- `results/phase5/analysis/ablation_statistics.csv`: ablation descriptive statistics and A0 differences.
- `results/phase5/analysis/scenario_statistics.csv`: per-scenario method ranking.
- `results/phase5/analysis/constraint_statistics.csv`: constraint violation statistics.
- `results/phase5/manifests/reproducibility_manifest.json`: environment, metadata, files, and model checksum manifest.
- `results/phase5/manifests/file_checksums.sha256`: SHA256 checksums for configs and Phase 4/5 reproducibility files.

## Metadata Verification

Run:

```bash
python scripts/validate_final_results.py
python scripts/build_reproducibility_manifest.py
```

The validation report must pass, and the manifest `git_commit` and `source_code_hash`
must match the formal Phase 4 training metadata.
