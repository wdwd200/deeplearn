# Phase 4.2 Experiment Spec

Phase 4 compares TD3, DDPG, SAC, and Phase 1 baselines under a unified protocol.
Phase 4.2 repairs experiment version metadata and strict reuse checks. It does
not change algorithms, communication formulas, reward semantics, baseline
definitions, or fixed evaluation scenarios.

## Scope

- Record the exact Git commit used to create every formal Phase 4 result.
- Record whether the training run started from a dirty Git worktree.
- Record a source-code hash covering training-relevant files.
- Reject stale model reuse after code, config, commit, or metadata changes.
- Require a clean Git worktree before formal training.
- Recreate Phase 4 formal results after the metadata repair.

The current outputs remain engineering validation results, not final paper
conclusions.

## Fixed Evaluation Scenarios

All DRL algorithms and baselines use:

```text
configs/phase4_eval_scenarios.yaml
```

The file defines exactly five scenarios:

```text
eval_0
eval_1
eval_2
eval_3
eval_4
```

Each result row records `scenario_id`. Training seed controls model
initialization and exploration only; it does not change evaluation scenario
content.

## Metadata Fields

Every `training_params.json` written under `results/phase4` records:

```text
algorithm
training_seed
episodes
max_steps
config_hash
source_code_hash
git_commit
git_dirty
official_result
created_at
```

`git_commit` is read from:

```bash
git rev-parse HEAD
```

`git_dirty` is derived from:

```bash
git status --porcelain
```

If that command has any output, `git_dirty = true`; otherwise
`git_dirty = false`.

`official_result = true` only when the run starts with `git_dirty = false`.
Runs created with `--allow-dirty` are debug outputs and are marked
`official_result = false`.

## Source Code Hash

`source_code_hash` is a SHA256 digest over relative file paths and file contents.
The hash covers at least:

```text
src/uav_relay_env/drl/
src/uav_relay_env/comm_env.py
src/uav_relay_env/reward.py
src/uav_relay_env/channel.py
src/uav_relay_env/rate.py
src/uav_relay_env/mobility.py
scripts/phase4_common.py
scripts/train_phase4_algorithms.py
scripts/run_phase4_ablations.py
scripts/evaluate_phase4_algorithms.py
configs/phase4_experiments.yaml
configs/phase4_eval_scenarios.yaml
```

Files are sorted by repository-relative path. Results, caches, model weights,
and figures are not part of the source hash.

`config_hash` hashes the normalized experiment payload, including environment,
reward, algorithm, training budget, action transform, and evaluation scenario
definitions. `source_code_hash` hashes the code and config files that generate
the experiment. Both must match for reuse.

## Formal Training Rules

Formal training commands require a clean Git worktree:

```bash
python scripts/train_phase4_algorithms.py --force
python scripts/run_phase4_ablations.py --force
```

If the worktree is dirty, the scripts stop before training. For temporary
debugging only:

```bash
python scripts/train_phase4_algorithms.py --force --allow-dirty
python scripts/run_phase4_ablations.py --force --allow-dirty
```

Outputs produced with `--allow-dirty` are not formal results and are ignored by
formal summary generation.

## Reuse Conditions

Existing models and logs may be reused only when all of the following match:

```text
algorithm
training_seed
config_hash
source_code_hash
git_commit
git_dirty = false
official_result = true
episodes
max_steps
batch_size
replay_size
hidden_sizes
environment config
reward config
algorithm config
evaluation scenarios
required model/log files
```

Any mismatch requires retraining with `--force`. Stale results must not be made
formal by editing JSON metadata manually.

## Reproducing From A Commit

To reproduce a formal result:

1. Check out the recorded `git_commit`.
2. Confirm `git status --porcelain` has no output.
3. Run `python -m pytest --basetemp .pytest_tmp`.
4. Run `python scripts/train_phase4_algorithms.py --force`.
5. Run `python scripts/run_phase4_ablations.py --force`.
6. Run `python scripts/evaluate_phase4_algorithms.py`.
7. Run `python scripts/plot_phase4_results.py`.
8. Run `python scripts/run_baselines.py`.

The regenerated metadata should contain the same `git_commit` and
`source_code_hash` for all formal Phase 4 model directories.

## Acceptance Commands

```bash
python -m pytest --basetemp .pytest_tmp
python scripts/train_phase4_algorithms.py --force
python scripts/run_phase4_ablations.py --force
python scripts/evaluate_phase4_algorithms.py
python scripts/plot_phase4_results.py
python scripts/run_baselines.py
```

Phase 5 is allowed only after all commands pass, every formal
`training_params.json` has `git_dirty = false` and `official_result = true`, all
formal results share one `git_commit` and one `source_code_hash`, and
`results/phase4` contains no `nan` or `inf` values.
