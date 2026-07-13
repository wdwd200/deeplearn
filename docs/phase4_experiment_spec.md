# Phase 4.1 Experiment Spec

Phase 4 compares TD3, DDPG, SAC, and Phase 1 baselines under a unified protocol.
Phase 4.1 fixes the Phase 4 research protocol before any Phase 5 analysis is
allowed.

## Scope

- Retrain TD3, DDPG, and SAC with the same training budget.
- Re-evaluate all DRL algorithms and Phase 1 baselines on the same fixed
  scenarios.
- Re-run the TD3 ablations after the protocol fixes.
- Keep the communication formulas, baseline definitions, and reward semantics
  unchanged.
- Treat previous Phase 4 results as obsolete after this repair.

The current outputs are still engineering validation results, not final paper
conclusions.

## SAC Action Distribution

SAC no longer applies tanh squashing followed by an extra radial projection.
The policy samples a Gaussian variable and maps it directly into the 3D relay
velocity ball:

```text
z ~ Normal(mean, std)
r = ||z||
action = V_max * z / (1 + r)
```

This transform guarantees:

```text
||action|| < V_max
```

For the three-dimensional action, the log-Jacobian determinant is:

```text
log_det_J = 3 * log(V_max) - 4 * log(1 + ||z||)
```

The SAC policy density is:

```text
log_prob(action) = log_prob_normal(z) - log_det_J
```

The same final transformed action is used by the critic, the entropy term, and
the environment. Deterministic SAC evaluation uses the Gaussian mean through the
same transform.

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
initialization and exploration only; it does not change the evaluation
scenario content.

## Algorithms

- TD3: twin critics, delayed policy updates, and target policy smoothing.
- DDPG: deterministic actor-critic with one critic and soft target updates.
- SAC: Gaussian actor with the Phase 4.1 velocity-ball transform and entropy
  regularization.

All three algorithms use the same observation, action dimension, replay
capacity, hidden sizes, training seeds, episode count, and max step count.

## Baselines

Phase 1 policies are evaluated on the same fixed scenarios:

- `RandomPolicy`
- `StaticRelayPolicy`
- `MidpointPolicy`
- `HorizontalMidpointPolicy`
- `GreedyRatePolicy`
- `BalancedLinkPolicy`

## Ablations

- `A0_full_td3`: complete TD3.
- `A1_isotropic_antenna`: set antenna model to `isotropic`.
- `A2_no_balance_penalty`: set `omega_B = 0`.
- `A3_no_energy_penalty`: set `omega_E = 0`.
- `A4_fixed_relay_height`: force relay vertical action to zero.

All ablations use the same training seeds, training budget, and fixed
evaluation scenario file.

## Config Hash And Reuse

Each training directory writes `training_params.json` with at least:

```text
algorithm
training_seed
config_hash
git_commit
created_at
```

`config_hash` is a SHA256 hash of the normalized experiment payload. The payload
includes the algorithm name, training seed, episodes, max steps, batch size,
replay size, hidden sizes, reward parameters, full environment config,
algorithm config, training protocol, action transform, fixed evaluation scenario
path, and scenario definitions.

If an output directory already exists and `--force` is not provided, the scripts
validate the saved metadata before reusing it. Any mismatch rejects reuse and
requires:

```bash
python scripts/train_phase4_algorithms.py --force
python scripts/run_phase4_ablations.py --force
```

`--force` retrains and overwrites the corresponding Phase 4 outputs. It does not
delete or overwrite Phase 1, Phase 2, or Phase 3 results.

## Metrics

Episode-level outputs include:

```text
algorithm
training_seed
evaluation_episode
scenario_id
average_rate_e2e
average_rate_HR
average_rate_RL
average_snr_HR
average_snr_RL
total_reward
outage_count
constraint_violation_count
trajectory_length
```

## Outputs

- `results/phase4/algorithms/<algorithm>/seed_<seed>/`
- `results/phase4/algorithm_episode_results.csv`
- `results/phase4/algorithm_summary.csv`
- `results/phase4/ablations/ablation_results.csv`
- `results/phase4/ablations/ablation_summary.csv`
- `results/phase4/figures/`

Saved paths in experiment metadata are relative to the repository where
applicable.

## Acceptance Commands

```bash
python -m pytest --basetemp .pytest_tmp
python scripts/train_phase4_algorithms.py --force
python scripts/evaluate_phase4_algorithms.py
python scripts/run_phase4_ablations.py --force
python scripts/plot_phase4_results.py
python scripts/run_baselines.py
```

Phase 5 is allowed only after all commands pass and `results/phase4` contains no
`nan` or `inf` values.
