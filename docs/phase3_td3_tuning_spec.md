# Phase 3 TD3 Tuning Specification

Phase 3 improves the Phase 2 TD3 training framework with configuration files,
multi-seed runs, small hyperparameter sweeps, best-model selection, and stable
baseline comparison. It does not implement SAC, DDPG, PPO, multi-agent methods,
final paper ablations, or large-scale comparison experiments.

## Purpose

The purpose is to make TD3 training more reproducible and easier to inspect
before longer Phase 4 experiments. Current results are engineering validation
outputs and are not final paper conclusions.

## TD3 Configuration

`configs/td3_default.yaml` stores the default seed, training horizon, replay
buffer size, batch size, evaluation interval, TD3 optimizer parameters,
exploration-noise schedule, network hidden sizes, normalizer settings, and
output root. Scripts read training parameters from this file and only use code
defaults when a field is missing.

`configs/td3_sweep.yaml` stores the small sweep ranges and run limits.

## Multi-Seed Training

`python scripts/train_td3_multiseed.py` trains seeds `0`, `1`, and `2` by
default. Each seed writes an isolated result directory:

```text
results/phase3/seed_<seed>/
```

Each seed saves training logs, evaluation logs, final model files, best model
files, the used config, and training parameters.

## Hyperparameter Search

`python scripts/sweep_td3.py` runs at most six small configurations from
`configs/td3_sweep.yaml`. It writes:

- `results/phase3/sweep_results.csv`
- `results/phase3/best_config.yaml`

The best config is selected by the largest best evaluation average end-to-end
rate.

## Best Model Rule

During training, `td3_actor.pt` and `td3_critic.pt` are the final model files.
`best_td3_actor.pt` and `best_td3_critic.pt` are updated when evaluation
average end-to-end rate improves. Evaluation scripts load the best model by
default and fall back to the final model if best files are unavailable.

## Evaluation Metrics

Training logs include reward, average rates, average SNR values, outage count,
constraint violation count, trajectory length, actor/critic loss, Q-value
means, replay buffer size, and exploration noise. Evaluation logs include rate,
SNR, outage, constraint, and trajectory summary fields.

## Baseline Comparison

`python scripts/evaluate_td3_multiseed.py` reads Phase 1 baseline results from
`results/phase1/baseline_results.csv` and compares them with TD3 multi-seed
mean/std. The output is `results/phase3/td3_vs_baseline_summary.csv`, including
the best baseline and the TD3 mean minus best-baseline difference.

## Outputs

Phase 3 writes:

- `results/phase3/multiseed_summary.csv`
- `results/phase3/multiseed_eval_summary.csv`
- `results/phase3/td3_multiseed_mean_std.csv`
- `results/phase3/td3_vs_baseline_summary.csv`
- `results/phase3/sweep_results.csv`
- `results/phase3/best_config.yaml`
- `results/phase3/figures/*.png`

The plots include multi-seed evaluation rate, training reward, actor/critic
loss, TD3 vs baseline mean/std, best-seed trajectory, best-seed rate, and
best-seed SNR curves.
