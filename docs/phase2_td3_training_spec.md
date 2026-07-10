# Phase 2 TD3 Training Specification

Phase 2 adds a minimal TD3 training framework for the existing UAV relay
communication environment. It keeps the Phase 0/0.5/1 communication model and
baseline code unchanged.

## Scope

Implemented components:

- `ReplayBuffer` for off-policy transition storage.
- `Actor` and `Critic` PyTorch MLP networks.
- `TD3Agent` with actor target, two critic targets, target policy smoothing,
  delayed actor update, soft target update, exploration noise, save, and load.
- Optional `ObservationNormalizer` for training input preprocessing.
- `scripts/train_td3.py` for a small CPU-friendly training run.
- `scripts/evaluate_td3.py` for deterministic model evaluation.
- `scripts/plot_training_curves.py` for diagnostic training figures.

Not implemented in this phase:

- SAC, DDPG, PPO, multi-agent methods, curriculum learning, ablation studies,
  large hyperparameter sweeps, or final paper experiments.

## Training Outputs

`python scripts/train_td3.py` writes:

- `results/phase2/training_log.csv`
- `results/phase2/eval_log.csv`
- `results/phase2/td3_actor.pt`
- `results/phase2/td3_critic.pt`
- `results/phase2/config_used.yaml`
- `results/phase2/training_params.json`

The default run uses 20 episodes, a 64-sample batch size, and a small replay
buffer. It is intended to verify the full loop, not to maximize performance.

## Evaluation Outputs

`python scripts/evaluate_td3.py` writes:

- `results/phase2/td3_eval_step_results.csv`
- `results/phase2/td3_eval_summary.csv`

The summary includes TD3 average end-to-end rate and a simple comparison against
the best Phase 1 baseline from `results/phase1/baseline_results.csv`.

## Figures

`python scripts/plot_training_curves.py` writes:

- `results/phase2/figures/training_reward_curve.png`
- `results/phase2/figures/eval_rate_curve.png`
- `results/phase2/figures/td3_vs_baseline_rate_bar.png`

## Phase Boundary

Passing Phase 2 means the TD3 training framework can run, save models, evaluate
deterministically, and generate logs/figures. It does not mean TD3 outperforms
the Phase 1 baselines. Longer training and tuning belong to Phase 3.
