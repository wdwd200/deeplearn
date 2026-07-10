# Phase 1 Baseline Specification

Phase 1 validates the H -> R -> L communication environment with deterministic
or random non-learning policies. It does not implement DDPG, TD3, SAC, PPO,
actor networks, critic networks, replay buffers, neural-network training, or
model save/load logic.

## Purpose

The goal is to build baseline references before later DRL work. Every policy
uses the same `UAVRelayCommEnv` environment and the same communication formulas
documented in `docs/communication_model_spec.md`.

## Policies

- `RandomPolicy`: samples a finite velocity vector inside the relay speed limit.
- `StaticRelayPolicy`: keeps the relay UAV fixed with action `[0, 0, 0]`.
- `MidpointPolicy`: moves toward the 3D midpoint between H and L.
- `HorizontalMidpointPolicy`: moves toward the H-L horizontal midpoint while
  keeping the relay altitude unchanged by default.
- `GreedyRatePolicy`: searches a finite candidate action set and selects the
  action with the largest predicted next-step end-to-end rate.
- `BalancedLinkPolicy`: searches the same candidate set and selects the action
  that minimizes the predicted two-hop rate imbalance, with higher end-to-end
  rate used as the tie-breaker.

The search policies evaluate candidate relay positions with pure communication
calculations and do not call `env.step()` during action selection.

## Output Files

`python scripts/run_baselines.py` writes:

- `results/phase1/baseline_results.csv`: one summary row per policy.
- `results/phase1/baseline_step_results.csv`: all per-step records.
- `results/phase1/trajectory_<policy>.csv`: per-policy trajectory records.

The step records include policy name, step, relay position, link distances,
elevation angles, SNR values, per-hop rates, end-to-end rate, reward, and
constraint violation indicator.

The summary records include episode length, total reward, average rates,
average SNR values, outage count, constraint violation count, trajectory length,
and final relay position.

## Figures

`python scripts/plot_baseline_results.py` reads the Phase 1 CSV files and writes:

- `results/phase1/figures/rate_e2e_curve.png`
- `results/phase1/figures/rate_hr_rl_curve.png`
- `results/phase1/figures/snr_curve.png`
- `results/phase1/figures/distance_curve.png`
- `results/phase1/figures/elevation_angle_curve.png`
- `results/phase1/figures/trajectory_3d.png`
- `results/phase1/figures/baseline_average_rate_bar.png`
- `results/phase1/figures/constraint_violation_bar.png`

These figures are diagnostic plots for checking baseline behavior, not final
paper-quality figures.

## Phase Boundary

Phase 1 produces comparison baselines for future DRL phases. It does not contain
any DRL algorithm or learning loop. Passing Phase 1 only means the environment
and non-learning baselines are ready to serve as references for later work.
