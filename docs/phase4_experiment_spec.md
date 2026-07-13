# Phase 4 Experiment Spec

Phase 4 compares TD3, DDPG, SAC, and Phase 1 baselines under a unified protocol.
It also runs four TD3-only ablations to isolate key model components.

## Purpose

- Verify that three DRL algorithms can train and evaluate under the same budget.
- Compare DRL methods against fixed baselines.
- Quantify the impact of antenna gain, balance penalty, energy penalty, and
  relay height control.
- Keep the communication model unchanged.

## Algorithms

- TD3: twin critics, delayed policy updates, and target policy smoothing.
- DDPG: deterministic actor-critic with one critic and soft target updates.
- SAC: stochastic Gaussian actor with twin critics and entropy regularization.

All three algorithms use the same observation, action space, replay capacity,
hidden sizes, seeds, episode count, and step count.

## Fair Comparison Rules

- Same environment and communication configuration.
- Same training seeds: `0, 1, 2`.
- Same training episodes: `300`.
- Same max steps: `100`.
- Same batch size and replay buffer size.
- No algorithm-specific change to the environment or reward other than the
  defined ablation switches.

## Baselines

Phase 1 policies are evaluated with the same environment and step budget:

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

## Metrics

- `average_rate_e2e`
- `average_rate_HR`
- `average_rate_RL`
- `average_snr_HR`
- `average_snr_RL`
- `total_reward`
- `outage_count`
- `constraint_violation_count`
- `trajectory_length`

## Outputs

- `results/phase4/algorithms/<algorithm>/seed_<seed>/`
- `results/phase4/algorithm_episode_results.csv`
- `results/phase4/algorithm_summary.csv`
- `results/phase4/ablations/ablation_results.csv`
- `results/phase4/ablations/ablation_summary.csv`
- `results/phase4/figures/`

## Notes

- The communication formulas are unchanged.
- The results are engineering validation, not final paper conclusions.
- No PPO or multi-agent methods are included in this phase.
