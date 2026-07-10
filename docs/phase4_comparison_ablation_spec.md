# Phase 4 Comparison and Ablation Spec

Phase 4 performs lightweight comparison and ablation validation after the Phase 3
TD3 workflow has passed local checks. It does not introduce SAC, DDPG, PPO,
multi-agent training, curriculum learning, or changes to the communication model.

## Scope

- Compare Phase 3 TD3 multi-seed results against the Phase 1 baseline policies.
- Run small TD3-only ablation variants to test implementation choices.
- Save reproducible CSV summaries and simple diagnostic figures.
- Treat the outputs as engineering validation, not final paper conclusions.

## Inputs

- `results/phase1/baseline_results.csv`
- `results/phase3/td3_multiseed_mean_std.csv`
- `configs/td3_default.yaml`
- `configs/phase4_ablation.yaml`

## Ablation Variants

The default Phase 4 config runs short TD3 training jobs with one seed:

- `td3_short_default`: short control run using Phase 3 defaults.
- `no_observation_normalizer`: disables observation normalization.
- `small_network`: changes hidden sizes to `[128, 128]`.
- `no_exploration_decay`: keeps exploration noise fixed by setting decay to `1.0`.

All variants reuse the existing TD3 agent, replay buffer, actor, critic, reward,
environment, and communication formulas.

## Outputs

`python scripts/run_phase4_experiments.py` writes:

- `results/phase4/phase4_method_comparison.csv`
- `results/phase4/phase4_ablation_summary.csv`
- `results/phase4/phase4_ablation_vs_reference.csv`
- `results/phase4/ablations/<variant>/`

`python scripts/plot_phase4_results.py` writes:

- `results/phase4/figures/phase4_td3_vs_baselines.png`
- `results/phase4/figures/phase4_ablation_rate_delta.png`
- `results/phase4/figures/phase4_ablation_constraints_outages.png`

## Metrics

The comparison table reports average end-to-end rate, rate standard deviation,
outage count, and constraint violation count for Phase 1 baselines and Phase 3
TD3. The ablation tables report best and final evaluation rate, reward, outage
count, constraint violation count, and deltas against the Phase 3 TD3 multi-seed
mean.

## Validation

Required commands:

```bash
pytest
python scripts/run_phase4_experiments.py
python scripts/plot_phase4_results.py
```

Optional sanity check:

```bash
python scripts/evaluate_td3_multiseed.py
```

Phase 4 passes when tests pass, comparison and ablation CSVs are generated,
figures are generated, no forbidden algorithm files are added, and result CSVs
contain no NaN or inf values.
