# Phase 5 Figure and Table Index

## Figures

| No. | File | Title | Data source | X axis | Y axis | Error bars | Suggested section | Main point | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 1 | `figure_1_system_model.png/.pdf` | System model and relay flight region | `configs/comm_env_default.yaml` | x (m) | y/z (m) | none | System model | Shows H, R, L, links, and relay flight region | Schematic only; not a new physical model |
| Figure 2 | `figure_2_modeling_pipeline.png/.pdf` | Modeling and training pipeline | communication model and scripts | pipeline stage | not applicable | none | Method | Shows position-to-reward-to-action loop | No experimental metric plotted |
| Figure 3 | `figure_3_training_curves.png/.pdf` | Training reward curves | `results/phase4/algorithms/*/training_log.csv` | episode | episode reward | seed standard deviation | Training results | SAC is more stable in current runs | No smoothing is applied |
| Figure 4 | `figure_4_algorithm_rate_comparison.png/.pdf` | Algorithm and baseline rate comparison | `results/phase4/algorithm_summary.csv` | method | average end-to-end rate (Mbps) | DRL seed x scenario std; baseline scenario std | Main results | SAC is highest; BalancedLinkPolicy is the best baseline | Do not claim significance |
| Figure 5 | `figure_5_scenario_comparison.png/.pdf` | Scenario-wise algorithm performance | `results/phase4/algorithm_episode_results.csv` | scenario_id | average end-to-end rate (Mbps) | none | Scenario analysis | Ranking varies by scenario | Baselines use fixed policy runs |
| Figure 6 | `figure_6_ablation_rate.png/.pdf` | TD3 ablation rate comparison | `results/phase4/ablations/ablation_summary.csv` | ablation group | average end-to-end rate (Mbps) | seed x scenario std | Ablation study | Isotropic antenna gives optimistic rate | Interpret A1 as model simplification |
| Figure 7 | `figure_7_constraint_violations.png/.pdf` | Constraint violations | `results/phase4/algorithm_summary.csv` | method | mean constraint violation count | none | Limitations | DRL constraint violations are high | Counts are not probabilities |
| Figure 8 | `figure_8_representative_trajectories.png/.pdf` | Representative relay trajectories | `results/phase5/source_data/representative_trajectories.csv` | x (m) | y/z (m) | none | Trajectory analysis | Compares same `eval_0` scene | Same coordinate range for all methods |
| Figure 9 | `figure_9_rate_over_time.png/.pdf` | Rate over time | `results/phase5/source_data/representative_rate_trace.csv` | step | rate (Mbps) | none | Link behavior | Shows H-R, R-L, and bottleneck rate | Uses best DRL and best baseline on `eval_0` |

## Tables

| No. | File | Title | Data source | Suggested section | Main point |
| --- | --- | --- | --- | --- | --- |
| Table 1 | `table_1_system_parameters.csv/.md` | System and communication parameters | `configs/comm_env_default.yaml` | System model | Lists environment, channel, antenna, and reward parameters |
| Table 2 | `table_2_training_parameters.csv/.md` | DRL training parameters | `configs/phase4_experiments.yaml` | Training setup | Lists common and algorithm-specific training settings |
| Table 3 | `table_3_algorithm_comparison.csv/.md` | Algorithm comparison results | `results/phase4/algorithm_summary.csv` | Main results | SAC has the highest mean rate; BalancedLinkPolicy is close |
| Table 4 | `table_4_ablation_results.csv/.md` | Ablation results | `results/phase4/ablations/ablation_summary.csv` | Ablation study | Shows A1 optimistic rate and A4 fixed-height degradation |
| Table 5 | `table_5_scenario_results.csv/.md` | Scenario results | `results/phase4/algorithm_episode_results.csv` | Scenario analysis | Shows method performance across `eval_0` to `eval_4` |
