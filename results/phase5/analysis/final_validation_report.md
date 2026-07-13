# Phase 5 Final Validation Report

- passed: `true`
- git_commit: `221c50db0924c5cd02ad69e0e440bbb39da79d6c`
- source_code_hash: `013686511bc9830caa578e6aa7d4ea21962f1c845348e8e7a6db66acce6e9f97`
- phase4_training_params_count: `24`

| check | status | details |
| --- | --- | --- |
| results/phase4/algorithm_episode_results.csv required fields | pass |  |
| results/phase4/algorithm_summary.csv required fields | pass |  |
| results/phase4/ablations/ablation_results.csv required fields | pass |  |
| results/phase4/ablations/ablation_summary.csv required fields | pass |  |
| TD3 has 3 training seeds | pass | seeds=[0, 1, 2] |
| TD3 evaluated on eval_0..eval_4 | pass |  |
| DDPG has 3 training seeds | pass | seeds=[0, 1, 2] |
| DDPG evaluated on eval_0..eval_4 | pass |  |
| SAC has 3 training seeds | pass | seeds=[0, 1, 2] |
| SAC evaluated on eval_0..eval_4 | pass |  |
| RandomPolicy uses fixed eval scenarios | pass |  |
| StaticRelayPolicy uses fixed eval scenarios | pass |  |
| MidpointPolicy uses fixed eval scenarios | pass |  |
| HorizontalMidpointPolicy uses fixed eval scenarios | pass |  |
| GreedyRatePolicy uses fixed eval scenarios | pass |  |
| BalancedLinkPolicy uses fixed eval scenarios | pass |  |
| A0_full_td3 has 3 training seeds | pass | seeds=[0, 1, 2] |
| A0_full_td3 evaluated on eval_0..eval_4 | pass |  |
| A1_isotropic_antenna has 3 training seeds | pass | seeds=[0, 1, 2] |
| A1_isotropic_antenna evaluated on eval_0..eval_4 | pass |  |
| A2_no_balance_penalty has 3 training seeds | pass | seeds=[0, 1, 2] |
| A2_no_balance_penalty evaluated on eval_0..eval_4 | pass |  |
| A3_no_energy_penalty has 3 training seeds | pass | seeds=[0, 1, 2] |
| A3_no_energy_penalty evaluated on eval_0..eval_4 | pass |  |
| A4_fixed_relay_height has 3 training seeds | pass | seeds=[0, 1, 2] |
| A4_fixed_relay_height evaluated on eval_0..eval_4 | pass |  |
| official metadata flags are valid | pass |  |
| all formal results use one git_commit | pass | commits=['221c50db0924c5cd02ad69e0e440bbb39da79d6c'] |
| all formal results use one source_code_hash | pass | source_hashes=['013686511bc9830caa578e6aa7d4ea21962f1c845348e8e7a6db66acce6e9f97'] |
| recorded result paths are relative | pass |  |
| no NaN or inf in Phase 4 CSV/JSON | pass |  |
| no duplicate method + seed + scenario_id | pass |  |
| algorithm summary matches episode data | pass |  |
| ablation summary matches episode data | pass |  |
