# 本轮任务完成情况：Phase 4 对比实验与消融实验

## 任务状态

- 状态：已完成本地验收。
- 当前阶段：Phase 4：对比实验与消融实验。
- 本轮基于 Phase 3 已通过的 TD3 多 seed 结果继续推进。
- 项目内没有更细化的 Phase 4 AGENTS 任务清单，因此本轮采用保守的轻量 Phase 4：TD3 与 Phase 1 baseline 对比、TD3-only 短程消融、结果汇总和图像生成。
- 未新增 SAC、DDPG、PPO、多智能体算法、课程学习或新的通信模型。
- 未修改 Phase 0/0.5/1 已验证的通信公式。

## 本轮新增文件

- `configs/phase4_ablation.yaml`
- `scripts/run_phase4_experiments.py`
- `scripts/plot_phase4_results.py`
- `tests/test_phase4_experiments.py`
- `docs/phase4_comparison_ablation_spec.md`
- `results/phase4/`

## 本轮实现内容

- 新增 Phase 4 轻量消融配置。
- 新增 Phase 4 实验脚本，自动读取 Phase 1 baseline 和 Phase 3 TD3 多 seed 汇总。
- 生成 TD3 与 baseline 的 Phase 4 方法对比表。
- 运行 4 个 TD3-only 短程消融变体：
  - `td3_short_default`
  - `no_observation_normalizer`
  - `small_network`
  - `no_exploration_decay`
- 为每个消融变体保存训练日志、评估日志、final model、best model、配置和参数。
- 生成消融结果相对 Phase 3 TD3 multi-seed mean 的差值表。
- 新增 Phase 4 绘图脚本，生成方法对比图、消融速率差值图、消融约束/中断计数图。
- 新增 Phase 4 测试，覆盖配置读取、禁用算法名称、消融汇总逻辑、差值计算和方法对比表。

## 验收命令结果

- `pytest`：当前 PATH 中无法直接找到 `pytest` 命令。
- `python -m pytest --basetemp .pytest_tmp`：`77 passed`。
- `python scripts/run_phase4_experiments.py`：通过。
- `python scripts/plot_phase4_results.py`：通过。
- `rg -n "nan|inf" results\phase4 -g "*.csv" -g "*.json" -g "*.yaml"`：无匹配，Phase 4 结果文件未发现 NaN/inf 字符串。

## 生成结果文件

- `results/phase4/phase4_method_comparison.csv`
- `results/phase4/phase4_ablation_summary.csv`
- `results/phase4/phase4_ablation_vs_reference.csv`
- `results/phase4/ablations/td3_short_default/`
- `results/phase4/ablations/no_observation_normalizer/`
- `results/phase4/ablations/small_network/`
- `results/phase4/ablations/no_exploration_decay/`
- `results/phase4/figures/phase4_td3_vs_baselines.png`
- `results/phase4/figures/phase4_ablation_rate_delta.png`
- `results/phase4/figures/phase4_ablation_constraints_outages.png`

## Phase 4 方法对比结果

- Phase 3 TD3 multi-seed mean average rate_e2e：`4.543915 Mbps`
- Phase 3 TD3 multi-seed std average rate_e2e：`0.524517 Mbps`
- 最优 Phase 1 baseline：`balanced_link`
- 最优 baseline average rate_e2e：`3.416483 Mbps`
- TD3 mean - best baseline：`1.127432 Mbps`

## Phase 4 消融结果

- `td3_short_default` best eval average rate_e2e：`3.505167 Mbps`
- `no_observation_normalizer` best eval average rate_e2e：`4.872519 Mbps`
- `small_network` best eval average rate_e2e：`4.885812 Mbps`
- `no_exploration_decay` best eval average rate_e2e：`3.523351 Mbps`

## 阶段观察

- 轻量短程消融显示 `small_network` 和 `no_observation_normalizer` 在本次单 seed 短训练中高于 Phase 3 TD3 multi-seed mean。
- `td3_short_default` 和 `no_exploration_decay` 明显低于 Phase 3 TD3 multi-seed mean。
- 这些结果只作为工程阶段诊断，不作为论文最终结论。

## 阶段判断

- Phase 4 轻量对比与消融流程已跑通。
- 当前仍未进入 SAC/DDPG/PPO 或最终论文级大规模实验。
