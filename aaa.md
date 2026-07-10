# 项目任务完成记录

维护说明：后续每完成一个阶段或关键任务，都在本文档中更新完成状态、验收命令、结果文件和提交信息。

## 当前状态

- 当前阶段：Phase 1 已完成本地验收。
- 最新阶段功能提交：`01ba0a3 Add phase 1 baseline policies and plots`
- 当前分支：`main`
- 远端状态：已推送到 `origin/main`
- 禁止事项状态：未新增 DDPG、TD3、SAC、PPO、actor、critic、replay buffer、train 等 DRL/训练相关文件。

## Phase 0.5：环境测试补强与代码清理

状态：已完成并推送。

完成内容：

- 删除无关文件 `11.txt`。
- 将 `configs/comm_env_default.yaml` 改为标准 YAML 缩进格式。
- 新增/补强通信环境测试：mobility、reward、geometry、antenna、channel、rate、comm env step、metrics。
- 完善 `scripts/run_comm_env_check.py` 输出指标。
- 完善 `EpisodeMetrics.summary()`。
- 更新 `docs/communication_model_spec.md`。

验收结果：

- `pytest`：通过。
- `python scripts/run_comm_env_check.py`：通过，随机动作 episode 可完整运行。

相关提交：

- `fac3e30 Complete phase 0.5 environment validation`
- `0507ce3 Add episode metrics summary tests`

## Phase 1：Baseline 策略与可视化验证

状态：已完成并推送。

完成内容：

- 新增 `src/uav_relay_env/baselines.py`。
- 新增 baseline 策略：
  - `RandomPolicy`
  - `StaticRelayPolicy`
  - `MidpointPolicy`
  - `HorizontalMidpointPolicy`
  - `GreedyRatePolicy`
  - `BalancedLinkPolicy`
- 新增 `scripts/run_baselines.py`。
- 新增 `scripts/plot_baseline_results.py`。
- 新增 `tests/test_baselines.py`。
- 新增 `docs/phase1_baseline_spec.md`。
- 生成 `results/phase1/` 下的 CSV 结果和图像。

验收结果：

- `pytest`：`52 passed`，存在 1 个 pytest cache warning，不影响结果。
- `python scripts/run_baselines.py`：通过。
- `python scripts/plot_baseline_results.py`：通过。

生成结果：

- `results/phase1/baseline_results.csv`
- `results/phase1/baseline_step_results.csv`
- `results/phase1/trajectory_<policy>.csv`
- `results/phase1/figures/*.png`

平均端到端速率排序：

| 排名 | Policy | average_rate_e2e |
| --- | --- | --- |
| 1 | `balanced_link` | 3.416483 Mbps |
| 2 | `midpoint` | 2.981117 Mbps |
| 3 | `greedy_rate` | 2.415220 Mbps |
| 4 | `random` | 1.833743 Mbps |
| 5 | `static` | 1.746559 Mbps |
| 6 | `horizontal_midpoint` | 1.722315 Mbps |

注意事项：

- `midpoint` 策略存在较多约束裁剪，因为 H-L 三维中点高度超过 relay 飞行区域上界。
- Greedy 和 Balanced 策略使用纯通信计算评估候选动作，不调用真实 `env.step()` 破坏环境状态。

相关提交：

- `01ba0a3 Add phase 1 baseline policies and plots`

## 后续待办

- 进入 Phase 2 前，先确认 Phase 1 结果和图像是否符合论文 baseline 对比需求。
- Phase 2 仍需先按新的 AGENTS 文档执行；没有明确进入 DRL 阶段前，不新增任何 DRL 算法代码。
