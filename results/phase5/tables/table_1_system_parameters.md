| parameter | description | value | unit | source |
| --- | --- | --- | --- | --- |
| task_duration | 任务周期 | 100 | s | configs/comm_env_default.yaml |
| slot_duration | 时隙长度 | 1 | s | configs/comm_env_default.yaml |
| episode_steps | 每个 episode 步数 | 100 | step | configs/comm_env_default.yaml |
| flight_area | 飞行区域 | x=[0.0, 1000.0], y=[0.0, 1000.0], z=[50.0, 500.0] | m | configs/comm_env_default.yaml |
| max_speed | 最大速度 | 30 | m/s | configs/comm_env_default.yaml |
| q_H_height | H 初始高度 | 1000 | m | configs/comm_env_default.yaml |
| q_R_height | R 初始高度 | 200 | m | configs/comm_env_default.yaml |
| q_L_height | L 初始高度 | 100 | m | configs/comm_env_default.yaml |
| power_HR | H-R 发射功率 | 1 | W | configs/comm_env_default.yaml |
| power_RL | R-L 发射功率 | 1 | W | configs/comm_env_default.yaml |
| bandwidth | 带宽 | 1000000 | Hz | configs/comm_env_default.yaml |
| noise_power | 噪声功率 | 1e-13 | W | configs/comm_env_default.yaml |
| beta0 | 参考信道增益 | 0.001 | - | configs/comm_env_default.yaml |
| path_loss_exponent | 路径损耗指数 | 2.2 | - | configs/comm_env_default.yaml |
| antenna_model | 天线模型 | {"g_max":1.0,"g_min":0.05,"model":"dipole"} | - | configs/comm_env_default.yaml |
| r_min | 最低速率阈值 | 1000000 | bps | configs/comm_env_default.yaml |
| reward_weights | 奖励函数权重 | {"epsilon":1e-09,"kappa":0.001,"omega_B":0.5,"omega_C":1.0,"omega_E":0.01,"omega_O":1.0,"omega_R":1e-06} | - | configs/comm_env_default.yaml |
