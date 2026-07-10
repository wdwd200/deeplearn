# Communication Model Specification

This document records the Phase 0 UAV relay communication environment. It only
models the H -> R -> L two-hop communication process. It does not include DDPG,
TD3, SAC, PPO, actor networks, critic networks, replay buffers, training loops,
or model persistence.

## Entities

- H: high-altitude UAV, fixed position `q_H = [x_H, y_H, z_H]` in meters.
- R: relay UAV, controlled by action velocity `v_R = [v_x, v_y, v_z]` in m/s.
- L: low-altitude task UAV, fixed position `q_L = [x_L, y_L, z_L]` in meters.

The communication chain is `H -> R -> L`.

## Configuration

The default configuration file is `configs/comm_env_default.yaml`. It uses these
top-level sections:

- `simulation`: time-step length `delta_t_s` and episode horizon `max_steps`.
- `area`: flight boundary `bounds_m.x`, `bounds_m.y`, `bounds_m.z` in meters.
- `uavs`: fixed `q_H_m`, initial relay `q_R_initial_m`, and fixed `q_L_m`.
- `communication`: path loss and rate parameters `beta0`,
  `path_loss_exponent`, `min_distance_m`, `bandwidth_hz`, `noise_power_w`,
  `power_HR_w`, `power_RL_w`, `half_duplex`, and `r_min_bps`.
- `antenna`: antenna `model`, `g_max`, and `g_min`.
- `reward`: reward weights `omega_R`, `omega_E`, `omega_O`, `omega_B`,
  `omega_C`, flight energy coefficient `kappa`, and denominator stabilizer
  `epsilon`.
- `environment`: relay speed limit `max_speed_mps`, `action_dim`, and
  `observation_dim`.

The loader also accepts the earlier internal section names
`scenario`, `mobility`, `channel`, and `rate` for backward compatibility.

## Geometry

For link `a-b`:

```text
d_ab = ||q_a - q_b||
rho_ab = sqrt((x_a - x_b)^2 + (y_a - y_b)^2)
theta_ab = arctan(|z_a - z_b| / rho_ab)
```

If `rho_ab = 0` and the altitude difference is non-zero, `theta_ab = pi/2`.
If the two points coincide, `theta_ab = 0`.

## Antenna Gain

The dipole antenna directional gain is:

```text
G(theta) = max(G_min, G_max * cos^2(theta))
```

An isotropic mode is also available for checks and returns `G(theta) = 1`.

## Channel Gain

For each hop `i`:

```text
h_i = beta0 * d_i^(-alpha) * G_tx(theta_i) * G_rx(theta_i)
```

Implementation detail: `d_i` is lower-bounded by `min_distance_m` to avoid a
singularity if two UAV positions coincide.

## SNR and Rate

```text
SNR_i = P_i * h_i / sigma2
R_i = B * log2(1 + SNR_i)
R_e2e = min(R_HR, R_RL)
```

If `half_duplex = true`:

```text
R_e2e = 0.5 * min(R_HR, R_RL)
```

All rates are in bit/s.

## Mobility

The action is interpreted as the relay UAV velocity in m/s:

```text
q_R_next = q_R + v_R * delta_t
```

The velocity is scaled down when:

```text
||v_R|| > V_max
```

The next position is then clamped to the configured 3D flight boundary.

## Reward

```text
reward =
    omega_R * R_e2e
    - omega_E * E_fly
    - omega_O * outage_penalty
    - omega_B * balance_penalty
    - omega_C * constraint_penalty
```

With:

```text
outage_penalty = 1 if R_e2e < R_min else 0
balance_penalty = abs(R_HR - R_RL) / (R_HR + R_RL + epsilon)
E_fly = kappa * ||v_R||^2 * delta_t
```

`constraint_penalty` is `1` when velocity clipping or boundary clipping occurs.

## Environment Interface

`reset()` returns:

```text
observation, info
```

`step(action)` returns:

```text
observation, reward, terminated, truncated, info
```

The observation is a 15-element list:

```text
[q_H(3), q_R(3), q_L(3), rate_HR, rate_RL, rate_e2e, snr_HR, snr_RL, step_fraction]
```

Element order:

```text
0:2   q_H = [x_H, y_H, z_H]
3:5   q_R = [x_R, y_R, z_R]
6:8   q_L = [x_L, y_L, z_L]
9     rate_HR
10    rate_RL
11    rate_e2e
12    snr_HR
13    snr_RL
14    step_count / max_steps
```

The action is a 3-element velocity command for the relay UAV:

```text
action = [v_x, v_y, v_z]  # m/s
```

The environment clips this velocity to `max_speed_mps` and then clamps the
resulting relay position to the configured 3D flight boundary.

The `info` dictionary contains:

```text
step: current step count after step()
q_H, q_R, q_L: UAV positions in meters
d_HR, d_RL: 3D link distances in meters
rho_HR, rho_RL: horizontal link distances in meters
theta_HR, theta_RL: elevation angles in radians
gain_tx_HR, gain_rx_HR, gain_tx_RL, gain_rx_RL: antenna gains
h_HR, h_RL: channel gains
snr_HR, snr_RL: per-hop SNR values
rate_HR, rate_RL: per-hop rates in bit/s
rate_e2e: end-to-end bottleneck rate in bit/s
reward_terms: reward component dictionary
constraint_violation: true if velocity or boundary clipping occurred
constraint_info: detailed clipping diagnostics
```

`reward_terms` contains:

```text
rate_reward: omega_R * R_e2e
energy_penalty: omega_E * E_fly
outage_penalty: omega_O * outage_indicator
balance_penalty: omega_B * balance_penalty_raw
constraint_penalty: omega_C * constraint_indicator
raw_rate_e2e_bps: unclipped R_e2e value in bit/s
raw_E_fly: kappa * ||v_R||^2 * delta_t
raw_outage_indicator: 1 if R_e2e < R_min else 0
raw_balance: abs(R_HR - R_RL) / (R_HR + R_RL + epsilon)
```

`terminated` is always `False` in Phase 0 because there is no task-success or
failure terminal condition. `truncated` becomes `True` when
`step_count >= max_steps`.

## Random Check Script

`python scripts/run_comm_env_check.py` runs one full random-action episode and
prints episode length, total reward, average rates, average SNR values, outage
count, constraint violation count, trajectory length, and final relay position.

## Phase Boundary

This phase contains no DRL algorithm code. It does not include DDPG, TD3, SAC,
PPO, actor networks, critic networks, replay buffers, training loops, algorithm
comparison experiments, or model save/load logic.
