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

The `info` dictionary contains at least:

```text
q_H, q_R, q_L,
d_HR, d_RL,
rho_HR, rho_RL,
theta_HR, theta_RL,
h_HR, h_RL,
snr_HR, snr_RL,
rate_HR, rate_RL, rate_e2e,
reward_terms,
constraint_violation
```
