# Phase 5 Result Analysis

## 6.1 Overall Algorithm Comparison

The final comparison uses the fixed Phase 4 evaluation scenarios `eval_0` to `eval_4`.
SAC has the highest mean end-to-end rate among the tested methods: 4.8765 Mbps.
BalancedLinkPolicy is the best baseline at 4.8145 Mbps. TD3 reaches 4.6390 Mbps and
DDPG reaches 4.4652 Mbps.

The observed SAC to BalancedLinkPolicy gap is about 0.0620 Mbps. This is small relative
to the scenario variation and is reported as a descriptive difference only. No
statistical significance claim is made. TD3 is below BalancedLinkPolicy by about
0.1755 Mbps in the current fixed-scenario evaluation.

The standard deviations also matter. SAC has a lower rate standard deviation
(0.1832 Mbps) than TD3 (0.4927 Mbps) and DDPG (0.7461 Mbps), suggesting more stable
behavior under the current seeds and scenarios. GreedyRatePolicy and MidpointPolicy
are close to TD3 in mean rate, but their constraint behavior differs.

## 6.2 TD3, DDPG, and SAC Differences

DDPG uses a single critic and can be more sensitive to value overestimation and policy
update noise. In this experiment DDPG has the lowest mean rate among the three DRL
methods and the largest rate standard deviation.

TD3 uses twin critics and delayed policy updates to improve training stability. It
performs better than DDPG by about 0.1738 Mbps on average, but it does not exceed SAC
or the best baseline in the final summary.

SAC uses a stochastic policy and entropy regularization, which improves exploration in
this setting. The final results are consistent with that mechanism: SAC has the
highest mean rate and the smallest DRL standard deviation. This is still a descriptive
observation under the current seed and scenario count, not a general superiority claim.

## 6.3 Baseline Results

BalancedLinkPolicy performs well because it directly balances the H-R and R-L links,
which matches the bottleneck nature of the end-to-end rate. It is the strongest
baseline in the final table.

MidpointPolicy is competitive in mean rate, but it produces a high constraint violation
count because the geometric midpoint can drive the relay toward the boundary or height
limit. StaticRelayPolicy and RandomPolicy provide lower-complexity references and are
useful lower baselines.

GreedyRatePolicy is strong in some scenarios because it optimizes immediate rate, but
it is not guaranteed to be best over a full episode. The current results show that DRL
does not clearly dominate every baseline: SAC is slightly above BalancedLinkPolicy on
mean rate, while TD3 and DDPG are below it.

## 6.4 Ablation Results

### A1 Isotropic Antenna

A1_isotropic_antenna has the highest mean rate in the ablation table at 5.7467 Mbps,
which is 1.1077 Mbps above A0_full_td3. This should be interpreted as an optimistic
communication estimate: ignoring the elevation-dependent dipole gain attenuation can
overestimate system communication performance.

This result does not mean the elevation-angle antenna model is unnecessary. It means
the simplified isotropic antenna removes a physical loss term and therefore changes the
communication model.

### A2 No Balance Penalty

A2_no_balance_penalty reaches 4.4812 Mbps, which is 0.1578 Mbps below A0. Removing the
balance penalty weakens the reward signal that discourages large differences between
the two hop rates. Because the end-to-end rate is limited by the bottleneck hop, this
can reduce final bottleneck performance.

### A3 No Energy Penalty

A3_no_energy_penalty reaches 4.6290 Mbps, only 0.0100 Mbps below A0. Under the current
reward weights, removing the energy penalty has limited effect on the communication
rate. The trajectory length remains 100 steps, and the constraint statistics are close
to A0, so the current energy term mainly acts as a mild regularizer.

### A4 Fixed Relay Height

A4_fixed_relay_height reaches 4.0290 Mbps, which is 0.6100 Mbps below A0. This indicates
that three-dimensional relay placement is useful in the current model. Restricting the
vertical action removes part of the optimization space and reduces the achievable
bottleneck rate.

## 6.5 Constraint Violation Problem

The DRL policies still have high constraint violation counts. Mean counts per 100-step
episode are 86.13 for TD3, 82.33 for DDPG, and 79.40 for SAC. The environment clips
velocity and boundary states so the actual state remains feasible, but the learned
actions can still rely on clipping.

This is a limitation of the current approach. The result should not be hidden in the
paper. Future work should consider action parameterization, a safety layer, stronger
constraint penalties, or constrained reinforcement learning.

Some baselines also have constraint issues. MidpointPolicy reaches 94.00 violations on
average, while BalancedLinkPolicy is much lower at 3.40. StaticRelayPolicy and
RandomPolicy have 0.00 in the fixed Phase 4 scenario evaluation.

## 6.6 Research Limitations

The current evaluation uses a limited number of fixed scenarios and three training
seeds per DRL method. The H and L motion model is simplified, and the channel model
does not include complex blockage, shadowing, or probabilistic LoS effects.

The environment does not model detailed UAV flight dynamics. Constraint violations are
frequent for the learned policies, even though clipping keeps the simulated state
inside the feasible region. The conclusions therefore apply only to the current
communication model, reward definition, scenario set, and training configuration.
