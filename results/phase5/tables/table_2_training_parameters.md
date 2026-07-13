| algorithm | episodes | training_seeds | batch_size | replay_buffer_size | hidden_sizes | actor_lr | critic_lr | gamma | tau | exploration_noise | algorithm_unique_parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TD3 | 300 | [0,1,2] | 128 | 100000 | [256,256] | 0.0003 | 0.0003 | 0.99 | 0.005 | 0.1 | {"exploration_noise_decay":0.995,"min_exploration_noise":0.02,"noise_clip":0.5,"policy_delay":2,"policy_noise":0.2} |
| DDPG | 300 | [0,1,2] | 128 | 100000 | [256,256] | 0.0003 | 0.0003 | 0.99 | 0.005 | 0.1 | {"exploration_noise_decay":0.995,"min_exploration_noise":0.02} |
| SAC | 300 | [0,1,2] | 128 | 100000 | [256,256] | 0.0003 | 0.0003 | 0.99 | 0.005 |  | {"entropy_coef":0.2} |
