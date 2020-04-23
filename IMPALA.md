# IMPALA: Scalable Distributed Deep-RL with ImportanceWeighted Actor-Learner Architectures

## Problem
Solving a large collection of tasks using a single reinforcement learning agent with a single set of parameters with a fast, scalable and powerful architecture. 

## Solution

IMPALA actors communicate trajectories of experience (sequences of states, actions, and rewards) to a centralised learner. Since the learner in IMPALA has access to full trajectories of experience we use a GPU to perform updates on mini-batches of trajectories while aggressively parallelising all time independent operations. This type of decoupled architecture can achieve very high throughput.
However, because the policy used to generate a trajectory can lag behind the policy on the learner by several updates at
the time of gradient calculation, learning becomes off-policy. Therefore, we introduce the V-trace off-policy actor-critic
algorithm to correct for this harmful discrepancy.

## Architecture
The architecture consists of a set of actors, repeatedly generating trajectories of experience, and one or more learners that use the experiences sent from actors to learn off-policy. At the beginning of each trajectory, an actor updates its own local policy to the latest learner policy and runs it for n steps in its environment. 
![Impala](https://drive.google.com/uc?id=1E7WfDnLOPtXEHFcnVwmiUUj59hMJmu7t)

## V-trace
The learner policy is potentially several updates ahead of the actor’s policy at the time of update, therefore there is a
policy-lag between the actors and learner(s). IMPALA thus uses off-policy observations, but a2c requires on-policy.
V-trace corrects this lag.

![V-trace](https://drive.google.com/uc?id=1xvRIxdMjU605ASkn_EQr9CLVYi0MMwob)

Truncation levels $c$ and $\rho$ depends on how much the policy changed since observetion generation. If the value is not on-policy, it is less important for value function approximation.

Truncation levels $c$ and $\rho$ represent different features of the algorithm: $\rho$ impacts the nature of
the value function we converge to, whereas $c$ impacts the speed at which we converge to this function.

V-trace is a general off-policy learning algorithm that is more stable and robust compared to other off-policy correction
methods for actor critic agents.

## Actor-Critic algorithm
![a2c_impala](https://drive.google.com/uc?id=1a6k4JnjE7L6mkdlPByhETxJ7rhIzai2N)

The overall update is obtained by summing these gradients and entropy regularuzation rescaled by appropriate coefficients, which are hyperparameters of the algorithm.

## Results

IMPALA outperforms the synchronous batched A2C on 2 out of 5 tasks while achieving much higher throughput.
IMPALA is also more robust to the choice of hyperparameters than A3C.

![results_impala_1](https://drive.google.com/open?id=1IqsBXOOPlbWyrJIWkl3G4_naNuiKgPZv)

Impala is much faster that previous alghoritms and achieves good quality results especially in multi-task setting.
Experiments DMLab-30 show that, in the multi-task setting, positive transfer between individual tasks lead IMPALA to achieve better performance compared to the expert training setting. (light green is one-task IMPALA, the other are multitask).

![results_impala_2](https://drive.google.com/uc?id=1M48yQR-PJt_6Db4xEqgcns4MTuE_-j9D)


