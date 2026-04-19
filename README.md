Experimental Protocol
To ensure fair and consistent comparison across all methods, the following rules need to be followed.

The baseline is Vanilla PPO, already implemented in:
'agents/train_ppo_multi_env_logging.py'

All methods (penalty, masking, etc.) will be compared against this baseline.

Required Code Usage

All experiments are expected to use the following shared files:

- Environment wrapper:
  - `env/lava_logging_wrapper.py`
- Evaluation:
  - `metrics/evaluation.py`
- Training logging:
  - `metrics/training_logger.py`
- Plotting / visualization:
  - `metrics/plot_results.py`

Separate evaluation or logging pipelines are not recommended, to keep results consistent.


Environments

All methods should be tested on:
- MiniGrid-LavaGapS5-v0  
- MiniGrid-LavaGapS6-v0  
- MiniGrid-LavaGapS7-v0  


Random Seeds
All experiments should use exactly the same seeds:[0, 1, 2, 3, 4]

Training Setup
- Same PPO hyperparameters as the baseline
- Same total training timesteps
- Same observation setup (FlatObsWrapper)

-----
- Evaluation Setup

All evaluation should:
- use `metrics/evaluation.py`
- run for multiple episodes (not single rollout)
- use deterministic policy during evaluation

---
Required Metrics

Every method should report:
- mean reward ± std
- success rate
- mean number of safety violations
- convergence episode / timestep

------

