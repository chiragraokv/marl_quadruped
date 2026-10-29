# Survival Reward Task Report

Date: 2026-10-29

## Outcome

The negative `episode_time` term was removed and replaced with a positive,
saturating survival reward:

```text
r_survival = (1 - exp(-episode_time / 1.0)) * warmup_ramp
weight     = 0.5
```

The ramp increases linearly from 0 to 1 over PPO iterations 0–200. Its
environment-step conversion comes from `agent_cfg.num_steps_per_env` during
task registration rather than a separately hard-coded environment value.
Tracking and gait reward weights were not changed as part of this experiment.

## Reward/config diff

The effective reward-module change is:

```diff
- "episode_time": -1.0
+ "survival": 0.5

- def episode_time_penalty(...):
-   return exp(-episode_time)
+ def survival_bonus(env, tau_s=1.0):
+   return 1.0 - exp(-episode_time / tau_s)
+
+ def warmup_ramp(env, start_iter=0, ramp_iters=200, steps_per_iter=24):
+   ... # clipped linear ramp
+
+ def survival_reward(...):
+   return survival_bonus(env, tau_s) * warmup_ramp(...)
```

The environment configuration changed from `episode_time_penalty` to:

```python
"survival": RewardTermCfg(
  func=mdp.survival_reward,
  weight=scales["survival"],
  params={
    "tau_s": 1.0,
    "start_iter": 0,
    "ramp_iters": 200,
    "steps_per_iter": steps_per_iter,
  },
)
```

`steps_per_iter` is passed from `agent_cfg.num_steps_per_env` in the Go2 task
registration. `agent/reward_v2.yaml` now documents the survival term and no
longer contains `episode_time`.

## Logging added

- `Episode_Reward/survival`
- `Episode_Reward/track_lin_vel` (alias of the existing full-name metric)
- `Metrics/survival_ramp`
- `Train/mean_episode_length`
- `Train/max_episode_length`
- `TerminationCount/time_out`
- `TerminationCount/fell_over`
- `TerminationCount/illegal_contact`
- `TerminationCount/fall_or_base_contact`

Every MASQ run now automatically post-processes its local event log after
training. The run's `plots/` directory contains grouped PNG graphs for episode
performance, survival/tracking, termination counts and rates, losses,
policy/performance, every reward term, and remaining metrics, plus
`all_scalars.csv` containing every logged scalar without downsampling.

## 500-iteration experiment

Command:

```bash
WANDB_MODE=disabled MASQ_GPU_IDS=cpu MASQ_NUM_ENVS=4 \
  ./train.sh --agent.max-iterations=500
```

Run directory:

```text
logs/rsl_rl/go2_masq/2026-10-29_18-50-01
```

The requested plot is [survival_first_500.png](survival_first_500.png). Survival
reward is multiplied by 1000 only for visual readability; table values below
are the unscaled logged values.

| Metric | Iteration 100 | Iteration 500 / end |
|---|---:|---:|
| Mean episode length | 10.23 | 10.27 |
| Max episode length | 19 | 26 |
| Survival reward | 0.000338 | 0.000577 |
| Linear tracking reward | 0.000350 | 0.000885 |
| Survival ramp | 0.497998 | 1.000000 |

The ramp was also checked at iteration 200: `0.998030`; it reaches 1.0 on the
following environment step and stays there.

## Termination breakdown

Counts are per 24-step PPO collection window across four environments.

| Snapshot | Fall | Illegal/base contact | Fall or base contact | Timeout |
|---|---:|---:|---:|---:|
| Iteration 100 | 0 | 29 | 29 | 0 |
| Iteration 500 | 4 | 22 | 26 | 0 |
| End (iteration 500) | 4 | 22 | 26 | 0 |

Most episodes end through illegal/non-foot contact, with no timeout episodes.
The current limitation is therefore gait/stability behavior rather than an
insufficient survival incentive.

## Reward magnitude conclusion

At iteration 500, survival (`0.000577`) is smaller than linear tracking
(`0.000885`) and considerably smaller than total velocity tracking when the
angular term is included. It does not dominate, so the `0.5` scale should not
be increased before addressing illegal-contact terminations.

## Checkpoint resume behavior

RSL-RL did not previously save `env.common_step_counter`; only model/optimizer
state and PPO iteration were present. MASQ checkpoints now store
`masq_common_step_counter` in checkpoint `infos` and restore it on load. For
legacy checkpoints, the counter is reconstructed as
`saved_iteration * num_steps_per_env`.

A real resume test saved counter `48`, loaded that checkpoint, ran one rollout,
and saved counter `72`. Thus the warm-up ramp is preserved across new MASQ
checkpoint resumes rather than restarting.

## Verification performed

- Six unit/configuration tests: passed.
- Python compilation for reward, environment, runner, and registration files:
  passed.
- Two-iteration telemetry smoke test: passed; all required scalar tags present.
- Checkpoint resume smoke test: passed (`48 -> 72`).
- Controlled 500-iteration CPU experiment: passed, 48,000 simulator samples.
- Plot generated directly from the run's TensorBoard event data.

## Known limitations

- The controlled experiment used four CPU environments and is diagnostic, not
  a production-scale gait-quality evaluation.
- Iteration 500 is also the end of this experiment, so those two requested
  termination snapshots are identical.
- W&B was disabled in the sandbox; the same metrics were recorded by its
  TensorBoard-compatible writer and verified from the event file.
