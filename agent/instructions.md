## Goal

Replace the episode_time penalty with a survival reward that encourages longer episodes without rewarding standing still. Keep the warm-up gating, but ramp it in smoothly. Verify the change with logged metrics before tuning further.

### Step 1: Remove the penalty
In the reward config (reward.yaml or the equivalent file), delete the episode_time entry.
Delete or stop referencing the episode_time_penalty function in the reward module.
Search the codebase for other references to episode_time_penalty and remove them.
### Step 2: Add the survival bonus
Add this function to the reward module, next to the other reward functions:
python
   def survival_bonus(env, tau_s: float = 1.0):
       """Saturating bonus for staying up. Rises over ~tau_s seconds, then flattens."""
       t = env.episode_length_buf.float() * env.step_dt
       return 1.0 - torch.exp(-t / tau_s)
Add the corresponding entry to the reward config:
yaml
   - id: survival
     name: Survival Bonus
     type: task
     scale: 0.5
     equation: "1 - exp(-t / tau_s)"   # tau_s = 1.0 s
### Step 3: Add the warm-up ramp
Add this function to the reward module:
python
   def warmup_ramp(env, start_iter: int = 0, ramp_iters: int = 200,
                   steps_per_iter: int = 24):
       """Linear 0 -> 1 ramp over ramp_iters PPO iterations, starting at start_iter."""
       step = env.common_step_counter
       start = start_iter * steps_per_iter
       ramp = ramp_iters * steps_per_iter
       frac = min(max((step - start) / ramp, 0.0), 1.0)
       return torch.full_like(env.episode_length_buf, frac, dtype=torch.float)
Multiply the survival term by the ramp. In the reward computation, do:
python
   r_survival = survival_bonus(env) * warmup_ramp(env, ramp_iters=200, steps_per_iter=steps_per_iter)
Read steps_per_iter from the runner config instead of hard-coding it:
python
   steps_per_iter = agent_cfg.num_steps_per_env  # use the actual attribute name in your config

Pass this value into the reward functions, or read it at setup.

If the warm-up is only needed to stabilize early learning, consider removing it entirely and relying on the survival bonus alone. The ramp is optional.
Step 4: Check checkpoint resume behavior
Confirm whether env.common_step_counter is saved in checkpoints.
If it isn’t, either save it alongside the model weights, or accept that resuming restarts the ramp. Record which behavior you chose in the run notes.
Step 5: Verify
Run a short training job and log these per-term values to the logger:
Episode_Reward/survival
Episode_Length (mean and max)
Episode_Reward/track_lin_vel
Check the termination reasons. Log the count of episodes ended by fall or base contact versus timeout. If most episodes end by falling, the stability and gait terms need work, not the survival term.
Confirm that Episode_Reward/survival is small compared with tracking reward once the policy is walking. If it dominates, reduce the scale.
Confirm that the ramp reaches 1.0 at start_iter + ramp_iters by logging the ramp value once per iteration.
Step 6: Report back

### Report these items:

The diff for the reward file and config.
The first 500 iterations of Episode_Length and survival reward, plotted together.
The termination reason breakdown at iterations 100, 500, and the end of training.
Whether checkpoint resume preserves the warm-up step counter.

Do not change tracking or gait weights in the same run. Make one change at a time so the effect of the survival term can be isolated.