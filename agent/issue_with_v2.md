Why standing still is currently the optimal policy

Walk through what a frozen, perfectly-balanced robot collects every step:

body_orientation, lin_vel_z, ang_vel_xy, base_height → all ≈ 0 penalty (easy to keep at 0 by never moving)
feet_air_time → 0, not negative. It only fires (t_air - 0.5) * first_contact, so a foot that never leaves the ground never triggers this term at all. Standing still is invisible to this reward, not punished by it.
feet_slip, undesired_contact → 0 (no motion, no slip, no bad contacts)
torques, dof_acc, action_rate → near-minimal (holding a static pose costs almost nothing)
survival → ramps to its full 0.5 × 1 = 0.5 after ~1s, completely independent of whether the robot is tracking any command
track_lin_vel_xy / track_ang_vel_z → this is the only term that would punish not-moving, but a tracking failure just gives you a low positive reward (exp(-large/0.25) ≈ 0), never a negative one.

So the only term that actually wants the robot to move gives it nothing to lose by failing, while every other term in the reward is either neutral or actively safest when the robot does nothing. Meanwhile, attempting locomotion is the single riskiest thing the policy can do early in training — a fall triggers termination, torching the base_height/body_orientation penalties on the way down and zeroing out all future survival + stability reward for that episode.

From the optimizer’s point of view: standing still is a low-variance, guaranteed ~0.5+/step income stream. Trying to walk is a high-variance gamble that frequently ends the episode early. PPO will converge to the low-variance option unless you change the incentives — this isn’t a bug in your code, it’s the reward function telling the policy to do exactly this.