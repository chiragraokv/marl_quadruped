def multi_gait_curriculum(env, env_ids, command_name="gait", threshold=.8):
  cmd = env.command_manager.get_term(command_name)
  # Track a per-environment EMA and reassign only when the current gait has
  # crossed the threshold. The reward scale is intentionally tunable because
  # the paper does not publish the threshold value.
  reward = getattr(env, "reward_buf", None)
  if reward is None:
    return None
  cmd.reward_ema[env_ids].mul_(.99).add_(.01 * reward[env_ids].detach())
  ids = env_ids[cmd.reward_ema[env_ids] > threshold]
  if len(ids):
    cmd.gait_id[ids] = cmd.gait_id[ids].random_(0, 4)
    cmd._update_offsets(ids)
