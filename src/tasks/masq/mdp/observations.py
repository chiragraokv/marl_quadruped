import torch

def previous_previous_action(env):
  return env.action_manager.prev_prev_action

def temporal_director(env, leg_index: int, command_name="gait"):
  cmd = env.command_manager.get_term(command_name)
  return torch.sin(2 * torch.pi * (cmd.phase +
                                   cmd.phase_offsets[:, leg_index:leg_index + 1]))

def temporal_director_all(env, command_name="gait"):
  cmd = env.command_manager.get_term(command_name)
  return torch.sin(2 * torch.pi * (cmd.phase + cmd.phase_offsets))

def leg_observation(env, leg_index: int, command_name="gait"):
  # The second action history slot resolves the paper's ambiguous a_t input.
  q = env.scene["robot"].data.joint_pos - env.scene["robot"].data.default_joint_pos
  qd = env.scene["robot"].data.joint_vel
  action = env.action_manager.prev_action
  action_prev2 = env.action_manager.prev_prev_action
  parts = (q[:, leg_index * 3:(leg_index + 1) * 3],
           qd[:, leg_index * 3:(leg_index + 1) * 3],
           action_prev2[:, leg_index * 3:(leg_index + 1) * 3],
           action[:, leg_index * 3:(leg_index + 1) * 3],
           temporal_director(env, leg_index, command_name),
           env.scene["robot"].data.projected_gravity_b,
           env.command_manager.get_command(command_name),
           env.scene["robot"].data.root_link_lin_vel_b,
           torch.full((env.num_envs, 1), leg_index / 3.0, device=env.device))
  return torch.cat(parts, dim=-1)
