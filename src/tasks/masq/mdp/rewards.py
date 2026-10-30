import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.mdp import rewards as velocity_rewards
from mjlab.envs import mdp as mjlab_mdp

# Exact V5 runtime reward manifest.  Every entry is registered by
# ``make_masq_env_cfg``; historical/disabled reward experiments belong in the
# versioned YAML documents, not this executable scale table.
MASQ_REWARD_SCALES = {
  "track_linear_velocity": 1.5,
  "track_angular_velocity": 0.75,
  "body_orientation_l2": -1.0,
  "lin_vel_z": -2.0,
  "ang_vel_xy": -0.05,
  "base_height": -10.0,
  "alive_moving": 0.2,
  "dof_acc": -2.5e-7,
  "collision": -1.0,
  "action_rate": -1e-2,
  "feet_slip": -0.1,
  "action_smooth_2": -5e-3,
  "gait_contact_match": 1.0,
  "support_deficit": -0.5,
  "all_air_nonpronk": -3.0,
  "fall_termination": -50.0,
}

# Explicit aliases make every reused reward discoverable from this MASQ file.
track_linear_velocity = velocity_rewards.track_linear_velocity
track_angular_velocity = velocity_rewards.track_angular_velocity
body_orientation_l2 = velocity_rewards.body_orientation_l2
self_collision_cost = velocity_rewards.self_collision_cost
body_angular_velocity_penalty = velocity_rewards.body_angular_velocity_penalty
angular_momentum_penalty = velocity_rewards.angular_momentum_penalty
torques = mjlab_mdp.joint_torques_l2
dof_vel = mjlab_mdp.joint_vel_l2
dof_acc = mjlab_mdp.joint_acc_l2
action_rate = mjlab_mdp.action_rate_l2
action_smooth_1 = mjlab_mdp.action_rate_l2
action_smooth_2 = mjlab_mdp.action_acc_l2
feet_slip = velocity_rewards.feet_slip

def track_lin_vel_xy(
  env, command_name="gait", sigma=.25, threshold=.2, threshold_scale=2.0
):
  """Track planar velocity, with a negative tail for large errors.

  The exponential kernel alone is always positive and therefore permits a
  stationary policy to settle on a reward plateau.  The v4 threshold term
  makes sufficiently poor tracking explicitly costly.
  """
  command = env.command_manager.get_command(command_name)
  actual = env.scene["robot"].data.root_link_lin_vel_b
  error = torch.linalg.vector_norm(command[:, :2] - actual[:, :2], dim=-1)
  _log_mean(env, "Metrics/linear_velocity_error", error)
  return torch.exp(-error.square() / sigma) - threshold_scale * torch.clamp(
    error - threshold, min=0.0
  )

def track_ang_vel_z(
  env, command_name="gait", sigma=.25, threshold=.2, threshold_scale=2.0
):
  """Track yaw rate, with the same negative large-error tail as v4."""
  command = env.command_manager.get_command(command_name)
  actual = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
  error = torch.abs(command[:, 2] - actual)
  _log_mean(env, "Metrics/angular_velocity_error", error)
  return torch.exp(-error.square() / sigma) - threshold_scale * torch.clamp(
    error - threshold, min=0.0
  )

def lin_vel_z(env):
  return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square()

def ang_vel_xy(env):
  return env.scene["robot"].data.root_link_ang_vel_b[:, :2].square().sum(-1)

def base_height(env, target_height=.30, tolerance=.03):
  """Penalize base height only outside the target's allowed deadband."""
  height = env.scene["robot"].data.root_link_pos_w[:, 2]
  error_outside_band = torch.clamp(
    torch.abs(height - target_height) - tolerance, min=0.0
  )
  return error_outside_band.square()

def feet_air_time(env, sensor_name="feet_ground_contact", threshold=.5):
  """Reward completed swing duration at the instant each foot touches down."""
  sensor = env.scene[sensor_name]
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  return ((sensor.data.last_air_time - threshold) * first_contact).sum(-1)

def feet_stance_time(
  env, sensor_name="feet_ground_contact", stance_limit=1.25
):
  """Penalize only the portion of active foot stance exceeding the limit."""
  contact_time = env.scene[sensor_name].data.current_contact_time
  return torch.clamp(contact_time - stance_limit, min=0.0).sum(-1)

def _foot_contacts(env, sensor_name):
  """Return binary foot contacts in the configured FR, FL, RR, RL order."""
  found = env.scene[sensor_name].data.found
  assert found is not None
  return found > 0


def _active_command(env, command_name, planar_threshold=.1, yaw_threshold=.2):
  command = env.command_manager.get_command(command_name)
  return ((torch.linalg.vector_norm(command[:, :2], dim=-1) > planar_threshold)
          | (torch.abs(command[:, 2]) > yaw_threshold))


def _log_mean(env, name, value):
  """Log scalar diagnostics without changing the reward returned to the manager."""
  env.extras.setdefault("log", {})[name] = value.float().mean()


def alive_moving(
  env, command_name="gait", sensor_name="feet_ground_contact", eps_cmd=.1, v_min=.1
):
  """Speed shaping only when a planar command and ground support both exist."""
  command = env.command_manager.get_command(command_name)
  command_speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
  body_speed = torch.linalg.vector_norm(
    env.scene["robot"].data.root_link_lin_vel_b[:, :2], dim=-1
  )
  supported = _foot_contacts(env, sensor_name).any(dim=-1)
  return (
    (2.0 * torch.tanh(body_speed / v_min) - 1.0)
    * (command_speed > eps_cmd).float()
    * supported.float()
  )


def gait_contact_match(
  env, command_name="gait", sensor_name="feet_ground_contact", duty_factor=.58,
  planar_threshold=.1, yaw_threshold=.2,
):
  """Signed whole-body contact match to the commanded gait phase."""
  cmd = env.command_manager.get_term(command_name)
  contact = _foot_contacts(env, sensor_name)
  desired = ((cmd.phase + cmd.phase_offsets) % 1.0) < duty_factor
  raw = ((2 * contact.float() - 1) * (2 * desired.float() - 1)).mean(dim=-1)
  active = _active_command(env, command_name, planar_threshold, yaw_threshold)
  result = raw * active.float()
  _log_mean(env, "Metrics/gait_contact_match", result)
  return result


def support_deficit(
  env, command_name="gait", sensor_name="feet_ground_contact",
  planar_threshold=.1, yaw_threshold=.2,
):
  """Normalized support shortfall; pronk is explicitly exempt."""
  cmd = env.command_manager.get_term(command_name)
  minimum = torch.tensor(
    (2, 2, 1, 0), device=env.device, dtype=torch.long
  )[cmd.gait_id]
  contacts = _foot_contacts(env, sensor_name).sum(dim=-1)
  raw = (minimum - contacts).clamp(min=0) / minimum.clamp_min(1)
  active = _active_command(env, command_name, planar_threshold, yaw_threshold)
  result = raw * active.float()
  _log_mean(env, "Metrics/support_deficit", result)
  _log_mean(env, "Metrics/contact_count_mean", contacts)
  return result


def all_air_nonpronk(
  env, command_name="gait", sensor_name="feet_ground_contact",
  planar_threshold=.1, yaw_threshold=.2,
):
  """Flag unsupported flight for gaits where all-leg flight is not commanded."""
  cmd = env.command_manager.get_term(command_name)
  all_air = ~_foot_contacts(env, sensor_name).any(dim=-1)
  active = _active_command(env, command_name, planar_threshold, yaw_threshold)
  result = (all_air & active & (cmd.gait_id != 3)).float()
  _log_mean(env, "Metrics/all_air_fraction", result)
  return result


def fall_termination(
  env, steps_per_iteration=24, zero_until_iteration=200,
  ramp_until_iteration=800,
):
  """Scheduled one-step cost for a non-timeout termination.

  The RewardManager applies this raw value as ``weight * value * step_dt``.
  Keeping the raw value at zero during early PPO iterations avoids suppressing
  exploration before a stepping policy has emerged.
  """
  if ramp_until_iteration <= zero_until_iteration:
    raise ValueError("ramp_until_iteration must be greater than zero_until_iteration")
  terminated = mjlab_mdp.is_terminated(env)
  iteration = env.common_step_counter / steps_per_iteration
  ramp = min(
    max(
      (iteration - zero_until_iteration)
      / (ramp_until_iteration - zero_until_iteration),
      0.0,
    ),
    1.0,
  )
  _log_mean(env, "Metrics/fall_rate", terminated)
  _log_mean(env, "Metrics/fall_termination_ramp", torch.ones_like(terminated) * ramp)
  return terminated * ramp

def jump(env, command_name="gait"):
  cmd = env.command_manager.get_term(command_name)
  z = env.scene["robot"].data.root_link_pos_w[:, 2]
  return -(z - cmd.height_cmd.squeeze(-1)).square()

def collision(env, sensor_name="nonfoot_ground_touch", force_threshold=.1):
  force = env.scene[sensor_name].data.force
  return (torch.linalg.norm(force, dim=-1) > force_threshold).float().sum(-1)

def feet_impact_velocity(env, sensor_name="feet_ground_contact"):
  sensor = env.scene[sensor_name]
  force = sensor.data.force
  vel = sensor.data.contact_vel if hasattr(sensor.data, "contact_vel") else force * 0
  return (torch.linalg.norm(vel, dim=-1).square() * (force.norm(dim=-1) > .1)).sum(-1)

def raibert(env, command_name="gait", asset_cfg=SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL")), nominal_xy=((.19, -.13), (.19, .13), (-.19, -.13), (-.19, .13)), stance_time=.25):
  """Raibert-style horizontal foot placement error in the body frame."""
  robot = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  foot = robot.data.site_pos_w[:, asset_cfg.site_ids, :2]
  root = robot.data.root_link_pos_w[:, None, :2]
  actual = foot - root
  nominal = torch.as_tensor(nominal_xy, device=env.device, dtype=actual.dtype).unsqueeze(0)
  velocity_offset = .5 * stance_time * cmd[:, None, :2]
  desired = nominal + velocity_offset
  return (actual - desired).square().sum(-1).sum(-1)

def tracking_contacts_shaped_velocity(env, command_name="gait", sensor_name="feet_ground_contact", sigma=.25, asset_cfg=SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL"))):
  sensor = env.scene[sensor_name]
  contact = (sensor.data.found > 0).float()
  vel = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :]
  return (contact * (1 - torch.exp(-vel.square().sum(-1) / sigma))).sum(-1)
