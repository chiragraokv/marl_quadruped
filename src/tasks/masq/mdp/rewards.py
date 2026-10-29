import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.mdp import rewards as velocity_rewards
from mjlab.envs import mdp as mjlab_mdp

# Complete MASQ Table-I registry. Several entries intentionally reuse
# implementations already validated by the stock velocity task/mjlab.
MASQ_REWARD_SCALES = {
  # reward_v3.yaml names these track_lin_vel_xy and track_ang_vel_z.
  "track_linear_velocity": 1.5,
  "track_angular_velocity": 0.75,
  "body_orientation_l2": -1.0,
  "self_collisions": 0.0,
  "body_ang_vel": 0.0,
  "angular_momentum": 0.0,
  "lin_vel_z": -2.0,
  "ang_vel_xy": -0.05,
  "base_height": -10.0,
  "feet_air_time": 1.0,
  "feet_stance_time": -0.05,
  "stand_still_penalty": -1.0,
  "torques": -1e-5,
  "dof_vel": 0.0,
  "dof_acc": -2.5e-7,
  # reward_v2.yaml calls this undesired_contact.
  "collision": -1.0,
  "action_rate": -1e-2,
  "survival": 0.5,
  "jump": 0.0,
  "feet_slip": -0.1,
  "action_smooth_1": 0.0,
  "action_smooth_2": 0.0,
  "feet_impact_vel": 0.0,
  "raibert": 0.0,
  "track_contact_shaped_vel": 0.0,
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

def track_lin_vel_xy(env, command_name="gait", std=.5):
  """Exact v3 XY-only command tracking kernel."""
  command = env.command_manager.get_command(command_name)
  actual = env.scene["robot"].data.root_link_lin_vel_b
  error = (command[:, :2] - actual[:, :2]).square().sum(-1)
  return torch.exp(-error / std**2)

def track_ang_vel_z(env, command_name="gait", std=.5):
  """Exact v3 yaw-only command tracking kernel."""
  command = env.command_manager.get_command(command_name)
  actual = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
  return torch.exp(-(command[:, 2] - actual).square() / std**2)

def lin_vel_z(env):
  return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square()

def ang_vel_xy(env):
  return env.scene["robot"].data.root_link_ang_vel_b[:, :2].square().sum(-1)

def base_height(env, target_height=.32):
  """Squared base-height error around Go2's nominal standing height."""
  height = env.scene["robot"].data.root_link_pos_w[:, 2]
  return (height - target_height).square()

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

def stand_still_penalty(
  env, command_name="gait", command_threshold=.1, sigma_still=.1
):
  """Penalize near-zero joint motion under a nonzero planar command."""
  command = env.command_manager.get_command(command_name)
  command_active = torch.linalg.vector_norm(command[:, :2], dim=-1) > command_threshold
  joint_speed_sq = env.scene["robot"].data.joint_vel.square().sum(-1)
  frozen = torch.exp(-joint_speed_sq / sigma_still)
  return frozen * command_active.float()

def survival_bonus(
  env, tau_s: float = 1.0, command_name: str = "gait",
  tracking_std: float = .5,
):
  """Saturating survival bonus gated by planar command-tracking quality."""
  episode_time = env.episode_length_buf.float() * env.step_dt
  alive = 1.0 - torch.exp(-episode_time / tau_s)
  tracking_quality = track_lin_vel_xy(env, command_name, tracking_std)
  return alive * tracking_quality

def warmup_ramp(
  env, start_iter: int = 0, ramp_iters: int = 200, steps_per_iter: int = 24
):
  """Linear 0 -> 1 ramp over PPO iterations, starting at start_iter."""
  start = start_iter * steps_per_iter
  ramp = ramp_iters * steps_per_iter
  if ramp <= 0:
    frac = 1.0 if env.common_step_counter >= start else 0.0
  else:
    frac = min(max((env.common_step_counter - start) / ramp, 0.0), 1.0)
  return torch.full_like(env.episode_length_buf, frac, dtype=torch.float)

def survival_reward(
  env, tau_s: float = 1.0, start_iter: int = 0, ramp_iters: int = 200,
  steps_per_iter: int = 24, command_name: str = "gait", tracking_std: float = .5,
):
  """Ramped survival bonus gated by planar command-tracking quality."""
  ramp = warmup_ramp(env, start_iter, ramp_iters, steps_per_iter)
  env.extras.setdefault("log", {})["Metrics/survival_ramp"] = ramp.mean()
  return survival_bonus(env, tau_s, command_name, tracking_std) * ramp

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
