import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.mdp import rewards as velocity_rewards
from mjlab.envs import mdp as mjlab_mdp

# Complete MASQ Table-I registry. Several entries intentionally reuse
# implementations already validated by the stock velocity task/mjlab.
MASQ_REWARD_SCALES = {
  "track_linear_velocity": 1.0,
  "track_angular_velocity": 0.5,
  "lin_vel_z": -2e-2,
  "ang_vel_xy": -1e-3,
  "torques": -1e-5,
  "dof_vel": -1e-4,
  "dof_acc": -2.5e-7,
  "collision": -5.0,
  "action_rate": -1e-2,
  "jump": 10.0,
  "feet_slip": -4e-2,
  "action_smooth_1": -0.1,
  "action_smooth_2": -0.1,
  "feet_impact_vel": 0.0,
  "raibert": -10.0,
  "track_contact_shaped_vel": 4.0,
}

# Explicit aliases make every reused reward discoverable from this MASQ file.
track_linear_velocity = velocity_rewards.track_linear_velocity
track_angular_velocity = velocity_rewards.track_angular_velocity
torques = mjlab_mdp.joint_torques_l2
dof_vel = mjlab_mdp.joint_vel_l2
dof_acc = mjlab_mdp.joint_acc_l2
action_rate = mjlab_mdp.action_rate_l2
action_smooth_1 = mjlab_mdp.action_rate_l2
action_smooth_2 = mjlab_mdp.action_acc_l2
feet_slip = velocity_rewards.feet_slip

def lin_vel_z(env):
  return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square()

def ang_vel_xy(env):
  return env.scene["robot"].data.root_link_ang_vel_b[:, :2].square().sum(-1)

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
