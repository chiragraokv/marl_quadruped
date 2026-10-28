"""MASQ Go2 environment configuration, derived from the stock velocity task."""
import math
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg
from src.tasks.masq import mdp
from src.tasks.masq.mdp.commands import GaitCommandCfg
from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

def make_masq_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # Start from the validated Go2 flat task so robot entities, foot sensors,
  # site names, and the raycast-frame overrides are all initialized correctly.
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.commands = {
    "gait": GaitCommandCfg(
      entity_name="robot", resampling_time_range=(3., 8.),
      ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1., 2.), lin_vel_y=(-1., 1.),
                                   ang_vel_z=(-1., 1.), heading=None),
    )
  }
  # Keep the manager-level four-way split explicit: each term is one 35D leg.
  cfg.observations["actor"].terms = {
    f"leg{i}_obs": ObservationTermCfg(func=mdp.leg_observation,
                                       params={"leg_index": i, "command_name": "gait"})
    for i in range(4)
  }
  cfg.observations["critic"].terms = {
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
    # Second action-history slot: paper ambiguity is resolved as a_{t-2}.
    "actions_prev2": ObservationTermCfg(func=mdp.previous_previous_action),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "gait"}),
    "base_lin_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}),
    "directors": ObservationTermCfg(func=mdp.temporal_director_all, params={"command_name": "gait"}),
  }
  cfg.rewards = {
    "track_linear_velocity": RewardTermCfg(func=mdp.track_linear_velocity, weight=1., params={"command_name":"gait", "std":.5}),
    "track_angular_velocity": RewardTermCfg(func=mdp.track_angular_velocity, weight=.5, params={"command_name":"gait", "std":.5}),
    "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z, weight=-2e-2),
    "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy, weight=-1e-3),
    "torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-1e-5),
    "dof_vel": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-1e-4),
    "dof_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-1e-2),
    "action_smooth_1": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-.1),
    "action_smooth_2": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-.1),
    "feet_impact_vel": RewardTermCfg(func=mdp.feet_impact_velocity, weight=0.0),
    "jump": RewardTermCfg(func=mdp.jump, weight=10., params={"command_name":"gait"}),
    "feet_slip": RewardTermCfg(func=mdp.feet_slip, weight=-.04, params={"sensor_name":"feet_ground_contact", "command_name":"gait", "asset_cfg":SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL"))}),
    "collision": RewardTermCfg(func=mdp.collision, weight=-5.),
    "raibert": RewardTermCfg(func=mdp.raibert, weight=-10., params={"command_name":"gait", "asset_cfg":SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL"))}),
    "track_contact_shaped_vel": RewardTermCfg(func=mdp.tracking_contacts_shaped_velocity, weight=4., params={"asset_cfg":SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL"))}),
  }
  cfg.curriculum = {
    "multi_gait": CurriculumTermCfg(
      func=mdp.multi_gait_curriculum,
      params={"command_name": "gait", "threshold": 0.8},
    )
  }
  cfg.events.update({
    "body_mass_masq": EventTermCfg(
      func=dr.body_mass, mode="startup",
      params={"ranges": (0.8, 1.2), "operation": "scale",
              "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))}),
    "motor_strength_masq": EventTermCfg(
      func=dr.pd_gains, mode="startup",
      params={"kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1),
              "operation": "scale", "asset_cfg": SceneEntityCfg("robot")}),
    "joint_calibration_masq": EventTermCfg(
      func=dr.joint_default_pos, mode="startup",
      params={"ranges": (-0.015, 0.015), "operation": "add",
              "asset_cfg": SceneEntityCfg("robot")}),
    "effort_limits_masq": EventTermCfg(
      func=dr.effort_limits, mode="startup",
      params={"effort_limit_range": (0.9, 1.1), "operation": "scale",
              "asset_cfg": SceneEntityCfg("robot")}),
    "gravity_impulse_masq": EventTermCfg(
      # mjlab's stateful impulse event supports the paper's 0.99s sustain
      # window and an approximately eight-second cooldown.
      func=envs_mdp.apply_body_impulse, mode="step",
      params={"force_range": (-30.0, 30.0), "torque_range": (-5.0, 5.0),
              "duration_s": (0.99, 0.99), "cooldown_s": (8.0, 8.0),
              "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))}),
  })
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
  return cfg
