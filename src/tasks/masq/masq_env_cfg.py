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
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg
from src.tasks.masq import mdp
from src.tasks.masq.mdp.commands import GaitCommandCfg
from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

def make_masq_env_cfg(
  play: bool = False, steps_per_iter: int = 24
) -> ManagerBasedRlEnvCfg:
  # Start from the validated Go2 flat task so robot entities, foot sensors,
  # site names, and the raycast-frame overrides are all initialized correctly.
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.commands = {
    "gait": GaitCommandCfg(
      entity_name="robot", resampling_time_range=(3., 8.),
      ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1., 2.), lin_vel_y=(-1., 1.),
                                   ang_vel_z=(-1., 1.), heading=None),
      # V5 starts with the controlled trot-only comparison.  Other gait ids
      # remain available through GaitCommandCfg.gait_ids for later ablations.
      gait_ids=(1,),
    )
  }
  # Observe robot-on-robot contacts separately from the existing non-foot
  # ground sensor.  Using the base subtree covers all Go2 collision geoms.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (self_collision_cfg,)
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
  scales = mdp.MASQ_REWARD_SCALES
  cfg.rewards = {
    "track_linear_velocity": RewardTermCfg(func=mdp.track_lin_vel_xy, weight=scales["track_linear_velocity"], params={"command_name":"gait", "sigma":.25, "threshold":.2}),
    "track_angular_velocity": RewardTermCfg(func=mdp.track_ang_vel_z, weight=scales["track_angular_velocity"], params={"command_name":"gait", "sigma":.25, "threshold":.2}),
    # Stability terms copied from the validated velocity task.  These close
    # the reward-hacking loophole where velocity can be tracked with a tilted,
    # violently rotating, or self-colliding body.
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2, weight=scales["body_orientation_l2"],
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))}),
    # Disabled for reward_v4 experiment; keep available for later ablations.
    # "self_collisions": RewardTermCfg(
    #   func=mdp.self_collision_cost, weight=scales["self_collisions"],
    #   params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0}),
    # "body_ang_vel": RewardTermCfg(
    #   func=mdp.body_angular_velocity_penalty, weight=scales["body_ang_vel"],
    #   params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))}),
    # "angular_momentum": RewardTermCfg(
    #   func=mdp.angular_momentum_penalty, weight=scales["angular_momentum"],
    #   params={"sensor_name": "robot/root_angmom"}),
    "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z, weight=scales["lin_vel_z"]),
    "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy, weight=scales["ang_vel_xy"]),
    "base_height": RewardTermCfg(
      func=mdp.base_height, weight=scales["base_height"],
      params={"target_height": 0.30, "tolerance": 0.03}),
    "alive_moving": RewardTermCfg(
      func=mdp.alive_moving, weight=scales["alive_moving"],
      params={"command_name": "gait", "sensor_name": "feet_ground_contact", "eps_cmd": 0.1, "v_min": 0.1}),
    "gait_contact_match": RewardTermCfg(
      func=mdp.gait_contact_match, weight=scales["gait_contact_match"],
      params={"command_name": "gait", "sensor_name": "feet_ground_contact", "duty_factor": .58}),
    "support_deficit": RewardTermCfg(
      func=mdp.support_deficit, weight=scales["support_deficit"],
      params={"command_name": "gait", "sensor_name": "feet_ground_contact"}),
    "all_air_nonpronk": RewardTermCfg(
      func=mdp.all_air_nonpronk, weight=scales["all_air_nonpronk"],
      params={"command_name": "gait", "sensor_name": "feet_ground_contact"}),
    "fall_termination": RewardTermCfg(
      func=mdp.fall_termination, weight=scales["fall_termination"],
      params={"steps_per_iteration": steps_per_iter, "zero_until_iteration": 200,
              "ramp_until_iteration": 800}),
    # Disabled for reward_v4 experiment; keep available for later ablations.
    # "torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=scales["torques"]),
    # "dof_vel": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=scales["dof_vel"]),
    "dof_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=scales["dof_acc"]),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=scales["action_rate"]),
    # "action_smooth_1": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=scales["action_smooth_1"]),
    "action_smooth_2": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=scales["action_smooth_2"]),
    # "feet_impact_vel": RewardTermCfg(func=mdp.feet_impact_velocity, weight=scales["feet_impact_vel"]),
    # "jump": RewardTermCfg(func=mdp.jump, weight=scales["jump"], params={"command_name":"gait"}),
    "feet_slip": RewardTermCfg(func=mdp.feet_slip, weight=scales["feet_slip"], params={"sensor_name":"feet_ground_contact", "command_name":"gait", "asset_cfg":SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL"))}),
    "collision": RewardTermCfg(func=mdp.collision, weight=scales["collision"]),
    # "raibert": RewardTermCfg(func=mdp.raibert, weight=scales["raibert"], params={"command_name":"gait", "asset_cfg":SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL"))}),
    # "track_contact_shaped_vel": RewardTermCfg(func=mdp.tracking_contacts_shaped_velocity, weight=scales["track_contact_shaped_vel"], params={"asset_cfg":SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL"))}),
  }
  cfg.curriculum = {
    "multi_gait": CurriculumTermCfg(
      func=mdp.multi_gait_curriculum,
      params={"command_name": "gait", "threshold": 0.8},
    )
  }
  cfg.events.update({
    "body_mass_masq": EventTermCfg(
      func=dr.pseudo_inertia, mode="startup",
      params={"alpha_range": (0.8, 1.2),
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
