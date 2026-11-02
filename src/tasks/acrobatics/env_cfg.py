"""Factory for the registered Go2 acrobatics environment variants."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import reset_joints_by_offset, reset_root_state_uniform, time_out
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.entity import EntityArticulationInfoCfg

from src.assets.robots import get_go2_robot_cfg
from src.tasks.acrobatics.cfg import AcrobaticsCfg
from src.tasks.acrobatics.mdp import observations, rewards, terminations
from src.tasks.acrobatics.mdp.actions import SagittalPDActionCfg
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg


Motion = Literal["in_place", "forward"]
Stage = Literal["ars", "imitate", "refine"]


def make_env_cfg(
  motion: Motion = "in_place",
  stage: Stage = "ars",
  soft: bool = False,
  randomize: bool = False,
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build one shared Go2 task config; policy/reward behavior varies by stage."""
  if soft:
    raise NotImplementedError("Soft springs are gated on successful Stage 2 evaluation (Gate 4).")
  if randomize:
    raise NotImplementedError("Mass/CoM randomization is gated on successful Stage 2 evaluation (Gate 4).")
  acro = AcrobaticsCfg(motion=motion, stage=stage, soft=soft, randomize=randomize)
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.scene.num_envs = 64
  cfg.episode_length_s = acro.episode_length_s
  cfg.scale_rewards_by_dt = False  # Sparse terminal rewards must not be divided by control dt.
  cfg.scene.entities = {"robot": _go2_motor_robot_cfg(soft=soft, acro=acro)}
  cfg.commands = {}
  cfg.curriculum = {}

  foot_names = ("FR", "FL", "RR", "RL")
  feet = ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(mode="geom", pattern=tuple(f"{n}_foot_collision" for n in foot_names), entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="none", num_slots=1,
    track_air_time=True,
  )
  nonfoot = ContactSensorCfg(
    name="nonfoot_contact",
    primary=ContactMatch(mode="geom", entity="robot", pattern=r".*_collision\d*$",
                         exclude=tuple(f"{n}_foot_collision" for n in foot_names)),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="none", num_slots=1, history_length=4,
  )
  cfg.scene.sensors = (feet, nonfoot)

  # Keep only small reset perturbations. No interval pushes or velocity-task DR.
  cfg.events = {
    "reset_base": EventTermCfg(
      func=reset_root_state_uniform, mode="reset",
      params={"pose_range": {"x": (-acro.reset_xy_noise, acro.reset_xy_noise),
                              "y": (-acro.reset_xy_noise, acro.reset_xy_noise),
                              "z": (-acro.reset_height_noise, acro.reset_height_noise),
                              "pitch": (-acro.reset_pitch_noise, acro.reset_pitch_noise),
                              "yaw": (-acro.reset_yaw_noise, acro.reset_yaw_noise)},
              "velocity_range": {}},
    ),
    "reset_robot_joints": EventTermCfg(
      func=reset_joints_by_offset, mode="reset",
      params={"position_range": (-acro.reset_joint_pos_noise, acro.reset_joint_pos_noise),
              "velocity_range": (-acro.reset_joint_vel_noise, acro.reset_joint_vel_noise),
              "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }

  noise = (0.0,) * 5 if play else (acro.sigma_h, acro.sigma_hdot, acro.sigma_pitch, acro.sigma_q, acro.sigma_qdot)
  include_flight = stage != "ars"
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={"jump_state": ObservationTermCfg(func=observations.actor_jump_observation,
                                               params={"sigmas": noise, "include_flight": include_flight})},
      concatenate_terms=True, enable_corruption=False, history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms={"privileged_jump_state": ObservationTermCfg(func=observations.critic_jump_observation,
                                                          params={"include_flight": include_flight})},
      concatenate_terms=True, enable_corruption=False, history_length=1,
    ),
  }
  cfg.actions = {
    "sagittal_pd": SagittalPDActionCfg(entity_name="robot", acrobatics=acro),
  }
  cfg.rewards = _reward_terms(stage, acro)
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "fall": TerminationTermCfg(func=terminations.fallen, params={"h_fall": acro.h_fall}),
    "illegal_contact": TerminationTermCfg(func=terminations.illegal_contact,
                                           params={"sensor_name": "nonfoot_contact",
                                                   "force_threshold": acro.illegal_contact_threshold}),
    "tilt": TerminationTermCfg(func=terminations.tilted, params={"threshold": acro.tilt_threshold}),
    # This side effect updates the state before the reward manager runs.
    "update_trackers": TerminationTermCfg(func=terminations.update_trackers),
  }
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0
  return cfg


def _go2_motor_robot_cfg(*, soft: bool, acro: AcrobaticsCfg):
  if soft:
    raise NotImplementedError("Soft springs are not yet implemented.")
  cfg = get_go2_robot_cfg()
  cfg.init_state = replace(cfg.init_state, pos=(0.0, 0.0, acro.h_stand))
  cfg.articulation = EntityArticulationInfoCfg(
    actuators=(
      BuiltinMotorActuatorCfg(target_names_expr=(".*hip_.*",), effort_limit=acro.effort_hip, armature=0.01),
      BuiltinMotorActuatorCfg(target_names_expr=(".*thigh_.*",), effort_limit=acro.effort_thigh, armature=0.01),
      BuiltinMotorActuatorCfg(target_names_expr=(".*calf_.*",), effort_limit=acro.effort_calf, armature=0.02),
    ),
    soft_joint_pos_limit_factor=0.9,
  )
  return cfg


def _reward_terms(stage: Stage, acro: AcrobaticsCfg):
  if stage == "ars":
    return {"sparse_terminal": RewardTermCfg(func=rewards.sparse_reward_term, weight=1.0,
                                              params={"acrobatics": acro})}
  if stage == "imitate":
    return {"imitation": RewardTermCfg(func=rewards.imitation_reward_term, weight=1.0,
                                        params={"acrobatics": acro, "policy_path": acro.ars_policy_path})}
  return {
    "height": RewardTermCfg(func=rewards.reward_height, weight=1.0, params={"acrobatics": acro}),
    "contact": RewardTermCfg(func=rewards.reward_contact, weight=1.0, params={"acrobatics": acro}),
    "distance": RewardTermCfg(func=rewards.reward_displacement, weight=1.0, params={"acrobatics": acro}),
    "smooth_torque": RewardTermCfg(func=rewards.reward_smooth_torque, weight=1.0),
    "pitch": RewardTermCfg(func=rewards.reward_pitch, weight=1.0),
    "terminal_bonus": RewardTermCfg(func=rewards.terminal_bonus, weight=1.0, params={"acrobatics": acro}),
  }


