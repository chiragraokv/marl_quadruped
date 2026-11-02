"""Sagittal policy action mapped to batched PD torques."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from src.tasks.acrobatics.cfg import AcrobaticsCfg
from src.tasks.acrobatics.mdp.phase import LANDING, TAKEOFF
from src.tasks.acrobatics.mdp.sagittal import action_to_joint_positions, home_action
from src.tasks.acrobatics.mdp.sagittal import average_sagittal
from src.tasks.acrobatics.mdp.trackers import AcrobaticsState


@dataclass(kw_only=True)
class SagittalPDActionCfg(ActionTermCfg):
  acrobatics: AcrobaticsCfg

  def build(self, env):
    return SagittalPDAction(self, env)


class SagittalPDAction(ActionTerm):
  def __init__(self, cfg: SagittalPDActionCfg, env):
    super().__init__(cfg, env)
    self.acro = cfg.acrobatics
    self._raw_action = torch.zeros((self.num_envs, 6), device=self.device)
    self.home = home_action(device=self.device)
    self.filtered_action = self.home.expand(self.num_envs, -1).clone()
    self.state = AcrobaticsState(env, self.acro)
    env.acrobatics_state = self.state
    self.joint_ids, _ = self._entity.find_joints((
      "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
      "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ), preserve_order=True)
    self.kp = torch.tensor([self.acro.kp_hip, self.acro.kp_thigh, self.acro.kp_calf] * 4, device=self.device)
    self.kd = torch.tensor([self.acro.kd_hip, self.acro.kd_thigh, self.acro.kd_calf] * 4, device=self.device)
    self.effort = torch.tensor([self.acro.effort_hip, self.acro.effort_thigh, self.acro.effort_calf] * 4, device=self.device)
    self.torque = torch.zeros((self.num_envs, 12), device=self.device)
    self.sagittal_torque = torch.zeros((self.num_envs, 6), device=self.device)
    self.pitch = torch.zeros(self.num_envs, device=self.device)

  @property
  def action_dim(self):
    return 6

  @property
  def raw_action(self):
    return self._raw_action

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_action.copy_(actions.clamp(-1.0, 1.0))
    action = self._raw_action
    if self.acro.stage == "ars":
      action = torch.where((self.state.phase.phase == TAKEOFF).unsqueeze(-1), action, self.home)
      action = self.acro.lpf_alpha * self.filtered_action + (1.0 - self.acro.lpf_alpha) * action
      self.filtered_action.copy_(action.clamp(-1.0, 1.0))
    else:
      self.filtered_action.copy_(action)

  def apply_actions(self) -> None:
    robot = self._entity
    q_ref = action_to_joint_positions(self.filtered_action)
    q = robot.data.joint_pos[:, self.joint_ids]
    qd = robot.data.joint_vel[:, self.joint_ids]
    kp = self.kp.expand_as(q)
    kd = self.kd.expand_as(q)
    if self.acro.stage == "ars":
      scale = torch.where((self.state.phase.phase == LANDING).unsqueeze(-1), self.acro.landing_gain_scale, 1.0)
      kp = kp * scale
      kd = kd * scale
    self.torque.copy_((kp * (q_ref - q) - kd * qd).clamp(-self.effort, self.effort))
    self.sagittal_torque.copy_(average_sagittal(self.torque))
    robot.set_joint_effort_target(self.torque, self.joint_ids)
    w, x, y, z = robot.data.root_link_quat_w.unbind(-1)
    self.pitch.copy_(torch.atan2(-2.0 * (x * z - w * y), 1.0 - 2.0 * (y.square() + z.square())))

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = slice(None)
    self.filtered_action[env_ids] = self.home
    self._raw_action[env_ids] = 0.0
    self.torque[env_ids] = 0.0
    self.sagittal_torque[env_ids] = 0.0
    self.state.reset(env_ids)
