"""Early termination checks and the pre-reward per-step tracker update."""

from __future__ import annotations

import torch

from src.tasks.acrobatics.mdp.trackers import get_state


def update_trackers(env) -> torch.Tensor:
  action_term = env.action_manager.get_term("sagittal_pd")
  robot = env.scene["robot"]
  root = robot.data.root_link_pos_w
  contact = env.scene["feet_contact"].data
  if contact.force is None:
    force = torch.zeros(env.num_envs, device=env.device)
  else:
    force = contact.force.reshape(env.num_envs, -1, 3).norm(dim=-1).sum(dim=-1)
  state = get_state(env)
  quat = robot.data.root_link_quat_w
  w, xq, yq, zq = quat.unbind(-1)
  fwd_x = 1.0 - 2.0 * (yq.square() + zq.square())
  fwd_y = 2.0 * (xq * yq + w * zq)
  forward_axis = torch.stack((fwd_x, fwd_y), dim=-1)
  forward_axis = torch.nn.functional.normalize(forward_axis, dim=-1)
  state.update(root[:, 2], root[:, 0], root[:, 1], forward_axis, action_term.pitch,
               robot.data.root_link_lin_vel_w[:, 2], force, action_term.sagittal_torque)
  return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def fallen(env, h_fall: float = 0.10) -> torch.Tensor:
  return env.scene["robot"].data.root_link_pos_w[:, 2] < h_fall


def tilted(env, threshold: float = 0.85) -> torch.Tensor:
  quat = env.scene["robot"].data.root_link_quat_w
  _, x, y, _ = quat.unbind(-1)
  return 1.0 - 2.0 * (x.square() + y.square()) < threshold


def illegal_contact(env, sensor_name: str = "nonfoot_contact", force_threshold: float = 10.0) -> torch.Tensor:
  data = env.scene[sensor_name].data
  if data.force_history is not None:
    return (data.force_history.norm(dim=-1) > force_threshold).any(dim=-1).any(dim=-1)
  return data.found.any(dim=-1).any(dim=-1)
