"""Go2 jump observations with configurable white noise."""

from __future__ import annotations

import torch

from src.tasks.acrobatics.mdp.sagittal import average_sagittal
from src.tasks.acrobatics.mdp.trackers import get_state


def _pitch_from_quaternion(quat: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(-2.0 * (x * z - w * y), 1.0 - 2.0 * (y.square() + z.square()))


def jump_observation(env, sigmas: tuple[float, ...], include_flight: bool = False) -> torch.Tensor:
  robot = env.scene["robot"]
  data = robot.data
  h = data.root_link_pos_w[:, 2:3]
  hdot = data.root_link_lin_vel_w[:, 2:3]
  pitch = _pitch_from_quaternion(data.root_link_quat_w).unsqueeze(-1)
  q = average_sagittal(data.joint_pos[:, :12])
  qdot = average_sagittal(data.joint_vel[:, :12])
  pieces = [h, hdot, pitch, q, qdot]
  for idx, sigma in enumerate(sigmas):
    if sigma:
      pieces[idx] = pieces[idx] + sigma * torch.randn_like(pieces[idx])
  result = torch.cat(pieces, dim=-1)
  if include_flight:
    result = torch.cat((result, get_state(env).phase.flight_flag.unsqueeze(-1)), dim=-1)
  return result


def actor_jump_observation(env, sigmas=(0.01, 0.05, 0.01, 0.01, 0.05), include_flight=False):
  return jump_observation(env, sigmas, include_flight)


def critic_jump_observation(env, sigmas=(0.0, 0.0, 0.0, 0.0, 0.0), include_flight=True):
  obs = jump_observation(env, sigmas, include_flight)
  state = get_state(env)
  phase = torch.nn.functional.one_hot(state.phase.phase, num_classes=3).to(obs.dtype)
  t = env.episode_length_buf.to(obs.dtype).unsqueeze(-1) / max(env.max_episode_length, 1)
  episode = torch.stack((state.h_bar, state.d_bar, state.theta_bar), dim=-1)
  return torch.cat((obs, phase, episode, t), dim=-1)
