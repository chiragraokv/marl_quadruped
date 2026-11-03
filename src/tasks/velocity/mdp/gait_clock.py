from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def gait_phase(env: ManagerBasedRlEnv, period: float) -> torch.Tensor:
  """Return a per-environment phase, randomized at startup and each reset."""
  steps = env.episode_length_buf
  phase_offsets = getattr(env, "_gait_phase_offsets", None)
  last_steps = getattr(env, "_gait_last_episode_steps", None)
  if phase_offsets is None or last_steps is None:
    phase_offsets = torch.rand(env.num_envs, device=env.device) * period
    env._gait_phase_offsets = phase_offsets
    env._gait_last_episode_steps = steps.clone()
  else:
    reset_mask = (steps == 0) & (last_steps > 0)
    if reset_mask.any():
      phase_offsets[reset_mask] = torch.rand(
        int(reset_mask.sum().item()), device=env.device
      ) * period
    last_steps.copy_(steps)
  return ((steps * env.step_dt + phase_offsets) / period).remainder(1.0)
