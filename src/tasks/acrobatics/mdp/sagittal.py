"""Six-DoF sagittal policy mapping for the Go2's twelve leg joints."""

from __future__ import annotations

import torch
import mujoco
import re
from functools import lru_cache

from src.assets.robots.unitree_go2.go2_constants import GO2_XML, INIT_STATE

# Simulator order from the Go2 MJCF: FL, FR, RL, RR, each hip/thigh/calf.
JOINT_NAMES = (
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
POLICY_TO_JOINT = (0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5)
HIP_SIGN = ( -1.0, 1.0, -1.0, 1.0 )


def _tensor(values, *, device=None, dtype=None):
  return torch.as_tensor(values, device=device, dtype=dtype or torch.float32)


def default_go2_limits(*, device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
  """Joint limits read from Go2's compiled MJCF, in simulator order."""
  lo, hi = _mjcf_joint_limits()
  return _tensor(lo, device=device, dtype=dtype), _tensor(hi, device=device, dtype=dtype)


@lru_cache(maxsize=1)
def _mjcf_joint_limits() -> tuple[tuple[float, ...], tuple[float, ...]]:
  model = mujoco.MjModel.from_xml_path(str(GO2_XML))
  lower, upper = [], []
  for name in JOINT_NAMES:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0 or not model.jnt_limited[joint_id]:
      raise ValueError(f"Go2 MJCF is missing a limited joint: {name}")
    lower.append(float(model.jnt_range[joint_id, 0]))
    upper.append(float(model.jnt_range[joint_id, 1]))
  return tuple(lower), tuple(upper)


def default_go2_home(*, device=None, dtype=None) -> torch.Tensor:
  """Resolve the entity config's default pose in Go2 simulator order."""
  positions = []
  for joint_name in JOINT_NAMES:
    for pattern, value in INIT_STATE.joint_pos.items():
      if re.fullmatch(pattern, joint_name):
        positions.append(float(value))
        break
    else:
      raise ValueError(f"No default joint position is configured for {joint_name}")
  return _tensor(positions, device=device, dtype=dtype)


def expand_sagittal(action: torch.Tensor) -> torch.Tensor:
  """Mirror [front hip/thigh/calf, rear hip/thigh/calf] into 12 joints."""
  if action.shape[-1] != 6:
    raise ValueError(f"Expected six sagittal actions, got {action.shape[-1]}")
  expanded = action[..., list(POLICY_TO_JOINT)]
  signs = torch.ones(12, device=action.device, dtype=action.dtype)
  signs[0], signs[3], signs[6], signs[9] = HIP_SIGN
  return expanded * signs


def action_to_joint_positions(
  action: torch.Tensor,
  lower: torch.Tensor | None = None,
  upper: torch.Tensor | None = None,
) -> torch.Tensor:
  """Scale normalized actions to per-joint limits and mirror them."""
  if action.shape[-1] != 6:
    raise ValueError(f"Expected six sagittal actions, got {action.shape[-1]}")
  lower = default_go2_limits(device=action.device, dtype=action.dtype)[0] if lower is None else lower.to(action)
  upper = default_go2_limits(device=action.device, dtype=action.dtype)[1] if upper is None else upper.to(action)
  if lower.shape[-1] != 12 or upper.shape[-1] != 12:
    raise ValueError("Joint limit tensors must have 12 entries in simulator order")
  a12 = expand_sagittal(action.clamp(-1.0, 1.0))
  return lower + 0.5 * (a12 + 1.0) * (upper - lower)


def inverse_sagittal(joint_pos: torch.Tensor) -> torch.Tensor:
  """Average mirrored legs and return the equivalent normalized six-vector."""
  if joint_pos.shape[-1] != 12:
    raise ValueError("Expected 12 joint positions in simulator order")
  lo, hi = default_go2_limits(device=joint_pos.device, dtype=joint_pos.dtype)
  signs = torch.tensor(HIP_SIGN, device=joint_pos.device, dtype=joint_pos.dtype)
  # Map the mirror partner back into the front/rear policy convention.
  front_left = joint_pos[..., 0:3].clone()
  front_right = joint_pos[..., 3:6].clone()
  rear_left = joint_pos[..., 6:9].clone()
  rear_right = joint_pos[..., 9:12].clone()
  front_left[..., 0] *= signs[0]
  front_right[..., 0] *= signs[1]
  rear_left[..., 0] *= signs[2]
  rear_right[..., 0] *= signs[3]
  front = (front_left + front_right) * 0.5
  rear = (rear_left + rear_right) * 0.5
  pair = torch.cat((front, rear), dim=-1)
  pair_lo = torch.stack((lo[0], lo[1], lo[2], lo[6], lo[7], lo[8]))
  pair_hi = torch.stack((hi[0], hi[1], hi[2], hi[6], hi[7], hi[8]))
  return (2.0 * (pair - pair_lo) / (pair_hi - pair_lo) - 1.0).clamp(-1.0, 1.0)


def home_action(*, device=None, dtype=None) -> torch.Tensor:
  """Return the normalized sagittal action corresponding to the Go2 home pose."""
  return inverse_sagittal(default_go2_home(device=device, dtype=dtype))


def average_sagittal(joint_values: torch.Tensor) -> torch.Tensor:
  """Average mirrored joint values after correcting the hip sign convention."""
  if joint_values.shape[-1] != 12:
    raise ValueError("Expected 12 joint values in simulator order")
  signs = torch.tensor(HIP_SIGN, device=joint_values.device, dtype=joint_values.dtype)
  front_left = joint_values[..., 0:3].clone()
  front_right = joint_values[..., 3:6].clone()
  rear_left = joint_values[..., 6:9].clone()
  rear_right = joint_values[..., 9:12].clone()
  front_left[..., 0] *= signs[0]
  front_right[..., 0] *= signs[1]
  rear_left[..., 0] *= signs[2]
  rear_right[..., 0] *= signs[3]
  front = (front_left + front_right) * 0.5
  rear = (rear_left + rear_right) * 0.5
  return torch.cat((front, rear), dim=-1)
