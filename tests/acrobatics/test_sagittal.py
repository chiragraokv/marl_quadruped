import torch

from src.tasks.acrobatics.mdp.sagittal import (
  action_to_joint_positions,
  default_go2_home,
  default_go2_limits,
  expand_sagittal,
  home_action,
  inverse_sagittal,
)


def test_home_round_trip_and_action_round_trip():
  home = default_go2_home()
  a_home = home_action()
  assert torch.allclose(action_to_joint_positions(a_home), home, atol=1e-5)
  action = torch.tensor([-0.2, 0.3, -0.1, 0.4, -0.5, 0.2])
  assert torch.allclose(inverse_sagittal(action_to_joint_positions(action)), action, atol=1e-5)


def test_symmetric_actions_mirror_lateral_hip_offsets():
  hip_only = torch.tensor([[0.4, 0.0, 0.0, -0.2, 0.0, 0.0]])
  joints = expand_sagittal(hip_only)
  assert joints[0, 0] == -joints[0, 3]
  assert joints[0, 6] == -joints[0, 9]
  lo, hi = default_go2_limits()
  positions = action_to_joint_positions(hip_only)
  assert torch.all(positions >= lo) and torch.all(positions <= hi)
