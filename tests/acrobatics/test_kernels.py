import torch

from src.tasks.acrobatics.mdp.rewards import (
  imitation_reward,
  kernel,
  normalize_clip,
  sparse_terminal_reward,
)
from src.tasks.acrobatics.cfg import AcrobaticsCfg


def test_kernel_and_normalization():
  x = torch.tensor([-1.0, 0.0, 1.0])
  assert torch.allclose(kernel(x, 2.0, 1.0), torch.tensor([2 / torch.e, 2.0, 2 / torch.e]))
  assert torch.equal(normalize_clip(torch.tensor([-1.0, 0.5, 2.0]), 1.0), torch.tensor([0.0, 0.5, 1.0]))


def test_sparse_rewards_match_both_equations_and_branches():
  h, d, theta = torch.tensor([0.5]), torch.tensor([0.25]), torch.tensor([0.0])
  normal = sparse_terminal_reward("in_place", h, d, theta, torch.tensor([False]), h_max=1, d_max=1,
                                  a_theta=1, b_theta=1, a_d=1, b_d=1, c_h=0,
                                  success_bonus=0.1)
  expected = (1 + torch.exp(torch.tensor(-0.25))) * 0.5 + 0.05
  assert torch.allclose(normal, expected.reshape(1))
  failed = sparse_terminal_reward("in_place", h, d, theta, torch.tensor([True]), h_max=1, d_max=1,
                                  a_theta=1, b_theta=1, a_d=1, b_d=1, c_h=0,
                                  failure_penalty=0.08, crash_scale=0.096)
  assert torch.allclose(failed, (1 + torch.exp(torch.tensor(-0.25))) * 0.5 - 0.128)
  fwd = sparse_terminal_reward("forward", h, d, theta, torch.tensor([False]), h_max=1, d_max=1,
                               a_theta=1, b_theta=1, c_d=2, c_h=0, success_bonus=0.1)
  assert torch.allclose(fwd, torch.tensor([1.5 * 0.5 + 0.075]))


def test_imitation_reward():
  action = torch.zeros((2, 6))
  target = torch.stack((torch.zeros(6), torch.ones(6) * 0.2))
  result = imitation_reward(action, target)
  assert result[0].item() == 1.0
  assert torch.allclose(result[1], torch.exp(torch.tensor(-0.96)))


def test_standing_height_has_zero_normalized_jump_reward():
  standing = torch.tensor([0.224])
  result = sparse_terminal_reward(
    "in_place", standing, torch.zeros_like(standing), torch.zeros_like(standing),
    torch.tensor([False]), h_max=0.524, h_stand=0.224, d_max=1.3,
    success_bonus=0.1,
  )
  assert torch.equal(result, torch.zeros_like(result))


def test_default_sparse_failure_reward_is_negative_at_max_jump_height():
  cfg = AcrobaticsCfg()
  height = torch.tensor([cfg.h_max])
  zero = torch.zeros_like(height)
  early = torch.ones_like(height, dtype=torch.bool)
  for motion in ("in_place", "forward"):
    result = sparse_terminal_reward(
      motion, height, zero, zero, early,
      h_max=cfg.h_max, h_stand=cfg.h_stand, d_max=cfg.d_max,
      a_theta=cfg.sparse_theta_amplitude, b_theta=cfg.sparse_theta_decay,
      c_h=cfg.sparse_height_bias, a_d=cfg.sparse_distance_amplitude,
      b_d=cfg.sparse_distance_decay, c_d=cfg.sparse_forward_distance_weight,
      success_bonus=cfg.sparse_success_bonus,
      failure_penalty=cfg.sparse_failure_penalty,
      crash_scale=cfg.sparse_crash_scale,
    )
    assert result.item() < 0.0
