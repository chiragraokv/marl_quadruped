import torch

from src.tasks.acrobatics.ars.ars import ars_iteration
from src.tasks.acrobatics.ars.linear_policy import BatchedLinearPolicy, RunningNorm


def test_batched_linear_policy_and_running_norm():
  norm = RunningNorm(2)
  norm.update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
  assert torch.allclose(norm.mean.float(), torch.tensor([2.0, 3.0]))
  policy = BatchedLinearPolicy(torch.eye(2), torch.zeros(2))
  assert torch.allclose(policy(torch.tensor([[2.0, 3.0]]), torch.zeros(2), torch.ones(2)), torch.tensor([[1.0, 1.0]]))


def test_ars_improves_quadratic_objective():
  torch.manual_seed(7)
  theta = torch.zeros(2)
  target = torch.tensor([1.0, -0.7])
  for _ in range(200):
    directions = torch.randn(48, 2)
    probes = torch.cat((theta + 0.05 * directions, theta - 0.05 * directions))
    rewards = -((probes - target) ** 2).sum(dim=1)
    theta = ars_iteration(theta, rewards, directions, top_b=16, nu=0.05, alpha=0.02)
  assert -((theta - target) ** 2).sum() > -((torch.zeros(2) - target) ** 2).sum()
  assert -((theta - target) ** 2).sum() > -0.05 * (target**2).sum()
