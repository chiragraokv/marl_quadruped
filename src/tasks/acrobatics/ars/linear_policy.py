"""Batched linear policy and V2 observation normalization."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RunningNorm:
  obs_dim: int
  device: torch.device | str = "cpu"
  epsilon: float = 1e-8

  def __post_init__(self):
    self.device = torch.device(self.device)
    self.count = torch.zeros((), device=self.device, dtype=torch.float64)
    self.mean = torch.zeros(self.obs_dim, device=self.device, dtype=torch.float64)
    self.m2 = torch.zeros_like(self.mean)

  @property
  def variance(self) -> torch.Tensor:
    return torch.where(self.count > 0, self.m2 / self.count.clamp_min(1.0), torch.ones_like(self.m2))

  @property
  def std(self) -> torch.Tensor:
    return self.variance.sqrt().clamp_min(self.epsilon)

  def normalize(self, obs: torch.Tensor) -> torch.Tensor:
    return ((obs - self.mean.to(obs)) / self.std.to(obs)).to(obs.dtype)

  @torch.no_grad()
  def update(self, batch: torch.Tensor) -> None:
    batch = batch.to(device=self.device, dtype=torch.float64).reshape(-1, self.obs_dim)
    if batch.numel() == 0:
      return
    batch_count = torch.as_tensor(batch.shape[0], device=self.device, dtype=torch.float64)
    batch_mean = batch.mean(0)
    batch_m2 = ((batch - batch_mean) ** 2).sum(0)
    delta = batch_mean - self.mean
    total = self.count + batch_count
    self.mean += delta * batch_count / total
    self.m2 += batch_m2 + delta.square() * self.count * batch_count / total
    self.count = total


@dataclass
class BatchedLinearPolicy:
  weights: torch.Tensor  # (E, A, S) or (A, S)
  bias: torch.Tensor  # (E, A) or (A,)

  def __call__(self, obs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    norm_obs = (obs - mean) / (std + 1e-8)
    if self.weights.ndim == 2:
      out = torch.nn.functional.linear(norm_obs, self.weights, self.bias)
    else:
      out = torch.bmm(self.weights, norm_obs.unsqueeze(-1)).squeeze(-1) + self.bias
    return out.clamp(-1.0, 1.0)
