"""One V2-t ARS policy update, separated from simulator evaluation."""

from __future__ import annotations

import torch


@torch.no_grad()
def ars_iteration(
  theta: torch.Tensor,
  rewards: torch.Tensor,
  directions: torch.Tensor,
  *,
  top_b: int,
  nu: float,
  alpha: float,
) -> torch.Tensor:
  """Apply one ARS V2 update from rewards for positive then negative probes."""
  n = directions.shape[0]
  if rewards.numel() != 2 * n:
    raise ValueError("rewards must contain N positive and N negative returns")
  if not 1 <= top_b <= n:
    raise ValueError("top_b must be between 1 and the number of directions")
  pos, neg = rewards[:n], rewards[n:]
  selected = torch.topk(torch.maximum(pos, neg), top_b).indices
  selected_returns = torch.cat((pos[selected], neg[selected]))
  sigma = selected_returns.std(unbiased=False).clamp_min(1e-6)
  step = ((pos[selected] - neg[selected]).unsqueeze(-1) * directions[selected]).sum(0)
  return theta + alpha / (top_b * sigma) * step
