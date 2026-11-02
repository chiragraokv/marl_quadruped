"""Evaluate an ARS checkpoint on a fixed batch and print episode metrics."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import tyro

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from scripts.train_ars import evaluate


@dataclass
class EvalARSConfig:
  policy_file: Path
  task_id: str = "Unitree-Go2-JumpInPlace-ARS"
  num_episodes: int = 64
  device: str = "cuda:0"
  seed: int = 123
  landing_gain_scale: float | None = None


def main(args: EvalARSConfig) -> None:
  if not 1 <= args.num_episodes <= 500:
    raise ValueError("num_episodes must be between 1 and 500")
  torch.manual_seed(args.seed)
  device = args.device if torch.cuda.is_available() else "cpu"
  policy = torch.load(args.policy_file, map_location=device, weights_only=False)
  if policy.get("obs_dim") != 15 or policy.get("act_dim") != 6:
    raise ValueError("Expected an ARS checkpoint with obs_dim=15 and act_dim=6")

  cfg = load_env_cfg(args.task_id, play=True)
  acro = cfg.actions["sagittal_pd"].acrobatics
  if acro.stage != "ars":
    raise ValueError("eval_ars.py requires an ARS task ID")
  if args.landing_gain_scale is not None:
    if not 0.2 <= args.landing_gain_scale <= 0.8:
      raise ValueError("landing_gain_scale must be between 0.2 and 0.8")
    acro.landing_gain_scale = args.landing_gain_scale
  cfg.scene.num_envs = args.num_episodes
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  theta = torch.cat((policy["W"].reshape(-1), policy["b"])).to(device)
  norm = SimpleNamespace(mean=policy["mu"].to(device), std=policy["sd"].to(device))
  try:
    returns, _ = evaluate(env, theta.unsqueeze(0).expand(args.num_episodes, -1).clone(), norm,
                          env.max_episode_length)
    metrics = env.acrobatics_last_metrics
    print(f"episodes={args.num_episodes} return_mean={returns.mean().item():.4f} "
          f"return_std={returns.std(unbiased=False).item():.4f}")
    for key in ("h_bar", "d_bar", "theta_bar", "peak_force", "n_air_steps"):
      values = metrics[key].float()
      print(f"{key}: mean={values.mean().item():.4f} std={values.std(unbiased=False).item():.4f}")
    early = metrics["terminated_early"].bool()
    jump_success = (metrics["n_air_steps"] >= 3) & ~early
    print(f"early_termination_rate={early.float().mean().item():.4f}")
    print(f"jump_success_rate={jump_success.float().mean().item():.4f}")
    for cause in ("fall", "illegal_contact", "tilt"):
      print(f"{cause}_rate={metrics[cause].float().mean().item():.4f}")
    if acro.motion == "forward":
      distance_success = metrics["d_bar"] >= acro.forward_success_distance
      print(f"forward_distance_success_rate={distance_success.float().mean().item():.4f} "
            f"threshold={acro.forward_success_distance:.3f}m")
  finally:
    env.close()


if __name__ == "__main__":
  main(tyro.cli(EvalARSConfig))
