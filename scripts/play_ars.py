"""View a saved ARS linear policy with the repository's native viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass
class PlayARS:
  policy_file: Path
  task_id: str = "Unitree-Go2-JumpInPlace-ARS"
  viewer: str = "auto"
  num_envs: int = 1
  device: str = "cuda:0"


def main(args: PlayARS):
  device = args.device if torch.cuda.is_available() else "cpu"
  policy = torch.load(args.policy_file, map_location=device, weights_only=False)
  if policy.get("obs_dim") != 15 or policy.get("act_dim") != 6:
    raise ValueError("Expected a 15-observation, 6-action ARS checkpoint")
  cfg = load_env_cfg(args.task_id, play=True)
  if cfg.actions["sagittal_pd"].acrobatics.stage != "ars":
    raise ValueError("play_ars.py requires an ARS task")
  cfg.scene.num_envs = args.num_envs
  base_env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=1.0)
  w = policy["W"].to(device)
  b = policy["b"].to(device)
  mean = policy["mu"].to(device)
  std = policy["sd"].to(device)

  def run(obs):
    actor_obs = obs["actor"] if isinstance(obs, dict) else obs
    return torch.nn.functional.linear((actor_obs - mean) / (std + 1e-8), w, b).clamp(-1.0, 1.0)

  viewer = args.viewer
  if viewer == "auto":
    import os
    viewer = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"
  if viewer == "native":
    NativeMujocoViewer(env, run).run()
  elif viewer == "viser":
    ViserPlayViewer(env, run).run()
  else:
    raise ValueError("viewer must be 'auto', 'native', or 'viser'")
  env.close()


if __name__ == "__main__":
  main(tyro.cli(PlayARS))
