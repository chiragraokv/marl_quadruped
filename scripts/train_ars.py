"""Train a batched linear Go2 policy with augmented random search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys

import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.registry import list_tasks
from src.tasks.acrobatics.ars.ars import ars_iteration
from src.tasks.acrobatics.ars.linear_policy import RunningNorm
from src.tasks.acrobatics.mdp.sagittal import home_action


@dataclass
class TrainARS:
  task_id: str = "Unitree-Go2-JumpInPlace-ARS"
  n_directions: int = 256
  top_b: int = 64
  nu: float = 0.05
  alpha: float = 0.02
  iterations: int = 500
  eval_every: int = 10
  device: str = "cuda:0"
  seed: int = 42


@torch.inference_mode()
def evaluate(env, candidates, normalizer, max_steps):
  """Run one first-terminal episode for each policy in the vector batch."""
  num_envs = candidates.shape[0]
  obs, _ = env.reset()
  weight = candidates[:, :90].reshape(num_envs, 6, 15)
  bias = candidates[:, 90:]
  returns = torch.zeros(num_envs, device=env.device)
  done = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
  seen_obs = []
  for _ in range(max_steps + 1):
    actor_obs = obs["actor"]
    seen_obs.append(actor_obs[~done].detach().clone())
    normalized = (actor_obs - normalizer.mean.to(actor_obs)) / (normalizer.std.to(actor_obs) + 1e-8)
    action = (torch.bmm(weight, normalized.unsqueeze(-1)).squeeze(-1) + bias).clamp(-1.0, 1.0)
    action[done] = 0.0
    obs, reward, terminated, timeouts, _ = env.step(action)
    newly_done = (~done) & (terminated | timeouts)
    returns[newly_done] = reward[newly_done]
    done |= newly_done
    if bool(done.all()):
      break
  if not bool(done.all()):
    raise RuntimeError("One or more ARS rollouts did not finish before the episode cap")
  return returns, torch.cat(seen_obs, dim=0)


def main(config: TrainARS) -> None:
  if config.n_directions < 1 or not 1 <= config.top_b <= config.n_directions:
    raise ValueError("Require n_directions >= top_b >= 1")
  if config.iterations < 1 or config.eval_every < 1:
    raise ValueError("iterations and eval_every must be positive")
  torch.manual_seed(config.seed)
  device = torch.device(config.device if torch.cuda.is_available() else "cpu")
  cfg = load_env_cfg(config.task_id)
  if cfg.actions["sagittal_pd"].acrobatics.stage != "ars":
    raise ValueError("train_ars.py requires an ARS task ID")
  cfg.scene.num_envs = 2 * config.n_directions
  env = ManagerBasedRlEnv(cfg=cfg, device=str(device))
  param_count = 96
  theta = torch.zeros(param_count, device=device)
  theta[90:] = home_action(device=device)
  normalizer = RunningNorm(15, device=device)
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  motion = cfg.actions["sagittal_pd"].acrobatics.motion
  run_dir = Path("logs/ars") / f"go2_{motion}" / stamp
  run_dir.mkdir(parents=True, exist_ok=True)
  writer = SummaryWriter(str(run_dir))
  best_return = float("-inf")
  max_steps = env.max_episode_length
  try:
    for iteration in range(config.iterations):
      directions = torch.randn(config.n_directions, param_count, device=device)
      candidates = torch.cat((theta + config.nu * directions, theta - config.nu * directions), dim=0)
      returns, observations = evaluate(env, candidates, normalizer, max_steps)
      theta = ars_iteration(theta, returns, directions, top_b=config.top_b, nu=config.nu, alpha=config.alpha)
      normalizer.update(observations)
      writer.add_scalar("Train/return_mean", returns.mean().item(), iteration)
      writer.add_scalar("Train/return_max", returns.max().item(), iteration)
      writer.add_scalar("Train/return_min", returns.min().item(), iteration)
      writer.add_scalar("Train/theta_norm", theta.norm().item(), iteration)
      print(f"iteration={iteration + 1}/{config.iterations} mean={returns.mean().item():.5f} max={returns.max().item():.5f}")
      if (iteration + 1) % config.eval_every == 0 or iteration + 1 == config.iterations:
        eval_returns, _ = evaluate(env, theta.expand(2 * config.n_directions, -1).clone(), normalizer, max_steps)
        mean_eval = eval_returns.mean().item()
        policy = _policy_payload(theta, normalizer, cfg.actions["sagittal_pd"].acrobatics)
        torch.save(policy, run_dir / "ars_policy_last.pt")
        if mean_eval > best_return:
          best_return = mean_eval
          torch.save(policy, run_dir / "ars_policy_best.pt")
        writer.add_scalar("Eval/return_mean", mean_eval, iteration)
        writer.add_scalar("Eval/return_std", eval_returns.std(unbiased=False).item(), iteration)
        _write_eval_metrics(writer, env, iteration, cfg.actions["sagittal_pd"].acrobatics)
  finally:
    writer.close()
    env.close()


def _policy_payload(theta, normalizer, acrobatics):
  return {
    "W": theta[:90].reshape(6, 15).detach().cpu(),
    "b": theta[90:].detach().cpu(),
    "mu": normalizer.mean.float().detach().cpu(),
    "sd": normalizer.std.float().detach().cpu(),
    "obs_dim": 15,
    "act_dim": 6,
    "a_home": home_action().cpu(),
    "cfg": asdict(acrobatics),
  }


def _write_eval_metrics(writer, env, step, acrobatics):
  metrics = getattr(env, "acrobatics_last_metrics", {})
  for key in ("h_bar", "d_bar", "theta_bar", "peak_force", "n_air_steps",
              "terminated_early", "fall", "illegal_contact", "tilt"):
    if key in metrics:
      writer.add_scalar(f"Eval/{key}_mean", metrics[key].float().mean().item(), step)
  if "h_bar" in metrics:
    writer.add_scalar("Eval/height_gain_mean", (metrics["h_bar"] - acrobatics.h_stand).mean().item(), step)
  if "terminated_early" in metrics:
    writer.add_scalar("Eval/early_termination_rate", metrics["terminated_early"].float().mean().item(), step)
  if "n_air_steps" in metrics:
    success = metrics["n_air_steps"] >= 3
    if "terminated_early" in metrics:
      success &= ~metrics["terminated_early"]
    writer.add_scalar("Eval/jump_success_rate", success.float().mean().item(), step)
  if acrobatics.motion == "forward" and "d_bar" in metrics:
    distance_success = metrics["d_bar"] >= acrobatics.forward_success_distance
    writer.add_scalar("Eval/forward_distance_success_rate", distance_success.float().mean().item(), step)


if __name__ == "__main__":
  import mjlab
  task_id, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    args=sys.argv[1:], add_help=False, return_unknown_args=True, config=mjlab.TYRO_FLAGS,
  )
  main(tyro.cli(TrainARS, args=remaining, default=TrainARS(task_id=task_id)))

