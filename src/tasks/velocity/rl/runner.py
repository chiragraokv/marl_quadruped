import os

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from src.tasks.velocity import mdp


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)

    # The environment reward is the proxy reward used by PPO.  The raw reward
    # terms are kept as a separate observational stream for reward auditing.
    manager = self.env.unwrapped.reward_manager
    self._reward_names = list(
      getattr(manager, "_term_names", [f"term_{i}" for i in range(len(manager._term_cfgs))])
    )
    self._proxy_reward_sum = torch.zeros(env.num_envs, device=env.device)
    self._true_reward_sum = torch.zeros(env.num_envs, device=env.device)
    self._weighted_term_sums = torch.zeros(
      (env.num_envs, len(self._reward_names)), device=env.device
    )
    self._raw_term_sums = torch.zeros_like(self._weighted_term_sums)
    original_process = self.logger.process_env_step

    def process_env_step(rewards, dones, extras, intrinsic_rewards=None):
      weighted_terms = manager._step_reward
      weights = torch.tensor(
        [cfg.weight for cfg in manager._term_cfgs],
        device=weighted_terms.device,
        dtype=weighted_terms.dtype,
      )
      raw_terms = torch.where(
        weights.unsqueeze(0) != 0, weighted_terms / weights.unsqueeze(0),
        torch.zeros_like(weighted_terms),
      )
      # track_lin_vel_xy is replaced by speed_proxy for PPO. Reinsert the
      # commanded-direction term here so true reward remains evaluation-only.
      if "track_lin_vel_xy" in self._reward_names:
        idx = self._reward_names.index("track_lin_vel_xy")
        true_value = mdp.track_linear_velocity_v4(
          self.env.unwrapped, sigma=0.25, threshold=0.2, command_name="twist"
        )
        raw_terms[:, idx] = true_value * self.env.unwrapped.step_dt
      self._proxy_reward_sum += rewards
      self._weighted_term_sums += weighted_terms
      self._raw_term_sums += raw_terms
      # raw_terms already includes dt. Apply the true reward scales before
      # accumulating the evaluation-only true return.
      self._true_reward_sum += (raw_terms * weights.unsqueeze(0)).sum(dim=1)
      done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
      if done_ids.numel():
        log = extras.setdefault("log", {})
        log["Reward/proxy"] = self._proxy_reward_sum[done_ids].mean().detach()
        log["Reward/true"] = self._true_reward_sum[done_ids].mean().detach()
        for idx, name in enumerate(self._reward_names):
          tag = "proxy" if name == "track_lin_vel_xy" else "shared"
          log[f"Reward/{tag}/weighted/{name}"] = self._weighted_term_sums[done_ids, idx].mean().detach()
          log[f"Reward/{tag}/raw/{name}"] = self._raw_term_sums[done_ids, idx].mean().detach()
          if name == "track_lin_vel_xy":
            log["Reward/true/raw/track_lin_vel_xy"] = self._raw_term_sums[done_ids, idx].mean().detach()
            log["Reward/true/weighted/track_lin_vel_xy"] = (
              self._raw_term_sums[done_ids, idx] * weights[idx]
            ).mean().detach()
        self._proxy_reward_sum[done_ids] = 0.0
        self._true_reward_sum[done_ids] = 0.0
        self._weighted_term_sums[done_ids] = 0.0
        self._raw_term_sums[done_ids] = 0.0
      original_process(rewards, dones, extras, intrinsic_rewards)

    self.logger.process_env_step = process_env_step

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
