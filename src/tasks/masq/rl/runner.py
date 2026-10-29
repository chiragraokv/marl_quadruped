import wandb
import torch

from rsl_rl.utils import wandb_utils
from rsl_rl.utils.logger import Logger
from src.tasks.velocity.rl.runner import VelocityOnPolicyRunner
from .plots import generate_experiment_plots


class _CompatibleWandbSummaryWriter(wandb_utils.WandbSummaryWriter):
  """RSL-RL W&B writer compatible with releases that removed start_method."""

  def __init__(self, *args, **kwargs):
    original_settings = wandb.Settings

    def compatible_settings(*settings_args, **settings_kwargs):
      settings_kwargs.pop("start_method", None)
      return original_settings(*settings_args, **settings_kwargs)

    wandb.Settings = compatible_settings
    try:
      super().__init__(*args, **kwargs)
    finally:
      wandb.Settings = original_settings


class _MASQLogger(Logger):
  """Adds survival diagnostics that the stock RSL-RL logger does not expose."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._max_episode_length = 0.0
    self._termination_counts = {
      "time_out": 0.0, "fell_over": 0.0, "illegal_contact": 0.0,
    }

  def process_env_step(self, rewards, dones, extras, intrinsic_rewards=None):
    done_ids = (dones > 0).nonzero(as_tuple=False)
    if done_ids.numel() > 0:
      completed = self.cur_episode_length[done_ids][:, 0] + 1
      self._max_episode_length = max(
        self._max_episode_length, float(completed.max().item())
      )
    log = extras.get("log", extras.get("episode", {}))
    for name in self._termination_counts:
      value = log.get(f"Episode_Termination/{name}", 0.0)
      self._termination_counts[name] += float(
        value.item() if isinstance(value, torch.Tensor) else value
      )
    super().process_env_step(rewards, dones, extras, intrinsic_rewards)

  def log(self, *args, **kwargs):
    iteration = kwargs["it"]
    if self.writer is not None:
      self.writer.add_scalar(
        "Train/max_episode_length", self._max_episode_length, iteration
      )
      for name, count in self._termination_counts.items():
        self.writer.add_scalar(f"TerminationCount/{name}", count, iteration)
      fall_count = (
        self._termination_counts["fell_over"]
        + self._termination_counts["illegal_contact"]
      )
      self.writer.add_scalar(
        "TerminationCount/fall_or_base_contact", fall_count, iteration
      )
      # Keep the requested abbreviated tracking tag alongside the existing
      # RewardManager name without changing the actual reward term.
      track_values = []
      for info in self.ep_extras:
        value = info.get("Episode_Reward/track_linear_velocity")
        if value is not None:
          track_values.append(float(value.item() if isinstance(value, torch.Tensor) else value))
      if track_values:
        self.writer.add_scalar(
          "Episode_Reward/track_lin_vel",
          sum(track_values) / len(track_values), iteration,
        )
    super().log(*args, **kwargs)
    self._max_episode_length = 0.0
    self._termination_counts = {name: 0.0 for name in self._termination_counts}

class MASQOnPolicyRunner(VelocityOnPolicyRunner):
  """MASQ runner with folded actor export and W&B experiment logging."""

  def __init__(self, *args, **kwargs):
    # Logger imports this symbol lazily when learn() starts.
    wandb_utils.WandbSummaryWriter = _CompatibleWandbSummaryWriter
    super().__init__(*args, **kwargs)
    self.logger = _MASQLogger(
      log_dir=self.logger.log_dir,
      cfg=self.cfg,
      env_cfg=self.env.cfg,
      num_envs=self.env.num_envs,
      is_distributed=self.is_distributed,
      gpu_world_size=self.gpu_world_size,
      gpu_global_rank=self.gpu_global_rank,
      device=self.device,
    )

  def learn(self, *args, **kwargs):
    super().learn(*args, **kwargs)
    if self.gpu_global_rank == 0 and self.logger.log_dir is not None:
      generated = generate_experiment_plots(self.logger.log_dir)
      print(f"[INFO] Generated {len(generated)} experiment artifacts in "
            f"{self.logger.log_dir}/plots")

  def save(self, path: str, infos=None):
    checkpoint_infos = dict(infos or {})
    checkpoint_infos["masq_common_step_counter"] = int(
      self.env.unwrapped.common_step_counter
    )
    super().save(path, checkpoint_infos)

  def load(self, path: str, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    if isinstance(infos, dict) and "masq_common_step_counter" in infos:
      self.env.unwrapped.common_step_counter = int(
        infos["masq_common_step_counter"]
      )
    else:
      # Legacy checkpoints have no environment state. Reconstruct the ramp
      # position from the saved PPO iteration instead of restarting at zero.
      self.env.unwrapped.common_step_counter = (
        self.current_learning_iteration * self.cfg["num_steps_per_env"]
      )
    return infos
