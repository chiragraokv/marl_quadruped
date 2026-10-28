from dataclasses import dataclass
import math
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg


GAIT_OFFSETS = torch.tensor(((0., .5, 0., .5), (0., .5, .5, 0.),
                             (0., 0., .5, .5), (0., 0., 0., 0.)))


class GaitCommand(UniformVelocityCommand):
  """Velocity command augmented with a concurrently-trained gait signal."""
  cfg: "GaitCommandCfg"

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.height_cmd = torch.zeros(self.num_envs, 1, device=self.device)
    self.gait_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.cadence = torch.ones(self.num_envs, 1, device=self.device)
    self.phase = torch.zeros(self.num_envs, 1, device=self.device)
    self.phase_offsets = torch.zeros(self.num_envs, 4, device=self.device)
    self.reward_ema = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self):
    gait = torch.nn.functional.one_hot(self.gait_id, 4).float()
    # 3 velocity + height + gait one-hot + cadence + phase + offsets + reserved.
    return torch.cat((self.vel_command_b, self.height_cmd, gait, self.cadence,
                      self.phase, self.phase_offsets,
                      torch.zeros(self.num_envs, 1, device=self.device)), dim=-1)

  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    self.height_cmd[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(
        *self.cfg.height_range)
    self.gait_id[env_ids] = torch.randint(0, 4, (len(env_ids),), device=self.device)
    self.cadence[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(
        *self.cfg.cadence_range)
    self.phase[env_ids] = 0
    self._update_offsets(env_ids)

  def _update_offsets(self, env_ids):
    self.phase_offsets[env_ids] = GAIT_OFFSETS.to(self.device)[self.gait_id[env_ids]]

  def compute(self, dt):
    super().compute(dt)
    self.phase = (self.phase + dt * self.cadence) % 1.0


@dataclass(kw_only=True)
class GaitCommandCfg(CommandTermCfg):
  entity_name: str
  ranges: UniformVelocityCommandCfg.Ranges
  resampling_time_range: tuple[float, float] = (3.0, 8.0)
  rel_standing_envs: float = .05
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_heading_envs: float = 1.0
  init_velocity_prob: float = 0.0
  height_range: tuple[float, float] = (.28, .42)
  cadence_range: tuple[float, float] = (1.0, 2.5)

  def build(self, env):
    return GaitCommand(self, env)
