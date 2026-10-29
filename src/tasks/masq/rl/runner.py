import wandb

from rsl_rl.utils import wandb_utils
from src.tasks.velocity.rl.runner import VelocityOnPolicyRunner


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

class MASQOnPolicyRunner(VelocityOnPolicyRunner):
  """MASQ runner with folded actor export and W&B experiment logging."""

  def __init__(self, *args, **kwargs):
    # Logger imports this symbol lazily when learn() starts.
    wandb_utils.WandbSummaryWriter = _CompatibleWandbSummaryWriter
    super().__init__(*args, **kwargs)
