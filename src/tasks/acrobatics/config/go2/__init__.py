"""Register the six rigid Go2 jump task variants."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.acrobatics.env_cfg import make_env_cfg
from src.tasks.acrobatics.rl_cfg import acrobatics_ppo_runner_cfg


for _motion, _suffix in (("in_place", "JumpInPlace"), ("forward", "JumpForward")):
  for _stage in ("ars", "imitate", "refine"):
    _stage_name = {"ars": "ARS", "imitate": "Imitate", "refine": "Refine"}[_stage]
    register_mjlab_task(
      task_id=f"Unitree-Go2-{_suffix}-{_stage_name}",
      env_cfg=make_env_cfg(_motion, _stage),
      play_env_cfg=make_env_cfg(_motion, _stage, play=True),
      rl_cfg=acrobatics_ppo_runner_cfg(_stage),
    )
