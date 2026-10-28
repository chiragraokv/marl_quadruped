"""MASQ-specific disturbance hooks.

These functions intentionally use mjlab's public entity/data interfaces so
they remain compatible with the installed 1.2.x API. Restitution is exposed
as a startup hook for versions that expose geom solref directly.
"""
import torch

def gravity_impulse(env, force_range=(-30.0, 30.0), asset_name="robot"):
  robot = env.scene[asset_name]
  force = torch.empty(env.num_envs, 3, device=env.device).uniform_(*force_range)
  robot.data.external_force_b[:] = force[:, None, :]

def randomize_restitution(env, ranges=(0.0, 0.5), asset_name="robot"):
  """Best-effort restitution hook; no-op when the backend lacks geom fields."""
  robot = env.scene[asset_name]
  if hasattr(robot, "write_geom_restitution_to_sim"):
    values = torch.empty(robot.num_geoms, device=env.device).uniform_(*ranges)
    robot.write_geom_restitution_to_sim(values)
