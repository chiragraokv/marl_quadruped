"""Pure reward kernels and episode terminal reward equations."""

from __future__ import annotations

import torch

from src.tasks.acrobatics.cfg import AcrobaticsCfg
from src.tasks.acrobatics.mdp.observations import jump_observation
from src.tasks.acrobatics.mdp.phase import TAKEOFF
from src.tasks.acrobatics.mdp.trackers import get_state


def kernel(x: torch.Tensor, amplitude: float, decay: float) -> torch.Tensor:
  return amplitude * torch.exp(-decay * torch.abs(x))


def normalize_clip(value: torch.Tensor, maximum: float) -> torch.Tensor:
  if maximum <= 0:
    raise ValueError("maximum must be positive")
  return (value / maximum).clamp(0.0, 1.0)


def sparse_terminal_reward(
  motion: str,
  h_bar: torch.Tensor,
  d_bar: torch.Tensor,
  theta_bar: torch.Tensor,
  early: torch.Tensor,
  *,
  h_max: float,
  h_stand: float = 0.0,
  d_max: float,
  a_theta: float = 1.0,
  b_theta: float = 1.0,
  c_h: float = 0.0,
  a_d: float = 1.0,
  b_d: float = 1.0,
  c_d: float = 1.0,
  success_bonus: float = 0.1,
  failure_penalty: float = 0.08,
  crash_scale: float = 0.096,
) -> torch.Tensor:
  """Sparse terminal reward for in-place or forward jumps (paper Eqs. 6–7)."""
  if h_max <= h_stand:
    raise ValueError("h_max must exceed h_stand")
  # Standing height is not jump height: normalize the gain above home pose.
  hn = normalize_clip(h_bar - h_stand, h_max - h_stand)
  dn = normalize_clip(d_bar, d_max)
  theta_term = kernel(theta_bar, a_theta, b_theta)
  if motion == "in_place":
    end = (theta_term + kernel(dn, a_d, b_d) + c_h) * hn
    finish = torch.where(early, -(failure_penalty + crash_scale * hn), success_bonus * hn)
  elif motion == "forward":
    end = (theta_term + c_d * dn + c_h) * hn
    finish = torch.where(early, -(failure_penalty + crash_scale * (hn + dn)), success_bonus * (hn + dn))
  else:
    raise ValueError("motion must be 'in_place' or 'forward'")
  return end + finish


def imitation_reward(action: torch.Tensor, target: torch.Tensor, w_a: float = 1.0, w_b: float = 4.0) -> torch.Tensor:
  return w_a * torch.exp(-w_b * torch.square(action - target).sum(dim=-1))


def sparse_reward_term(env, acrobatics: AcrobaticsCfg) -> torch.Tensor:
  state = get_state(env)
  early = env.termination_manager.terminated
  ended = early | env.termination_manager.time_outs
  if not hasattr(env, "acrobatics_last_metrics"):
    env.acrobatics_last_metrics = {}
  for name, value in (("h_bar", state.h_bar), ("d_bar", state.d_bar),
                      ("theta_bar", state.theta_bar), ("peak_force", state.peak_force),
                      ("n_air_steps", state.n_air_steps), ("terminated_early", early)):
    previous = env.acrobatics_last_metrics.get(name, torch.zeros_like(value))
    env.acrobatics_last_metrics[name] = torch.where(ended, value, previous)
  for name in ("fall", "illegal_contact", "tilt"):
    value = env.termination_manager.get_term(name)
    previous = env.acrobatics_last_metrics.get(name, torch.zeros_like(value))
    env.acrobatics_last_metrics[name] = torch.where(ended, value, previous)
  reward = sparse_terminal_reward(
    acrobatics.motion, state.h_bar, state.d_bar, state.theta_bar, early,
    h_max=acrobatics.h_max, h_stand=acrobatics.h_stand, d_max=acrobatics.d_max,
    a_theta=acrobatics.sparse_theta_amplitude,
    b_theta=acrobatics.sparse_theta_decay,
    c_h=acrobatics.sparse_height_bias,
    a_d=acrobatics.sparse_distance_amplitude,
    b_d=acrobatics.sparse_distance_decay,
    c_d=acrobatics.sparse_forward_distance_weight,
    success_bonus=acrobatics.sparse_success_bonus,
    failure_penalty=acrobatics.sparse_failure_penalty,
    crash_scale=acrobatics.sparse_crash_scale,
  )
  return torch.where(ended, reward, torch.zeros_like(reward))


def _stage2_end_bonus(env, acrobatics: AcrobaticsCfg) -> tuple[torch.Tensor, torch.Tensor]:
  state = get_state(env)
  height_gain = (state.h_bar - acrobatics.h_stand).clamp_min(0.0)
  early = env.termination_manager.terminated
  ended = early | env.termination_manager.time_outs
  if acrobatics.motion == "in_place":
    bonus = torch.where(early, -acrobatics.stage2_failure_height_weight * height_gain, torch.zeros_like(state.h_bar))
  else:
    bonus = torch.where(early, torch.zeros_like(state.h_bar), acrobatics.stage2_forward_bonus_weight * (height_gain + state.d_bar))
  return torch.where(ended, bonus, torch.zeros_like(bonus)), early


def terminal_bonus(env, acrobatics: AcrobaticsCfg) -> torch.Tensor:
  return _stage2_end_bonus(env, acrobatics)[0]


def reward_height(env, acrobatics: AcrobaticsCfg) -> torch.Tensor:
  h = env.scene["robot"].data.root_link_pos_w[:, 2]
  return torch.where((h >= acrobatics.stage2_h_min) & (h <= acrobatics.stage2_h_max), acrobatics.stage2_height_weight * h, torch.zeros_like(h))


def reward_contact(env, acrobatics: AcrobaticsCfg, sensor_name: str = "feet_contact") -> torch.Tensor:
  contact = env.scene[sensor_name].data
  if contact.force is None:
    return torch.zeros(env.num_envs, device=env.device)
  forces = contact.force.reshape(env.num_envs, -1, 3).norm(dim=-1)
  counts = (forces > acrobatics.foot_force_count_epsilon).sum(dim=-1).clamp_min(1)
  per_foot = forces.sum(dim=-1) / counts
  return torch.where(per_foot >= acrobatics.stage2_contact_min,
                     -acrobatics.stage2_contact_weight * per_foot, torch.zeros_like(per_foot))


def reward_displacement(env, acrobatics: AcrobaticsCfg) -> torch.Tensor:
  state = get_state(env)
  if acrobatics.motion == "in_place":
    return kernel(state.d_current, acrobatics.stage2_distance_amplitude, acrobatics.stage2_distance_decay)
  return acrobatics.stage2_forward_distance_weight * state.d_current


def reward_smooth_torque(env) -> torch.Tensor:
  cfg = get_state(env).cfg
  return cfg.stage2_smooth_weight * torch.exp(-cfg.stage2_smooth_decay * get_state(env).torque_delta)


def reward_pitch(env) -> torch.Tensor:
  cfg = get_state(env).cfg
  pitch = env.action_manager.get_term("sagittal_pd").pitch
  return cfg.stage2_pitch_weight * torch.exp(-cfg.stage2_pitch_decay * pitch.abs())


def imitation_reward_term(env, acrobatics: AcrobaticsCfg, policy_path: str) -> torch.Tensor:
  # Load a frozen first-stage policy once per environment, then evaluate it batched.
  cached = getattr(env, "acrobatics_ars_policy", None)
  if cached is None:
    policy_path = (policy_path or acrobatics.ars_policy_path or
                   env.action_manager.get_term("sagittal_pd").acro.ars_policy_path)
    if not policy_path:
      raise ValueError("Set env.actions.sagittal_pd.acrobatics.ars_policy_path for imitation")
    cached = torch.load(policy_path, map_location=env.device, weights_only=False)
    env.acrobatics_ars_policy = cached
  obs = jump_observation(env, (0.0,) * 5, include_flight=False)
  w = cached["W"].to(obs)
  b = cached["b"].to(obs)
  mean, std = cached["mu"].to(obs), cached["sd"].to(obs)
  target = torch.nn.functional.linear((obs - mean) / (std + 1e-8), w, b).clamp(-1.0, 1.0)
  state = get_state(env)
  home = cached.get("a_home", ())
  home = torch.as_tensor(home, device=obs.device, dtype=obs.dtype)
  if home.numel() != 6:
    from src.tasks.acrobatics.mdp.sagittal import home_action
    home = home_action(device=obs.device, dtype=obs.dtype)
  target = torch.where((state.phase.phase == TAKEOFF).unsqueeze(-1), target, home)
  action = env.action_manager.get_term("sagittal_pd").raw_action
  return imitation_reward(action, target, acrobatics.w_a, acrobatics.w_b)
