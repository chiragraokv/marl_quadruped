"""Batched per-episode state for jumps and evaluation metrics."""

from __future__ import annotations

import torch

from src.tasks.acrobatics.mdp.phase import PhaseTracker


class AcrobaticsState:
  def __init__(self, env, cfg):
    self.num_envs = env.num_envs
    self.device = torch.device(env.device)
    self.cfg = cfg
    self.phase = PhaseTracker(
      self.num_envs, self.device, repeat=cfg.repeat,
      landing_settle_steps=cfg.landing_settle_steps,
      airborne_debounce_steps=cfg.airborne_debounce_steps,
    )
    z = torch.zeros(self.num_envs, device=self.device)
    self.h_bar = z.clone()
    self.d_bar = z.clone()
    self.d_current = z.clone()
    self.theta_bar = z.clone()
    self.peak_force = z.clone()
    self.n_air_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.terminated_early = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.initial_xy = torch.zeros((self.num_envs, 2), device=self.device)
    self.initial_forward = torch.zeros((self.num_envs, 2), device=self.device)
    self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.previous_torque = torch.zeros((self.num_envs, 6), device=self.device)
    self.torque_delta = z.clone()
    self.last_contact = z.clone()

  def reset(self, env_ids):
    if env_ids is None:
      env_ids = slice(None)
    self.phase.reset(env_ids)
    for value in (self.h_bar, self.d_bar, self.d_current, self.theta_bar, self.peak_force, self.n_air_steps,
                  self.terminated_early, self.initial_xy, self.initial_forward, self.initialized,
                  self.previous_torque, self.torque_delta, self.last_contact):
      value[env_ids] = 0

  def update(self, h, x, y, forward_axis, pitch, hdot, contact_force, torque):
    fresh = ~self.initialized
    self.initial_xy[fresh, 0] = x[fresh]
    self.initial_xy[fresh, 1] = y[fresh]
    self.initial_forward[fresh] = forward_axis[fresh]
    self.initialized[fresh] = True
    self.h_bar = torch.maximum(self.h_bar, h)
    forward_delta = x - self.initial_xy[:, 0]
    lateral_delta = y - self.initial_xy[:, 1]
    signed_forward = forward_delta * self.initial_forward[:, 0] + lateral_delta * self.initial_forward[:, 1]
    if self.cfg.motion == "forward":
      displacement = signed_forward.clamp_min(0.0)
    else:
      displacement = signed_forward.abs()
    self.d_current = displacement
    self.d_bar = torch.maximum(self.d_bar, displacement)
    self.theta_bar = torch.maximum(self.theta_bar, pitch.abs())
    self.peak_force = torch.maximum(self.peak_force, contact_force)
    self.last_contact = contact_force
    self.n_air_steps += (contact_force < self.cfg.contact_force_epsilon).to(torch.long)
    self.torque_delta = (torque - self.previous_torque[:, :torque.shape[-1]]).norm(dim=-1)
    self.previous_torque[:, :torque.shape[-1]].copy_(torque)
    self.phase.update(contact_force, hdot, self.cfg.contact_force_epsilon)


def get_state(env) -> AcrobaticsState:
  state = getattr(env, "acrobatics_state", None)
  if state is None:
    raise RuntimeError("Acrobatics state is not initialized by the sagittal action term")
  return state
