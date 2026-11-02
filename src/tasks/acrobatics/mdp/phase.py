"""Batched takeoff, flight, and landing state machine."""

from __future__ import annotations

from dataclasses import dataclass

import torch

TAKEOFF, FLIGHT, LANDING = 0, 1, 2


@dataclass
class PhaseTracker:
  num_envs: int
  device: torch.device | str = "cpu"
  repeat: bool = False
  landing_settle_steps: int = 2
  airborne_debounce_steps: int = 1

  def __post_init__(self):
    self.device = torch.device(self.device)
    self.phase = torch.full((self.num_envs,), TAKEOFF, dtype=torch.long, device=self.device)
    self.land_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.air_counter = torch.zeros_like(self.land_counter)
    self.jump_count = torch.zeros_like(self.land_counter)

  @property
  def flight_flag(self) -> torch.Tensor:
    return (self.phase == FLIGHT).to(torch.float32)

  def update(self, contact_force: torch.Tensor, hdot: torch.Tensor, force_epsilon: float = 1.0) -> torch.Tensor:
    contact_force = contact_force.to(self.device).reshape(self.num_envs)
    hdot = hdot.to(self.device).reshape(self.num_envs)
    airborne = contact_force < force_epsilon
    self.air_counter = torch.where(airborne, self.air_counter + 1, torch.zeros_like(self.air_counter))
    stable_air = self.air_counter >= max(self.airborne_debounce_steps, 1)
    stable_contact = ~airborne
    self.land_counter = torch.where(self.phase == LANDING, self.land_counter + 1, torch.zeros_like(self.land_counter))
    to_flight = (self.phase == TAKEOFF) & stable_air
    to_landing = (self.phase == FLIGHT) & (hdot < 0) & stable_contact
    to_takeoff = self.repeat & (self.phase == LANDING) & (self.land_counter >= self.landing_settle_steps)
    self.jump_count += to_landing.to(torch.long)
    self.phase = torch.where(to_flight, FLIGHT, self.phase)
    self.phase = torch.where(to_landing, LANDING, self.phase)
    self.phase = torch.where(to_takeoff, TAKEOFF, self.phase)
    return self.phase

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.phase[env_ids] = TAKEOFF
    self.land_counter[env_ids] = 0
    self.air_counter[env_ids] = 0
    self.jump_count[env_ids] = 0
