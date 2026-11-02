import torch

from src.tasks.acrobatics.mdp.phase import FLIGHT, LANDING, TAKEOFF, PhaseTracker


def test_phase_transitions_repeat_and_reset():
  tracker = PhaseTracker(1, repeat=True, landing_settle_steps=2)
  tracker.update(torch.tensor([0.0]), torch.tensor([0.5]))
  assert tracker.phase.item() == FLIGHT
  assert tracker.flight_flag.item() == 1
  tracker.update(torch.tensor([4.0]), torch.tensor([-0.1]))
  assert tracker.phase.item() == LANDING
  assert tracker.jump_count.item() == 1
  tracker.update(torch.tensor([4.0]), torch.tensor([0.0]))
  tracker.update(torch.tensor([4.0]), torch.tensor([0.0]))
  assert tracker.phase.item() == TAKEOFF
  tracker.reset()
  assert tracker.phase.item() == TAKEOFF
  assert tracker.jump_count.item() == 0
