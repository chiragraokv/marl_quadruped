"""Automatic post-training plots for MASQ experiments.

The plot set is deliberately limited to terms configured for the run.  This
prevents a reward revision from inheriting obsolete plots (for example V4's
survival ramp) and making a missing signal look like a zero-valued one.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PLOT_GROUPS = {
  "episode_length": (
    "Train/mean_episode_length",
    "Train/max_episode_length",
  ),
  "termination_counts": (
    "TerminationCount/time_out",
    "TerminationCount/fell_over",
    "TerminationCount/illegal_contact",
    "TerminationCount/fall_or_base_contact",
  ),
  "termination_rates": (
    "Episode_Termination/time_out",
    "Episode_Termination/fell_over",
    "Episode_Termination/illegal_contact",
  ),
}

GAIT_DIAGNOSTIC_PREFIXES = ("Metrics/gait_", "Metrics/contact_")
GAIT_DIAGNOSTIC_TAGS = (
  "Metrics/all_air_fraction",
  "Metrics/support_deficit",
  "Metrics/fall_rate",
  "Metrics/fall_termination_ramp",
  "Metrics/linear_velocity_error",
  "Metrics/angular_velocity_error",
)


def _plot_group(accumulator, tags, output, title):
  available = set(accumulator.Tags().get("scalars", ()))
  tags = [tag for tag in tags if tag in available]
  if not tags:
    return False
  figure, axes = plt.subplots(len(tags), 1, figsize=(11, 2.8 * len(tags)), squeeze=False)
  for axis, tag in zip(axes[:, 0], tags, strict=True):
    values = accumulator.Scalars(tag)
    axis.plot([value.step for value in values], [value.value for value in values])
    axis.set_ylabel(tag.split("/", 1)[-1])
    axis.grid(alpha=.2)
  axes[-1, 0].set_xlabel("PPO iteration")
  figure.suptitle(title.replace("_", " ").title())
  figure.tight_layout()
  figure.savefig(output, dpi=160)
  plt.close(figure)
  return True


def _current_reward_tags(scalar_tags: list[str]) -> list[str]:
  """Return only reward terms actually emitted by the current run."""
  return [tag for tag in scalar_tags if tag.startswith("Episode_Reward/")]


def _gait_diagnostic_tags(scalar_tags: list[str]) -> list[str]:
  """Return opt-in contact/gait metrics without plotting unrelated scalars."""
  return [
    tag for tag in scalar_tags
    if tag in GAIT_DIAGNOSTIC_TAGS or tag.startswith(GAIT_DIAGNOSTIC_PREFIXES)
  ]


def generate_experiment_plots(run_dir: str | Path) -> list[Path]:
  """Generate current-reward and gait/safety plots from one run directory.

  Reward terms are discovered from TensorBoard rather than kept in a historic
  hard-coded list.  Videos remain the source of truth for gait quality.
  """
  run_dir = Path(run_dir)
  output_dir = run_dir / "plots"
  output_dir.mkdir(parents=True, exist_ok=True)
  accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  accumulator.Reload()
  scalar_tags = sorted(accumulator.Tags().get("scalars", ()))
  if not scalar_tags:
    return []

  generated = []
  reward_tags = _current_reward_tags(scalar_tags)
  csv_path = output_dir / "current_reward_terms.csv"
  with csv_path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("tag", "iteration", "wall_time", "value"))
    for tag in reward_tags:
      for value in accumulator.Scalars(tag):
        writer.writerow((tag, value.step, value.wall_time, value.value))
  generated.append(csv_path)

  assigned = set()
  for group_name, tags in PLOT_GROUPS.items():
    output = output_dir / f"{group_name}.png"
    if _plot_group(accumulator, tags, output, group_name):
      generated.append(output)
      assigned.update(tags)

  reward_output = output_dir / "current_reward_terms.png"
  if _plot_group(accumulator, reward_tags, reward_output, "current_reward_terms"):
    generated.append(reward_output)
    assigned.update(reward_tags)

  gait_tags = _gait_diagnostic_tags(scalar_tags)
  metric_output = output_dir / "gait_diagnostics.png"
  if _plot_group(accumulator, gait_tags, metric_output, "gait_diagnostics"):
    generated.append(metric_output)

  return generated
