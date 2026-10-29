"""Automatic post-training plots for MASQ experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PLOT_GROUPS = {
  "episode_performance": (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Train/max_episode_length",
  ),
  "survival_and_tracking": (
    "Episode_Reward/survival",
    "Episode_Reward/track_lin_vel",
    "Episode_Reward/track_angular_velocity",
    "Metrics/survival_ramp",
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
  "losses": (
    "Loss/value",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",
  ),
  "policy_and_performance": (
    "Policy/mean_std",
    "Perf/total_fps",
    "Perf/collection_time",
    "Perf/learning_time",
  ),
}


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


def generate_experiment_plots(run_dir: str | Path) -> list[Path]:
  """Generate grouped PNGs and a complete scalar CSV from one run directory."""
  run_dir = Path(run_dir)
  output_dir = run_dir / "plots"
  output_dir.mkdir(parents=True, exist_ok=True)
  accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  accumulator.Reload()
  scalar_tags = sorted(accumulator.Tags().get("scalars", ()))
  if not scalar_tags:
    return []

  generated = []
  csv_path = output_dir / "all_scalars.csv"
  with csv_path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("tag", "iteration", "wall_time", "value"))
    for tag in scalar_tags:
      for value in accumulator.Scalars(tag):
        writer.writerow((tag, value.step, value.wall_time, value.value))
  generated.append(csv_path)

  assigned = set()
  for group_name, tags in PLOT_GROUPS.items():
    output = output_dir / f"{group_name}.png"
    if _plot_group(accumulator, tags, output, group_name):
      generated.append(output)
      assigned.update(tags)

  reward_tags = [tag for tag in scalar_tags if tag.startswith("Episode_Reward/")]
  reward_output = output_dir / "all_reward_terms.png"
  if _plot_group(accumulator, reward_tags, reward_output, "all_reward_terms"):
    generated.append(reward_output)
    assigned.update(reward_tags)

  metric_tags = [
    tag for tag in scalar_tags
    if (tag.startswith("Metrics/") or tag.startswith("Episode_Metrics/"))
    and tag not in assigned
  ]
  metric_output = output_dir / "all_metrics.png"
  if _plot_group(accumulator, metric_tags, metric_output, "all_metrics"):
    generated.append(metric_output)

  return generated
