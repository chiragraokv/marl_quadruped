"""Plot current MASQ reward terms and episode length from a TensorBoard run.

The filename is retained for callers of the former V4 utility.  It no longer
looks for V4-only survival/ramp scalars.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def series(accumulator, tag, limit):
  values = [value for value in accumulator.Scalars(tag) if value.step < limit]
  return [value.step + 1 for value in values], [value.value for value in values]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("run_dir", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--iterations", type=int, default=500)
  args = parser.parse_args()

  accumulator = EventAccumulator(str(args.run_dir), size_guidance={"scalars": 0})
  accumulator.Reload()
  available = set(accumulator.Tags().get("scalars", ()))
  reward_tags = sorted(
    tag for tag in available if tag.startswith("Episode_Reward/")
  )
  length_tags = [
    tag for tag in ("Train/mean_episode_length", "Train/max_episode_length")
    if tag in available
  ]
  if not reward_tags and not length_tags:
    raise ValueError("No current reward or episode-length scalars found")

  tags = length_tags + reward_tags
  figure, axes = plt.subplots(
    len(tags), 1, figsize=(11, 2.8 * len(tags)), squeeze=False
  )
  for axis, tag in zip(axes[:, 0], tags, strict=True):
    x_values, y_values = series(accumulator, tag, args.iterations)
    axis.plot(x_values, y_values)
    axis.set_ylabel(tag.split("/", 1)[-1])
    axis.grid(alpha=.2)
  axes[-1, 0].set_xlabel("PPO iteration")
  figure.suptitle("MASQ current rewards and episode length")
  figure.tight_layout()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(args.output, dpi=180)
  plt.close(figure)


if __name__ == "__main__":
  main()
