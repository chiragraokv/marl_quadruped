"""Plot MASQ survival diagnostics from an RSL-RL TensorBoard run."""

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

  accumulator = EventAccumulator(str(args.run_dir))
  accumulator.Reload()

  mean_x, mean_length = series(
    accumulator, "Train/mean_episode_length", args.iterations
  )
  max_x, max_length = series(
    accumulator, "Train/max_episode_length", args.iterations
  )
  survival_x, survival = series(
    accumulator, "Episode_Reward/survival", args.iterations
  )
  ramp_x, ramp = series(accumulator, "Metrics/survival_ramp", args.iterations)

  figure, left = plt.subplots(figsize=(10, 5.5))
  left.plot(mean_x, mean_length, label="Mean episode length", color="#1769aa")
  left.plot(max_x, max_length, label="Max episode length", color="#70a1d7", alpha=.7)
  left.set_xlabel("PPO iteration")
  left.set_ylabel("Episode length (environment steps)", color="#1769aa")
  left.tick_params(axis="y", labelcolor="#1769aa")
  left.grid(alpha=.2)

  right = left.twinx()
  right.plot(
    survival_x, [value * 1000 for value in survival],
    label="Episode survival reward ×1000", color="#d1495b"
  )
  right.plot(ramp_x, ramp, label="Survival ramp", color="#f28e2b", linestyle="--")
  right.set_ylabel("Survival reward ×1000 / ramp", color="#d1495b")
  right.tick_params(axis="y", labelcolor="#d1495b")

  lines = left.lines + right.lines
  left.legend(lines, [line.get_label() for line in lines], loc="upper right")
  left.set_title("MASQ survival experiment: first 500 iterations")
  figure.tight_layout()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
  main()
