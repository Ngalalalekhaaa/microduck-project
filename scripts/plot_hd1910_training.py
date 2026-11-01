"""Plot the HD-1910 TensorBoard training history across resumed runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/20261001_hd1910_velocity_eval"
TAGS = {
    "mean_reward": "Train/mean_reward",
    "velocity_reward": "Episode_Reward/track_linear_velocity",
    "training_velocity_error": "Metrics/twist/error_vel_xy",
}


def history(variant: str) -> dict[str, dict[int, float]]:
    runs = ROOT / "microduck_rl_hd1910/logs/rsl_rl" / f"local_hd1910_{variant}_train"
    series: dict[str, dict[int, float]] = {name: {} for name in TAGS}
    for file in sorted(runs.glob("*/events.out.tfevents*")):
        event = event_accumulator.EventAccumulator(str(file), size_guidance={"scalars": 0})
        event.Reload()
        available = set(event.Tags()["scalars"])
        # Resuming starts with an empty episode-statistics buffer. Its first
        # values are low even when the loaded policy has not degraded.
        resumed = "continued_" in file.parent.name or "continuous_" in file.parent.name
        for name, tag in TAGS.items():
            if tag in available:
                points = event.Scalars(tag)
                warmup_end = points[0].step + 75 if resumed else points[0].step
                series[name].update({point.step: point.value for point in points
                                     if point.step >= warmup_end})
    return series


def smooth(values: np.ndarray, window: int = 100) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    middle = np.convolve(values, kernel, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), middle])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    histories = {v: history(v) for v in ("flat", "backlash")}
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    colors = {"flat": "#1565c0", "backlash": "#e65100"}
    summary = {}
    for variant, series in histories.items():
        steps = sorted(set().union(*(set(s) for s in series.values())))
        with (OUTPUT / f"{variant}_training_curve.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["iteration", *TAGS])
            for step in steps:
                writer.writerow([step, *(series[name].get(step, "") for name in TAGS)])
        summary[variant] = {}
        for axis, (name, values) in zip(axes, series.items()):
            x = np.asarray(sorted(values), dtype=int)
            y = np.asarray([values[step] for step in x], dtype=float)
            axis.plot(x, smooth(y), color=colors[variant], label=variant, lw=1.7)
            summary[variant][name] = {
                "last_iteration": int(x[-1]),
                "last_value": float(y[-1]),
                "last_100_mean": float(y[-100:].mean()),
            }
    for axis, label in zip(axes, ["Mean training reward", "Linear velocity reward", "Training velocity error"]):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("PPO iteration")
    fig.suptitle("HD-1910 training history (100-iteration moving average)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "training_curves.png", dpi=170)
    plt.close(fig)
    (OUTPUT / "training_curve_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
