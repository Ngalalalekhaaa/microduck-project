"""Plot fixed-speed MuJoCo evaluations for the latest HD-1910 checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "runs/20261001_hd1910_velocity_eval"
CASES = {
    "flat policy / flat physics": "flat9750_on_flat",
    "flat policy / backlash physics": "flat9750_on_backlash",
    "backlash policy / flat physics": "backlash9729_on_flat",
    "backlash policy / backlash physics": "backlash9729_on_backlash",
}
COLORS = ("#1565c0", "#42a5f5", "#e65100", "#ffb74d")


def load_reports() -> dict[str, dict]:
    reports = {}
    for name, directory in CASES.items():
        file = ROOT / directory / "evaluation.json"
        if not file.exists():
            raise FileNotFoundError(file)
        reports[name] = json.loads(file.read_text())
    return reports


def main() -> None:
    reports = load_reports()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot([0, .3], [0, .3], "k--", alpha=.6, label="command = actual")
    summary = {}
    for color, (name, report) in zip(COLORS, reports.items()):
        rows = sorted(report["results"].values(), key=lambda x: x["command_vx_vy_wz"][0])
        target = np.asarray([r["command_vx_vy_wz"][0] for r in rows])
        actual = np.asarray([r["mean_actual_vx_vy_wz"][0] for r in rows])
        survival = np.asarray([r["survival_fraction"] for r in rows])
        error = np.asarray([r["mean_abs_error_vx_vy_wz"][0] for r in rows])
        axes[0].plot(target, actual, "o-", color=color, label=name)
        axes[1].plot(target, error, "o-", color=color, label=name)
        summary[name] = {
            "mean_abs_vx_error_positive_speeds": float(error[target > 0].mean()),
            "min_survival_fraction": float(survival.min()),
            "target_vx": target.tolist(), "actual_vx": actual.tolist(),
            "survival_fraction": survival.tolist(),
        }
    axes[0].set_ylabel("Measured forward speed (m/s)")
    axes[1].set_ylabel("Mean absolute speed error (m/s)")
    axes[1].set_xlabel("Commanded forward speed (m/s)")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("HD-1910 MuJoCo velocity tracking: first episode, 64 starts, 10 s")
    fig.tight_layout()
    fig.savefig(ROOT / "velocity_tracking.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, target in zip(axes, (.10, .20)):
        key = f"vx_{target:+.2f}".replace(".", "p")
        ax.axhline(target, ls="--", color="black", alpha=.6, label="command")
        for color, (name, directory) in zip(COLORS, CASES.items()):
            trace = np.genfromtxt(ROOT / directory / f"{key}_trace.csv",
                                  delimiter=",", skip_header=1)
            ax.plot(trace[:, 0], trace[:, 2], color=color, lw=1.5, label=name)
        ax.set_ylabel(f"Actual vx (m/s)\ncommand {target:.2f}")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(ROOT / "velocity_traces_010_020.png", dpi=170)
    plt.close(fig)
    (ROOT / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
