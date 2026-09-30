"""Auto-select the latest local ``model_*.pt`` when `play` omits a checkpoint.

mjlab's play CLI requires ``--wandb-run-path`` unless ``--checkpoint-file`` is
set. XGODUCK (and any local TensorBoard run) saves checkpoints under
``logs/rsl_rl/<experiment_name>/``, so this hook injects the newest ``.pt``
into ``sys.argv`` before tyro parses.

Called at the end of ``mjlab_microduck.tasks`` import (after task registration)
so ``load_rl_cfg`` works. A no-op unless argv[0] is ``play``. Explicit
``--checkpoint-file`` / ``--wandb-run-path`` still win.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _invoked_as_play() -> bool:
    prog = Path(sys.argv[0]).name
    return prog.removesuffix(".py").removesuffix("-script") == "play"


def _flag_present(args: list[str], name: str) -> bool:
    prefix = f"--{name}"
    return any(a == prefix or a.startswith(prefix + "=") for a in args)


def _flag_value(args: list[str], name: str) -> str | None:
    prefix = f"--{name}"
    for i, a in enumerate(args):
        if a == prefix and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(prefix + "="):
            return a.split("=", 1)[1]
    return None


def find_latest_checkpoint(log_root: Path) -> Path:
    """Newest ``model_<iter>.pt`` under ``log_root/<run>/``.

    Run directories are timestamp-prefixed, so lexicographic max is latest.
    Within a run, the highest iteration number wins.
    """
    pts = [
        p
        for p in log_root.glob("*/model_*.pt")
        if p.is_file() and p.parent.name != "wandb_checkpoints"
    ]
    if not pts:
        raise FileNotFoundError(
            f"No model_*.pt under {log_root}. Train first, or pass "
            "--checkpoint-file / --wandb-run-path."
        )

    def _key(p: Path) -> tuple[str, int]:
        try:
            iteration = int(p.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            iteration = -1
        return (p.parent.name, iteration)

    return max(pts, key=_key)


def maybe_inject_latest_checkpoint(argv: list[str] | None = None) -> list[str]:
    """If this is a trained `play` with no checkpoint flag, inject the latest pt.

    Returns the (possibly modified) argv. Mutates ``sys.argv`` when ``argv`` is
    None (the import-time path).
    """
    owned = argv is None
    argv = list(sys.argv if owned else argv)

    if owned and not _invoked_as_play():
        return argv
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        return argv

    rest = argv[2:]
    if any(a in ("-h", "--help") for a in rest):
        return argv
    if _flag_present(rest, "checkpoint-file") or _flag_present(rest, "wandb-run-path"):
        return argv
    agent = _flag_value(rest, "agent")
    if agent in ("zero", "random"):
        return argv

    task_id = argv[1]
    from mjlab.tasks.registry import load_rl_cfg

    try:
        agent_cfg = load_rl_cfg(task_id)
    except KeyError:
        return argv

    log_root = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    try:
        ckpt = find_latest_checkpoint(log_root)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Auto-selected latest checkpoint: {ckpt}")
    argv = [argv[0], task_id, "--checkpoint-file", str(ckpt), *rest]
    if owned:
        sys.argv = argv
    return argv
