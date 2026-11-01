#!/usr/bin/env python3
"""离线检查六模型，预览/保存选择，或显式启动现有 Xbox 流程。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from microduck_deploy.config import (
    calibration_arrays, load_config, require_calibration, save_config,
)
from microduck_deploy.policy import HOME, JOINT_NAMES, OnnxPolicy, q_to_ticks

KINDS = {"walk": "walking", "roulade": "roulade", "sitstand_original": "sitstand",
         "sitstand_friction": "sitstand", "sitstand_com": "sitstand", "ground_pick": "ground_pick"}
MOTION_SCRIPTS = {
    "policy_trial.py", "home_then_policy.py", "supported_home.py", "straight_then_home.py",
    "xbox_autostart.py", "xbox_model_hd1910.sh", "xbox_walk.sh", "xbox_friction_hd1910.sh",
    "xbox_friction_5000_hd1910.sh", "imu_home_assist.py", "adjust_joint11_home.py",
    "apply_joint11_live.py", "readpose.py", "benchmark_control_readonly.py", "check_policy_readonly.py",
}
TARGET_MODE_NOTE = (
    "run 继承 xbox_model_hd1910.sh 的显式 --log-only-target-angles："
    "目标角范围和单步目标跳变仅记录，不拦截、不裁剪；"
    "单圈编码器、实际关节范围、IMU/通信/时效及动作专用保护仍强制执行。"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(root, value):
    if not isinstance(value, str) or not value or any(x in value for x in "\0\n\r"):
        raise ValueError("目录清单路径须为非空字符串")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("目录清单仅允许部署目录内的相对路径")
    resolved = (root / rel).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("目录清单路径不能跳出部署目录")
    return resolved


def load_catalog(root=ROOT):
    root = Path(root)
    catalog = json.loads((root / "models/suite_20261028.json").read_text())
    if catalog.get("schema_version") != 1 or set(catalog.get("models", {})) != set(KINDS):
        raise ValueError("模型清单版本或六模型键不匹配")
    for name, kind in KINDS.items():
        item = catalog["models"][name]
        if item.get("kind") != kind:
            raise ValueError(f"{name}: 模型类别不匹配")
        for key in ("path", "samples"):
            local_path(root, item.get(key))
        for key in ("sha256", "samples_sha256"):
            value = item.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name}: {key} 无效")
    return catalog


def check_calibration(cfg):
    require_calibration(cfg, imu=True)
    zero, direction = calibration_arrays(cfg)
    ticks = q_to_ticks(HOME, zero, direction)
    rotation = np.asarray(cfg["imu"]["mounting_rotation"], dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
            or np.linalg.det(rotation) < .99):
        raise ValueError("已保存的 IMU 轴映射不是右手正交旋转")
    return {"status": "ok", "home_ticks": ticks.tolist(), "hardware_read": False}


def feasibility(item, cfg):
    """Report sampled encoder goals; unused policies never block a CPU check."""
    warnings = []
    if cfg is None:
        return warnings
    if "observed_target_min" not in item or "observed_target_max" not in item:
        return ["没有保存目标范围；仅验证契约和推理，未验证硬件可表示范围。"]
    low, high = (np.asarray(item[k], dtype=float) for k in ("observed_target_min", "observed_target_max"))
    if low.shape != (14,) or high.shape != (14,) or not np.isfinite([low, high]).all() or np.any(low > high):
        raise ValueError("清单 observed_target_min/max 无效")
    zero, direction = calibration_arrays(cfg)
    raw = zero[None, :] + direction[None, :] * np.array([low, high]) * (4096 / (2*np.pi))
    bad = np.flatnonzero(np.any((raw < 0) | (raw > 4095), axis=0))
    if len(bad):
        warnings.append("已保存仿真样本的原始目标可能超出当前单圈编码器：" + ", ".join(JOINT_NAMES[i] for i in bad)
                        + "。这只是可行性提示；未选模型不影响检查，运行时仍按各动作的实际保护拒绝/约束目标。")
    return warnings


def check_suite(*, root=ROOT, cfg=None):
    root = Path(root)
    started = time.perf_counter()
    catalog = load_catalog(root)
    calibration = check_calibration(cfg) if cfg is not None else {"status": "not_supplied", "hardware_read": False}
    result = {"status": "ok", "hardware_access": False, "calibration": calibration, "models": {},
              "parity_tolerance": {"rtol": 3e-4, "atol": 3e-5},
              "scope": "SHA256、ONNX Runtime CPU 图加载/契约和保存样本推理一致性；不代表真机动作通过。"}
    for name, item in catalog["models"].items():
        begin = time.perf_counter()
        try:
            model, samples = (local_path(root, item[key]) for key in ("path", "samples"))
            if digest(model) != item["sha256"]:
                raise ValueError("模型 SHA256 不匹配")
            if digest(samples) != item["samples_sha256"]:
                raise ValueError("样本 SHA256 不匹配")
            policy = OnnxPolicy(model, policy_kind=item["kind"])
            if policy.session.get_providers() != ["CPUExecutionProvider"]:
                raise ValueError("推理未使用纯 CPU provider")
            load_ms = (time.perf_counter() - begin) * 1000
            with np.load(samples, allow_pickle=False) as saved:
                obs, expected = saved["obs"], saved["action"]
            if (obs.ndim != 2 or obs.shape[1] != 61 or len(obs) == 0
                    or expected.shape != (len(obs), 14) or not np.isfinite(obs).all()
                    or not np.isfinite(expected).all()):
                raise ValueError("样本必须是有限 obs[N,61] 和 action[N,14]")
            durations, outputs = [], []
            for observation in obs:
                tick = time.perf_counter()
                outputs.append(policy.infer(observation))
                durations.append((time.perf_counter()-tick)*1000)
            actual = np.asarray(outputs)
            if not np.allclose(actual, expected, rtol=3e-4, atol=3e-5):
                raise ValueError(f"样本输出不一致，最大绝对误差 {float(np.max(np.abs(actual-expected))):.8g}")
            result["models"][name] = {"status": "ok", "kind": item["kind"], "path": item["path"],
                                      "sha256": item["sha256"], "samples_count": len(obs),
                                      "max_abs_error": float(np.max(np.abs(actual-expected))),
                                      "load_ms": load_ms, "infer_mean_ms": float(np.mean(durations)),
                                      "infer_p95_ms": float(np.percentile(durations, 95)),
                                      "warnings": feasibility(item, cfg)}
        except Exception as exc:
            result["status"] = "failed"
            result["models"][name] = {"status": "failed", "error": str(exc)}
    result["elapsed_ms"] = (time.perf_counter()-started)*1000
    return result


def selection(cfg, catalog, *, walking="keep", sitstand=None, roulade=None, four_actions=False):
    selected = copy.deepcopy(cfg)
    models = catalog["models"]
    if four_actions:
        if walking != "keep":
            raise ValueError("四动作遥操从当前行走模型开始，用十字键↑切换；请保留 --walking keep")
        if sitstand is not None or roulade == "off":
            raise ValueError("--four-actions 启用滚翻并保留当前主坐站；用十字键选择，不同时指定 --sitstand 或 --roulade off")
        # Keep the robot's established Y/joystick models. The D-pad variants
        # have separate slots, so switching back never loses the base policy.
        if not selected.get("sitstand_model"):
            selected["sitstand_model"] = models["sitstand_original"]["path"]
        selected["updated_walking_model"] = models["walk"]["path"]
        selected["sitstand_friction_model"] = models["sitstand_friction"]["path"]
        selected["sitstand_com_model"] = models["sitstand_com"]["path"]
        selected["roulade_model"] = models["roulade"]["path"]
        selected["control"]["roulade_enabled"] = True
        selected["control"]["model_suite_buttons_enabled"] = True
        return selected
    if walking == "new":
        selected["model"] = models["walk"]["path"]
    selected["sitstand_model"] = models["sitstand_" + (sitstand or "com")]["path"]
    selected["ground_pick_model"] = models["ground_pick"]["path"]
    selected["roulade_model"] = models["roulade"]["path"]
    selected["control"]["roulade_enabled"] = roulade == "on"
    if "model_suite_buttons_enabled" in selected["control"]:
        selected["control"]["model_suite_buttons_enabled"] = False
    return selected


def motion_processes(proc_root=Path("/proc")):
    found = []
    for process in Path(proc_root).iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            argv = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise RuntimeError(f"无法核验进程 {process.name}；请使用可查看控制进程的权限") from exc
        names = {Path(arg).name for arg in argv if arg}
        native_controller = bool(argv and Path(argv[0]).name in {"padd", "robotd"})
        web_server = False
        if "server.py" in names:
            try:
                web_server = ("servo_web" in str((process / "cwd").resolve())
                              or any("servo_web" in arg for arg in argv))
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError as exc:
                raise RuntimeError(f"无法核验网页进程 {process.name}") from exc
        if (native_controller or "microduck_deploy" in argv or names & MOTION_SCRIPTS
                or web_server
                or ("model_suite.py" in names and "run" in argv)):
            found.append({"pid": int(process.name), "argv": [x for x in argv if x]})
    return found


def assert_idle(cfg):
    """Read OS state only: never open serial/IMU or stop another controller."""
    service_states = {}
    if Path("/run/systemd/system").is_dir():
        for service in ("microduck-xbox.service", "servo-web.service", "padd.service", "robotd.service"):
            run = subprocess.run(["systemctl", "show", service, "--property=LoadState", "--property=ActiveState"],
                                 capture_output=True, text=True, timeout=5)
            state = dict(line.split("=", 1) for line in run.stdout.splitlines() if "=" in line)
            if state.get("LoadState") == "not-found" and state.get("ActiveState") == "inactive":
                service_states[service] = state
                continue
            if run.returncode != 0 or state.get("ActiveState") != "inactive":
                raise RuntimeError(f"服务仍活动或无法确认已停止：{service} {state}；请先停止对应服务。")
            service_states[service] = state
    processes = motion_processes()
    if processes:
        raise RuntimeError("检测到活动控制进程，拒绝切换/启动：" + json.dumps(processes, ensure_ascii=False))
    port = Path(cfg["serial"]["port"])
    if not port.is_absolute() or not str(port).startswith("/dev/"):
        raise ValueError("串口配置须为 /dev/ 下的绝对路径")
    if port.exists():
        if not shutil.which("fuser"):
            raise RuntimeError("缺少 fuser；先安装 psmisc 以核验串口占用")
        usage = subprocess.run(["fuser", str(port)], capture_output=True, text=True, timeout=5)
        if usage.returncode != 1 or usage.stdout.strip() or usage.stderr.strip():
            raise RuntimeError("串口被占用或无法核验：" + usage.stdout + usage.stderr)
    return {"services": service_states, "serial": str(port), "serial_present": port.exists(), "device_opened": False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("check", "select", "run"):
        child = sub.add_parser(name)
        child.add_argument("--config", type=Path, help="默认部署目录 robot.json；check 缺省时允许尚无配置")
        if name != "check":
            child.add_argument("--four-actions", action="store_true",
                               help="启用十字键四模型：↑新旧行走、↓滚翻、←摩擦坐站、→重心坐站；同时启用滚翻")
            child.add_argument("--walking", choices=("keep", "new"), default="keep", help="默认保留已有行走模型")
            child.add_argument("--sitstand", choices=("original", "friction", "com"), default=None,
                               help="普通选择默认com；四动作入口保留当前主坐站")
            child.add_argument("--roulade", choices=("off", "on"), default=None,
                               help="普通入口默认关闭；--four-actions同时启用实验前滚翻")
        if name == "select":
            child.add_argument("--apply", action="store_true", help="显式写入并备份配置；缺省只预览")
        if name == "run":
            child.add_argument("--seconds", type=float, default=10., help="默认10秒；0持续至B停止，其他值至少3秒")
            child.add_argument("--max-vx", type=float, default=.2, help="默认0.2m/s，上限0.4m/s")
    return p


def main(argv=None, *, root=ROOT):
    args = parser().parse_args(argv)
    root = Path(root).resolve()
    config_path = (args.config if args.config is not None else root / "robot.json").expanduser().resolve()
    try:
        if args.command == "check":
            cfg = load_config(config_path) if config_path.exists() or args.config is not None else None
            report = check_suite(root=root, cfg=cfg)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["status"] == "ok" else 1
        if args.command == "run":
            if config_path != (root / "robot.json").resolve():
                raise ValueError("现有 Xbox 包装器固定使用部署目录 robot.json；run 不接受其他 --config 路径")
            if not math.isfinite(args.seconds) or args.seconds < 0 or 0 < args.seconds < 3:
                raise ValueError("seconds 须为0或至少3秒")
            if not math.isfinite(args.max_vx) or not 0 < args.max_vx <= .4:
                raise ValueError("max-vx 须在(0,0.4] m/s内")
        cfg = load_config(config_path)
        report = check_suite(root=root, cfg=cfg)
        if report["status"] != "ok":
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1
        chosen = selection(cfg, load_catalog(root), walking=args.walking, sitstand=args.sitstand,
                           roulade=args.roulade, four_actions=args.four_actions)
        # Retained walking models may be outside the six-model catalog. Verify
        # their normal contract without silently selecting the new walk model.
        walking_path = Path(chosen["model"])
        walking_path = walking_path if walking_path.is_absolute() else root / walking_path
        if not walking_path.is_file():
            raise ValueError(f"当前行走模型不存在：{walking_path}")
        OnnxPolicy(walking_path)
        standing_path = Path(chosen["sitstand_model"])
        standing_path = standing_path if standing_path.is_absolute() else root / standing_path
        if not standing_path.is_file():
            raise ValueError(f"当前坐站模型不存在：{standing_path}")
        OnnxPolicy(standing_path, policy_kind="sitstand")
        changes = {key: {"before": cfg.get(key), "after": chosen.get(key)}
                   for key in ("model", "sitstand_model", "ground_pick_model", "roulade_model",
                               "updated_walking_model", "sitstand_friction_model", "sitstand_com_model")
                   if cfg.get(key) != chosen.get(key)}
        changes["control.roulade_enabled"] = {"before": cfg["control"].get("roulade_enabled", False),
                                               "after": chosen["control"]["roulade_enabled"]}
        if args.four_actions or "model_suite_buttons_enabled" in cfg["control"]:
            changes["control.model_suite_buttons_enabled"] = {
                "before": cfg["control"].get("model_suite_buttons_enabled", False),
                "after": chosen["control"]["model_suite_buttons_enabled"]}
        output = {"mode": "run" if args.command == "run" else ("apply" if args.apply else "preview"),
                  "config": str(config_path), "changes": changes, "walking_choice": args.walking,
                  "target_check_mode": TARGET_MODE_NOTE,
                  "four_actions": args.four_actions,
                  "warnings": {key: entry["warnings"] for key, entry in report["models"].items() if entry["warnings"]},
                  "note": "保留现有标定/IMU；仿真及离线检查不等于真机验证。select 不启动硬件。"}
        print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
        if args.command == "select" and not args.apply:
            return 0
        assert_idle(cfg)
        if args.command == "run" and not (root / "xbox_model_hd1910.sh").is_file():
            raise ValueError("缺少 xbox_model_hd1910.sh")
        if chosen != cfg:
            save_config(config_path, chosen)
        if args.command == "run":
            # The inherited wrapper requires a fresh A then Menu press. We do
            # not press buttons, start services, or skip its hardware checks.
            command = ["bash", str(root / "xbox_model_hd1910.sh"), str(chosen["model"]),
                       str(args.seconds), str(args.max_vx)]
            print(TARGET_MODE_NOTE, flush=True)
            return subprocess.run(command, cwd=root).returncode
        print("已保存选择并备份原配置；未打开硬件。", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
