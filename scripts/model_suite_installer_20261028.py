#!/usr/bin/env python3
"""Offline model-suite installer. Preview by default; --apply is explicit consent.

Never opens devices, changes robot.json/system service settings, or starts motion.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tempfile

PROTECTED = ("robot.json*", "*.user.json", "*.bak*", "*.pyc", "*.pyo", "*.log", "*.pid", ".*")
EXCLUDED_DIRS = {"logs", "backups", "backup", ".venv", "__pycache__", ".git"}
DEFAULT_SERVICES = ("microduck-xbox.service", "servo-web.service", "padd.service", "robotd.service")
NATIVE_CONTROLLERS = frozenset(("padd", "robotd"))
MOTION_SCRIPTS = {"xbox_autostart.py", "home_then_policy.py", "policy_trial.py",
                  "supported_home.py", "straight_then_home.py", "imu_home_assist.py",
                  "adjust_joint11_home.py", "apply_joint11_live.py", "readpose.py"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_relative(text):
    p = PurePosixPath(text)
    if (not text or p.is_absolute() or str(p) != text or ".." in p.parts
            or "\\" in text or "\n" in text or "\r" in text):
        raise ValueError(f"Unsafe relative path: {text!r}")
    return Path(*p.parts)


def payload_allowed(path):
    return not (any(x in EXCLUDED_DIRS or x.startswith(".") for x in path.parts[:-1])
                or any(fnmatch.fnmatch(path.name, pattern) for pattern in PROTECTED))


def no_symlink_path(root, relative):
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symbolic links are refused: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"Expected regular file: {path}")
    return path


def verify_package(package):
    manifest_file = package / "MANIFEST.json"
    if manifest_file.is_symlink():
        raise ValueError("Manifest must be a regular file")
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("format") != "microduck-model-suite-v1":
        raise ValueError("Unknown package format")
    records = manifest["files"]
    if not isinstance(records, dict) or not records:
        raise ValueError("Empty package manifest")
    payload = {}
    for text, expected in records.items():
        rel = safe_relative(text)
        source = no_symlink_path(package, rel)
        if not source.is_file() or digest(source) != expected["sha256"] or source.stat().st_size != expected["size"]:
            raise ValueError(f"Package checksum mismatch: {rel}")
        if rel.parts[0] == "payload":
            sub = Path(*rel.parts[1:])
            if len(rel.parts) < 2 or not payload_allowed(sub):
                raise ValueError(f"Protected path in payload: {rel}")
            payload[sub] = source
    actual = set()
    for directory, dirs, files in os.walk(package, followlinks=False):
        for name in dirs + files:
            p = Path(directory) / name
            if p.is_symlink():
                raise ValueError(f"Package contains symlink: {p}")
        for name in files:
            actual.add((Path(directory) / name).relative_to(package).as_posix())
    if actual != set(records) | {"MANIFEST.json"}:
        raise ValueError(f"Unexpected/missing package files: {actual ^ (set(records) | {'MANIFEST.json'})}")
    if not payload or Path("scripts/model_suite.py") not in payload:
        raise ValueError("Model-suite selector missing")
    return payload, manifest


def checked_run(command, *, cwd=None, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"Check failed: {' '.join(map(str, command))}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def assert_service_stopped(service):
    if not service.endswith(".service") or "/" in service or service.startswith("-"):
        raise ValueError(f"Invalid service name: {service}")
    # No systemd runtime means there cannot be an active systemd system service.
    if not Path("/run/systemd/system").is_dir():
        return {"service": service, "state": "systemd_not_running"}
    result = subprocess.run(["systemctl", "show", service, "--property=LoadState", "--property=ActiveState"],
                            text=True, capture_output=True, timeout=15)
    state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    # Some systemd releases return nonzero for an explicitly missing unit even
    # though show supplied its state. Absence is safe; inability to query isn't.
    if state.get("LoadState") == "not-found" and state.get("ActiveState") in (None, "inactive"):
        return {"service": service, "state": "not_installed"}
    if result.returncode or state.get("LoadState") not in ("loaded", "masked"):
        raise RuntimeError(f"Cannot verify service state: {service}: {state}: {result.stderr}")
    if state.get("ActiveState") != "inactive":
        raise RuntimeError(f"Stop service before installing: {service}: {state}. Use sudo systemctl stop {service}")
    return {"service": service, "state": "inactive"}


def motion_processes(proc_root=Path("/proc")):
    found = []
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            argv = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise RuntimeError(f"Cannot inspect PID {process.name}; run installer with sufficient process visibility") from exc
        names = {Path(x).name for x in argv if x}
        native_controller = bool(argv and Path(argv[0]).name in NATIVE_CONTROLLERS)
        try:
            native_controller |= Path(os.readlink(process / "exe")).name in NATIVE_CONTROLLERS
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            pass  # argv[0] still identifies the normal/manual launch path.
        servo_server = any(Path(arg).name == "server.py" and
                           ("servo_web" in arg or "servo-web" in arg) for arg in argv if arg)
        if "server.py" in names and not servo_server:
            try:
                cwd = os.readlink(process / "cwd")
                servo_server = "servo_web" in cwd or "servo-web" in cwd
            except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
                pass
        module_runtime = "microduck_deploy" in argv and "self-test" not in argv and "--help" not in argv
        if (native_controller or servo_server or module_runtime or (names & MOTION_SCRIPTS and "--help" not in argv)
                or ("model_suite.py" in names and "run" in argv)):
            found.append({"pid": int(process.name), "argv": argv})
    return found


def assert_idle(config, services=DEFAULT_SERVICES):
    states = [assert_service_stopped(service) for service in services]
    processes = motion_processes()
    if processes:
        raise RuntimeError(f"Motion/controller process still running: {processes}")
    port = Path(config["serial"]["port"])
    if not port.is_absolute() or not str(port).startswith("/dev/"):
        raise ValueError("Configured serial port must be an absolute /dev path")
    if port.exists():
        if not shutil.which("fuser"):
            raise RuntimeError("Install psmisc (fuser) to verify that the serial port is unused")
        # psmisc fuser versions differ on '--'; the validated absolute /dev/
        # pathname cannot be an option, so pass it directly for portability.
        result = subprocess.run(["fuser", str(port)], text=True, capture_output=True, timeout=15)
        if result.returncode == 0 or result.stdout.strip():
            raise RuntimeError(f"Serial port occupied: {port}: {result.stdout} {result.stderr}")
        if result.returncode != 1 or result.stderr.strip():
            raise RuntimeError(f"Cannot verify serial-port ownership: {result.stderr}")
    return {"services": states, "motion_processes": [], "serial": str(port),
            "serial_present": port.exists(), "device_opened": False}


def offline_validate(python, code_root, config):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    # Import modules, compile all Python files, validate saved calibration and
    # HOME encoder range. No constructor that can access hardware is called.
    source = """
import pathlib, sys, importlib, pkgutil, numpy as np
root=pathlib.Path(sys.argv[1]); sys.path.insert(0,str(root))
for p in root.rglob('*.py'):
    if not any(x in {'.venv','__pycache__','backups','logs'} for x in p.relative_to(root).parts):
        compile(p.read_bytes(),str(p),'exec')
import microduck_deploy
for m in pkgutil.iter_modules(microduck_deploy.__path__):
    if m.name != '__main__': importlib.import_module('microduck_deploy.'+m.name)
from microduck_deploy.config import load_config,require_calibration,calibration_arrays
from microduck_deploy.policy import HOME,q_to_ticks
cfg=load_config(sys.argv[2]); require_calibration(cfg,imu=True)
zero,direction=calibration_arrays(cfg); q_to_ticks(HOME,zero,direction)
saved=cfg['calibration']
assert isinstance(saved,dict) and saved.get('pose') in {'straight','zero','home'}
assert saved.get('joint_ids')==[j['id'] for j in cfg['joints']]
assert saved.get('directions')==direction.tolist()
for field in ('captured_ticks','reference_q_rad','home_ticks'):
    values=np.asarray(saved.get(field),dtype=float)
    assert values.shape==(14,) and np.isfinite(values).all(), 'Incomplete calibration '+field
rotation=np.asarray(cfg['imu']['mounting_rotation'],dtype=float)
assert rotation.shape==(3,3) and np.isfinite(rotation).all()
assert np.allclose(rotation@rotation.T,np.eye(3),atol=1e-5) and np.linalg.det(rotation)>.99
print('Imports, syntax, mapping, zero, IMU and HOME range: PASS')
"""
    imports = checked_run([str(python), "-B", "-c", source, str(code_root), str(config)], cwd=code_root, env=env)
    models = checked_run([str(python), "-B", str(code_root / "scripts/model_suite.py"),
                          "check", "--config", str(config)], cwd=code_root, env=env)
    return {"imports_and_calibration": imports, "model_suite_check": models}


def atomic_copy(source, target):
    fd, temp = tempfile.mkstemp(prefix=".model-suite-", dir=target.parent)
    os.close(fd)
    temp = Path(temp)
    try:
        shutil.copy2(source, temp)
        with temp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def protected_state(target):
    paths = [target / "robot.json", Path("/etc/default/microduck-xbox")]
    return {str(p): digest(p) if p.exists() else None for p in paths}


def install_transaction(target, payload, backup, *, postcheck, copy_file=atomic_copy):
    """Backup first; atomic replacement per file; rollback on any failure."""
    for rel in payload:
        if not payload_allowed(rel):
            raise ValueError(f"Protected destination: {rel}")
        no_symlink_path(target, rel)
    no_symlink_path(target, backup.relative_to(target) / "path-check")
    changed = {rel: src for rel, src in payload.items()
               if not (target / rel).is_file() or digest(target / rel) != digest(src)}
    original = protected_state(target)
    backup.mkdir(parents=True, exist_ok=False)
    records = {}
    for rel in changed:
        dest = no_symlink_path(target, rel)
        if dest.exists():
            saved = backup / "files" / rel
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, saved)
            records[rel.as_posix()] = {"existed": True, "sha256": digest(saved)}
        else:
            records[rel.as_posix()] = {"existed": False}
    journal = {"state": "prepared", "target": str(target), "protected": original, "files": records}
    log = backup / "transaction.json"
    log.write_text(json.dumps(journal, indent=2) + "\n")
    touched, new_dirs = [], []
    try:
        for rel, src in changed.items():
            dest = no_symlink_path(target, rel)
            missing = []
            parent = dest.parent
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for parent in reversed(missing):
                parent.mkdir()
                new_dirs.append(parent)
            touched.append(rel)  # journal covers the file even if replace fails.
            copy_file(src, dest)
            if digest(dest) != digest(src):
                raise RuntimeError(f"Installed hash mismatch: {rel}")
        postcheck()
        if protected_state(target) != original:
            raise RuntimeError("Protected robot.json or service settings changed during installation")
        journal["state"] = "committed"
        log.write_text(json.dumps(journal, indent=2) + "\n")
    except BaseException:
        errors = []
        for rel in reversed(touched):
            try:
                if records[rel.as_posix()]["existed"]:
                    atomic_copy(backup / "files" / rel, target / rel)
                else:
                    (target / rel).unlink(missing_ok=True)
            except BaseException as exc:
                errors.append(f"{rel}: {exc}")
        for directory in reversed(new_dirs):
            try:
                directory.rmdir()
            except OSError:
                pass
        journal.update(state="rollback_failed" if errors else "rolled_back", rollback_errors=errors)
        log.write_text(json.dumps(journal, indent=2) + "\n")
        if errors:
            raise RuntimeError(f"Rollback incomplete; use backup {backup}: {errors}")
        raise
    return {"changed_files": len(changed), "backup": str(backup), "protected_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("~/deploy_hd1910"))
    parser.add_argument("--python", type=Path, help="Default: TARGET/.venv/bin/python")
    parser.add_argument("--service", action="append", default=list(DEFAULT_SERVICES), help="Additional service that must be stopped")
    parser.add_argument("--apply", action="store_true", help="Install only after all checks; default is preview")
    args = parser.parse_args()
    if args.target.expanduser().is_symlink():
        raise ValueError("Target deployment directory must not be a symlink")
    target = args.target.expanduser().resolve()
    if not target.is_dir():
        raise ValueError("Target must be an existing calibrated deployment directory")
    package = Path(__file__).resolve().parent
    if target == package or target in package.parents or package in target.parents:
        raise ValueError("Package and target must be separate directories")
    config = no_symlink_path(target, Path("robot.json"))
    if not config.is_file():
        raise ValueError("Existing target robot.json is required; this bundle never supplies calibration")
    payload, manifest = verify_package(package)
    for rel in payload:
        no_symlink_path(target, rel)
    # Preserve the venv executable path: resolving its symlink would discard
    # pyvenv.cfg and run the system interpreter without deployment packages.
    python = (args.python.expanduser() if args.python else target / ".venv/bin/python").absolute()
    if not python.is_file():
        raise ValueError(f"Python environment missing: {python}; create/install dependencies first")
    before = protected_state(target)
    cfg = json.loads(config.read_text())
    idle = assert_idle(cfg, args.service)
    validation = offline_validate(python, package / "payload", config)
    if protected_state(target) != before:
        raise RuntimeError("Protected configuration changed while checking; retry with all controllers stopped")
    summary = {"mode": "apply" if args.apply else "preview", "target": str(target),
               "payload_files": len(payload), "bundle": manifest.get("bundle"),
               "robot_json_sha256": before[str(config)], "idle": idle, "validation": validation}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.apply:
        print("Preview complete; no target files changed. Add --apply to install.")
        return
    lock_path = no_symlink_path(target, Path(".model_suite_install.lock"))
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle(cfg, args.service)
        if protected_state(target) != before:
            raise RuntimeError("Protected configuration changed before commit")
        backup = target / "backups" / ("model_suite_20261028_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        def postcheck():
            assert_idle(cfg, args.service)
            offline_validate(python, target, config)
        result = install_transaction(target, payload, backup, postcheck=postcheck)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Installed. No service started and no device opened. Use scripts/model_suite.py select/run explicitly.")


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Installation refused/rolled back: {exc}", file=sys.stderr)
        raise SystemExit(1)
