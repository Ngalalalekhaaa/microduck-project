#!/usr/bin/env python3
"""Build a local, checksummed deployment bundle without machine calibration."""
from __future__ import annotations
import argparse
from datetime import datetime
import fnmatch
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "microduck-model-suite-20261028"
EPOCH = int(datetime.fromisoformat("2026-10-28T00:00:00+08:00").timestamp())
SKIP_DIRS = {".git", ".venv", "venv", "env", "logs", "backups", "backup", "__pycache__", ".pytest_cache", "node_modules"}
SKIP_NAMES = ("robot.json*", "*.user.json", "*.bak*", "*.pyc", "*.pyo", "*.log", "*.pid", "*.tmp", "*.swp", "*~", "*SHA256SUMS*", "*.sha256", "*.pem", "*.key", "credentials*", "secrets*",
              "*.tar", "*.tar.gz", "*.tgz", "*.zip", "*.7z", "*.rar", "*.zst",
              "*calibration*.json", "*snapshot*.json", "*session*.json", "*capture*.json", "imu_bias*.json")
SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".onnx", ".npz", ".png", ".jpg", ".in", ".toml", ".html", ".js", ".css"}
REQUIRED = {"scripts/model_suite.py", "models/suite_20261028.json", "microduck_deploy/runtime.py", "requirements.txt"}
REQUIRED |= {f"models/hd1910_{key}.onnx" for key in ("walk_1849", "roulade_1600", "sitstand_1699", "sitstand_friction_2598", "sitstand_com_2598", "ground_pick_2249")}

INSTALL_TEXT = """Microduck 2026-10-28 六模型离线更新包

在电脑校验：sha256sum -c microduck-model-suite-20261028.tar.gz.sha256
解压：tar -xzf microduck-model-suite-20261028.tar.gz
把解压目录复制到机器人用户目录，与现有 ~/deploy_hd1910 并列。

机器人端先按 B 卸力，结束手动控制程序，再停止服务：
sudo systemctl stop microduck-xbox.service
若装有 servo-web.service、padd.service、robotd.service，也先停止这些服务；勿同时运行原生遥操和 HD1910 控制。
cd ~/microduck-model-suite-20261028
python3 install.py --target ~/deploy_hd1910
确认预览通过后安装：
python3 install.py --target ~/deploy_hd1910 --apply

安装器使用目标 .venv/bin/python，缺依赖时先在该环境按 payload/requirements.txt 安装。
可用 --python /完整路径/python 指定已装好依赖的 Python 3.11+。
安装前后均离线检查代码、现有14关节/IMU标定、HOME单圈刻度与全部六模型CPU推理。
包校验、标定、服务/进程/串口占用任一检查不通过时拒绝安装；覆盖失败自动回滚。
备份保存在目标 backups/model_suite_20261028_*/，含 transaction.json。
不覆盖 robot.json，不修改 /etc/default/microduck-xbox，不安装/启动服务，不驱动真机。

安装完成：
cd ~/deploy_hd1910
.venv/bin/python scripts/model_suite.py --help
.venv/bin/python scripts/model_suite.py check --config robot.json
使用 select 明确选择模型，再按 selector 的 run 帮助进行真机试用。
前滚翻属于实验动作；仿真通过不等于真实设备已验证。保留现有机械标定，勿复制他人 robot.json。
"""


def validate_name(name):
    """A bundle name is a single safe ASCII directory/filename component."""
    if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", name)
            or ".." in name or name.endswith(".")):
        raise ValueError("Bundle name must be 1..96 ASCII letters/digits/._-, start with a letter/digit, and contain no '..' or trailing dot")
    return name


def selected_files(source):
    selected = []
    for directory, dirs, files in os.walk(source, followlinks=False):
        base = Path(directory)
        kept = []
        for name in sorted(dirs):
            if name in SKIP_DIRS or name.startswith("."):
                continue
            if (base / name).is_symlink():
                raise ValueError(f"Symlink refused: {base / name}")
            kept.append(name)
        dirs[:] = kept
        for name in sorted(files):
            if name.startswith(".") or any(fnmatch.fnmatch(name, p) for p in SKIP_NAMES):
                continue
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Only regular files allowed: {path}")
            if path.suffix.lower() not in SUFFIXES and name not in {"LICENSE", "NOTICE"}:
                raise ValueError(f"Review unexpected file type: {path}")
            selected.append(path.relative_to(source))
    available = {x.as_posix() for x in selected}
    if REQUIRED - available:
        raise ValueError(f"Required payload not ready: {sorted(REQUIRED - available)}")
    samples = [x for x in selected if x.parts[:2] == ("samples", "suite_20261028") and x.suffix == ".npz"]
    if len(samples) < 6:
        raise ValueError("Need at least 6 suite_20261028 sample NPZ files")
    return sorted(selected)


def build(source, output, files, name=NAME):
    name = validate_name(name)
    entries = {}
    for rel in files:
        p = source / rel
        entries["payload/" + rel.as_posix()] = (p.read_bytes(), 0o755 if p.stat().st_mode & 0o111 else 0o644)
    entries["install.py"] = ((ROOT / "scripts/model_suite_installer_20261028.py").read_bytes(), 0o755)
    entries["INSTALL.txt"] = (INSTALL_TEXT.replace(NAME, name).encode(), 0o644)
    manifest = {"format": "microduck-model-suite-v1", "bundle": name, "date": "2026-10-28", "timezone": "Asia/Shanghai",
                "calibration_included": False, "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                                                         for name, (data, mode) in entries.items()}}
    entries["MANIFEST.json"] = ((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(), 0o644)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=EPOCH) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as archive:
                    for relative, (data, mode) in entries.items():
                        entry = tarfile.TarInfo(name + "/" + relative)
                        entry.size, entry.mode, entry.mtime = len(data), mode, EPOCH
                        archive.addfile(entry, io.BytesIO(data))
        temporary.replace(output)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n")
    print(json.dumps({"archive": str(output), "sha256": digest, "bytes": output.stat().st_size, "payload_files": len(files)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "deploy_hd1910")
    parser.add_argument("--name", default=NAME, help="Archive root / manifest name; default preserves the original suite name")
    parser.add_argument("--output", type=Path, help="Default: workspace root / NAME.tar.gz")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    name = validate_name(args.name)
    source = args.source.expanduser().resolve()
    output = (args.output.expanduser() if args.output else ROOT / (name + ".tar.gz")).resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the payload source")
    files = selected_files(source)
    if args.check:
        print(json.dumps({"payload_files": len(files), "bytes": sum((source / p).stat().st_size for p in files)}, indent=2))
    else:
        build(source, output, files, name=name)


if __name__ == "__main__":
    main()
