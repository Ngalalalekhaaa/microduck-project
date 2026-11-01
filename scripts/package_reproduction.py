#!/usr/bin/env python3
"""Build the source/model bundle used by README-从零复现.md (Python 3.10+).

Run from any directory: python3 /path/to/microduck/scripts/package_reproduction.py
Use --check to inspect the selection without creating an archive.
No virtual environment, training checkpoint, machine calibration or Wi-Fi
configuration is copied. The generated archive is a local artifact, not a
publication or an installation.
"""

import argparse
from datetime import datetime
import fnmatch
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DATE = "2026-10-28"
ARCHIVE_NAME = "microduck-repro-20261028.tar.gz"
ARCHIVE_MTIME = int(datetime.fromisoformat(RELEASE_DATE + "T00:00:00+08:00").timestamp())
SOURCE_PATHS = (
    "README-从零复现.md",
    "README.md",
    "README-训练与部署.md",
    "microduck_rl",
    "microduck_rl_hd1910",
    "deploy_hd1910",
    "servo_web_hd1910",
    "radxa_j5_i2c_fix",
    "scripts",
    "microduck-replica/tools/飞特",
    "microduck-replica/tools/radxa",
    "microduck-replica/LICENSE",
    "microduck-replica/NOTICE.md",
)
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "env", "logs", "runs", "cache", "caches",
    "poses", "backups", "backup", "__pycache__", "node_modules", "wandb",
}
EXCLUDED_PATTERNS = (
    "robot.json", "robot.json.*", "card.conf", "directions.json",
    "*.pyc", "*.pyo", "*.bak*", "*.pid", "pid", "*.log", "*.pt", "*.pth",
    "*.ckpt", "*.tmp", "*.swp", "*~", "*SHA256SUMS*", "*.sha256",
    "*.pem", "*.key", "*.p12", "*.pfx", "credentials*", "secrets*",
)
# Fail on unexpected file types instead of quietly including arbitrary local data.
PUBLIC_SUFFIXES = {
    ".py", ".md", ".toml", ".lock", ".json", ".xml", ".stl", ".part",
    ".sh", ".txt", ".png", ".in", ".html", ".js", ".cjs", ".bin",
    ".dts", ".ps1", ".onnx",
}
PUBLIC_NAMES = {"LICENSE", "NOTICE", "UPSTREAM_COMMIT", ".gitignore"}
UPSTREAM_FILES = {
    "microduck-replica/LICENSE",
    "microduck-replica/NOTICE.md",
    "microduck-replica/tools/飞特/CH340_URT.rar",
    "microduck-replica/tools/飞特/FD1.9.8.5(250729).7z",
    "microduck-replica/tools/飞特/LICENSE",
    "microduck-replica/tools/飞特/README.md",
    "microduck-replica/tools/radxa/1-做卡.ps1",
    "microduck-replica/tools/radxa/build-armbian-card.sh",
    "microduck-replica/tools/radxa/README.md",
    "microduck-replica/tools/radxa/镜像使用说明.md",
}
REQUIRED_FILES = (
    "microduck_rl/uv.lock",
    "microduck_rl/pyproject.toml",
    "microduck_rl_hd1910/src/mjlab_microduck/robot/hd1910/1910_m6.json",
    "scripts/train_pair.py",
    "scripts/verify_hd1910_export.py",
    "scripts/package_reproduction.py",
    "deploy_hd1910/robot.example.json",
    "deploy_hd1910/models/hd1910_backlash_30000.onnx",
    "deploy_hd1910/xbox_model_hd1910.sh",
    "servo_web_hd1910/readpose.py",
    "radxa_j5_i2c_fix/enable_j5.py",
)


def excluded(relative: Path) -> bool:
    """Exclude mutable state, credentials, caches and inherited checksums."""
    if any(part.lower() in EXCLUDED_DIRS or part.startswith(".")
           for part in relative.parts[:-1]):
        return True
    name = relative.name
    if name.startswith(".") and name != ".gitignore":
        return True
    return any(fnmatch.fnmatch(name.lower(), pattern.lower())
               for pattern in EXCLUDED_PATTERNS)


def selected_files(root: Path) -> list[Path]:
    files = []
    for selection in SOURCE_PATHS:
        selected = root / selection
        if not selected.exists():
            raise FileNotFoundError(f"Required source is missing: {selected}")
        if selected.is_symlink():
            raise ValueError(f"Refusing symbolic link: {selected}")
        if selected.is_file():
            candidates = [selected]
        else:
            candidates = []
            for directory, names, filenames in os.walk(selected, followlinks=False):
                base = Path(directory)
                kept = []
                for name in sorted(names):
                    child = base / name
                    if name.startswith(".") or name.lower() in EXCLUDED_DIRS:
                        continue
                    if child.is_symlink():
                        raise ValueError(f"Refusing symbolic link: {child}")
                    kept.append(name)
                names[:] = kept
                candidates.extend(base / name for name in sorted(filenames))
        for path in candidates:
            relative = path.relative_to(root)
            if excluded(relative):
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Only regular files may be packaged: {path}")
            if relative.parts[0] == "microduck-replica":
                if relative.as_posix() not in UPSTREAM_FILES:
                    continue
            elif path.suffix.lower() not in PUBLIC_SUFFIXES and path.name not in PUBLIC_NAMES:
                raise ValueError(f"Review unexpected file type before packaging: {relative}")
            files.append(relative)
    available = {path.as_posix() for path in files}
    missing = set(REQUIRED_FILES) - available
    if missing:
        raise FileNotFoundError(f"Required bundle files missing: {sorted(missing)}")
    return sorted(files, key=lambda path: path.as_posix())


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metadata(files: list[Path], byte_count: int) -> bytes:
    version = {
        "bundle": "microduck-repro",
        "date": RELEASE_DATE,
        "timezone": "Asia/Shanghai",
        "archive_root": "microduck",
        "source": "local workspace snapshot; not a GitHub release or a Git commit",
        "guide": "README-从零复现.md",
        "source_file_count": len(files),
        "source_bytes": byte_count,
        "included_roots": list(SOURCE_PATHS),
        "constraints": {
            "virtual_environments_included": False,
            "pt_training_checkpoints_included": False,
            "machine_calibration_included": False,
            "wifi_configuration_included": False,
            "pretrained_onnx_included": True,
            "requires_fresh_dependency_install": True,
            "historical_paths_in_provenance_and_optional_scripts_may_need_adjustment": True,
            "checksum_manifest": "REPRO_SHA256SUMS (all payload files except itself)",
            "radxa_card_scripts_require_your_own_card_conf": True,
        },
        "excluded_directories": sorted(EXCLUDED_DIRS),
        "excluded_file_patterns": list(EXCLUDED_PATTERNS),
        "hidden_files": "excluded except .gitignore",
        "symlinks": "refused",
    }
    return (json.dumps(version, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    entry = tarfile.TarInfo("microduck/" + name)
    entry.size = len(data)
    entry.mode = mode
    entry.mtime = ARCHIVE_MTIME
    archive.addfile(entry, io.BytesIO(data))


def build_archive(files: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    file_bytes = sum((ROOT / path).stat().st_size for path in files)
    version = metadata(files, file_bytes)
    checksum_lines = []
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=ARCHIVE_MTIME) as zipped:
                with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for path in files:
                        data = (ROOT / path).read_bytes()
                        name = path.as_posix()
                        mode = 0o755 if (ROOT / path).stat().st_mode & 0o111 else 0o644
                        add_bytes(archive, name, data, mode)
                        checksum_lines.append(f"{sha256(data)}  {name}\n")
                    add_bytes(archive, "REPRO_VERSION.json", version)
                    checksum_lines.append(f"{sha256(version)}  REPRO_VERSION.json\n")
                    add_bytes(archive, "REPRO_SHA256SUMS", "".join(checksum_lines).encode("utf-8"))
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    with output.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    checksum = output.with_name(output.name + ".sha256")
    checksum.write_text(f"{digest.hexdigest()}  {output.name}\n", encoding="utf-8")
    print(f"Created {output} ({output.stat().st_size / 2**20:.2f} MiB)")
    print(f"Created {checksum}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate selection and print size without writing")
    parser.add_argument("--output", type=Path, default=ROOT / ARCHIVE_NAME,
                        help="Output .tar.gz path (default: workspace root)")
    args = parser.parse_args()
    files = selected_files(ROOT)
    size = sum((ROOT / path).stat().st_size for path in files)
    print(f"Selected {len(files)} source files; uncompressed {size / 2**20:.2f} MiB")
    if not args.check:
        output = args.output.expanduser().resolve()
        if output in {ROOT / path for path in files}:
            parser.error("Output must not replace a selected source file")
        build_archive(files, output)


if __name__ == "__main__":
    main()
