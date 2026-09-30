"""Seed an isolated venv with identical CUDA wheels already installed locally.

Only binary shared libraries use hard links (installers unlink on replacement).
Python sources and distribution metadata are copied. Never symlink package
directories, so installing a different version cannot modify the source env.
"""
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/luckysir/anaconda3/envs/hvgen/lib/python3.10/site-packages')
TARGET = ROOT / 'microduck_rl/.venv/lib/python3.12/site-packages'
lock = tomllib.loads((ROOT / 'microduck_rl/uv.lock').read_text())
allowed = {(p['name'], p['version']) for p in lock['package'] if p['name'].startswith('nvidia-')}
report = []
for dist in importlib.metadata.distributions(path=[str(SOURCE)]):
    name, version = dist.metadata['Name'].lower(), dist.version
    if (name, version) not in allowed:
        continue
    linked_bytes = 0
    for entry in dist.files or []:
        if '..' in entry.parts or '__pycache__' in entry.parts:
            continue
        src, dst = SOURCE / entry, TARGET / entry
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        if '.so' in src.name and not src.is_symlink():
            os.link(src, dst)
            linked_bytes += src.stat().st_size
        else:
            shutil.copy2(src, dst)
    report.append({'name': name, 'version': version, 'linked_bytes': linked_bytes})
(ROOT / 'reports/reused_cuda.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
print('Shared binary GiB:', sum(r['linked_bytes'] for r in report) / 2**30)
