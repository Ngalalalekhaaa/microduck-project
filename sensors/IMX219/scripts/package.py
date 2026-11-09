#!/usr/bin/env python3
"""Build a standalone deployment tarball, excluding all original machine logs/photos."""
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('scripts', 'vendor', 'docs', 'references', 'packages', 'tests', 'evidence')


def package(output):
    subprocess.run([sys.executable, str(ROOT/'scripts/fetch_assets.py'), '--check'], check=True)
    files = [ROOT/'README.md', ROOT/'.gitignore']
    for name in DIRECTORIES:
        files.extend(p for p in (ROOT/name).rglob('*')
                     if p.is_file() and not p.is_symlink()
                     and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo'))
    files = sorted(set(files))
    if output.exists():
        raise SystemExit(f'Refusing to overwrite: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    checksums = []
    with tarfile.open(output, 'w:gz') as archive:
        for path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise SystemExit(f'External file or symlink: {path}')
            relative = path.relative_to(ROOT).as_posix()
            checksums.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+relative)
            info = archive.gettarinfo(str(path), 'IMX219/'+relative)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            with path.open('rb') as stream:
                archive.addfile(info, stream)
        data = ('\n'.join(checksums)+'\n').encode()
        info = tarfile.TarInfo('IMX219/SHA256SUMS')
        info.size = len(data)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name+'.sha256').write_text(sha+'  '+output.name+'\n')
    print(json.dumps({'archive': str(output), 'bytes': output.stat().st_size,
                      'files': len(files), 'sha256': sha,
                      'excluded': ['archive/', 'captures/', 'logs/', 'dist/']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT/'dist'/('microduck-camera-IMX219-'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.tar.gz')
    package(output.resolve())


if __name__ == '__main__':
    main()
