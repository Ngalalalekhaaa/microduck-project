"""Record disk/GPU availability; refuse training below a disk reserve."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--reserve-gib', type=float, default=3.0)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
disk = shutil.disk_usage(root)
gpu = subprocess.check_output([
    'nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu',
    '--format=csv,noheader,nounits',
], text=True).strip()
report = {
    'time': datetime.now().astimezone().isoformat(),
    'workspace': str(root),
    'disk_total_gib': disk.total / 2**30,
    'disk_free_gib': disk.free / 2**30,
    'reserve_gib': args.reserve_gib,
    'gpu_name_totalMiB_usedMiB_freeMiB_tempC': gpu,
    'ok': disk.free >= args.reserve_gib * 2**30,
}
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(report, indent=2, ensure_ascii=False))
if not report['ok']:
    raise SystemExit('Disk reserve would be violated; training not started.')
