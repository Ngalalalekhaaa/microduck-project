"""Wait for the existing training process, then produce a local result report."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[1]
active = json.loads((root / 'runs/active_run.json').read_text())
pid = active['supervisor_pid']
while True:
    paths = sorted((root / 'runs').glob('*_train/status.json'))
    if paths:
        status_path = paths[-1]
        state = json.loads(status_path.read_text())
        if state['status'] in ('completed', 'failed', 'interrupted'):
            break
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        if not paths:
            raise SystemExit('Training exited before creating status; inspect runs/train_supervisor.log')
        state['status'] = 'failed'
        state['error'] = 'Supervisor exited before completion; see runs/train_supervisor.log'
        temp = status_path.with_suffix('.watcher.tmp')
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(status_path)
        break
    time.sleep(30)
subprocess.run([str(root / 'microduck_rl/.venv/bin/python'),
                str(root / 'scripts/finish_report.py'), '--run', str(status_path.parent)], check=True)
