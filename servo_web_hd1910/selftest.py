"""Run adapted profile and servo protocol tests, without hardware access."""
from pathlib import Path
import subprocess
import sys
raise SystemExit(subprocess.call([sys.executable, '-m', 'pytest', '-q',
    'test_user_profile.py', 'test_sim.py', 'test_servo_report.py'], cwd=Path(__file__).parent))
