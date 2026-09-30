"""Print the latest local training progress without loading GPU libraries."""
from datetime import datetime
import json
from pathlib import Path
import re
import shutil

root = Path(__file__).resolve().parents[1]
paths = sorted((root / 'runs').glob('*_train/status.json'))
if not paths:
    raise SystemExit('No training run has started.')
path = paths[-1]
state = json.loads(path.read_text())
print('状态文件:', path)
print('流水线状态:', state['status'])
print('开始时间:', state['started'])
for run in state['runs']:
    print('\n任务:', run['task'], '| 状态:', run['status'])
    log = Path(run['stdout'])
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - 32768))
            text = f.read().decode(errors='replace')
        text = re.sub(r'\x1b\[[0-9;]*m', '', text)
        for label, pattern in [
            ('迭代', r'Learning iteration\s+(\d+/\d+)'),
            ('平均回报', r'Mean reward:\s*([-\d.]+)'),
            ('平均回合长度（步）', r'Mean episode length:\s*([\d.]+)'),
            ('单轮耗时（秒）', r'Iteration time:\s*([\d.]+)s'),
            ('本任务剩余估计', r'ETA:\s*([\d:]+)'),
        ]:
            matches = re.findall(pattern, text)
            if matches:
                print(label + ':', matches[-1])
        print('日志更新:', datetime.fromtimestamp(log.stat().st_mtime).astimezone().isoformat())
    print('日志:', log)
    for directory in run.get('training_dirs', []):
        print('检查点目录:', directory)
    if run.get('artifacts'):
        print('导出和评估:', run['artifacts'])
print('\n磁盘剩余 GiB:', round(shutil.disk_usage(root).free / 2**30, 2))
queue_path = root / 'runs/hd1910_queue.json'
if queue_path.exists():
    queue = json.loads(queue_path.read_text())
    print('\nHD-1910 后续队列:', queue['status'])
    print('队列进程 PID:', queue['pid'])
    print('等待的原版任务:', queue['predecessor'])
    print('每版轮数 / 环境数:', queue['iterations_per_model'], '/', queue['num_envs'])
    print('队列状态文件:', queue_path)
    if queue.get('error'):
        print('队列错误:', queue['error'])
    if 'estimated_training_hours_excluding_exports' in queue:
        print('容量验证后估计两版训练总小时数（不含导出）:',
              round(queue['estimated_training_hours_excluding_exports'], 2))
    for stage in queue['stages']:
        print('阶段:', stage['name'], stage['status'], '| 日志:', stage['log'])
