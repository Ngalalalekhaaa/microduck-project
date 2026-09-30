"""Wait for a specific XL330 pair, validate HD1910, then train its pair.

No GPU subprocess starts while the predecessor is running. A failed dependency,
changed source snapshot, failed smoke test or low disk reserve stops the queue.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

from train_pair import HD1910_TASKS, ROOT, VENV, run_monitored

PROJECT = ROOT / 'microduck_rl_hd1910'


def now():
    return datetime.now().astimezone().isoformat()


def source_snapshot():
    files = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        cwd=PROJECT).decode().split('\0')
    paths = [PROJECT / p for p in files if p and (PROJECT / p).is_file()]
    paths += [ROOT / 'scripts' / p for p in
              ('train_pair.py', 'evaluate.py', 'preflight.py', 'finish_report.py', 'queue_hd1910.py')]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def process_live(pid):
    try:
        # Zombies no longer execute code or hold a CUDA context.
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return stat[0] != 'Z'
    except FileNotFoundError:
        return False


def dependency_ready(path):
    predecessor = json.loads(path.read_text())
    state = predecessor['status']
    if state in ('failed', 'interrupted'):
        raise RuntimeError(f'Original training ended with {state}; see {path}')
    if state == 'completed':
        if len(predecessor['runs']) != 2 or any(r['status'] != 'completed' for r in predecessor['runs']):
            raise RuntimeError('Original pair does not contain two completed tasks.')
        for run in predecessor['runs']:
            artifact = Path(run['artifacts'])
            for relative in ('policy.onnx', 'eval_flat/evaluation.json', 'eval_backlash/evaluation.json'):
                if not (artifact / relative).is_file():
                    raise RuntimeError(f'Missing original artifact: {artifact / relative}')
        return not process_live(predecessor['pid'])
    if not process_live(predecessor['pid']):
        raise RuntimeError(f'Original supervisor exited without completing: {path}')
    return False


def validate_training_logs(run_dir):
    state = json.loads((run_dir / 'status.json').read_text())
    if state['status'] != 'completed' or len(state['runs']) != 2:
        raise RuntimeError('HD1910 pair did not complete.')
    for run in state['runs']:
        if run['task'] != HD1910_TASKS.get(run['variant']):
            raise RuntimeError('Unexpected task: HD1910 source selection failed.')
        log = re.sub(r'\x1b\[[0-9;]*m', '', Path(run['stdout']).read_text())
        nan_rates = re.findall(r'Episode_Termination/nan_state:\s*(\S+)', log)
        if not nan_rates or any(not math.isfinite(float(x)) or float(x) != 0 for x in nan_rates):
            raise RuntimeError(f'Invalid/absent NaN termination metric: {run["stdout"]}')
    return state


def validate_smoke(run_dir):
    state = validate_training_logs(run_dir)
    for run in state['runs']:
        for variant in ('flat', 'backlash'):
            result = json.loads((Path(run['artifacts']) / f'eval_{variant}/evaluation.json').read_text())
            if result['task'] != HD1910_TASKS[variant]:
                raise RuntimeError('Evaluation used the wrong actuator task.')
            expected_commands = {'stand', 'forward', 'forward_fast', 'backward', 'left', 'turn'}
            if set(result['results']) != expected_commands:
                raise RuntimeError('Evaluation did not finish all required commands.')
            if not result.get('onnx_parity_pass'):
                raise RuntimeError('ONNX/checkpoint output parity failed.')
            if result.get('onnx_input_shape') != [1, 61] or result.get('onnx_output_shape') != [1, 14]:
                raise RuntimeError('Policy interface is not 61 observations / 14 actions.')
            if any(metrics['nonfinite_first_episode_states'] for metrics in result['results'].values()):
                raise RuntimeError('Nonfinite states found during smoke evaluation.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after', type=Path, required=True, help='Specific predecessor status.json')
    parser.add_argument('--state', type=Path, default=ROOT / 'runs/hd1910_queue.json')
    parser.add_argument('--num-envs', type=int, default=2048)
    parser.add_argument('--iterations', type=int, default=4000)
    args = parser.parse_args()
    lock_file = (ROOT / 'runs/hd1910_queue.lock').open('a')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('An HD1910 queue is already active.')
    if args.state.exists():
        raise SystemExit(f'Refusing to overwrite queue history: {args.state}')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    queue_dir = ROOT / 'runs' / f'{stamp}_hd1910_queue'
    queue_dir.mkdir()
    state = {
        'created': now(), 'pid': os.getpid(), 'status': 'waiting_originals',
        'predecessor': str(args.after.resolve()), 'queue_dir': str(queue_dir),
        'source_project': str(PROJECT), 'initialization': 'random (no XL330 checkpoint)',
        'iterations_per_model': args.iterations, 'num_envs': args.num_envs, 'seed': 42,
        'stages': [], 'source_snapshot': str(queue_dir / 'source_sha256.json'),
    }
    snapshot = source_snapshot()
    (queue_dir / 'source_sha256.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    (queue_dir / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=PROJECT))

    def save():
        state['updated'] = now()
        temp = args.state.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
        temp.replace(args.state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    env = dict(os.environ, PYTHONPATH=str(PROJECT / 'src'), PYTHONUNBUFFERED='1',
               MUJOCO_GL='egl', WANDB_MODE='disabled', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env['PATH'] = str(VENV / 'bin') + os.pathsep + env.get('PATH', '')
    env.pop('MICRODUCK_WARM_START', None)

    def stage(name, command, *, extra_env=None):
        if source_snapshot() != snapshot:
            raise RuntimeError('HD1910 source/scripts changed after queuing; revalidate before restarting.')
        subprocess.run(['python3', str(ROOT / 'scripts/preflight.py'), '--output',
                        str(queue_dir / f'{name}_preflight.json')], check=True)
        state['status'] = name
        item = {'name': name, 'command': command, 'started': now(),
                'status': 'running', 'log': str(queue_dir / f'{name}.log')}
        state['stages'].append(item)
        save()
        print(f'{now()} Starting {name}; {item["log"]}', flush=True)
        run_monitored(command, project=PROJECT, env={**env, **(extra_env or {})},
                      log_path=Path(item['log']), item=item, save=save, stop_timeout=45)
        item.update(status='completed', finished=now())
        save()

    save()
    print(f'{now()} Waiting for {args.after}', flush=True)
    try:
        while not dependency_ready(args.after):
            save()
            time.sleep(30)
        state['predecessor_completed_at'] = now()
        save()
        stage('gpu_validation', [str(VENV / 'bin/python'), '-m', 'pytest', '-q', '-s',
                                 'tests/test_hd1910_rollout.py'],
              extra_env={'MICRODUCK_TEST_GPU': '1'})
        outputs = {}
        for phase, envs, iterations, finalize in (
            ('smoke', 64, 5, True), ('benchmark', args.num_envs, 10, False),
            ('train', args.num_envs, args.iterations, True),
        ):
            output = ROOT / 'runs' / f'{stamp}_hd1910_{phase}'
            outputs[phase] = str(output)
            state['outputs'] = outputs
            command = ['python3', str(ROOT / 'scripts/train_pair.py'), '--robot', 'hd1910',
                       '--phase', phase, '--num-envs', str(envs), '--iterations', str(iterations),
                       '--output', str(output)]
            if finalize:
                command.append('--finalize')
            stage(phase, command)
            validate_training_logs(output)
            if phase == 'smoke':
                validate_smoke(output)
                state['smoke_validation_passed'] = True
                save()
            elif phase == 'benchmark':
                benchmark = json.loads((output / 'status.json').read_text())
                times = {}
                for run in benchmark['runs']:
                    matches = re.findall(r'Iteration time:\s*([\d.]+)s', Path(run['stdout']).read_text())
                    if matches:
                        times[run['variant']] = statistics.median(float(x) for x in matches[-8:])
                state['benchmark_seconds_per_iteration'] = times
                state['estimated_training_hours_excluding_exports'] = sum(times.values()) * args.iterations / 3600
                save()
        stage('reporting', [str(VENV / 'bin/python'), str(ROOT / 'scripts/finish_report.py'),
                            '--run', outputs['train'], '--output',
                            str(ROOT / 'reports/HD1910训练结果.md')])
        state.update(status='completed', finished=now())
        save()
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     error=f'{type(exc).__name__}: {exc}', finished=now())
        if state['stages'] and state['stages'][-1]['status'] == 'running':
            state['stages'][-1]['status'] = state['status']
        save()
        raise
    print(f'{now()} Completed HD1910 pair', flush=True)


if __name__ == '__main__':
    main()
