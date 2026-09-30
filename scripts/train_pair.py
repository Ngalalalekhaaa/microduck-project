"""Run a walking pair sequentially, reusing one venv with isolated source trees."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'microduck_rl'
VENV = PROJECT / '.venv'
TASKS = {
    'flat': 'Mjlab-Velocity-Flat-MicroDuck',
    'backlash': 'Mjlab-Velocity-Flat-Backlash-MicroDuck',
}
HD1910_TASKS = {
    'flat': 'Mjlab-Velocity-Flat-MicroDuck-HD1910',
    'backlash': 'Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910',
}


def stop_child(process, timeout=20):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_monitored(command, *, project, env, log_path, item, save, stop_timeout=20):
    """Guard training, export and evaluation alike; terminate only our process group."""
    started = time.monotonic()
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=project, stdout=log,
                                   stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)
        item['active_pid'] = process.pid
        item['active_command'] = command
        try:
            save()
            while process.poll() is None:
                free = shutil.disk_usage(ROOT).free / 2**30
                item['disk_free_gib'] = free
                item['stage_elapsed_seconds'] = time.monotonic() - started
                if free < 2.5:
                    raise RuntimeError('Free disk space below 2.5 GiB; stopped this stage.')
                save()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            stop_child(process, timeout=stop_timeout)
            raise
        finally:
            item['active_pid'] = None
            item['last_returncode'] = process.returncode
            save()
    if process.returncode:
        raise RuntimeError(f'Command exited {process.returncode}; see {log_path}')
    return time.monotonic() - started


def main():
    def stop_requested(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, stop_requested)
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke', 'benchmark', 'train'], required=True)
    parser.add_argument('--variant', choices=['both', *TASKS], default='both')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--finalize', action='store_true', help='Export and evaluate completed training runs')
    parser.add_argument('--robot', choices=['xl330', 'hd1910'], default='xl330')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    project = PROJECT if args.robot == 'xl330' else ROOT / 'microduck_rl_hd1910'
    tasks = TASKS if args.robot == 'xl330' else HD1910_TASKS
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = '' if args.robot == 'xl330' else '_hd1910'
    output = args.output or ROOT / 'runs' / f'{stamp}{suffix}_{args.phase}'
    output = output.resolve()
    output.mkdir(parents=True)
    manifest = {'started': datetime.now().astimezone().isoformat(), 'pid': os.getpid(),
                'args': {**vars(args), 'output': str(output)}, 'project': str(project),
                'python': str(VENV / 'bin/python'), 'runs': [], 'status': 'running'}

    def save():
        tmp = output / 'status.tmp'
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(output / 'status.json')

    save()
    # The installed entry point resolves this source tree without changing the
    # shared venv or the source imported by an already-running XL330 process.
    env = dict(os.environ, PYTHONUNBUFFERED='1', WANDB_MODE='disabled',
               MUJOCO_GL='egl', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONPATH=str(project / 'src'))
    env['PATH'] = str(VENV / 'bin') + os.pathsep + env.get('PATH', '')
    env.pop('MICRODUCK_WARM_START', None)
    selected = tasks if args.variant == 'both' else {args.variant: tasks[args.variant]}
    try:
        run_pair(args, output, manifest, project, tasks, selected, suffix, stamp, env, save)
    except BaseException as exc:
        manifest['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        if manifest['runs'] and manifest['runs'][-1]['status'] != 'completed':
            manifest['runs'][-1]['status'] = manifest['status']
        manifest['finished'] = datetime.now().astimezone().isoformat()
        save()
        raise
    manifest['status'] = 'completed'
    manifest['finished'] = datetime.now().astimezone().isoformat()
    save()
    print(f'Completed: {output / "status.json"}', flush=True)


def run_pair(args, output, manifest, project, tasks, selected, suffix, stamp, env, save):
    for variant, task in selected.items():
        subprocess.run(['python3', str(ROOT / 'scripts/preflight.py'), '--output',
                        str(output / f'{variant}_preflight.json')], check=True)
        experiment = f'local{suffix}_{variant}_{args.phase}'
        run_name = f'{variant}_{stamp}'
        command = [str(VENV / 'bin/train'), task,
                   '--env.scene.num-envs', str(args.num_envs),
                   '--agent.max-iterations', str(args.iterations),
                   '--agent.logger', 'tensorboard',
                   '--agent.experiment-name', experiment,
                   '--agent.run-name', run_name, '--agent.save-interval', '250',
                   '--agent.seed', '42', '--agent.upload-model', 'False']
        item = {'variant': variant, 'task': task, 'command': command,
                'stdout': str(output / f'{variant}.log'), 'experiment': experiment,
                'run_name': run_name, 'status': 'running'}
        manifest['runs'].append(item)
        save()
        print(f'Starting {variant}: {item["stdout"]}', flush=True)
        item['elapsed_seconds'] = run_monitored(command, project=project, env=env,
                                               log_path=Path(item['stdout']), item=item, save=save)
        item['returncode'] = 0
        item['status'] = 'completed'
        logroot = project / 'logs/rsl_rl' / experiment
        item['training_dirs'] = [str(p) for p in logroot.glob(f'*_{run_name}')]
        save()
        if len(item['training_dirs']) != 1:
            raise RuntimeError(f'Expected one training directory: {item["training_dirs"]}')
        if args.finalize:
            train_dir = Path(item['training_dirs'][0])
            checkpoint = max(train_dir.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
            artifact_dir = output / variant
            artifact_dir.mkdir(exist_ok=True)
            item['checkpoint'] = str(checkpoint)
            item['artifacts'] = str(artifact_dir)
            item['status'] = 'exporting'
            save()
            export_cmd = [str(VENV / 'bin/python'), str(project / 'scripts/export.py'), task,
                          '--checkpoint-file', str(checkpoint), '--num-envs', '1',
                          '--onnx-file', str(artifact_dir / 'policy.onnx'), '--device', 'cuda:0']
            run_monitored(export_cmd, project=project, env=env,
                          log_path=artifact_dir / 'export.log', item=item, save=save)
            item['status'] = 'evaluating'
            save()
            for eval_variant, eval_task in tasks.items():
                eval_dir = artifact_dir / f'eval_{eval_variant}'
                eval_cmd = [str(VENV / 'bin/python'), str(ROOT / 'scripts/evaluate.py'),
                            '--task', eval_task, '--checkpoint', str(checkpoint),
                            '--output', str(eval_dir), '--onnx', str(artifact_dir / 'policy.onnx')]
                if eval_variant == variant:
                    eval_cmd.append('--video')
                if args.robot == 'hd1910':
                    eval_cmd.extend(['--commands', 'stand', 'forward', 'forward_fast',
                                     'backward', 'left', 'turn'])
                run_monitored(eval_cmd, project=project, env=env,
                              log_path=artifact_dir / f'eval_{eval_variant}.log', item=item, save=save)
            item['status'] = 'completed'
            save()


if __name__ == '__main__':
    main()
