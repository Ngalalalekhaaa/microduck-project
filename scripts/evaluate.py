"""Headless fixed-command evaluation of an upstream Microduck checkpoint.

Reports first-episode survival separately from pre-failure tracking error.
No pushes; startup/reset domain randomization remains enabled. Head/body
commands are held at zero. This is simulation validation, not hardware testing.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('WANDB_MODE', 'disabled')
import numpy as np
import torch
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

COMMANDS = {
    'stand': (0.0, 0.0, 0.0),
    'forward': (0.15, 0.0, 0.0),
    'forward_fast': (0.30, 0.0, 0.0),
    'backward': (-0.15, 0.0, 0.0),
    'left': (0.0, 0.10, 0.0),
    'turn': (0.0, 0.0, 0.5),
}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--commands', nargs='+', choices=list(COMMANDS),
                        default=['stand', 'forward', 'backward', 'left', 'turn'])
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--onnx', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Compare ONNX CPU FP32 with checkpoint FP32, avoiding TF32 roundoff in
    # the validation itself. Training keeps the upstream fast TF32 setting.
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 123
    cfg.auto_reset = False
    cfg.episode_length_s = args.seconds + 1
    cfg.curriculum = {}
    cfg.events.pop('push_robot', None)
    # Fix all command samplers as well as the live buffers; resampling cannot
    # inject different targets between observations and reward calculation.
    twist = cfg.commands['twist']
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.rel_world_envs = 0.0
    twist.init_velocity_prob = 0.0
    twist.resampling_time_range = (1000., 1000.)
    for name in ('head_pose', 'body_pose'):
        cfg.commands[name].ranges = tuple((0., 0.) for _ in cfg.commands[name].ranges)
        cfg.commands[name].resampling_time_range = (1000., 1000.)
    cfg.viewer.distance = 0.75
    cfg.viewer.elevation = -15
    cfg.viewer.azimuth = 135
    cfg.viewer.height, cfg.viewer.width = 480, 640
    cfg.viewer.max_extra_envs = 0
    env = ManagerBasedRlEnv(cfg, device='cuda:0', render_mode='rgb_array' if args.video else None)
    agent = load_rl_cfg(args.task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = load_runner_cls(args.task)(wrapped, asdict(agent), device='cuda:0')
    runner.load(str(args.checkpoint), map_location='cuda:0')
    policy = runner.get_inference_policy(device='cuda:0')
    report = {
        'task': args.task, 'checkpoint': str(args.checkpoint.resolve()),
        'seed': 123, 'num_envs_per_command': args.num_envs,
        'horizon_seconds': args.seconds, 'pushes': False,
        'randomization': 'upstream startup/reset ranges; curricula disabled',
        'metric_scope': 'first episode only; tracking excludes first 1 second and failed steps',
        'results': {},
    }
    onnx_session = None
    if args.onnx:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        onnx_session = ort.InferenceSession(str(args.onnx), options, providers=['CPUExecutionProvider'])
        report['onnx_input_shape'] = onnx_session.get_inputs()[0].shape
        report['onnx_output_shape'] = onnx_session.get_outputs()[0].shape
        assert report['onnx_input_shape'] == [1, 61]
        assert report['onnx_output_shape'] == [1, 14]
    parity_errors = []
    try:
        for index, name in enumerate(args.commands):
            command = torch.tensor(COMMANDS[name], device='cuda:0')
            live = env.command_manager.get_term('twist')
            live.cfg.ranges.lin_vel_x = (float(command[0]),) * 2
            live.cfg.ranges.lin_vel_y = (float(command[1]),) * 2
            live.cfg.ranges.ang_vel_z = (float(command[2]),) * 2
            env.reset(seed=123 + index)
            alive = torch.ones(args.num_envs, dtype=torch.bool, device='cuda:0')
            survived_steps = torch.zeros(args.num_envs, device='cuda:0')
            velocity_samples, tilt_samples = [], []
            nan_failures = 0
            writer = None
            if args.video and name in ('forward', 'forward_fast'):
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(args.output / f'{name}.mp4'), fps=25,
                                            codec='libx264', quality=7)
            with torch.inference_mode():
                for step in range(round(args.seconds / env.step_dt)):
                    live.command[:] = command
                    for term in ('head_pose', 'body_pose'):
                        env.command_manager.get_term(term).command.zero_()
                    obs = wrapped.get_observations()
                    actions = policy(obs)
                    if not torch.isfinite(actions).all():
                        raise RuntimeError('Policy produced non-finite actions')
                    if onnx_session and step < 5:
                        ort_action = onnx_session.run(None, {onnx_session.get_inputs()[0].name:
                                                           obs['actor'][0:1].cpu().numpy()})[0]
                        assert np.isfinite(ort_action).all()
                        parity_errors.append(float(np.max(np.abs(ort_action - actions[0:1].cpu().numpy()))))
                    _, _, _, _ = wrapped.step(actions)
                    terminated = env.reset_terminated.clone()
                    done = terminated | env.reset_time_outs
                    data = env.scene['robot'].data
                    finite = (torch.isfinite(data.root_link_pos_w).all(dim=1)
                              & torch.isfinite(data.joint_pos).all(dim=1)
                              & torch.isfinite(data.joint_vel).all(dim=1))
                    nan_term = env.termination_manager.get_term('nan_state')
                    nan_failures += int((alive & (~finite | nan_term)).sum().item())
                    survived_steps += alive.float()
                    valid = alive & ~terminated & finite
                    if step * env.step_dt >= 1.0 and valid.any():
                        v = torch.cat([data.root_link_lin_vel_b[:, :2],
                                       data.root_link_ang_vel_b[:, 2:3]], dim=1)
                        velocity_samples.append(v[valid].cpu().numpy())
                        tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1))
                        tilt_samples.append(torch.rad2deg(tilt[valid]).cpu().numpy())
                    if writer is not None and step % 2 == 0:
                        writer.append_data(env.render())
                    alive &= ~done & finite
                    if done.any():
                        env.reset(env_ids=done.nonzero().flatten())
            if writer is not None:
                writer.close()
            velocities = np.concatenate(velocity_samples) if velocity_samples else np.empty((0, 3))
            tilts = np.concatenate(tilt_samples) if tilt_samples else np.empty(0)
            result = {
                'command_vx_vy_wz': list(COMMANDS[name]),
                'survival_fraction': float(alive.float().mean().item()),
                'mean_first_episode_seconds': float(survived_steps.mean().item() * env.step_dt),
                'mean_actual_vx_vy_wz': velocities.mean(0).tolist() if len(velocities) else None,
                'mean_abs_error_vx_vy_wz': np.abs(velocities - COMMANDS[name]).mean(0).tolist() if len(velocities) else None,
                'tilt_p95_deg': float(np.percentile(tilts, 95)) if len(tilts) else None,
                'nonfinite_first_episode_states': nan_failures,
                'valid_tracking_samples': len(velocities),
            }
            report['results'][name] = result
            print(name, json.dumps(result), flush=True)
            (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
        if parity_errors:
            report['onnx_max_abs_error_vs_checkpoint'] = max(parity_errors)
            report['onnx_parity_pass'] = max(parity_errors) < 1e-4
            assert report['onnx_parity_pass'], report['onnx_max_abs_error_vs_checkpoint']
        (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        wrapped.close()


if __name__ == '__main__':
    main()
