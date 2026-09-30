"""Simulation-only comparison of deployed policy target jumps and joint motion."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('WANDB_MODE', 'disabled')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import onnxruntime as ort
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deploy_hd1910'))
from microduck_deploy.policy import HOME, JOINT_NAMES, JOINT_LIMITS


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--seconds', type=float, default=10)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT/'microduck_rl_hd1910/logs/rsl_rl/local_hd1910_flat_train/2026-09-28_01-36-32_flat_20260928_013621/model_3999.pt'
    model = ROOT/'deploy_hd1910/models/hd1910_flat.onnx'
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    task = 'Mjlab-Velocity-Flat-MicroDuck-HD1910'
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 123
    cfg.auto_reset = False
    cfg.episode_length_s = args.seconds + 2
    cfg.curriculum = {}
    cfg.events.pop('push_robot', None)
    twist = cfg.commands['twist']
    for field in ('rel_standing_envs', 'rel_heading_envs', 'rel_turn_in_place_envs',
                  'rel_forward_envs', 'rel_world_envs', 'init_velocity_prob'):
        setattr(twist, field, 0.)
    twist.resampling_time_range = (1000., 1000.)
    for field in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z'):
        setattr(twist.ranges, field, (0., 0.))
    for name in ('head_pose', 'body_pose'):
        cfg.commands[name].ranges = tuple((0., 0.) for _ in cfg.commands[name].ranges)
        cfg.commands[name].resampling_time_range = (1000., 1000.)
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    agent = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = load_runner_cls(task)(wrapped, asdict(agent), device='cuda:0')
    runner.load(str(checkpoint), map_location='cuda:0')
    policy = runner.get_inference_policy(device='cuda:0')
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    onnx = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
    action_term = env.action_manager.get_term('joint_pos')
    assert list(action_term.target_names) == list(JOINT_NAMES)
    report = {'task': task, 'num_envs': args.num_envs, 'seconds': args.seconds,
              'seed':123, 'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
              'action_clip':str(action_term.cfg.clip), 'simulation_only':True,
              'scope':'first episode, startup randomization retained, no pushes; no deployment guards applied',
              'results':{}}
    home = torch.tensor(HOME.copy(), device='cuda:0', dtype=torch.float32)
    bounds = JOINT_LIMITS.copy(); bounds[[1,10]] = np.deg2rad([-30,30])
    try:
        for vx, ramp in [(0.05,3.),(.1,3.),(.2,3.),(.2,0.)]:
            name = f'vx{vx:g}_ramp{ramp:g}'
            env.reset(seed=123)
            live = env.command_manager.get_term('twist')
            alive = torch.ones(args.num_envs, dtype=torch.bool, device='cuda:0')
            previous = home.expand(args.num_envs,-1).clone()
            trace = {k:[] for k in ('t','command','target','jump','q','qd','vx','tilt','valid')}
            parity = []
            for step in range(round(args.seconds/env.step_dt)):
                t = (step+1)*env.step_dt
                fraction = 1. if ramp == 0 else min(1., t/ramp)
                command = vx*fraction*fraction*(3-2*fraction)
                live.command.zero_(); live.command[:,0] = command
                for term in ('head_pose','body_pose'):
                    env.command_manager.get_term(term).command.zero_()
                obs = wrapped.get_observations()
                # Update command slots even if the wrapper caches prior observations.
                obs['actor'][:,48:51] = torch.tensor([command,0.,0.],device='cuda:0')
                actions = policy(obs)
                assert torch.isfinite(actions).all()
                if step < 5:
                    pred = onnx.run(None, {onnx.get_inputs()[0].name:obs['actor'][:1].cpu().numpy()})[0]
                    parity.append(float(np.max(np.abs(pred-actions[:1].cpu().numpy()))))
                target = home + actions
                jump = target-previous
                pre_q = env.scene['robot'].data.joint_pos[:,action_term.target_ids].clone()
                pre_qd = env.scene['robot'].data.joint_vel[:,action_term.target_ids].clone()
                wrapped.step(actions)
                assert torch.max(torch.abs(action_term._processed_actions-target)) < 1e-5
                data = env.scene['robot'].data
                finite = torch.isfinite(data.joint_pos).all(1) & torch.isfinite(data.root_link_lin_vel_b).all(1)
                valid = alive & ~env.reset_terminated & finite
                values = dict(t=t,command=command,target=target,jump=jump,q=pre_q,qd=pre_qd,
                              vx=data.root_link_lin_vel_b[:,0],
                              tilt=torch.rad2deg(torch.acos((-data.projected_gravity_b[:,2]).clamp(-1,1))), valid=valid)
                for key, value in values.items():
                    trace[key].append(value.cpu().numpy().copy() if isinstance(value,torch.Tensor) else value)
                previous = target.clone()
                alive &= ~(env.reset_terminated | env.reset_time_outs) & finite
            arrays = {k:np.asarray(v) for k,v in trace.items()}
            np.savez_compressed(args.output/(name+'.npz'), **arrays)
            mask = arrays['valid']; settled = mask & (arrays['t'][:,None]>=4)
            jumps = np.abs(arrays['jump']); exceeds = np.any(jumps>.6,axis=-1)&mask
            outside = np.any((arrays['target']<bounds[:,0])|(arrays['target']>bounds[:,1]),axis=-1)&mask
            moving_envs = []
            for i in range(args.num_envs):
                v = arrays['vx'][settled[:,i],i]
                moving_envs.append(float(v.mean()) if len(v) else None)
            summary = {'vx':vx,'ramp_s':ramp,'survival_fraction':float(alive.float().mean()),
                       'mean_vx_after4s':float(arrays['vx'][settled].mean()),
                       'mean_vx_per_env_after4s':moving_envs,
                       'jump_deg_p95_p99_max':np.rad2deg(np.percentile(jumps.max(-1)[mask],[95,99,100])).tolist(),
                       'left_knee_jump_deg_p95_p99_max':np.rad2deg(np.percentile(jumps[:,:,3][mask],[95,99,100])).tolist(),
                       'frames_exceeding_34_38deg':int(exceeds.sum()),
                       'envs_exceeding_34_38deg':int(exceeds.any(0).sum()),
                       'frames_outside_deployed_target_bounds':int(outside.sum()),
                       'median_abs_tracking_error_deg_after4s':np.rad2deg(np.median(np.abs(arrays['target']-arrays['q'])[settled],axis=0)).tolist(),
                       'onnx_max_difference':max(parity)}
            assert max(parity)<1e-4
            report['results'][name] = summary
            (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            print(name,json.dumps(summary),flush=True)
    finally:
        wrapped.close()


if __name__ == '__main__':
    main()
