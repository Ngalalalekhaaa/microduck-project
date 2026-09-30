"""Read-only ONNX evaluation in a selected simulation; no hardware imports."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('WANDB_MODE', 'disabled')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--task',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=10)
    p.add_argument('--num-envs',type=int,default=16)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    sess=ort.InferenceSession(str(a.model),sess_options=opts,providers=['CPUExecutionProvider'])
    meta=sess.get_modelmeta().custom_metadata_map
    assert sess.get_inputs()[0].shape==[1,61] and sess.get_outputs()[0].shape==[1,14]
    names=meta['joint_names'].split(',')
    # Exact XgoDuck HOME from the matching public robot configuration.
    home=np.array([0,-.0873,-np.deg2rad(24),-.0049,np.deg2rad(24),.3491,.3491,0,0,
                   0,.0873,np.deg2rad(24),.0049,-np.deg2rad(24)],dtype=np.float32)
    assert np.allclose(home,np.fromstring(meta['default_joint_pos'],sep=','),atol=.000501)
    assert meta['action_scale']=='1.0'
    cfg=load_env_cfg(a.task,play=True)
    cfg.scene.entities['robot']=copy.deepcopy(cfg.scene.entities['robot'])
    cfg.scene.entities['robot'].init_state.joint_pos=dict(zip(names,home.tolist()))
    cfg.scene.num_envs=a.num_envs;cfg.seed=123;cfg.auto_reset=False
    cfg.episode_length_s=a.seconds+2;cfg.curriculum={};cfg.events.pop('push_robot',None)
    twist=cfg.commands['twist']
    for field in ('rel_standing_envs','rel_heading_envs','rel_turn_in_place_envs','rel_forward_envs','rel_world_envs','init_velocity_prob'):
        setattr(twist,field,0.)
    twist.resampling_time_range=(1000.,1000.)
    for field in ('lin_vel_x','lin_vel_y','ang_vel_z'):setattr(twist.ranges,field,(0.,0.))
    for name in ('head_pose','body_pose'):
        cfg.commands[name].ranges=tuple((0.,0.) for _ in cfg.commands[name].ranges)
        cfg.commands[name].resampling_time_range=(1000.,1000.)
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    term=env.action_manager.get_term('joint_pos')
    assert list(term.target_names)==names and term.cfg.clip is None
    h=torch.tensor(home,device='cuda:0')
    report={'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'metadata':meta,
            'task':a.task,'policy_home_rad':home.tolist(),'num_envs':a.num_envs,'seconds':a.seconds,
            'seed':123,'scope':'first episode, no pushes, startup randomization retained, raw action history; no deployment guards',
            'results':{}}
    try:
        for alpha in (0.,.45):
            for vx in (0.,.1,.2,.3):
                env.reset(seed=123)
                prior=np.zeros((a.num_envs,14),dtype=np.float32);filtered=prior.copy()
                previous=home+filtered;alive=np.ones(a.num_envs,dtype=bool)
                trace={k:[] for k in ('time','command','vx','vy','tilt','valid','jump','raw_jump','position','target')}
                initial=env.scene['robot'].data.root_link_pos_w.cpu().numpy().copy()
                last_valid_position=initial.copy()
                failure_time=[None]*a.num_envs
                for step in range(round(a.seconds/env.step_dt)):
                    t=(step+1)*env.step_dt;frac=min(1.,t/3.);cmd=vx*frac*frac*(3-2*frac)
                    live=env.command_manager.get_term('twist');live.command.zero_();live.command[:,0]=cmd
                    obs=env.obs_buf['actor'].cpu().numpy().copy()
                    obs[:,34:48]=prior;obs[:,48:51]=[cmd,0,0];obs[:,51:61]=0
                    action=np.concatenate([sess.run(['actions'],{'obs':row[None,:]})[0] for row in obs])
                    assert np.isfinite(action).all()
                    raw_jump=action-prior
                    filtered=alpha*filtered+(1-alpha)*action
                    target=home+filtered;jump=target-previous
                    env.step(torch.tensor(filtered,device='cuda:0'))
                    assert torch.max(torch.abs(term._processed_actions-(h+torch.tensor(filtered,device='cuda:0'))))<1e-5
                    d=env.scene['robot'].data
                    xyz=d.root_link_pos_w.cpu().numpy().copy();vel=d.root_link_lin_vel_b.cpu().numpy()
                    tilt=torch.rad2deg(torch.acos((-d.projected_gravity_b[:,2]).clamp(-1,1))).cpu().numpy()
                    # In addition to task termination, mark >45deg tilt as a fall.
                    before=alive.copy()
                    task_done=(env.reset_terminated|env.reset_time_outs).cpu().numpy()
                    alive &= ~task_done & (tilt<45) & np.isfinite(xyz).all(1)
                    last_valid_position[alive]=xyz[alive]
                    for i in np.flatnonzero(before&~alive):failure_time[int(i)]=t
                    values=dict(time=t,command=cmd,vx=vel[:,0].copy(),vy=vel[:,1].copy(),tilt=tilt,
                                valid=alive.copy(),jump=jump.copy(),raw_jump=raw_jump.copy(),position=xyz,target=target.copy())
                    for k,v in values.items():trace[k].append(v)
                    prior=action;previous=target
                    # The API requires reset before stepping a terminated world.
                    # Keep these worlds permanently excluded from first-episode stats.
                    if task_done.any():
                        ids=torch.tensor(np.flatnonzero(task_done),device='cuda:0')
                        env.reset(env_ids=ids)
                        prior[task_done]=0;filtered[task_done]=0;previous[task_done]=home
                arrays={k:np.asarray(v) for k,v in trace.items()}
                valid=arrays['valid'];settled=valid&(arrays['time'][:,None]>=4)
                each=[float(arrays['vx'][settled[:,i],i].mean()) if settled[:,i].any() else None for i in range(a.num_envs)]
                eligible=[each[i] for i in range(a.num_envs) if alive[i]]
                name=f'alpha{alpha:g}_vx{vx:g}'
                result={'vx':vx,'alpha':alpha,'survived':int(alive.sum()),
                        'mean_vx_after4s_survivors':float(np.mean(eligible)) if eligible else None,
                        'mean_vx_per_env_after4s':each,
                        'walk_count_speed_above_half_command':int(sum(alive[i] and each[i] is not None and each[i]>max(.02,.5*vx) for i in range(a.num_envs))),
                        'net_displacement_xy_m':(last_valid_position[:,:2]-initial[:,:2]).tolist(),
                        'first_failure_time_s':failure_time,
                        'max_tilt_deg':float(arrays['tilt'].max()),
                        'max_sent_jump_deg_valid':float(np.rad2deg(np.abs(arrays['jump'][valid]).max())) if valid.any() else None,
                        'max_raw_jump_deg_valid':float(np.rad2deg(np.abs(arrays['raw_jump'][valid]).max())) if valid.any() else None,
                        'max_abs_vy_after4s':float(np.abs(arrays['vy'][settled]).max()) if settled.any() else None}
                np.savez_compressed(a.output/(name+'.npz'),**arrays)
                report['results'][name]=result
                (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
                print(name,json.dumps({k:v for k,v in result.items() if k not in ('net_displacement_xy_m','mean_vx_per_env_after4s')}),flush=True)
    finally:env.close()


if __name__=='__main__':main()
