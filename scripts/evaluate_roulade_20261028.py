#!/usr/bin/env python3
"""Offline forward-roll evaluation, saved movies and full 50 Hz replay traces."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import time
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_action_comparison import load_inference, cpu_session, Video, FONT
from render_hd1910_controller import load_scene

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'reports/20261028_model_suite/input'
SCENE = ROOT/'microduck_rl_hd1910/src/mjlab_microduck/robot/microduck/scene.xml'


def artifact_path(path):
    """Use repo-relative report links where possible, including CLI paths."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def render_frame(renderer, model, data, xyz, cameras, label, t, stage, tilt, rotation):
    canvas = Image.new('RGB', (1280,720), (17,25,39))
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(FONT,24); small=ImageFont.truetype(FONT,19)
    draw.text((24,8),'Microduck | HD-1910 | 前滚翻 model_1600',font=font,fill='white')
    draw.text((24,45),f'{label}  |  {t:.2f} s  |  {stage}',font=small,fill=(110,213,242))
    vis=mujoco.MjvOption();vis.geomgroup[3:]=0
    for i,camera in enumerate(cameras):
        camera.lookat[:]=[xyz[0],xyz[1],.13]
        renderer.update_scene(data,camera=camera,scene_option=vis)
        canvas.paste(Image.fromarray(renderer.render()),(640*i,88))
    draw=ImageDraw.Draw(canvas)
    draw.text((20,98),'斜前视角',font=small,fill='white',stroke_width=1,stroke_fill='black')
    draw.text((660,98),'侧面视角',font=small,fill='white',stroke_width=1,stroke_fill='black')
    draw.text((24,654),f'躯干高 {xyz[2]*100:.1f} cm  |  倾角 {tilt:.1f}°  |  支撑时前转积分 {rotation:.1f}°',font=small,fill='white')
    draw.text((24,685),'MuJoCo + BAM | 50 Hz 原始策略 | 无重置/裁剪/平滑 | 仿真不等于真机验收',font=small,fill=(157,175,197))
    return canvas


def evaluate(mode,seed,output,videos,record):
    label=f'{mode}_seed{seed}'
    roll_path=INPUT/'model_1600_roulade.onnx'
    stand_path=INPUT/('model_2598_sitstand_com.onnx' if mode=='monitor' else 'model_1699_sitstand.onnx')
    pre,active,post=(0.,5.,0.) if mode=='direct5' else (2.,2. if mode=='handoff2' else 5.,3.)
    log=io.StringIO()
    with redirect_stdout(log):
        module=load_inference()
        model,data,controller,physics=load_scene(SCENE,'hd1910',timestep=.005,lag_steps=4)
        model.opt.iterations=30;model.opt.ls_iterations=50
        with patch.object(module.ort,'InferenceSession',cpu_session):
            policy=module.PolicyInference(model,data,sitstand_onnx_path=str(stand_path),
                roulade_onnx_path=str(roll_path),bam_ctrl=controller,new_cmd_obs=True,
                use_projected_gravity=True,roulade_duration=active)
    (output/(label+'.init.log')).write_text(log.getvalue())
    model.vis.global_.offwidth=640;model.vis.global_.offheight=560
    root_joint=model.joint('trunk_base_freejoint');qa=int(root_joint.qposadr[0])
    rng=np.random.default_rng(seed)
    mujoco.mj_resetData(model,data)
    data.qpos[qa:qa+7]=[0,0,.12,1,0,0,0]
    if seed:
        rot=rng.uniform(-np.deg2rad(2),np.deg2rad(2),size=3)
        q=np.zeros(4);mujoco.mju_euler2Quat(q,rot,'xyz');data.qpos[qa+3:qa+7]=q
    data.qpos[policy.joint_qpos_indices]=policy.default_pose+(rng.uniform(-.01,.01,14) if seed else 0)
    mujoco.mj_forward(model,data);controller.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    joint_ids=np.array(model.actuator_trnid[:,0],dtype=int)
    limits=model.jnt_range[joint_ids]
    head_id=int(model.body('jaw_soft').id)
    feet={int(model.geom(n).id) for n in ['left_foot_collision','right_foot_collision']}
    cameras=[]
    for azimuth in (145,90):
        cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth=azimuth;cam.elevation=-13;cam.distance=.65;cameras.append(cam)
    keys=('time','stage','obs','action','joint_target','joint_pos','joint_vel','xyz','gravity','gyro','tilt_deg',
          'rotation_deg','head_ground','feet_ground','robot_ground','max_contact_force_n','joint_torque')
    trace={k:[] for k in keys}; rotation=0.;forward=0.;finite=True
    monitor=None; completed_at=None; runtime_failure=None
    previous_q=data.qpos[policy.joint_qpos_indices].copy()
    if mode=='monitor':
        sys.path.insert(0,str(ROOT/'deploy_hd1910'))
        from microduck_deploy.runtime import RouladeMonitor,check_posture,constrain_posture_target,validate_targets
        from microduck_deploy.policy import build_obs61,HOME,q_to_ticks
        cfg=json.loads((ROOT/'reports/20261024_recenter_id13/source_after/robot.json').read_text())
        cfg['control']['roulade_enabled']=True
        # Existing Xbox wrapper sets target-angle checks to log-only, as on the recorded robot.
        cfg['control']['log_only_target_angles']=True
        zero=np.array([j['zero_tick'] for j in cfg['joints']]); signs=np.array([j['direction'] for j in cfg['joints']])
        previous_target=HOME.copy()
    writer=Video(videos/(label+'.mp4'),25) if record else None
    renderer=mujoco.Renderer(model,height=560,width=640) if record else None
    snapshots=[];next_snap=0.;begun=time.perf_counter()
    try:
        for step in range(round((pre+active+post)/.02)):
            t=round(step*.02,8)
            rolling=pre<=t<pre+active
            if mode=='monitor':
                rolling=t>=pre and completed_at is None
                state=SimpleNamespace(projected_gravity=policy.get_projected_gravity(),base_ang_vel=policy.get_base_ang_vel())
                current_q=data.qpos[policy.joint_qpos_indices].copy()
                try:
                    tilt_check=check_posture(current_q,state,cfg,policy_mode='roulade' if rolling else 'stand')
                    if rolling:
                        if monitor is None: monitor=RouladeMonitor(t,state,cfg)
                        if monitor.update(t,state,tilt_check): completed_at=t;rolling=False
                    if np.max(np.abs(current_q-previous_q))>.45: raise ValueError('feedback jump >0.45rad')
                except Exception as exc:
                    runtime_failure={'time':t,'reason':str(exc)};break
            policy.command[:]=0
            policy.ort_session=policy.behavior_sessions['roulade'] if rolling else policy.sit_session
            stage='前滚翻' if rolling else ('初始站立' if t<pre else '切回坐站模型站立')
            obs=policy.get_observations().copy()
            assert obs.shape==(61,) and np.all(obs[48:]==0)
            if mode=='monitor':
                obs=build_obs61(state.base_ang_vel,state.projected_gravity,current_q,(current_q-previous_q)/.02,policy.last_action)
                action=policy.ort_session.run([policy.output_name],{policy.input_name:obs.reshape(1,-1)})[0][0]
                policy.last_action=action.copy();target=HOME+action
                try:
                    if not rolling: target,_=constrain_posture_target(target,cfg)
                    validate_targets(target,previous_target,cfg);q_to_ticks(target,zero,signs)
                except Exception as exc:
                    runtime_failure={'time':t,'reason':str(exc)};break
                previous_q=current_q.copy();previous_target=target.copy()
            else:
                action=policy.infer();target=policy.default_pose+action
            if not (np.isfinite(obs).all() and np.isfinite(action).all()): finite=False;break
            if mode=='monitor':policy.set_position_targets(target)
            else:policy.apply_action(action)
            head_contact=False;feet_contact=False;support=False;max_force=0.
            for _ in range(4):
                controller.update();mujoco.mj_step(model,data)
                grounded=False
                for ci in range(data.ncon):
                    c=data.contact[ci];g1,g2=int(c.geom1),int(c.geom2)
                    b1,b2=int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])
                    if b1!=0 and b2!=0: continue
                    body=b2 if b1==0 else b1;geom=g2 if b1==0 else g1
                    grounded=True;head_contact |= body==head_id;feet_contact |= geom in feet
                    force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
                    max_force=max(max_force,float(np.linalg.norm(force[:3])))
                support |= grounded
                if rolling:
                    omega=float(policy.get_base_ang_vel()[1])
                    forward+=omega*.005
                    if grounded: rotation+=omega*.005
            gravity=policy.get_projected_gravity();gyro=policy.get_base_ang_vel()
            tilt=float(np.rad2deg(np.arccos(np.clip(-gravity[2],-1,1))))
            xyz=data.xpos[policy.trunk_base_id].copy()
            values=(t,int(rolling),obs,action,target,data.qpos[policy.joint_qpos_indices].copy(),
                    data.qvel[policy.joint_qvel_indices].copy(),xyz,gravity,gyro,tilt,
                    np.rad2deg(rotation),head_contact,feet_contact,support,max_force,data.actuator_force.copy())
            for k,v in zip(keys,values):trace[k].append(v)
            if writer is not None and step%2==0:
                frame=render_frame(renderer,model,data,xyz,cameras,label,t,stage,tilt,np.rad2deg(rotation))
                writer.write(frame)
                if t+1e-6>=next_snap:
                    snapshots.append(frame.resize((640,360)));next_snap+=1.
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all(): finite=False;break
    finally:
        if renderer:renderer.close()
        if writer:writer.close()
    arrays={k:np.asarray(v) for k,v in trace.items()}
    np.savez_compressed(output/(label+'.npz'),**arrays)
    if snapshots:
        sheet=Image.new('RGB',(1280,360*((len(snapshots)+1)//2)))
        for i,frame in enumerate(snapshots):sheet.paste(frame,((i%2)*640,(i//2)*360))
        sheet.save(videos/(label+'.jpg'),quality=90)
    active_mask=arrays['stage'].astype(bool);final=arrays['time']>=arrays['time'][-1]-.5
    supported_rotation=float(np.max(arrays['rotation_deg']))
    final_tilt=float(np.mean(arrays['tilt_deg'][final]));final_height=float(np.mean(arrays['xyz'][final,2]))
    inverted=bool(np.any(arrays['tilt_deg'][active_mask]>150))
    final_upright=bool(final_tilt<25 and final_height>.09 and np.mean(arrays['feet_ground'][final])>.8)
    result={'case':label,'seed':seed,'mode':mode,'duration_s':pre+active+post,'roll_duration_s':active,
        'finite':finite,'runtime_failure':runtime_failure,'monitor_completed_at_s':completed_at,'inverted_during_roll':inverted,'max_supported_forward_rotation_deg':supported_rotation,
        'net_forward_rotation_deg':float(np.rad2deg(forward)),'final_tilt_deg_mean':final_tilt,
        'final_height_m_mean':final_height,'final_feet_contact_fraction':float(np.mean(arrays['feet_ground'][final])),
        'final_upright':final_upright,'roll_and_stand':bool(finite and runtime_failure is None and inverted and supported_rotation>=330 and final_upright),
        'max_tilt_deg':float(arrays['tilt_deg'].max()),'max_gyro_rad_s':float(np.linalg.norm(arrays['gyro'],axis=1).max()),
        'head_contact_during_roll':bool(np.any(arrays['head_ground'][active_mask])),
        'max_contact_force_n':float(arrays['max_contact_force_n'].max()),
        'max_target_step_deg':float(np.rad2deg(np.abs(np.diff(arrays['joint_target'],axis=0))).max()),
        'max_measured_joint_limit_excess_deg':float(np.rad2deg(np.maximum(limits[:,0]-arrays['joint_pos'],arrays['joint_pos']-limits[:,1])).max()),
        'max_target_limit_excess_deg':float(np.rad2deg(np.maximum(limits[:,0]-arrays['joint_target'],arrays['joint_target']-limits[:,1])).max()),
        'trace':artifact_path(output/(label+'.npz')),'video':artifact_path(videos/(label+'.mp4')) if record else None,
        'wall_seconds':time.perf_counter()-begun}
    (output/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result,physics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modes',nargs='+',choices=('direct5','handoff2','handoff5','monitor'),default=['direct5','handoff2','handoff5'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--no-video',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'reports/20261028_model_suite/roulade')
    p.add_argument('--videos',type=Path,default=ROOT/'videos/20261028_model_suite/roulade')
    a=p.parse_args()
    a.output=a.output.resolve();a.videos=a.videos.resolve()
    a.output.mkdir(parents=True,exist_ok=True);a.videos.mkdir(parents=True,exist_ok=True)
    report={'model':'model_1600_roulade.onnx','sha256':hashlib.sha256((INPUT/'model_1600_roulade.onnx').read_bytes()).hexdigest(),
        'scope':'Local HD1910 nominal CPU BAM simulation; full external training configuration unavailable. No hardware execution.',
        'success_definition':'inversion >150deg plus supported forward rotation >=330deg and final .5s upright <25deg, height>.09m, feet contact >80%',
        'seed_perturbation':'seed0 exact HOME; others initial Euler +/-2deg and joint +/-0.01rad','results':[]}
    for mode in a.modes:
        for seed in a.seeds:
            result,physics=evaluate(mode,seed,a.output,a.videos,not a.no_video and seed==0)
            report['physics']=physics;report['results'].append(result)
            (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(result),flush=True)

if __name__=='__main__':main()
