"""One supported policy trial, up to 60 seconds, with calibrated IMU bias.

Stop the web server before opening this script. Explicit --move required.
The operator supports the trunk throughout; motion completion relaxes motors.
"""
import argparse
import math
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--move',action='store_true')
    bias=p.add_mutually_exclusive_group(required=True)
    bias.add_argument('--bias-session',type=Path)
    bias.add_argument('--calibrate-imu',action='store_true',help='Hold still for a fresh gyro bias measurement')
    p.add_argument('--seconds',type=float,default=2)
    p.add_argument('--ramp-seconds',type=float,default=0)
    p.add_argument('--log-only-target-angles', action='store_true',
                   help='Log target angle/jump violations without stopping; preserve encoder and feedback checks')
    p.add_argument('--max-target-jump-deg',type=float,default=None,
                   help='Optional per-trial target jump bound, at most 45 degrees')
    p.add_argument('--vx',type=float,default=0,help='Forward command in m/s (0..0.3)')
    args=p.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0<args.seconds<=60:
        p.error('--seconds must be finite and in (0,60]')
    if not math.isfinite(args.vx) or not 0<=args.vx<=.3:
        p.error('--vx must be finite and in [0,0.3] m/s')
    if not math.isfinite(args.ramp_seconds) or not 0<=args.ramp_seconds<=args.seconds:
        p.error('--ramp-seconds must be in [0,seconds]')
    if args.max_target_jump_deg is not None and (
            not math.isfinite(args.max_target_jump_deg) or not 0<args.max_target_jump_deg<=45):
        p.error('--max-target-jump-deg must be finite and in (0,45]')
    return args

def main():
    args=parse_args()
    from microduck_deploy.config import load_config
    from microduck_deploy.runtime import Controller, check_posture, check_feedback
    from microduck_deploy.policy import OnnxPolicy, ticks_to_q, HOME
    from microduck_deploy.config import calibration_arrays
    from microduck_deploy.imu_process import ProcessImuReader
    from microduck_deploy.servo import ServoBus
    cfg=load_config(ROOT/'robot.json')
    if args.log_only_target_angles:
        cfg['control']['log_only_target_angles'] = True
        print('本次策略目标角度范围/单周期跳变：仅记录，不拦截、不裁剪。其他检查保留。', flush=True)
    if args.max_target_jump_deg is not None:
        cfg['control']['max_target_jump_rad']=math.radians(args.max_target_jump_deg)
        print(f'本次目标跳变上限：{args.max_target_jump_deg:g}°；仅本次生效。',flush=True)
    if args.bias_session:
        saved=json.loads(args.bias_session.read_text())
        if cfg['imu']!=saved['config']['imu'] or saved['status']!='completed':
            raise ValueError('Bias source must match IMU installation and be a completed session')
        age=(datetime.now().astimezone()-datetime.fromisoformat(saved['finished'])).total_seconds()
        if not 0<=age<=900:raise ValueError('Use a successful stationary calibration from the last 15 minutes')
        cfg['imu']['gyro_bias_rad_s']=saved['gyro_bias_rad_s']
        cfg['trial_bias_source']=str(args.bias_session)
    policy=OnnxPolicy(ROOT/cfg['model'])
    reader=ProcessImuReader(cfg,calibrate=args.calibrate_imu)
    try:
        if args.calibrate_imu:
            cfg['imu']['gyro_bias_rad_s']=reader.imu.gyro_bias_rad_s.tolist()
            cfg['trial_bias_source']='fresh stationary calibration '+datetime.now().astimezone().isoformat()
        with ServoBus(**cfg['serial']) as bus:
            ids=[j['id'] for j in cfg['joints']]
            conf={sid:bus.read_configuration(sid) for sid in ids}
            torques={c['torque'] for c in conf.values()}
            if torques not in ({0},{1}):raise ValueError('Mixed torque states: no trial started')
            takeover=torques=={1}
            if takeover and any(c['ram_gains']!={'p':5,'d':0,'i':0} for c in conf.values()):
                raise ValueError('Held joints must already use P5/D0/I0')
            feedback=bus.read_feedback_many(ids,sync=True);check_feedback(feedback,cfg)
            z,d=calibration_arrays(cfg)
            q=ticks_to_q([feedback[i].position_ticks for i in ids],z,d)
            state=reader.latest(.06);tilt=check_posture(q,state,cfg)
            print(json.dumps({'event':'preflight','body_tilt_deg':tilt,
                'torque_states':sorted(torques),'joint_home_errors_deg':[
                    {'id':sid,'name':cfg['joints'][i]['name'],
                     'ticks':feedback[sid].position_ticks,
                     'error_deg':float(np.rad2deg(q[i]-HOME[i]))}
                    for i,sid in enumerate(ids)]}),flush=True)
            ctl=Controller(cfg,bus=bus,imu_reader=reader,policy=policy)
            plan=ctl._home_plan(q)
            print(json.dumps({'move':args.move,'take_over_hold':takeover,'body_tilt_deg':tilt,'home_ticks':plan['home_ticks'],'policy_seconds':args.seconds,'command':[args.vx,0,0]}),flush=True)
            if args.move:
                try:
                    result=ctl.execute(seconds=args.seconds,twist=(args.vx,0,0),motion=True,
                                       take_over_hold=takeover,command_ramp_s=args.ramp_seconds)
                    print(json.dumps(result),flush=True)
                finally:
                    bus.set_torque(ids,False,verify=True)
                    print('全部舵机卸力已回读确认。',flush=True)
    finally:reader.close()

if __name__=='__main__':main()
