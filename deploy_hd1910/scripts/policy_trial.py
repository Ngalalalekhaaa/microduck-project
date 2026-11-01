"""Supported policy execution with calibrated IMU bias; --seconds 0 has no timer.

Stop the web server before opening this script. Explicit --move required.
The operator supports the trunk throughout; process exit relaxes motors.
With --seconds 0 --zero-after-seconds N, the policy continues at zero velocity
after N seconds instead of exiting.
"""
import argparse
import math
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

IMU_STARTUP_FAILED = 78


def start_imu(cfg, calibrate, factory, *, attempts=3, sleep=time.sleep, calibration_seconds=None):
    from microduck_deploy.imu import ImuError
    for attempt in range(1, attempts+1):
        duration = f'，静止采样约{calibration_seconds:g}秒' if calibrate and calibration_seconds is not None else ''
        print(f'IMU启动/静止校偏 {attempt}/{attempts}{duration}，请保持躯干不动、松开摇杆。', flush=True)
        try:
            options = {'calibrate': calibrate}
            if calibrate and calibration_seconds is not None:
                options['calibration_seconds'] = calibration_seconds
            return factory(cfg, **options)
        except ImuError as exc:
            print(f'IMU启动失败：{exc}', flush=True)
            if attempt == attempts:
                raise
            sleep(1)

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'robot.json',
                   help='Calibration shared with the supported HOME stage')
    p.add_argument('--move',action='store_true')
    p.add_argument('--keyboard',action='store_true',help='数字按键即时控制，无需回车；0停步保持模型，q退出卸力')
    bias=p.add_mutually_exclusive_group(required=True)
    bias.add_argument('--bias-session',type=Path)
    bias.add_argument('--calibrate-imu',action='store_true',help='Hold still for a fresh gyro bias measurement')
    p.add_argument('--imu-calibration-seconds',type=float,default=3.,help='Stationary IMU sample duration after WALK')
    p.add_argument('--policy-home-seconds',type=float,default=1.,help='Final HOME alignment duration before policy execution')
    p.add_argument('--seconds',type=float,default=2,help='Runtime in seconds; 0 runs until stopped')
    p.add_argument('--zero-after-seconds',type=float,default=None,
                   help='Set all velocity commands to zero after N policy seconds and continue; requires --seconds 0, no gamepad')
    p.add_argument('--ramp-seconds',type=float,default=0)
    p.add_argument('--log-only-target-angles', action='store_true',
                   help='Record target bounds and jumps without blocking or clipping policy outputs')
    p.add_argument('--max-target-jump-deg',type=float,default=None,
                   help='Optional per-trial target jump bound, at most 45 degrees')
    p.add_argument('--jaw-command-fd', type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument('--vx',type=float,default=0,help='Forward command in m/s (0..0.3)')
    from microduck_deploy.gamepad import add_gamepad_args, validate_gamepad_args
    add_gamepad_args(p)
    args=p.parse_args(argv)
    validate_gamepad_args(p, args)
    if args.keyboard and (args.gamepad or args.seconds != 0 or args.vx != 0
                          or args.zero_after_seconds is not None or args.ramp_seconds != 0):
        p.error('--keyboard requires --seconds 0 --vx 0, without gamepad, timed-zero or ramp options')
    if not math.isfinite(args.imu_calibration_seconds) or not .5<=args.imu_calibration_seconds<=5:
        p.error('--imu-calibration-seconds must be in [0.5,5]')
    if not math.isfinite(args.policy_home_seconds) or not 0<args.policy_home_seconds<=30:
        p.error('--policy-home-seconds must be in (0,30]')
    if not math.isfinite(args.seconds) or args.seconds<0:
        p.error('--seconds must be finite and nonnegative; 0 runs until stopped')
    if not math.isfinite(args.vx) or not 0<=args.vx<=.3:
        p.error('--vx must be finite and in [0,0.3] m/s')
    if (not math.isfinite(args.ramp_seconds) or args.ramp_seconds<0
            or (args.seconds>0 and args.ramp_seconds>args.seconds)):
        p.error('--ramp-seconds must be in [0,seconds]')
    if args.zero_after_seconds is not None:
        if not math.isfinite(args.zero_after_seconds) or args.zero_after_seconds <= 0:
            p.error('--zero-after-seconds must be finite and positive')
        if args.seconds != 0 or args.gamepad:
            p.error('--zero-after-seconds requires --seconds 0 and no --gamepad')
        if args.ramp_seconds > args.zero_after_seconds:
            p.error('--ramp-seconds must not exceed --zero-after-seconds')
    if args.max_target_jump_deg is not None and (
            not math.isfinite(args.max_target_jump_deg) or not 0<args.max_target_jump_deg<=45):
        p.error('--max-target-jump-deg must be finite and in (0,45]')
    return args


def load_suite_policies(cfg, *, gamepad, policy_loader=None):
    """Load all optional D-pad sessions before any device/IMU is opened."""
    if not gamepad or not cfg['control'].get('model_suite_buttons_enabled', False):
        return None, None, None
    from microduck_deploy.policy import OnnxPolicy, build_obs61, HOME
    loader = OnnxPolicy if policy_loader is None else policy_loader
    policies = []
    for key, kind in (('updated_walking_model', 'walking'),
                      ('sitstand_friction_model', 'sitstand'),
                      ('sitstand_com_model', 'sitstand')):
        policy = loader(ROOT / cfg[key], policy_kind=kind)
        flags = (0., 0., 0.) if kind == 'walking' else (0., 1.)
        for flag in flags:
            policy.infer(build_obs61(np.zeros(3), [0, 0, -1], HOME,
                                    np.zeros(14), np.zeros(14), twist=(flag, 0., 0.)))
        policies.append(policy)
    return tuple(policies)


def load_action_policies(cfg, *, enabled, policy_loader=None):
    """Warm the same action sessions for keyboard and gamepad before device IO."""
    if not enabled:
        return None, None, None
    from microduck_deploy.policy import OnnxPolicy, build_obs61, HOME
    loader = OnnxPolicy if policy_loader is None else policy_loader
    sitstand = ground_pick = roulade = None
    if cfg.get('sitstand_model'):
        sitstand = loader(ROOT/cfg['sitstand_model'], policy_kind='sitstand')
        for flag in (0., 1.):
            sitstand.infer(build_obs61(np.zeros(3), [0, 0, -1], HOME,
                                      np.zeros(14), np.zeros(14), twist=(flag, 0, 0)))
    if cfg.get('ground_pick_model'):
        if sitstand is None:
            raise ValueError('俯身动作需要站坐权重用于返回站立保持')
        ground_pick = loader(ROOT/cfg['ground_pick_model'], policy_kind='ground_pick')
        for phase in np.linspace(0., 1., 9):
            ground_pick.infer(build_obs61(np.zeros(3), [0, 0, -1], HOME,
                np.zeros(14), np.zeros(14),
                twist=(np.cos(2*np.pi*phase), np.sin(2*np.pi*phase), 0)))
    if cfg['control'].get('roulade_enabled', False):
        if sitstand is None:
            raise ValueError('滚翻需要站坐权重用于返回站立保持')
        roulade = loader(ROOT/cfg['roulade_model'], policy_kind='roulade')
        for _ in range(3):
            roulade.infer(build_obs61(np.zeros(3), [0, 0, -1], HOME,
                                     np.zeros(14), np.zeros(14)))
    return sitstand, ground_pick, roulade


def main():
    args=parse_args()
    keyboard = None
    from microduck_deploy.config import load_config
    from microduck_deploy.runtime import Controller, check_posture, check_feedback
    from microduck_deploy.policy import OnnxPolicy, ticks_to_q, HOME
    from microduck_deploy.config import calibration_arrays
    from microduck_deploy.imu_process import ProcessImuReader
    from microduck_deploy.imu import ImuError
    from microduck_deploy.servo import ServoBus
    cfg=load_config(args.config)
    from microduck_deploy.jaw import JawCommandReader, jaw_ids
    jaw_source = JawCommandReader(args.jaw_command_fd) if args.jaw_command_fd is not None else None
    cfg['control']['home_duration_s']=args.policy_home_seconds
    if args.log_only_target_angles:
        cfg['control']['log_only_target_angles'] = True
    if cfg['control'].get('log_only_target_angles', False):
        print('策略目标角度范围/单周期目标跳变：仅记录，不拦截、不裁剪。', flush=True)
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
    # Load and warm up all ONNX sessions before IMU acquisition/torque takeover.
    from microduck_deploy.policy import build_obs61
    warmup_obs = build_obs61(np.zeros(3), [0, 0, -1], HOME, np.zeros(14), np.zeros(14))
    for _ in range(3):
        policy.infer(warmup_obs)
    from microduck_deploy.keyboard_walking import load_walking_policies, walking_options
    keyboard_walking_policies = (load_walking_policies(cfg, ROOT, base_policy=policy)
                                 if args.keyboard else {})
    sitstand_policy, ground_pick_policy, roulade_policy = load_action_policies(
        cfg, enabled=args.gamepad or args.keyboard)
    from microduck_deploy.keyboard_sitstand import load_sitstand_policies, sitstand_options
    keyboard_sitstand_policies = (load_sitstand_policies(cfg, ROOT, base_policy=sitstand_policy)
                                  if args.keyboard else {})
    if args.keyboard:
        from microduck_deploy.terminal_command import TerminalCommand
        from microduck_deploy.terminal_actions import TerminalActionCommand
        keyboard = (TerminalActionCommand(
            max_vx=args.max_vx, max_vy=args.max_vy, max_wz=args.max_wz,
            single_key=True,
            sitstand_enabled=True, ground_pick_enabled=ground_pick_policy is not None,
            roulade_enabled=roulade_policy is not None,
            walking_models={key: item['label'] for key, item in walking_options(cfg).items()},
            sitstand_models={key: item['label'] for key, item in sitstand_options(cfg).items()},
            default_sitstand_key=cfg.get('keyboard_sitstand_default'),
            roulade_timeout_s=cfg['control'].get('roulade_timeout_s', 5.))
            if sitstand_policy is not None else TerminalCommand(
                max_vx=args.max_vx, max_vy=args.max_vy, max_wz=args.max_wz,
                single_key=True))
    updated_walking_policy, sitstand_friction_policy, sitstand_com_policy = load_suite_policies(
        cfg, gamepad=args.gamepad)
    suite_active = updated_walking_policy is not None
    from microduck_deploy.gamepad import (GamepadReader, GamepadCommand, GamepadStop,
                                        SitStandGamepadCommand, ActionGamepadCommand,
                                        RouladeGamepadCommand)
    pad = GamepadReader(args.gamepad_device) if args.gamepad else None
    command_class = (RouladeGamepadCommand if roulade_policy is not None else
                     ActionGamepadCommand if ground_pick_policy is not None else
                     SitStandGamepadCommand if sitstand_policy is not None else GamepadCommand)
    if suite_active:
        from microduck_deploy.gamepad import SuiteGamepadCommand
        command_class = SuiteGamepadCommand
    command_options = ({'roulade_enabled': True,
                        'roulade_timeout_s': cfg['control'].get('roulade_timeout_s', 5.),
                        'ground_pick_enabled': ground_pick_policy is not None}
                       if roulade_policy is not None else {})
    command_provider = (command_class(pad, max_vx=args.max_vx, max_vy=args.max_vy,
                                     max_wz=args.max_wz, **command_options) if pad else None)
    if keyboard is not None:
        command_provider = keyboard
    if args.gamepad and sitstand_policy is not None:
        print('Y键：停步后坐下／再次按Y站起；站起等待3秒后可推摇杆行走。', flush=True)
    if args.gamepad and ground_pick_policy is not None:
        print('RB右肩键：停步后俯身，完整4秒后站立；动作期间摇杆/Y/RB不插队，B可停止卸力。', flush=True)
    if args.gamepad and roulade_policy is not None:
        print('LB左肩键：停步后单次前滚翻；恢复直立并稳定后保持站立。'
              '动作中B/断连立即卸力，未恢复则超时卸力。', flush=True)
    elif args.gamepad and cfg.get('roulade_model'):
        print('滚翻权重已配置但未启用：LB不执行动作。', flush=True)
    if suite_active:
        print('方向键：↑停步后切换原/新行走模型；↓单次前滚翻；'
              '←选择摩擦坐站，→选择重心坐站。坐着按左右键先用当前模型站起；'
              '站起等待3秒后可操作。', flush=True)
    try:
        reader=start_imu(cfg,args.calibrate_imu,ProcessImuReader,
                         calibration_seconds=args.imu_calibration_seconds)
    except ImuError:
        if pad: pad.close()
        print('IMU未就绪，策略未启动，未发送策略动作。', flush=True)
        return IMU_STARTUP_FAILED
    except BaseException:
        if pad: pad.close()
        raise
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
            state=reader.latest(cfg['control']['imu_max_age_s']);tilt=check_posture(q,state,cfg)
            print(json.dumps({'event':'preflight','body_tilt_deg':tilt,
                'torque_states':sorted(torques),'joint_home_errors_deg':[
                    {'id':sid,'name':cfg['joints'][i]['name'],
                     'ticks':feedback[sid].position_ticks,
                     'error_deg':float(np.rad2deg(q[i]-HOME[i]))}
                    for i,sid in enumerate(ids)]}),flush=True)
            ctl=Controller(cfg,bus=bus,imu_reader=reader,policy=policy,jaw_source=jaw_source,
                           sitstand_policy=sitstand_policy,ground_pick_policy=ground_pick_policy,
                           roulade_policy=roulade_policy,
                           updated_walking_policy=updated_walking_policy,
                           sitstand_friction_policy=sitstand_friction_policy,
                           sitstand_com_policy=sitstand_com_policy,
                           keyboard_walking_policies=keyboard_walking_policies,
                           keyboard_sitstand_policies=keyboard_sitstand_policies)
            plan=ctl._home_plan(q)
            print(json.dumps({'move':args.move,'take_over_hold':takeover,'body_tilt_deg':tilt,'home_ticks':plan['home_ticks'],'policy_seconds':args.seconds,'zero_after_seconds':args.zero_after_seconds,'command':[args.vx,0,0]}),flush=True)
            if args.move:
                try:
                    result=ctl.execute(seconds=args.seconds,twist=(args.vx,0,0),motion=True,
                                       take_over_hold=takeover,command_ramp_s=args.ramp_seconds,
                                       command_provider=command_provider,
                                       zero_command_after_s=args.zero_after_seconds)
                    print(json.dumps(result),flush=True)
                except GamepadStop as exc:
                    print(str(exc), flush=True)
                finally:
                    bus.set_torque(ids + jaw_ids(cfg),False,verify=True)
                    print('全部舵机卸力已回读确认。',flush=True)
    finally:
        if keyboard is not None: keyboard.close()
        reader.close()
        if pad: pad.close()
        if jaw_source is not None: jaw_source.close()

if __name__=='__main__':sys.exit(main())
