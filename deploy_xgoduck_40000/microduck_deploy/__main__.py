from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .config import (ROOT, default_config, load_config, save_config, require_mapping,
                     require_calibration, record_reference, axes_rotation, apply_joint_ids)
from .policy import (HOME, STRAIGHT_REFERENCE, JOINT_NAMES, JOINT_POSITIVE_AXES_AT_ZERO, OnnxPolicy,
                     build_obs61, policy_calibration_reference, ticks_to_q)


def output(value):
    def serialize(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if dataclasses.is_dataclass(value):
            return dataclasses.asdict(value)
        raise TypeError(type(value).__name__)
    print(json.dumps(value, ensure_ascii=False, indent=2, default=serialize), flush=True)


def self_test():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    result = {}
    for name, info in manifest["models"].items():
        path = ROOT / info["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != info["sha256"]:
            raise ValueError(f"模型{name}校验和与manifest不符")
        policy = OnnxPolicy(path)
        obs = build_obs61([0, 0, 0], [0, 0, -1], HOME, np.zeros(14), np.zeros(14))
        action = policy.infer(obs)
        timing = []
        for _ in range(100):
            start = time.perf_counter()
            policy.infer(obs)
            timing.append(1000 * (time.perf_counter() - start))
        result[name] = {"sha256": digest, "input": [1, 61], "output": [1, 14],
                        "home_first_action": action.tolist(),
                        "inference_ms_median": float(np.median(timing)),
                        "inference_ms_p95": float(np.percentile(timing, 95))}
    print("两份模型契约/哈希/CPU推理通过。此命令不打开串口或IMU。")
    output(result)
    return result


def doctor(cfg):
    from .imu import discover_i2c
    info = {"platform": platform.platform(), "architecture": platform.machine(),
            "python": sys.version, "os_release": {}, "serial_config": cfg["serial"],
            "serial_devices": sorted(set(glob.glob('/dev/ttyS*') + glob.glob('/dev/ttyUSB*')
                                         + glob.glob('/dev/ttyACM*'))),
            "i2c": discover_i2c(), "spi_devices": glob.glob('/dev/spidev*'),
            "pin_note": "物理针脚编号≠Linux总线号；只读取0x6A/0x6B的WHO_AM_I识别LSM6DSV16X。"}
    try:
        info["os_release"] = dict(line.strip().split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    except OSError:
        pass
    try:
        info["kernel_command_line"] = Path('/proc/cmdline').read_text().strip()
    except OSError:
        pass
    path = ROOT / 'logs' / 'doctor.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    output(info)
    print(f"诊断已保存到 {path}。没有发送舵机动作或改写系统启动配置。")


def main(argv=None):
    parser = argparse.ArgumentParser(description="HD-1910 Microduck 初次部署：先读数、标定，再短时上力。")
    parser.add_argument('--config', type=Path, default=ROOT/'robot.json')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('self-test', help='仅校验模型和CPU推理，不连接硬件')
    sub.add_parser('init', help='创建待标定的robot.json；已有文件不会覆盖')
    sub.add_parser('doctor', help='读系统/设备节点/IMU身份，保存logs/doctor.json')
    sub.add_parser('inspect', help='只读14颗舵机的身份、配置、反馈')
    scan = sub.add_parser('scan', help='PING查找舵机，不分配ID')
    scan.add_argument('--all', action='store_true', help='扫描0..253；默认配置中的14ID和参考嘴部34')
    scan.add_argument('--baudrate', type=int)
    watch = sub.add_parser('watch', help='只读各舵机原始位置，手动转动以核对ID和方向')
    watch.add_argument('--id', type=int)
    watch.add_argument('--seconds', type=float, default=10)
    watch.add_argument('--angles', action='store_true',
                       help='标定后显示模型角度和相对HOME差值；不需IMU，不发动作')
    id_mapping = sub.add_parser('map-ids', help='导入14关节ID表；仅保存本地配置，不确认方向或标定')
    id_mapping.add_argument('--profile', type=Path, required=True,
                            help='关节名到ID的JSON文件，例如servo_ids.user.json')
    mapping = sub.add_parser('map-joint', help='将已实际确认的关节ID、方向存入配置')
    mapping.add_argument('--joint', choices=JOINT_NAMES, required=True)
    mapping.add_argument('--id', type=int, required=True)
    mapping.add_argument('--direction', type=int, choices=(-1, 1), required=True)
    calibration = sub.add_parser('calibrate', help='在已知物理姿态记录软件零位，不写舵机EEPROM')
    calibration.add_argument('--pose', choices=('zero', 'home', 'straight'), required=True)
    planning = sub.add_parser('home-plan', help='只读当前刻度并预览回HOME目标/角差；不需IMU、不发动作')
    planning.add_argument('--from-pose', choices=('straight',))
    planning.add_argument('--move-seconds', type=float)
    sub.add_parser('pose', help='显示14关节HOME/零姿正转轴及姿态图路径')
    imu = sub.add_parser('imu', help='只读IMU，静止校偏后显示gyro/gravity')
    imu.add_argument('--seconds', type=float, default=10)
    imap = sub.add_parser('imu-map', help='保存已核对的传感器到机身轴映射')
    imap.add_argument('--axes', required=True, help='逗号分隔，例如 +z,+y,-x；必须按实际安装确认')
    sub.add_parser('relax', help='关闭配置中14颗舵机扭矩；先托住机器人')
    for name, help_text in [('check', '只读真实传感器并运行策略，不发送动作'),
                            ('home', '上力并缓慢进入HOME、短时保持，结束卸力'),
                            ('run', '进入HOME后运行策略，结束卸力')]:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument('--seconds', type=float, default=5)
        cmd.add_argument('--vx', type=float, default=0)
        cmd.add_argument('--vy', type=float, default=0)
        cmd.add_argument('--wz', type=float, default=0)
        if name != 'check':
            cmd.add_argument('--enable-motion', action='store_true', required=True,
                             help='明确请求本次命令上力；不保存为自动启动设置')
        if name == 'home':
            cmd.add_argument('--from-pose', choices=('straight',),
                             help='仅从已标定且当前接近的直腿参考姿态进入HOME')
            cmd.add_argument('--move-seconds', type=float,
                             help='移动时长；直腿参考默认8秒，允许8..30秒；seconds为到位后保持时长')
    args = parser.parse_args(argv)
    if args.command == 'self-test':
        self_test()
        return 0
    if args.command == 'init':
        if args.config.exists():
            print(f"已存在 {args.config}，保留原配置。")
        else:
            save_config(args.config, default_config())
            print(f"已创建 {args.config}；ID是参考值，方向和零位必须标定。")
        return 0
    if args.command == 'pose':
        output([{'name':name, 'home_rad':float(angle), 'home_deg':float(np.rad2deg(angle)),
                 'straight_deg':float(np.rad2deg(STRAIGHT_REFERENCE[index])),
                 'positive_axis_at_zero':axis}
                for index, (name, angle, axis) in enumerate(zip(JOINT_NAMES, HOME, JOINT_POSITIVE_AXES_AT_ZERO))])
        print(f"姿态图: {ROOT/'docs/姿态与零位.md'}")
        return 0
    cfg = load_config(args.config)
    if args.command == 'map-ids':
        changed = apply_joint_ids(cfg, json.loads(args.profile.read_text(encoding='utf-8')))
        if changed:
            save_config(args.config, cfg)
            print('已导入ID表并备份原配置；变更关节需重新确认方向，全局零位标定已失效。')
        else:
            print('ID表与当前配置一致，保留已有方向和标定，未重写配置。')
        output({'changed_joints': changed,
                'joints': [{'name': j['name'], 'id': j['id'], 'direction': j['direction'],
                            'mapping_verified': j['mapping_verified']} for j in cfg['joints']]})
        print('未连接硬件；此命令不修改舵机内部ID，也不执行动作。')
        return 0
    if args.command == 'doctor':
        doctor(cfg)
        return 0
    if args.command == 'map-joint':
        if not 0 <= args.id <= 253:
            raise ValueError('ID必须在0..253范围内')
        for joint in cfg['joints']:
            if joint['name'] == args.joint:
                joint.update(id=args.id, direction=args.direction, mapping_verified=True, zero_tick=None)
        cfg['calibration'] = None
        # ID remapping may temporarily duplicate another unconfirmed default;
        # allow saving only if no confirmed joints collide.
        confirmed = [j['id'] for j in cfg['joints'] if j['mapping_verified']]
        if len(confirmed) != len(set(confirmed)):
            raise ValueError('已确认关节之间ID重复')
        save_config(args.config, cfg)
        print('已保存软件映射；零位标定已失效，需要重新calibrate。没有写舵机寄存器。')
        return 0
    if args.command == 'imu-map':
        cfg['imu']['mounting_rotation'] = axes_rotation(args.axes.split(','))
        cfg['imu_mount_verified'] = True
        save_config(args.config, cfg)
        print('已保存已核对的轴映射。再次运行imu，核对抬头/左倾/右倾的符号。')
        return 0
    if args.command in ('home', 'run', 'check'):
        from .runtime import Controller
        result = Controller(cfg).execute(seconds=args.seconds, twist=(args.vx,args.vy,args.wz),
                                         motion=args.command != 'check', home_only=args.command == 'home',
                                         from_pose=getattr(args, 'from_pose', None),
                                         move_seconds=getattr(args, 'move_seconds', None))
        output(result)
        return 0
    if args.command == 'imu':
        from .runtime import ImuReader
        if not math.isfinite(args.seconds) or not 0 < args.seconds <= 300:
            raise ValueError('观察时长必须为0..300秒')
        reader = ImuReader(cfg)
        try:
            until = time.monotonic() + args.seconds
            while time.monotonic() < until:
                state = reader.latest(.1)
                print('gyro(rad/s)=',np.round(state.base_ang_vel,3),'gravity=',np.round(state.projected_gravity,3),flush=True)
                time.sleep(.2)
        finally:
            reader.close()
        return 0
    if args.command == 'watch' and args.angles:
        require_calibration(cfg, imu=False)
        if args.id is not None and args.id not in [j['id'] for j in cfg['joints']]:
            raise ValueError('这个ID不在14关节标定表内，无法显示模型角度')
    if args.command == 'calibrate':
        require_mapping(cfg)
    if args.command == 'home-plan':
        from .homing import validate_reference_calibration
        require_calibration(cfg, imu=False)
        if args.from_pose is not None:
            validate_reference_calibration(cfg, args.from_pose)
    from .servo import ServoBus, ServoError, ServoTimeout
    serial = dict(cfg['serial'])
    if args.command == 'scan' and args.baudrate:
        serial['baudrate'] = args.baudrate
    ids = [j['id'] for j in cfg['joints']]
    with ServoBus(**serial) as bus:
        if args.command == 'scan':
            found, errors = [], []
            for sid in range(254) if args.all else sorted(set(ids + [34])):
                try:
                    bus.ping(sid)
                    found.append(sid)
                    print(f'找到ID {sid}',flush=True)
                except ServoTimeout:
                    pass
                except ServoError as exc:
                    errors.append({'id':sid,'error':str(exc)})
            output({'found_ids':found,'errors':errors,'baudrate':serial['baudrate']})
        elif args.command == 'inspect':
            result = []
            for joint in cfg['joints']:
                sid = joint['id']
                item = {'joint':joint['name'],'id':sid}
                try:
                    item.update(identity=bus.read_identity(sid),configuration=bus.read_configuration(sid),
                                feedback=dataclasses.asdict(bus.read_feedback(sid)))
                except ServoError as exc:
                    item['error'] = str(exc)
                result.append(item)
            (ROOT/'logs').mkdir(exist_ok=True)
            (ROOT/'logs/servos.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            output(result)
        elif args.command == 'watch':
            if not math.isfinite(args.seconds) or not 0 < args.seconds <= 300:
                raise ValueError('观察时长必须为0..300秒')
            selected = [args.id] if args.id is not None else ids
            until = time.monotonic() + args.seconds
            while time.monotonic() < until:
                readings = bus.read_feedback_many(selected, sync=len(selected)>1)
                if args.angles:
                    lines=[]
                    for sid in selected:
                        index=ids.index(sid)
                        joint=cfg['joints'][index]
                        raw=readings[sid].position_ticks
                        q=float(ticks_to_q(raw,joint['zero_tick'],joint['direction']))
                        lines.append(f"{joint['name']} ID={sid} raw={raw} "
                                     f"q={np.rad2deg(q):+.2f}deg "
                                     f"HOME_delta={np.rad2deg(q-HOME[index]):+.2f}deg")
                    print('\n'.join(lines)+'\n',flush=True)
                else:
                    print(' '.join(f'{sid}:{readings[sid].position_ticks}' for sid in selected),flush=True)
                time.sleep(.15)
        elif args.command == 'relax':
            bus.set_torque(ids, False, verify=True)
            print('已读回确认14颗舵机卸力；嘴部不属于策略，未改动。')
        elif args.command == 'home-plan':
            from .config import calibration_arrays
            from .homing import plan_home, check_servo_limits
            from .runtime import check_feedback, log_path
            if cfg['serial']['register_profile'] != 'hls_2':
                raise ValueError('回HOME要求HD1910 HLS_2寄存器配置')
            configs = {sid: bus.read_configuration(sid) for sid in ids}
            readings = bus.read_feedback_many(ids, sync=True)
            check_feedback(readings, cfg)
            zero, direction = calibration_arrays(cfg)
            q = ticks_to_q([readings[sid].position_ticks for sid in ids], zero, direction)
            plan = plan_home(cfg, q, from_pose=args.from_pose, move_seconds=args.move_seconds)
            check_servo_limits(plan, configs)
            path = log_path('home_plan').with_suffix('.json')
            path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
            print('关节                    ID   当前刻度  HOME刻度  当前角度   HOME角度    移动角度')
            for row in plan['joints']:
                print(f"{row['name']:<23} {row['id']:>3} {row['current_tick']:>8} {row['home_tick']:>8}"
                      f" {row['current_deg']:>9.2f} {row['home_deg']:>10.2f} {row['move_deg']:>11.2f}")
            print(f"计划移动{plan['duration_s']:.1f}秒，峰值目标速度约"
                  f"{plan['peak_target_speed_deg_s']:.2f}°/s。已保存{path}。")
            print('只读预览通过；未检查实体碰撞或IMU方向，未写目标/增益/扭矩。')
        elif args.command == 'calibrate':
            require_mapping(cfg)
            for sid in ids:
                conf = bus.read_configuration(sid)
                if conf['torque'] != 0 or conf['angle_resolution'] != 1:
                    raise ValueError(f'ID{sid}需先托住并relax，分辨率应为1；当前{conf}')
            samples=[]
            for _ in range(12):
                readings=bus.read_feedback_many(ids,sync=True)
                samples.append([readings[sid].position_ticks for sid in ids])
                time.sleep(.02)
            samples=np.asarray(samples)
            if np.max(np.ptp(samples,axis=0)) > 8:
                raise ValueError('标定期间关节移动超过8刻度，请固定已知姿态后重试')
            ticks=np.rint(np.mean(samples,axis=0)).astype(int)
            record_reference(cfg,ticks,args.pose,policy_calibration_reference(args.pose))
            save_config(args.config,cfg)
            output(cfg['calibration'])
            if args.pose == 'straight':
                print('已记录直腿参考刻度（髋/膝为±58.4119°，并非全零）。下一步：')
                print('./run.sh home-plan --from-pose straight --move-seconds 8')
            print('只保存软件零位，没有调用舵机硬件校零。姿态是否正确仍由实际摆放决定。')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\n已中断。', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'停止: {exc}', file=sys.stderr)
        sys.exit(2)
