"""Read, release, and save corrected hip-roll HOME (IDs 2, 11 or duck1's 12)."""
import argparse
import copy
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from microduck_deploy.config import load_config, require_calibration, save_config, calibration_arrays
from microduck_deploy.policy import HOME, q_to_ticks, ticks_to_q
from supported_home import PositionOnlyBus


def joint_index(cfg, servo_id=11):
    require_calibration(cfg, imu=False)
    names = {2: 'left_hip_roll', 11: 'right_hip_roll', 12: 'right_hip_roll'}
    if servo_id not in names:
        raise ValueError('仅支持2号、11号或duck1的12号髋侧倾标定')
    index = next((i for i, joint in enumerate(cfg['joints']) if joint['id'] == servo_id), None)
    if index is None or cfg['joints'][index]['name'] != names[servo_id]:
        raise ValueError(f'ID{servo_id}不是已标定的{names[servo_id]}')
    return index


def shifted_config(cfg, tick, servo_id=11):
    """Shift one encoder mapping, including its saved reference, by whole ticks."""
    index = joint_index(cfg, servo_id)
    if type(tick) is not int or not 0 <= tick <= 4095:
        raise ValueError('HOME刻度必须是0..4095的整数')
    old_home = q_to_ticks(HOME, *calibration_arrays(cfg))
    delta = tick - int(old_home[index])
    result = copy.deepcopy(cfg)
    result['joints'][index]['zero_tick'] += delta
    # Keep the existing straight reference numerically consistent. Replacing
    # its pose with HOME would break the supported recovery entry point.
    result['calibration']['captured_ticks'][index] += delta
    if not 0 <= result['calibration']['captured_ticks'][index] <= 4095:
        raise ValueError('修正会使保存的参考刻度超出单圈范围')
    home = q_to_ticks(HOME, *calibration_arrays(result))
    result['calibration']['home_ticks'] = home.tolist()
    result['calibration'].setdefault('manual_home_adjustments', []).append({
        'id': servo_id, 'old_home_tick': int(old_home[index]), 'new_home_tick': tick,
        'delta_ticks': delta, 'utc': datetime.now(timezone.utc).isoformat(),
        'method': 'manual correction with all other joints holding HOME',
    })
    return result


def snapshot(bus, ids):
    return {
        str(sid): {'torque': bus.read(sid, 40, 1)[0],
                   'tick': (feedback := bus.read_feedback(sid)).position_ticks,
                   'goal': feedback.goal_ticks}
        for sid in ids
    }


def check_hold(cfg, readings, *, released=False, ids=(11,)):
    home = q_to_ticks(HOME, *calibration_arrays(cfg))
    tolerance = cfg['control']['home_tolerance_rad'] * 4096 / (2 * 3.141592653589793)
    for joint, target in zip(cfg['joints'], home):
        sid = joint['id']
        row = readings[str(sid)]
        if sid in ids and released:
            if row['torque'] != 0:
                raise ValueError(f'{sid}号未卸力，不能保存手调位置')
            continue
        if row['torque'] != 1 or row['goal'] != int(target) or abs(row['tick'] - target) > tolerance:
            raise ValueError(f'ID{sid}未保持现有HOME；先执行慢速HOME命令')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hold_selected(bus, targets):
    """Treat a target write as motion: HD1910 may enable torque on address 42."""
    for sid, tick in targets.items():
        if bus.read(sid, 40, 1)[0] != 0:
            raise ValueError(f'ID{sid}恢复保持前必须处于卸力状态')
        if abs(bus.read_feedback(sid).position_ticks - tick) > 2:
            raise ValueError(f'ID{sid}当前位置偏离已保存HOME={tick}；扶稳并摆回附近后重试hold')
    bus.sync_positions(targets)
    for sid, tick in targets.items():
        # Read the actual target register, not only feedback register 67.
        if int.from_bytes(bus.read(sid, 42, 2), 'little') != tick:
            raise ValueError(f'ID{sid}新HOME目标回读不匹配，拒绝恢复上力')
        if abs(bus.read_feedback(sid).position_ticks - tick) > 2:
            raise ValueError(f'ID{sid}写入目标后位置偏离新HOME')
        state = bus.read(sid, 40, 1)[0]
        if state not in (0, 1):
            raise ValueError(f'ID{sid}扭矩状态异常：{state}')
        if state == 0:
            bus.set_torque([sid], True, verify=True)
        # Both auto-enable and explicit-enable paths require a final readback.
        if bus.read(sid, 40, 1)[0] != 1:
            raise ValueError(f'ID{sid}恢复上力未确认')
        if (int.from_bytes(bus.read(sid, 42, 2), 'little') != tick
                or abs(bus.read_feedback(sid).position_ticks - tick) > 2):
            raise ValueError(f'ID{sid}上力后目标或位置不匹配')


def finish_hold(bus, selected, targets, session, session_path):
    """Keep saved calibration on any failure; release only the selected joints."""
    try:
        session_path.write_text(json.dumps(session, indent=2) + '\n')
        hold_selected(bus, targets)
        session['phase'] = 'holding_new_home'
        session_path.write_text(json.dumps(session, indent=2) + '\n')
    except BaseException:
        try:
            bus.set_torque(selected, False, verify=True)
            session['phase'] = 'config_saved_selected_torque_off'
        except Exception as exc:
            session['phase'] = 'config_saved_torque_unknown'
            print(f'{selected}号卸力未确认：{exc}；请继续托稳机器人。', file=sys.stderr)
        try:
            session_path.write_text(json.dumps(session, indent=2) + '\n')
        except OSError:
            pass
        print('软件标定已保存，但新HOME保持未完成；请托稳并摆回新HOME附近，再执行hold。其他关节标定未改。',
              file=sys.stderr)
        raise
    print(f'{selected}号已在新位置恢复上力，其他关节标定不变；后续HOME和行走使用新零位。')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status', 'release', 'save', 'hold'),
                        help='hold只恢复已保存的新HOME保持，不重新标定')
    parser.add_argument('--ids', type=int, nargs='+', choices=(2, 11, 12), default=[11],
                        help='要调整的髋侧倾舵机；duck1右侧使用 --ids 12，旧机器人默认11号')
    parser.add_argument('--samples', type=int, choices=range(1, 21), default=5,
                        help='保存时的位置采样数；--samples 1记录单次位置，仍复核上力前未移动')
    parser.add_argument('--config', type=Path, default=ROOT / 'robot.json')
    parser.add_argument('--config-only', action='store_true', help='status只读配置，不访问串口')
    args = parser.parse_args(argv)
    if args.config_only and args.command != 'status':
        parser.error('--config-only只适用于status')
    path = args.config.resolve()
    cfg = load_config(path)
    selected = sorted(set(args.ids))
    indices = {sid: joint_index(cfg, sid) for sid in selected}
    home = q_to_ticks(HOME, *calibration_arrays(cfg))
    for sid, index in indices.items():
        joint = cfg['joints'][index]
        print(f"{sid}号 {joint['name']}: HOME={int(home[index])}, zero_tick={joint['zero_tick']}, direction={joint['direction']}")
    if args.config_only:
        return
    active = subprocess.run(['systemctl', 'is-active', '--quiet', 'microduck-xbox.service']).returncode
    if active == 0:
        raise ValueError('先托稳机器人，再sudo systemctl stop microduck-xbox.service；停止时会卸力')
    logs = path.parent / 'logs'
    logs.mkdir(exist_ok=True)
    session_path = logs / 'joint11_home_adjustment.json'
    (ROOT / 'logs').mkdir(exist_ok=True)
    with (ROOT / 'logs' / 'supported_sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cfg != load_config(path):
            raise ValueError('等待锁期间配置已变化，请重新执行')
        usage = subprocess.run(['fuser', cfg['serial']['port']], capture_output=True, text=True)
        if usage.returncode != 1:
            raise ValueError('串口被占用，请先停止舵机网页/其他控制程序：' + usage.stdout + usage.stderr)
        options = dict(cfg['serial'])
        options['timeout'] = max(.05, options['timeout'])
        with PositionOnlyBus(**options) as bus:
            readings = snapshot(bus, [j['id'] for j in cfg['joints']])
            if args.command == 'status':
                for sid in selected:
                    row = readings[str(sid)]
                    print(f"{sid}号实时刻度={row['tick']}, 目标寄存器={row['goal']}, 扭矩={row['torque']}")
                return
            if args.command == 'release':
                # An active single-joint adjustment can be extended to both
                # hips without rehoming and losing the hand-adjusted position.
                released_ids = [int(sid) for sid, row in readings.items() if row['torque'] == 0]
                if released_ids:
                    session = json.loads(session_path.read_text())
                    previous = session.get('ids', [11])
                    if (session['saved'] or session['config_sha256'] != digest(path)
                            or not set(previous) <= set(selected) or set(released_ids) != set(previous)):
                        raise ValueError('卸力状态与未保存的调整记录不匹配；先执行慢速HOME')
                    check_hold(cfg, readings, released=True, ids=previous)
                else:
                    check_hold(cfg, readings)
                    session = {'config_sha256': digest(path), 'before': readings, 'saved': False}
                session['ids'] = selected
                session_path.write_text(json.dumps(session, indent=2) + '\n')
                bus.set_torque(selected, False, verify=True)
                after = snapshot(bus, [j['id'] for j in cfg['joints']])
                check_hold(cfg, after, released=True, ids=selected)
                print(f'仅{selected}号卸力，其他{14-len(selected)}颗保持HOME。扶稳躯干，调整好后使用相同--ids执行save。')
                return
            session = json.loads(session_path.read_text())
            if args.command == 'hold':
                if not session.get('saved') or session.get('ids', [11]) != selected:
                    raise ValueError('没有与所选ID匹配的已保存调整记录；不能恢复保持')
                targets = {sid: int(home[indices[sid]]) for sid in selected}
                saved_targets = {int(sid): tick for sid, tick in session.get('new_home_ticks', {}).items()}
                if saved_targets != targets:
                    raise ValueError('已保存HOME与当前配置不一致；未执行保持')
                current_hash = digest(path)
                if session.get('saved_config_sha256', current_hash) != current_hash:
                    raise ValueError('保存后配置已变化；未执行保持')
                check_hold(cfg, readings, released=True, ids=selected)
                for sid, tick in targets.items():
                    if abs(readings[str(sid)]['tick'] - tick) > 2:
                        raise ValueError(f'ID{sid}当前位置偏离已保存HOME={tick}；扶稳并摆回附近后重试hold')
                for sid, row in readings.items():
                    if int(sid) not in selected and row['goal'] != session['before'][sid]['goal']:
                        raise ValueError(f'ID{sid}的保持目标已改变')
                if current_hash != digest(path):
                    raise ValueError('检查期间配置已变化；未执行保持')
                finish_hold(bus, selected, targets, session, session_path)
                return
            if session['saved'] or session['config_sha256'] != digest(path):
                raise ValueError('调整记录已保存或配置已变化；不能覆盖标定')
            if session.get('ids', [11]) != selected:
                raise ValueError('save的--ids必须与release一致')
            check_hold(cfg, readings, released=True, ids=selected)
            for sid, row in readings.items():
                if int(sid) not in selected and row['goal'] != session['before'][sid]['goal']:
                    raise ValueError(f'ID{sid}的保持目标已改变')
            samples = {sid: [] for sid in selected}
            for sample_index in range(args.samples):
                for sid in selected:
                    samples[sid].append(bus.read_feedback(sid).position_ticks)
                if sample_index + 1 < args.samples:
                    time.sleep(.05)
            candidate = cfg
            targets = {}
            for sid, values in samples.items():
                if max(values) - min(values) > 2:
                    raise ValueError(f'{sid}号仍在移动；扶住调整好的位置后再执行save')
                targets[sid] = sorted(values)[len(values) // 2]
                candidate = shifted_config(candidate, targets[sid], sid)
            # Verify all other mappings and the stored reference relationship.
            for i in range(14):
                if i not in indices.values() and candidate['joints'][i] != cfg['joints'][i]:
                    raise ValueError('其他关节映射发生变化')
            old_ref_q = ticks_to_q(cfg['calibration']['captured_ticks'], *calibration_arrays(cfg))
            new_ref_q = ticks_to_q(candidate['calibration']['captured_ticks'], *calibration_arrays(candidate))
            if max(abs(old_ref_q - new_ref_q)) > 1e-10:
                raise ValueError('保存的参考姿态关系不一致')
            final = snapshot(bus, [j['id'] for j in cfg['joints']])
            check_hold(cfg, final, released=True, ids=selected)
            for sid, tick in targets.items():
                if abs(final[str(sid)]['tick'] - tick) > 2:
                    raise ValueError(f'{sid}号在采样后移动，请扶稳后重新save')
            if session['config_sha256'] != digest(path):
                raise ValueError('采样期间配置已变化，未保存')
            save_config(path, candidate)
            for sid, tick in targets.items():
                print(f'已备份并保存：{sid}号HOME {int(home[indices[sid]])} → {tick}。', flush=True)
            session.update(saved=True, new_home_ticks=targets, phase='config_saved',
                           saved_config_sha256=digest(path))
            finish_hold(bus, selected, targets, session, session_path)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
