"""Passive motion-log analysis; servo voltage is never treated as IMU voltage."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics


LIMITS = {'imu_vdd': (1.71, 3.6), 'imu_vdd_io': (1.08, 3.6)}


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def feedback_min(feedback):
    values = [number(v.get('voltage_v')) for v in feedback.values() if isinstance(v, dict)]
    return min((v for v in values if v is not None), default=None)


def inventory(iio_root=Path('/sys/bus/iio/devices'), hwmon_root=Path('/sys/class/hwmon')):
    """Discover ADCs without claiming that an unlabelled channel measures 3V3."""
    channels = []
    for device in sorted(Path(iio_root).glob('iio:device*')):
        name = (device / 'name').read_text().strip() if (device / 'name').exists() else device.name
        for raw in sorted(device.glob('in_voltage*_raw')):
            scale = device / (raw.name[:-4] + '_scale')
            if not scale.exists():
                scale = device / 'in_voltage_scale'
            try:
                channels.append({'device': name, 'raw_path': str(raw), 'raw': raw.read_text().strip(),
                                 'scale_mv_per_count': scale.read_text().strip() if scale.exists() else None,
                                 'measurement_point': 'unverified; not an IMU voltage measurement'})
            except OSError as exc:
                channels.append({'raw_path': str(raw), 'error': str(exc)})
    return {'iio_voltage_channels': channels,
            'hwmon_voltage_paths': [str(p) for p in sorted(Path(hwmon_root).glob('hwmon*/in*_input'))],
            'imu_3v3_measurement_verified': False,
            'note': 'ADC存在不代表接到了IMU电源；未确认接线/分压/参考电压前不换算。'}


def read_rails(path, session_name):
    """External calibrated CSV: session,t,point,voltage_v; t matches motion t."""
    rows = []
    if path is None:
        return rows
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'session', 't', 'point', 'voltage_v'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('电压CSV须包含 session,t,point,voltage_v，t须同步到该动作日志的秒数')
        for i, row in enumerate(reader, 2):
            if row['session'] != session_name:
                continue
            try:
                stamp, volts = float(row['t']), float(row['voltage_v'])
            except (ValueError, TypeError) as exc:
                raise ValueError(f'电压CSV第{i}行数值无效') from exc
            if not math.isfinite(stamp) or not math.isfinite(volts) or volts < 0:
                raise ValueError(f'电压CSV第{i}行须为有限数值，电压不得为负')
            if row['point'] not in (*LIMITS, 'hat_5v'):
                raise ValueError(f'电压CSV第{i}行测点须为imu_vdd / imu_vdd_io / hat_5v')
            rows.append({'t': stamp, 'point': row['point'], 'voltage_v': volts})
    if not rows:
        raise ValueError('电压CSV没有与该session文件名匹配的样本；拒绝关联其他回合')
    return rows


def rail_summary(rows, fault_t):
    result = {}
    # This is a correlation window, not a sensor timing guarantee. Software's
    # fault observation may follow the actual electrical event.
    for point in sorted({r['point'] for r in rows}):
        selected = sorted((r for r in rows if r['point'] == point), key=lambda r: r['t'])
        near = [] if fault_t is None else [r for r in selected if fault_t - .1 <= r['t'] <= fault_t + .05]
        times = [r['t'] for r in selected]
        limits = LIMITS.get(point)
        result[point] = {
            'samples': len(selected), 'min_v': min(r['voltage_v'] for r in selected),
            'max_gap_s': max((b-a for a, b in zip(times, times[1:])), default=None),
            'fault_window_samples': len(near),
            'fault_window_min_v': min((r['voltage_v'] for r in near), default=None),
            'operating_limits_v': limits,
            'outside_operating_limits_near_fault': bool(limits and any(
                not limits[0] <= r['voltage_v'] <= limits[1] for r in near)),
            'nominal_3v3_sag_below_3v0_near_fault': bool(point in LIMITS and any(
                r['voltage_v'] < 3.0 for r in near)),
        }
    return result


def analyze(session_path, output_root, *, rail_csv=None):
    session_path = Path(session_path)
    session = json.loads(session_path.read_text())
    if not session.get('finished'):
        raise ValueError('回合仍在运行或尚未写完；动作结束后再分析')
    log_path = session_path.with_name(session_path.name.replace('.session.json', '.jsonl'))
    frames, skipped = [], 0
    if log_path.exists():
        with log_path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if row.get('kind') == 'sample' and number(row.get('t')) is not None:
                    volts = feedback_min(row.get('feedback', {}))
                    frames.append({'t': row['t'], 'policy_mode': row.get('policy_mode', ''),
                                   'servo_min_v': volts, 'imu_fault': False})
    frames.sort(key=lambda r: r['t'])
    reason = session.get('stop_reason', '')
    fault = session.get('imu_fault', {})
    fault_t = None
    if number(fault.get('observed_at_monotonic_s')) is not None and number(session.get('policy_start_monotonic_s')) is not None:
        fault_t = fault['observed_at_monotonic_s'] - session['policy_start_monotonic_s']
    fault_time_source = 'observed_at_monotonic_s' if fault_t is not None else 'unavailable'
    imu_failed = bool(fault) or 'IMU' in reason
    if imu_failed and fault_t is None and frames:
        # Old logs have no fault timestamp. Report the last good sample as a
        # reference only; never present it as the exact I2C failure time.
        fault_t = frames[-1]['t']
        fault_time_source = 'last_good_frame_approximation'
    fault_voltage = feedback_min(fault.get('feedback', {}))
    if fault and fault_t is not None:
        frames.append({'t': fault_t, 'policy_mode': 'imu_fault',
                       'servo_min_v': fault_voltage, 'imu_fault': True})
    # Analyze the most recent contiguous roll, rather than mixing separate
    # attempts with different battery conditions in one long Xbox session.
    roll_indices = [i for i, r in enumerate(frames) if r['policy_mode'] == 'roulade']
    roll = []
    if roll_indices:
        end = roll_indices[-1]
        start = end
        while start > 0 and frames[start - 1]['policy_mode'] == 'roulade':
            start -= 1
        roll = frames[start:end + 1]
    pre = [] if not roll else [r['servo_min_v'] for r in frames
                               if roll[0]['t'] - .8 <= r['t'] < roll[0]['t'] and r['servo_min_v'] is not None]
    during = [r['servo_min_v'] for r in roll if r['servo_min_v'] is not None]
    if roll and frames[-2 if fault else -1]['policy_mode'] == 'roulade' and fault_voltage is not None:
        during.append(fault_voltage)
    baseline = statistics.median(pre) if pre else None
    minimum = min(during) if during else None
    rails = rail_summary(read_rails(rail_csv, session_path.name), fault_t)
    out_of_spec = any(v['outside_operating_limits_near_fault'] for v in rails.values())
    sag = any(v['nominal_3v3_sag_below_3v0_near_fault'] for v in rails.values())
    conclusion = '未测量IMU电源，不能确认或排除供电故障；舵机电压只用于关联。'
    if out_of_spec and imu_failed:
        conclusion = '导入的IMU电源数据在故障参考窗口内超出芯片工作范围，支持供电异常；仍须核对测点、时钟同步及波形先后。'
    elif sag and imu_failed:
        conclusion = '导入的3.3V测点在故障参考窗口内低于3.0V；这是压降标记，不等于低于芯片最低工作电压，不能单独确定根因。'
    elif rails:
        conclusion = '本次导入电压未提供故障窗口内IMU超限的证据；稀疏采样或不完整测点不能排除短时电源/地线干扰。'
    report = {
        'session': session_path.name, 'started': session.get('started'), 'finished': session.get('finished'),
        'stop_reason': reason, 'imu_failed': imu_failed,
        'i2c_address_nack': imu_failed and ('[Errno 6]' in reason or '[Errno 6]' in fault.get('error', '')),
        'fault_reference_t_s': fault_t, 'fault_time_source': fault_time_source,
        'fault_frame_servo_min_v': fault_voltage,
        'pre_roll_median_servo_min_v': baseline, 'roll_min_servo_feedback_v': minimum,
        'roll_servo_drop_v': round(baseline - minimum, 3) if baseline is not None and minimum is not None else None,
        'imu_supply_measured': any(p in LIMITS for p in rails),
        'external_measurement_source': str(rail_csv) if rail_csv else None,
        'external_rails': rails, 'conclusion': conclusion,
        'data_quality': {'sample_frames': sum(not r['imu_fault'] for r in frames), 'malformed_lines_skipped': skipped,
                         'log_writer': session.get('log_writer'), 'motion_log_exists': log_path.exists()},
        'limitations': ['舵机反馈不是电池端/HAT5V/IMU3V3测量，控制日志通常约50Hz。',
                        '故障时间是软件发现时间；旧日志仅用最后成功帧作近似参考。',
                        '0.1秒前至0.05秒后的窗口只用于关联，采样间隙可能漏掉更短的毛刺。',
                        '外部CSV须由真实测量生成并人工校准时间；程序不验证物理接线。'],
    }
    destination = Path(output_root) / session_path.name.removesuffix('.session.json')
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'timeline.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('t', 'policy_mode', 'servo_min_v', 'imu_fault'))
        writer.writeheader()
        writer.writerows(frames)
    (destination / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report, destination
