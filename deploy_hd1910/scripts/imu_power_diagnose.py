#!/usr/bin/env python3
"""无硬件写入：分析IMU故障、电压关联，并可等待下一次动作日志。"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from microduck_deploy.power_diagnostics import analyze, inventory


def completed_sessions(logs):
    result = []
    for path in sorted(Path(logs).glob('motion_*.session.json')):
        try:
            if json.loads(path.read_text()).get('finished'):
                result.append(path)
        except (OSError, json.JSONDecodeError):
            continue  # A running controller may be finishing the JSON file.
    return result


def show(path, args):
    report, destination = analyze(path, args.output, rail_csv=args.rail_csv)
    print(json.dumps({k: report[k] for k in ('session', 'stop_reason', 'pre_roll_median_servo_min_v',
          'roll_min_servo_feedback_v', 'roll_servo_drop_v', 'conclusion')}, ensure_ascii=False, indent=2), flush=True)
    print(f'结果目录：{destination}\n  report.json：结论和证据\n  timeline.csv：逐帧舵机电压', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--latest', action='store_true', help='分析最近一个已结束的动作回合')
    mode.add_argument('--session', type=Path, help='指定 .session.json 文件')
    mode.add_argument('--watch', action='store_true', help='等待启动后结束的动作回合；Ctrl+C退出')
    mode.add_argument('--inventory', action='store_true', help='列出板载ADC；不猜测测点')
    p.add_argument('--logs', type=Path, default=ROOT / 'logs')
    p.add_argument('--output', type=Path, default=ROOT / 'logs' / 'imu_power_diagnosis')
    p.add_argument('--rail-csv', type=Path, help='可选：真实测量且同步到回合t的外部电压CSV，格式见文档')
    p.add_argument('--seconds', type=float, default=0, help='watch持续秒数；0表示持续等待')
    args = p.parse_args(argv)
    if not 0 <= args.seconds < float('inf'):
        p.error('--seconds须为有限非负数')
    if args.rail_csv and not (args.session or args.latest):
        p.error('--rail-csv请与--session或--latest一起使用')
    try:
        if args.inventory:
            print(json.dumps(inventory(), ensure_ascii=False, indent=2))
        elif args.session:
            show(args.session, args)
        elif args.latest:
            paths = completed_sessions(args.logs)
            if not paths:
                raise ValueError('没有已结束的motion_*.session.json日志')
            show(paths[-1], args)
        else:
            seen = set(completed_sessions(args.logs))
            begin = time.monotonic()
            print('等待新的动作日志结束；按原有Xbox流程操作。程序不访问I²C/串口，不启动动作。', flush=True)
            while args.seconds == 0 or time.monotonic() - begin < args.seconds:
                for path in completed_sessions(args.logs):
                    if path not in seen:
                        show(path, args)
                        seen.add(path)
                time.sleep(.5)
    except KeyboardInterrupt:
        print('\n诊断监听结束；手柄服务不受影响。')
    except (OSError, ValueError) as exc:
        p.exit(2, f'诊断失败：{exc}\n')


if __name__ == '__main__':
    main()
