"""Pure diagnostic capture of software calibration inputs; no hardware or files.

Candidate zero may lie outside 0..4095: it is a coordinate offset, not a servo
command. Only captured reference and HOME ticks are checked here, matching the
existing calibration checks; full model travel is deliberately not assessed.
"""
from __future__ import annotations

import numpy as np

from .config import require_mapping
from .policy import HOME, policy_calibration_reference


def capture_report(cfg, samples, pose):
    """Return one or 12 observations and candidate values without mutating cfg."""
    require_mapping(cfg)
    values = np.asarray(samples)
    if values.shape not in ((1, 14), (12, 14)) or not np.all(np.isfinite(values)):
        raise ValueError('标定采样记录需要1或12组、每组14个有限位置读数')
    sample_count = len(values)
    reference = policy_calibration_reference(pose)
    directions = np.asarray([j['direction'] for j in cfg['joints']])
    ticks = np.rint(np.mean(values, axis=0)).astype(int)
    spans = np.ptp(values, axis=0)
    zeros = ticks - directions * reference * (4096 / (2 * np.pi))
    home = zeros + directions * HOME * (4096 / (2 * np.pi))
    rows = []
    for index, joint in enumerate(cfg['joints']):
        issues = []
        if sample_count == 12 and spans[index] > 8:
            issues.append(f'采样跨度{float(spans[index]):g}刻度超过8')
        if ticks[index] < 0 or ticks[index] > 4095:
            issues.append(f'参考刻度{int(ticks[index])}超出0..4095')
        if home[index] < 0 or home[index] > 4095:
            issues.append(f'HOME未取整刻度{float(home[index]):.6f}超出0..4095')
        rows.append({'id': joint['id'], 'name': joint['name'],
                     'direction': joint['direction'], 'reference_tick': int(ticks[index]),
                     'sample_min': float(np.min(values[:, index])),
                     'sample_max': float(np.max(values[:, index])),
                     'span_ticks': float(spans[index]),
                     'reference_q_rad': float(reference[index]),
                     'candidate_zero_tick': float(zeros[index]),
                     'home_tick_unrounded': float(home[index]), 'issues': issues})
    return {'kind': 'software_calibration_capture', 'declared_pose': pose,
            'sample_count': sample_count,
            'sample_mode': 'single_snapshot' if sample_count == 1 else 'stable_average',
            'stability_checked': sample_count == 12,
            'joint_ids': [j['id'] for j in cfg['joints']],
            'directions': directions.tolist(), 'raw_samples': values.tolist(),
            'rounded_mean_ticks': ticks.tolist(), 'spans_ticks': spans.tolist(),
            'reference_q_rad': reference.tolist(), 'candidate_zero_ticks': zeros.tolist(),
            'home_ticks_unrounded': home.tolist(), 'joints': rows,
            'hardware_writes': False,
            'note': ('原始采样及候选计算记录；保存此文件不表示软件标定已通过。'
                     '每次读取中各舵机经总线依次回复，不保证14关节严格同时采样。'
                     + ('单次读取未检查姿态稳定性；仍须实际摆在声明的参考姿态。'
                        if sample_count == 1 else '12次读取取平均并检查采样跨度。'))}


def describe_issues(report):
    return '; '.join(f"ID{row['id']} {row['name']}: " + '，'.join(row['issues'])
                     for row in report['joints'] if row['issues'])
