"""Stationary two-axis trimming using only duck1's four hip joints.

Trim coordinates are integer encoder-equivalent ticks ordered (pitch, roll).
Model changes are left/right hip pitch (+u, -u) and hip roll (+v, +v):
robot_walk.xml gives pitch axes +Y/-Y and both roll axes +X at model zero.
The measured response, not this geometry, determines the feet-on-ground sign.
This routine does not identify anatomical joint zeros or alter calibration.
"""
from __future__ import annotations

import math
import operator

import numpy as np

from .config import calibration_arrays, require_calibration
from .policy import HOME, JOINT_LIMITS, JOINT_NAMES, q_to_ticks, ticks_to_q

MAX_OFFSET_TICKS = 44
STEP_TICKS = 5
MAX_PROBE_TICKS = MAX_OFFSET_TICKS
MIN_JOINT_RESPONSE_TICKS = 2
AXIS_JOINT_SIGNS = (((3, 1), (13, -1)), ((2, 1), (12, 1)))


def selected_indices(cfg):
    """Return the verified duck1 ID -> model index mapping for four hip joints."""
    joints = cfg.get('joints', [])
    if [joint.get('name') for joint in joints] != list(JOINT_NAMES):
        raise ValueError('关节顺序/名称与模型不一致')
    indices = {2: 1, 3: 2, 12: 10, 13: 11}
    names = ('left_hip_roll', 'left_hip_pitch', 'right_hip_roll', 'right_hip_pitch')
    for (sid, index), name in zip(indices.items(), names):
        joint = joints[index]
        if (joint.get('id') != sid or joint.get('name') != name
                or joint.get('direction') not in (-1, 1)
                or not joint.get('mapping_verified')):
            raise ValueError(f'必须先确认 ID{sid}={name} 及其正方向')
    ids = [joint.get('id') for joint in joints]
    if len(set(ids)) != len(ids):
        raise ValueError('舵机 ID 不得重复')
    return indices


def _integer(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} 必须为 {lower}..{upper} 内的整数')
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} 必须为整数') from exc
    if not lower <= result <= upper:
        raise ValueError(f'{name} 必须在 {lower}..{upper} 内')
    return result


def _offset(value, maximum=MAX_OFFSET_TICKS):
    try:
        if len(value) != 2:
            raise ValueError('offset_ticks 必须依次包含 pitch、roll 两个整数')
        return tuple(_integer(item, 'offset_ticks', -maximum, maximum) for item in value)
    except TypeError as exc:
        raise ValueError('offset_ticks 必须依次包含 pitch、roll 两个整数') from exc


def hip_targets(cfg, offset_ticks):
    """Return targets for IDs 2,3,12,13 only; never mutate cfg or other goals."""
    indices = selected_indices(cfg)
    require_calibration(cfg, imu=False)
    pitch, roll = _offset(offset_ticks)
    zero, direction = calibration_arrays(cfg)
    targets = q_to_ticks(HOME, zero, direction)
    for sid, amount in ((3, pitch), (13, -pitch), (2, roll), (12, roll)):
        index = indices[sid]
        targets[index] += int(direction[index]) * amount
    angles = ticks_to_q(targets, zero, direction)
    for index in indices.values():
        if not JOINT_LIMITS[index, 0] <= angles[index] <= JOINT_LIMITS[index, 1]:
            raise ValueError(f'{JOINT_NAMES[index]} 目标超出模型角度范围')
    return {sid: int(targets[index]) for sid, index in indices.items()}


class _RejectedSample(ValueError):
    pass


def _measurement(raw):
    try:
        result = {key: float(raw[key]) for key in
                  ('pitch_deg', 'roll_deg', 'pitch_std_deg', 'roll_std_deg')}
        gravity = [float(value) for value in raw['gravity']]
    except (KeyError, TypeError, ValueError) as exc:
        raise _RejectedSample('invalid_measurement') from exc
    if (len(gravity) != 3 or not all(math.isfinite(value) for value in
                                    list(result.values()) + gravity)):
        raise _RejectedSample('invalid_measurement')
    norm = math.sqrt(sum(value * value for value in gravity))
    if not .95 <= norm <= 1.05:
        raise _RejectedSample('invalid_gravity_norm')
    for axis in ('pitch', 'roll'):
        if not 0 <= result[f'{axis}_std_deg'] <= .20:
            raise _RejectedSample(f'unstable_{axis}_window')
    try:
        ticks = {int(sid): float(value) for sid, value in raw['joint_ticks'].items()}
        directions = {int(sid): value for sid, value in raw['joint_directions'].items()}
        for sid in (2, 3, 12, 13):
            if (not math.isfinite(ticks[sid]) or not 0 <= ticks[sid] <= 4095
                    or isinstance(directions[sid], (bool, np.bool_))
                    or directions[sid] not in (-1, 1)):
                raise ValueError('invalid joint sample')
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise _RejectedSample('invalid_joint_feedback') from exc
    result['joint_ticks'] = {str(sid): ticks[sid] for sid in (2, 3, 12, 13)}
    result['joint_directions'] = {str(sid): int(directions[sid]) for sid in (2, 3, 12, 13)}
    result['gravity'] = gravity
    result['tilt_deg'] = math.degrees(math.acos(max(-1., min(1., -gravity[2] / norm))))
    if result['tilt_deg'] >= 12:
        raise _RejectedSample('body_tilt_limit')
    return result


def _attitude(measurement):
    return np.array([measurement['pitch_deg'], measurement['roll_deg']], dtype=float)


def _noise(first, second):
    combined = sum(first[f'{axis}_std_deg'] ** 2 + second[f'{axis}_std_deg'] ** 2
                   for axis in ('pitch', 'roll'))
    return max(.20, 4 * math.sqrt(combined))


def _joint_progress(first, second, axis, sign):
    """Both members of a mirrored pair must actually travel in the desired direction."""
    progress = {}
    for sid, model_sign in AXIS_JOINT_SIGNS[axis]:
        key = str(sid)
        direction = first['joint_directions'][key]
        if direction != second['joint_directions'][key]:
            raise _RejectedSample('joint_direction_changed')
        progress[key] = (second['joint_ticks'][key] - first['joint_ticks'][key]) * direction * model_sign * sign
    return progress


def level_hips(plant, report, *, tolerance_deg=.5, max_ticks=MAX_OFFSET_TICKS,
               probe_ticks=9, iterations=24):
    """Measure a 2x2 response then trim in <=5 tick steps per coordinate.

    ``plant.sample()`` returns stationary mean pitch_deg, roll_deg, both
    pitch_std_deg/roll_std_deg, normalized body-frame ``gravity``, and mappings
    ``joint_ticks``/``joint_directions`` containing IDs 2, 3, 12, 13. Probe
    amplitude increases up to 44 ticks only while each sample remains safe;
    both active joints must move and the body response must exceed noise.
    ``plant.move((pitch_ticks, roll_ticks))`` moves only the four selected hips.
    Rows of ``response_deg_per_tick`` are (pitch, roll); columns are trim
    coordinates (pitch, roll).  ``report`` is updated before hardware calls,
    remains JSON-compatible, and is returned.  Hardware exceptions propagate;
    the caller must preserve torque and its last sent targets on failure.

    Measured rejection freezes at the latest offset, without automatic rollback.
    Convergence requires both angles within tolerance in two stable windows.
    There are no writes to configuration, joint zero, EEPROM or model HOME here.
    """
    max_ticks = _integer(max_ticks, 'max_ticks', 1, MAX_OFFSET_TICKS)
    probe_ticks = _integer(probe_ticks, 'probe_ticks', 1, min(9, max_ticks))
    iterations = _integer(iterations, 'iterations', 1, 30)
    if (isinstance(tolerance_deg, bool) or not isinstance(tolerance_deg, (int, float))
            or not math.isfinite(tolerance_deg) or not .2 <= tolerance_deg <= 1.):
        raise ValueError('tolerance_deg 必须在 0.2..1.0 度内')
    report.update(status='running', reason=None, offset_ticks=[0, 0],
                  pending_offset_ticks=None, samples=[], moves=[],
                  tolerance_deg=float(tolerance_deg), max_ticks=max_ticks,
                  probe_ticks=probe_ticks, coordinate_order=['pitch', 'roll'],
                  calibration_saved=False, probes=[], corrections=[])
    baseline = None
    last = None

    def finish(status, reason):
        report.update(status=status, reason=reason)
        if last is not None:
            report['final'] = last.copy()
            report['vertical_within_tolerance'] = bool(np.all(np.abs(_attitude(last)) <= tolerance_deg))
        return report

    def sample(stage):
        nonlocal last
        raw = plant.sample()
        try:
            value = _measurement(raw)
        except _RejectedSample as exc:
            report['samples'].append({'stage': stage,
                                      'offset_ticks': report['offset_ticks'].copy(),
                                      'rejected': str(exc)})
            raise
        last = value
        report['samples'].append({'stage': stage,
                                  'offset_ticks': report['offset_ticks'].copy(), **value})
        if baseline is not None and value['tilt_deg'] > baseline['tilt_deg'] + 2:
            raise _RejectedSample('tilt_growth_limit')
        return value

    def move(offset):
        offset = _offset(offset, max_ticks)
        if list(offset) == report['offset_ticks']:
            return
        report['pending_offset_ticks'] = list(offset)
        plant.move(offset)
        report['offset_ticks'] = list(offset)
        report['pending_offset_ticks'] = None
        report['moves'].append(list(offset))

    def stable(previous, current):
        return bool(np.max(np.abs(_attitude(previous) - _attitude(current))) <= .3)

    def within(value):
        return bool(np.all(np.abs(_attitude(value)) <= tolerance_deg))

    def converged(value, stage):
        if not within(value):
            return False
        confirmation = sample(stage)
        if not stable(value, confirmation):
            raise _RejectedSample('unstable_confirmation')
        return within(confirmation)

    def returned_baseline(stage):
        restored = sample(stage)
        if not stable(baseline, restored):
            # Allow one further settling window, but never redefine zero to
            # hide a persistent hysteresis or support shift.
            restored = sample(stage + '_settled')
            if not stable(baseline, restored):
                raise _RejectedSample('baseline_did_not_return')
        return restored

    try:
        first = sample('baseline_1')
        baseline = sample('baseline_2')
        if not stable(first, baseline):
            raise _RejectedSample('unstable_baseline')
        report['baseline'] = baseline.copy()
        if within(first) and within(baseline):
            return finish('converged', 'already_within_tolerance')

        columns = []
        accepted_amplitudes = []
        current = baseline
        for axis, axis_name in enumerate(('pitch', 'roll')):
            def probe(sign, origin):
                side = 'positive' if sign == 1 else 'negative'
                limit = min(MAX_PROBE_TICKS, max_ticks)
                amplitudes = sorted(set([min(probe_ticks * multiplier, limit)
                                         for multiplier in range(1, math.ceil(limit / probe_ticks) + 1)]))
                for amplitude in amplitudes:
                    offset = [0, 0]
                    offset[axis] = sign * amplitude
                    move(offset)
                    measured = sample(f'{axis_name}_{side}_probe_{amplitude}')
                    change = _attitude(measured) - _attitude(origin)
                    size = float(np.linalg.norm(change))
                    noise = _noise(measured, origin)
                    travel = _joint_progress(origin, measured, axis, sign)
                    followed = all(value >= MIN_JOINT_RESPONSE_TICKS for value in travel.values())
                    useful = size >= noise
                    report['probes'].append(dict(axis=axis_name, sign=sign,
                                                  amplitude_ticks=amplitude,
                                                  joint_progress_ticks=travel,
                                                  body_change_deg=change.tolist(),
                                                  noise_threshold_deg=noise,
                                                  both_joints_followed=followed,
                                                  body_response_measurable=useful))
                    if any(value <= -MIN_JOINT_RESPONSE_TICKS for value in travel.values()):
                        raise _RejectedSample('reversed_joint_probe_response')
                    if size > 2.5:
                        raise _RejectedSample('excessive_probe_response')
                    if followed and useful:
                        return change, size, amplitude
                raise _RejectedSample('weak_body_probe_response' if followed
                                      else 'insufficient_joint_probe_response')

            positive_change, positive_size, positive_amplitude = probe(1, current)
            move((0, 0))
            restored = returned_baseline(f'baseline_after_{axis_name}_positive_probe')
            negative_change, negative_size, negative_amplitude = probe(-1, restored)
            opposition = -float(positive_change @ negative_change) / (positive_size * negative_size)
            slope_ratio = (positive_size / positive_amplitude) / (negative_size / negative_amplitude)
            if opposition < .7 or not .4 <= slope_ratio <= 2.5:
                raise _RejectedSample('inconsistent_probe_response')
            columns.append((positive_change - negative_change) / (positive_amplitude + negative_amplitude))
            accepted_amplitudes.extend((positive_amplitude, negative_amplitude))
            move((0, 0))
            current = returned_baseline(f'baseline_after_{axis_name}_negative_probe')

        jacobian = np.column_stack(columns)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        report['response_deg_per_tick'] = jacobian.tolist()
        report['response_singular_values'] = singular.tolist()
        if singular[-1] < .015:
            raise _RejectedSample('weak_or_dependent_response')
        if singular[0] > .4:
            raise _RejectedSample('excessive_response')
        condition = float(singular[0] / singular[-1])
        report['response_condition'] = condition
        if condition > 20:
            raise _RejectedSample('ill_conditioned_response')

        # A five-tick correction can be absorbed by the same mechanical play
        # that required a larger probe. Accumulate small corrections in the
        # learned direction before calling a plateau, without larger steps.
        plateau_limit = min(MAX_PROBE_TICKS, max(accepted_amplitudes) + probe_ticks)
        report['plateau_travel_limit_ticks'] = plateau_limit
        plateau_origin = current
        plateau_offset = np.array(report['offset_ticks'], dtype=int)
        for iteration in range(iterations):
            if converged(current, f'confirmation_{iteration}'):
                return finish('converged', 'attitude_tolerance_met')
            current = last
            error = _attitude(current)
            desired = np.linalg.solve(jacobian, -error)
            scale = max(1., float(np.max(np.abs(desired))) / STEP_TICKS)
            delta = np.rint(desired / scale).astype(int)
            previous_offset = np.array(report['offset_ticks'], dtype=int)
            next_offset = np.clip(previous_offset + delta, -max_ticks, max_ticks)
            if np.array_equal(next_offset, previous_offset):
                raise _RejectedSample('offset_limit' if np.any(np.abs(previous_offset) == max_ticks)
                                      else 'encoder_resolution_limit')
            move(tuple(map(int, next_offset)))
            new = sample(f'trim_{iteration + 1}')
            change = _attitude(new) - error
            expected_change = jacobian @ (next_offset - previous_offset)
            improvement = float(np.linalg.norm(error) - np.linalg.norm(_attitude(new)))
            progress = {}
            for axis, command_delta in enumerate(next_offset - previous_offset):
                if command_delta:
                    progress.update(_joint_progress(current, new, axis,
                                                    1 if command_delta > 0 else -1))
            report['corrections'].append(dict(iteration=iteration + 1,
                                              offset_ticks=next_offset.tolist(),
                                              joint_progress_ticks=progress,
                                              body_change_deg=change.tolist(),
                                              expected_body_change_deg=expected_change.tolist()))
            if any(value <= -MIN_JOINT_RESPONSE_TICKS for value in progress.values()):
                raise _RejectedSample('reversed_joint_trim_response')
            if improvement < -.3 or (change @ expected_change < 0 and np.linalg.norm(change) > .20):
                raise _RejectedSample('response_worsened')
            cumulative_improvement = float(np.linalg.norm(_attitude(plateau_origin)) - np.linalg.norm(_attitude(new)))
            if cumulative_improvement >= .05:
                plateau_origin = new
                plateau_offset = next_offset.copy()
            elif np.max(np.abs(next_offset - plateau_offset)) >= plateau_limit:
                following = True
                for axis, travel in enumerate(next_offset - plateau_offset):
                    if abs(travel) >= STEP_TICKS:
                        progress = _joint_progress(plateau_origin, new, axis, 1 if travel > 0 else -1)
                        following = following and all(value >= MIN_JOINT_RESPONSE_TICKS for value in progress.values())
                raise _RejectedSample('body_response_stalled' if following else 'joint_response_stalled')
            current = new
        if converged(current, 'final_confirmation'):
            return finish('converged', 'attitude_tolerance_met')
        return finish('not_converged', 'iteration_limit')
    except _RejectedSample as exc:
        return finish('not_converged', str(exc))
