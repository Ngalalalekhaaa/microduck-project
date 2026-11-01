"""Power diagnostics must distinguish measured IMU voltage from servo droop."""
import csv
import json
from pathlib import Path
import pytest
from microduck_deploy.power_diagnostics import analyze, inventory, read_rails


def make_session(tmp_path, *, fault=True, finished=True):
    path = tmp_path / 'motion_test.session.json'
    session = {'finished': '2026-10-28T11:09:40Z' if finished else None,
               'stop_reason': 'IMU register 0x1e: [Errno 6]' if fault else '用户停止',
               'policy_start_monotonic_s': 100}
    if fault:
        session['imu_fault'] = {'observed_at_monotonic_s': 101.04,
                                'feedback': {'1': {'voltage_v': 5.5}}}
    path.write_text(json.dumps(session))
    rows = [{'kind': 'sample', 't': t, 'policy_mode': mode,
             'feedback': {'1': {'voltage_v': v}, '2': {'voltage_v': v + .1}}}
            for t, mode, v in [(.4, 'walking', 6.6), (.8, 'stopping', 6.6),
                               (1., 'roulade', 5.8), (1.02, 'roulade', 5.6)]]
    path.with_name('motion_test.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    return path


def rails(tmp_path, path, point, voltage, *, t=1.04, name=None):
    dest = tmp_path / 'rail.csv'
    with dest.open('w', newline='') as stream:
        w = csv.writer(stream); w.writerow(('session', 't', 'point', 'voltage_v'))
        w.writerow((name or path.name, t, point, voltage))
    return dest


def test_servo_droop_never_proves_imu_power_failure(tmp_path):
    path = make_session(tmp_path)
    report, dest = analyze(path, tmp_path / 'results')
    assert report['i2c_address_nack']
    assert report['roll_servo_drop_v'] == 1.1
    assert not report['imu_supply_measured']
    assert '不能确认或排除' in report['conclusion']
    assert report['fault_frame_servo_min_v'] == 5.5
    assert report['fault_time_source'] == 'observed_at_monotonic_s'
    assert (dest / 'timeline.csv').exists()


@pytest.mark.parametrize('point,volts,outside,sag', [
    ('imu_vdd', 1.6, True, True), ('imu_vdd', 1.71, False, True),
    ('imu_vdd_io', 1.1, False, True), ('imu_vdd_io', 1., True, True),
    ('imu_vdd', 2.8, False, True), ('imu_vdd', 3.3, False, False),
    ('imu_vdd', 3.7, True, False), ('hat_5v', 4., False, False)])
def test_only_measured_rails_are_compared_with_their_own_limits(tmp_path, point, volts, outside, sag):
    path = make_session(tmp_path)
    report, _ = analyze(path, tmp_path / 'results', rail_csv=rails(tmp_path, path, point, volts))
    result = report['external_rails'][point]
    assert result['outside_operating_limits_near_fault'] == outside
    assert result['nominal_3v3_sag_below_3v0_near_fault'] == sag
    assert report['imu_supply_measured'] == (point != 'hat_5v')


def test_distant_measurement_is_not_evidence_of_fault_coincidence(tmp_path):
    path = make_session(tmp_path)
    report, _ = analyze(path, tmp_path / 'results', rail_csv=rails(tmp_path, path, 'imu_vdd', 0, t=.1))
    assert not report['external_rails']['imu_vdd']['outside_operating_limits_near_fault']
    assert report['external_rails']['imu_vdd']['fault_window_samples'] == 0


@pytest.mark.parametrize('volts', ['nan', 'inf', '-1', 'bad'])
def test_invalid_external_voltage_rejected(tmp_path, volts):
    path = make_session(tmp_path)
    with pytest.raises(ValueError):
        read_rails(rails(tmp_path, path, 'imu_vdd', volts), path.name)


def test_other_session_measurement_rejected(tmp_path):
    path = make_session(tmp_path)
    with pytest.raises(ValueError, match='匹配'):
        read_rails(rails(tmp_path, path, 'imu_vdd', 1, name='different.session.json'), path.name)


def test_old_logs_explicitly_label_last_good_time_as_approximation(tmp_path):
    path = make_session(tmp_path)
    data = json.loads(path.read_text()); del data['imu_fault']; del data['policy_start_monotonic_s']
    path.write_text(json.dumps(data))
    report, _ = analyze(path, tmp_path / 'results')
    assert report['fault_time_source'] == 'last_good_frame_approximation'
    assert report['fault_frame_servo_min_v'] is None


def test_active_session_rejected_and_broken_line_reported(tmp_path):
    path = make_session(tmp_path, finished=False)
    with pytest.raises(ValueError, match='仍在运行'):
        analyze(path, tmp_path / 'results')
    path = make_session(tmp_path)
    with path.with_name('motion_test.jsonl').open('a') as stream:
        stream.write('{truncated\n')
    report, _ = analyze(path, tmp_path / 'results')
    assert report['data_quality']['malformed_lines_skipped'] == 1


def test_unlabelled_saradc_not_presented_as_imu_3v3(tmp_path):
    device = tmp_path / 'iio:device0'; device.mkdir()
    (device / 'name').write_text('rk3568-saradc')
    (device / 'in_voltage0_raw').write_text('1023')
    (device / 'in_voltage_scale').write_text('1.757812500')
    result = inventory(tmp_path, tmp_path)
    assert not result['imu_3v3_measurement_verified']
    assert result['iio_voltage_channels'][0]['measurement_point'].startswith('unverified')


def test_latest_roll_baseline_not_mixed_with_previous_roll(tmp_path):
    path = make_session(tmp_path, fault=False)
    with path.with_name('motion_test.jsonl').open('a') as stream:
        for t, mode, v in [(4.5, 'walking', 6.2), (5., 'roulade', 5.9)]:
            stream.write(json.dumps({'kind': 'sample', 't': t, 'policy_mode': mode,
                                     'feedback': {'1': {'voltage_v': v}}}) + '\n')
    report, _ = analyze(path, tmp_path / 'results')
    assert report['pre_roll_median_servo_min_v'] == 6.2
    assert report['roll_min_servo_feedback_v'] == 5.9
