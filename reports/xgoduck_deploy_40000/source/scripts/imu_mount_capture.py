"""Capture sensor-axis IMU vectors for a user-held, known body posture.

Only accesses the IMU. Does not open a servo port or change mounting calibration.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from microduck_deploy.config import load_config
from microduck_deploy.imu import ImuConfig, open_imu, GRAVITY


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pose', choices=('upright','forward','left','upright_check'))
    args = p.parse_args()
    values = dict(load_config(ROOT/'robot.json')['imu'])
    values.update(mounting_rotation=np.eye(3).tolist(), mount_verified=False)
    sensor = open_imu(ImuConfig.from_dict(values))
    try:
        sensor.initialize()
        samples = [sensor.read_sample() for _ in range(240)]
    finally:
        sensor.close()
    accel=np.stack([s.accel_m_s2 for s in samples])
    gyro=np.stack([s.gyro_rad_s for s in samples])
    result={'pose':args.pose,'sensor_axes':True,'accel_mean':accel.mean(0).tolist(),
            'accel_std':accel.std(0).tolist(),'gyro_mean':gyro.mean(0).tolist(),
            'gyro_std':gyro.std(0).tolist(),'gyro_peak_norm':float(np.linalg.norm(gyro,axis=1).max()),
            'accel_norm_g':float(np.linalg.norm(accel.mean(0))/GRAVITY),
            'duration_s':samples[-1].timestamp-samples[0].timestamp,
            'samples':[{'t':s.timestamp,'a':s.accel_m_s2.tolist(),'g':s.gyro_rad_s.tolist()} for s in samples]}
    folder=ROOT/'logs/imu_mount';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(time.strftime('%Y%m%d_%H%M%S')+'_'+args.pose+'.json')
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)
    print(path,flush=True)


if __name__=='__main__':
    main()
