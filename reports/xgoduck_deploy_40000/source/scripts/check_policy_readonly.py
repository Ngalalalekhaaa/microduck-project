"""Read-only policy check; optional reuse of an explicitly named calibrated session."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    from microduck_deploy.config import load_config
    from microduck_deploy.runtime import Controller
    from microduck_deploy.imu_process import ProcessImuReader
    from microduck_deploy.policy import OnnxPolicy
    from microduck_deploy.servo import ServoBus
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bias-session',type=Path)
    parser.add_argument('--seconds',type=float,default=5)
    args=parser.parse_args()
    cfg=load_config(ROOT/'robot.json')
    if args.bias_session:
        saved=json.loads(args.bias_session.read_text())
        assert saved['config']['imu']==cfg['imu'], 'IMU configuration changed since bias capture'
        cfg['imu']['gyro_bias_rad_s']=saved['gyro_bias_rad_s']
    policy=OnnxPolicy(ROOT/cfg['model'])
    class ReadOnlyBus(ServoBus):
        def _send(self,packet):
            if packet[4] not in (2,130):raise RuntimeError('This check forbids motor writes')
            return super()._send(packet)
    reader=ProcessImuReader(cfg,calibrate=args.bias_session is None)
    try:
        with ReadOnlyBus(**cfg['serial']) as bus:
            ctl=Controller(cfg,bus=bus,imu_reader=reader,policy=policy)
            result=ctl.execute(seconds=args.seconds,motion=False,twist=(0,0,0))
            print(json.dumps({'result':result,'retries':ctl.session.get('servo_read_retries',0),'bias_source':str(args.bias_session),'motor_writes':0}),flush=True)
    finally:reader.close()

if __name__=='__main__':main()
