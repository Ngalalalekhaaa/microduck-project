"""Isolate Linux IMU polling from the foreground controller's Python thread."""
import multiprocessing as mp
from queue import Empty
import time
from types import SimpleNamespace
import numpy as np
from .imu import ImuConfig, ImuError, ImuState, open_imu


def _worker(values, calibrate, samples, bias, ready, stop, errors):
    sensor = None
    try:
        sensor = open_imu(ImuConfig.from_dict(values))
        sensor.initialize()
        if calibrate:
            sensor.calibrate_gyro(sample_count=240)
        bias[:] = sensor.gyro_bias_rad_s.tolist()
        while not stop.is_set():
            state = sensor.read_state(timeout_s=.05)
            values = [state.timestamp, *state.base_ang_vel, *state.projected_gravity, *state.accel_m_s2]
            with samples.get_lock():
                samples[:] = values
            ready.set()
    except BaseException as exc:
        errors.put(repr(exc))
        ready.set()
    finally:
        if sensor is not None:
            sensor.close()


class ProcessImuReader:
    def __init__(self, cfg, *, calibrate=True):
        context = mp.get_context('spawn')
        self.samples = context.Array('d', 10)
        bias = context.Array('d', 3)
        ready = context.Event()
        self.stop = context.Event()
        self.errors = context.Queue(maxsize=1)
        self.process = context.Process(target=_worker, args=(cfg['imu'],calibrate,self.samples,bias,ready,self.stop,self.errors), daemon=True)
        self.process.start()
        try:
            if not ready.wait(8):
                raise ImuError('IMU process startup timed out')
            self.latest(.2)
            self.imu = SimpleNamespace(config=ImuConfig.from_dict(cfg['imu']), gyro_bias_rad_s=np.array(bias[:]))
        except BaseException:
            self.close()
            raise

    def latest(self, max_age):
        try:
            error = self.errors.get_nowait()
        except Empty:
            error = None
        if error is not None:
            raise ImuError('IMU process: '+error)
        if not self.process.is_alive():
            raise ImuError('IMU process exited')
        with self.samples.get_lock():
            data = np.array(self.samples[:])
        if not np.isfinite(data).all() or data[0] <= 0 or time.monotonic()-data[0] > max_age:
            raise ImuError('IMU process sample is stale or invalid')
        return ImuState(data[0],data[1:4],data[4:7],data[7:10])

    def close(self):
        self.stop.set()
        self.process.join(timeout=.3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=.5)
        self.errors.close()

