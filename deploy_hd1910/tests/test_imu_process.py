import multiprocessing as mp
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import pytest
from microduck_deploy.imu import ImuState, ImuError
from microduck_deploy.imu_process import _worker, ProcessImuReader


def test_worker_publishes_complete_sample_and_bias_then_closes():
    samples=mp.Array('d',10);bias=mp.Array('d',3)
    ready,stop=threading.Event(),threading.Event();errors=queue.Queue()
    sensor=Mock();sensor.gyro_bias_rad_s=np.array([.01,.02,.03])
    state=ImuState(time.monotonic(),np.array([1.,2.,3.]),np.array([0.,0.,-1.]),np.array([0.,0.,9.81]))
    def read(**kwargs):stop.set();return state
    sensor.read_state.side_effect=read
    with patch('microduck_deploy.imu_process.open_imu',return_value=sensor):
        _worker({},True,samples,bias,ready,stop,errors)
    assert ready.is_set() and errors.empty()
    assert samples[:]==[state.timestamp,1.,2.,3.,0.,0.,-1.,0.,0.,9.81]
    assert bias[:]==[.01,.02,.03]
    sensor.calibrate_gyro.assert_called_once_with(sample_count=240)
    sensor.close.assert_called_once()


def test_worker_error_is_reported_and_closed():
    ready=threading.Event();errors=queue.Queue();sensor=Mock()
    sensor.initialize.side_effect=OSError('sensor missing')
    with patch('microduck_deploy.imu_process.open_imu',return_value=sensor):
        _worker({},False,mp.Array('d',10),mp.Array('d',3),ready,threading.Event(),errors)
    assert ready.is_set() and 'sensor missing' in errors.get_nowait()
    sensor.close.assert_called_once()


def test_latest_rejects_stale_data_and_dead_process():
    reader=ProcessImuReader.__new__(ProcessImuReader)
    reader.failed=threading.Event()
    reader.errors=queue.Queue();reader.process=Mock();reader.process.is_alive.return_value=True
    reader.samples=mp.Array('d',[time.monotonic(),0,0,0,0,0,-1,0,0,9.81])
    assert reader.latest(.1).projected_gravity.tolist()==[0,0,-1]
    reader.samples[0]=time.monotonic()-1
    with pytest.raises(ImuError,match='stale'):reader.latest(.1)
    reader.process.is_alive.return_value=False
    with pytest.raises(ImuError,match='exited'):reader.latest(.1)


def test_startup_error_waits_for_queue_feeder_instead_of_reporting_zero_sample():
    reader=ProcessImuReader.__new__(ProcessImuReader)
    reader.failed=threading.Event();reader.failed.set()
    reader.errors=queue.Queue()
    reader.process=Mock();reader.process.is_alive.return_value=True
    reader.samples=mp.Array('d',10)
    thread=threading.Timer(.03,lambda:reader.errors.put('calibration rejected motion'))
    thread.start()
    try:
        with pytest.raises(ImuError,match='calibration rejected motion'):
            reader.latest(.2)
    finally:
        thread.join()
