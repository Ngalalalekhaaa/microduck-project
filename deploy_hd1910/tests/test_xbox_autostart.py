import importlib.util
from pathlib import Path
import signal
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS/(name+'.py'))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


daemon = module('xbox_autostart')
installer = module('install_xbox_service')


class Stop:
    def __init__(self):
        self.stopped = False
    def is_set(self):
        return self.stopped
    def set(self):
        self.stopped = True
    def wait(self, seconds):
        return self.stopped


class TestBootListener(unittest.TestCase):
    def test_disconnected_and_manual_session_do_not_launch(self):
        for busy in (True, False):
            stop = Stop()
            stop.wait = lambda _: stop.set()
            daemon.supervise(['flat','60','0.4'], stop, busy=lambda: busy,
                             ready=lambda: False,
                             launch=lambda *a, **kw: self.fail('unexpected launch'),
                             emit=lambda *a, **kw: None)

    def test_connected_then_shutdown_forwards_signal_and_joins(self):
        stop = Stop()
        signals = []
        joined = []
        calls = []
        class Child:
            returncode = None
            def poll(self): return self.returncode
            def send_signal(self, value): signals.append(value)
            def wait(self, timeout):
                joined.append(timeout)
                self.returncode = 130
        def launch(command, **kwargs):
            calls.append((command, kwargs))
            stop.set()
            return Child()
        daemon.supervise(['backlash','60','0.4'], stop, busy=lambda: False,
                         ready=lambda: True, launch=launch, emit=lambda *a, **kw: None)
        self.assertEqual(signals, [signal.SIGTERM])
        self.assertEqual(joined, [15])
        self.assertEqual(calls[0][0][-3:], ['backlash','60','0.4'])
        self.assertEqual(Path(calls[0][0][1]).name, 'xbox_friction_5000_hd1910.sh')
        self.assertTrue(calls[0][1]['start_new_session'])

    def test_model_override_launches_generic_entry_with_existing_limits(self):
        stop = Stop()
        calls = []
        class Child:
            returncode = 0
            def poll(self): return 0
        def launch(command, **kwargs):
            calls.append(command)
            stop.set()
            return Child()
        arguments = daemon.settings({'MICRODUCK_MODEL': 'models/hd1910_backlash_30000.onnx',
                                     'MICRODUCK_MAX_VX': '0.2'})
        daemon.supervise(arguments, stop, model_override=True, busy=lambda: False,
                         ready=lambda: True, launch=launch, emit=lambda *a, **kw: None)
        self.assertEqual(calls, [['bash', str(SCRIPTS.parent/'xbox_model_hd1910.sh'),
                                 'models/hd1910_backlash_30000.onnx', '0.0', '0.2']])

    def test_retries_after_controller_error(self):
        stop = Stop()
        probes = []
        def ready():
            probes.append(1)
            if len(probes) == 1:
                raise daemon.GamepadError('not connected')
            stop.set()
            return True
        daemon.supervise(['flat','60','0.4'], stop, busy=lambda: False, ready=ready,
                         launch=lambda *a, **kw: self.fail('stop should prevent launch'),
                         emit=lambda *a, **kw: None)
        self.assertEqual(len(probes), 2)

    def test_settings_and_unit(self):
        self.assertEqual(daemon.settings({}), ['backlash','0.0','0.3'])
        self.assertEqual(daemon.settings({'MICRODUCK_SECONDS':'120'}), ['backlash','120.0','0.3'])
        self.assertEqual(daemon.settings({'MICRODUCK_MODEL': ''}), ['backlash','0.0','0.3'])
        self.assertEqual(daemon.settings({'MICRODUCK_MODEL': '/tmp/new model.onnx'}),
                         ['/tmp/new model.onnx','0.0','0.3'])
        for env in ({'MICRODUCK_VARIANT':'wrong'}, {'MICRODUCK_SECONDS':'nan'},
                    {'MICRODUCK_SECONDS':'1'}, {'MICRODUCK_MAX_VX':'0.5'},
                    {'MICRODUCK_MODEL':'model.pt'}, {'MICRODUCK_MODEL':'bad\nname.onnx'},
                    {'MICRODUCK_MODEL':'name.onnx\n'},
                    {'MICRODUCK_MODEL':'bad\0name.onnx'}):
            with self.assertRaises(ValueError): daemon.settings(env)
        template = (SCRIPTS.parent/'systemd/microduck-xbox.service.in').read_text()
        unit = installer.render_unit(template, 'wmh', Path('/home/wmh/deploy_hd1910'))
        self.assertNotIn('@', unit)
        self.assertIn('User=wmh', unit)
        self.assertIn('WantedBy=multi-user.target', unit)
        self.assertIn('KillMode=mixed', unit)
        self.assertNotIn('network-online', unit)
