"""CPU tests of queue dependency gates and subprocess cleanup; no simulation."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import queue_hd1910 as queue
import train_pair


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state_path = self.root / 'status.json'

    def predecessor(self, status='running', artifacts=True):
        runs = []
        for variant in ('flat', 'backlash'):
            artifact = self.root / variant
            if artifacts:
                for file in ('policy.onnx', 'eval_flat/evaluation.json', 'eval_backlash/evaluation.json'):
                    p = artifact / file
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text('{}')
            runs.append({'status': 'completed', 'artifacts': str(artifact)})
        self.state_path.write_text(json.dumps({'status': status, 'runs': runs, 'pid': 123}))
        return self.state_path

    def test_waits_while_originals_are_running(self):
        with patch.object(queue, 'process_live', return_value=True):
            self.assertFalse(queue.dependency_ready(self.predecessor()))

    def test_waits_for_supervisor_exit_after_artifacts_finish(self):
        with patch.object(queue, 'process_live', return_value=True):
            self.assertFalse(queue.dependency_ready(self.predecessor('completed')))
        with patch.object(queue, 'process_live', return_value=False):
            self.assertTrue(queue.dependency_ready(self.predecessor('completed')))

    def test_failed_or_vanished_originals_do_not_launch_next_task(self):
        with self.assertRaises(RuntimeError):
            queue.dependency_ready(self.predecessor('failed'))
        with patch.object(queue, 'process_live', return_value=False), self.assertRaises(RuntimeError):
            queue.dependency_ready(self.predecessor())

    def test_missing_exports_block_queue(self):
        with self.assertRaises(RuntimeError):
            queue.dependency_ready(self.predecessor('completed', artifacts=False))

    def smoke_fixture(self):
        runs = []
        for variant, task in queue.HD1910_TASKS.items():
            artifact = self.root / variant
            artifact.mkdir()
            log = artifact / 'train.log'
            log.write_text('Episode_Termination/nan_state: 0.0000\n')
            for eval_variant, eval_task in queue.HD1910_TASKS.items():
                folder = artifact / f'eval_{eval_variant}'
                folder.mkdir()
                result = {'task': eval_task, 'onnx_parity_pass': True,
                          'onnx_input_shape': [1, 61], 'onnx_output_shape': [1, 14],
                          'results': {name: {'nonfinite_first_episode_states': 0}
                                      for name in ('stand', 'forward', 'forward_fast', 'backward', 'left', 'turn')}}
                (folder / 'evaluation.json').write_text(json.dumps(result))
            runs.append({'task': task, 'variant': variant, 'stdout': str(log), 'artifacts': str(artifact)})
        self.state_path.write_text(json.dumps({'status': 'completed', 'runs': runs}))

    def test_smoke_requires_complete_finite_matching_evaluation(self):
        self.smoke_fixture()
        queue.validate_smoke(self.root)
        p = self.root / 'flat/eval_backlash/evaluation.json'
        result = json.loads(p.read_text())
        result['results'].pop('forward')
        p.write_text(json.dumps(result))
        with self.assertRaises(RuntimeError):
            queue.validate_smoke(self.root)

    def test_nan_log_rejects_long_run(self):
        self.smoke_fixture()
        (self.root / 'flat/train.log').write_text('Episode_Termination/nan_state: 0.0100\n')
        with self.assertRaises(RuntimeError):
            queue.validate_smoke(self.root)

    def test_later_nan_token_is_not_ignored_after_valid_metrics(self):
        self.smoke_fixture()
        (self.root / 'flat/train.log').write_text('Episode_Termination/nan_state: 0.0000\n'
                                                 'Episode_Termination/nan_state: nan\n')
        with self.assertRaises(RuntimeError):
            queue.validate_smoke(self.root)

    def test_failed_subprocess_propagates(self):
        item = {}
        with self.assertRaises(RuntimeError):
            train_pair.run_monitored([sys.executable, '-c', 'raise SystemExit(7)'],
                                     project=self.root, env=os.environ.copy(),
                                     log_path=self.root / 'child.log', item=item, save=lambda: None)
        self.assertEqual(item['last_returncode'], 7)
        self.assertIsNone(item['active_pid'])

    def test_disk_guard_stops_owned_child(self):
        item = {}
        disk = type('Disk', (), {'free': 2 * 2**30})()
        with patch.object(train_pair.shutil, 'disk_usage', return_value=disk), self.assertRaises(RuntimeError):
            train_pair.run_monitored([sys.executable, '-c', 'import time; time.sleep(60)'],
                                     project=self.root, env=os.environ.copy(),
                                     log_path=self.root / 'child.log', item=item, save=lambda: None)
        self.assertIsNotNone(item['last_returncode'])
        self.assertIsNone(item['active_pid'])

    def test_first_status_write_failure_stops_owned_child(self):
        item = {}
        calls = 0

        def failing_save():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError('State storage unavailable')

        with self.assertRaises(OSError):
            train_pair.run_monitored([sys.executable, '-c', 'import time; time.sleep(60)'],
                                     project=self.root, env=os.environ.copy(),
                                     log_path=self.root / 'child.log', item=item, save=failing_save)
        self.assertIsNotNone(item['last_returncode'])
        self.assertIsNone(item['active_pid'])


if __name__ == '__main__':
    unittest.main()
