import threading
from types import SimpleNamespace

from microduck_deploy.background_log import BackgroundLineLog


def test_blocked_disk_does_not_block_enqueue_and_backlog_is_bounded():
    entered, release = threading.Event(), threading.Event()
    lines = []
    class SlowFile:
        def write(self, line):
            entered.set()
            assert release.wait(2)
            lines.append(line)
        def close(self):
            pass
    log = BackgroundLineLog(SimpleNamespace(open=lambda *a, **k: SlowFile()), capacity=2)
    try:
        with log:
            log.write('header\n')
            assert entered.wait(1)
            log.write('one\n')
            log.write('two\n')
            log.write('overflow\n')
            assert log.lines.qsize() == 2 and log.dropped == 1
        assert log.thread.is_alive()  # exit doesn't wait for blocked storage
    finally:
        release.set()
        stats = log.finish()
    assert lines == ['header\n', 'one\n', 'two\n']
    assert stats['written_lines'] == 3 and not stats['writer_running']


def test_failed_storage_is_reported_without_raising_in_control_thread():
    class BadFile:
        def write(self, line):
            raise OSError('storage unavailable')
        def close(self):
            pass
    log = BackgroundLineLog(SimpleNamespace(open=lambda *a, **k: BadFile()))
    with log:
        log.write('header\n')
    stats = log.finish()
    assert 'storage unavailable' in stats['error']
    assert not stats['writer_running']
