"""Keep storage writes out of the foreground robot control loop."""
from __future__ import annotations

import queue
import threading


class BackgroundLineLog:
    def __init__(self, path, *, capacity=256):
        self.path = path
        self.lines = queue.Queue(maxsize=capacity)
        self.stop = threading.Event()
        self.written = 0
        self.dropped = 0
        self.error = None
        self.thread = None

    def __enter__(self):
        # Open before starting the periodic loop so startup errors remain visible.
        self.file = self.path.open('w', buffering=1)
        self.thread = threading.Thread(target=self._run, name='motion-log-writer', daemon=True)
        self.thread.start()
        return self

    def write(self, line):
        # Storage delays must not block sampling. The bounded queue prevents
        # an unavailable disk from consuming all memory.
        if self.error is not None:
            self.dropped += 1
            return
        try:
            self.lines.put_nowait(line)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        try:
            while not self.stop.is_set() or not self.lines.empty():
                try:
                    line = self.lines.get(timeout=.02)
                except queue.Empty:
                    continue
                self.file.write(line)
                self.written += 1
        except Exception as exc:
            self.error = repr(exc)
        finally:
            try:
                self.file.close()
            except Exception as exc:
                self.error = self.error or repr(exc)

    def __exit__(self, *exc):
        # Do not wait for disk I/O here: Controller must release torque first.
        self.stop.set()

    def finish(self, timeout=2):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=timeout)
        return {'written_lines': self.written, 'dropped_lines': self.dropped,
                'pending_lines': self.lines.qsize(), 'error': self.error,
                'writer_running': self.thread is not None and self.thread.is_alive()}
