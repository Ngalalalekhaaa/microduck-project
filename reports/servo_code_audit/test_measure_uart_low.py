"""Check automatic BREAK release without opening a real serial device."""
import unittest

from measure_uart_low import TIOCSBRK, TIOCCBRK, hold_low


class BreakReleaseTests(unittest.TestCase):
    def test_normal_completion(self):
        calls, events = [], []
        hold_low(99, 1, ioctl=lambda fd, cmd: calls.append((fd, cmd)),
                 sleep=lambda seconds: None, report=events.append)
        self.assertEqual(calls, [(99, TIOCSBRK), (99, TIOCCBRK)])
        self.assertEqual([e["event"] for e in events], ["break_enabled", "break_disabled"])

    def test_interrupt_during_wait_releases(self):
        calls = []
        def interrupted(seconds):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            hold_low(99, 1, ioctl=lambda fd, cmd: calls.append(cmd),
                     sleep=interrupted, report=lambda event: None)
        self.assertEqual(calls, [TIOCSBRK, TIOCCBRK])

    def test_lost_output_connection_releases(self):
        calls = []
        def broken(event):
            raise BrokenPipeError()
        with self.assertRaises(BrokenPipeError):
            hold_low(99, 1, ioctl=lambda fd, cmd: calls.append(cmd),
                     sleep=lambda seconds: self.fail("Should not sleep"), report=broken)
        self.assertEqual(calls, [TIOCSBRK, TIOCCBRK])

    def test_enable_failure_still_attempts_release(self):
        calls = []
        def failed_enable(fd, cmd):
            calls.append(cmd)
            if cmd == TIOCSBRK:
                raise OSError("enable failed")
        with self.assertRaises(OSError):
            hold_low(99, 1, ioctl=failed_enable, sleep=lambda seconds: None,
                     report=lambda event: None)
        self.assertEqual(calls, [TIOCSBRK, TIOCCBRK])

    def test_invalid_duration_does_not_touch_uart(self):
        for seconds in (0, -1, 31, float("nan"), float("inf")):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                hold_low(99, seconds, ioctl=lambda *args: self.fail("Touched UART"))


if __name__ == "__main__":
    unittest.main()
