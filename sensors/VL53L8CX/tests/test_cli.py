#!/usr/bin/env python3
"""No hardware access: CLI validation and lifecycle linked to fake ST calls."""
import csv
import io
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(binary, args, mode="ok"):
    return subprocess.run(
        [str(ROOT / "build" / binary), *args],
        env={**os.environ, "TOF_MOCK": mode},
        text=True, capture_output=True, timeout=10, check=False,
    )


for args in ([], ["--help"], ["range", "--help"], ["probe", "--help"]):
    result = run("vl53l8cx_test", args)
    assert result.returncode == 0 and "Usage:" in result.stdout, result
for args in (["unknown"], ["range", "--hz", "16"], ["range", "--frames", "0"],
             ["probe", "--address", "0x80"], ["range", "--chunk", "4097"],
             ["probe", "--bus"], ["range", "--hz", "5junk"]):
    result = run("vl53l8cx_test", args)
    assert result.returncode == 2, result
    assert "Cannot open" not in result.stderr, result

result = run("test_cli_mock", ["probe"])
assert result.returncode == 0 and "MOCK_STOP" not in result.stderr
result = run("test_cli_mock", ["range", "--frames", "2", "--csv"])
rows = list(csv.DictReader(io.StringIO(result.stdout)))
assert result.returncode == 0 and len(rows) == 128, result
assert [r["quality"] for r in rows[:4]] == ["valid", "reduced", "invalid", "none"]
assert rows[64]["frame"] == "2" and rows[63]["zone"] == "63"
assert result.stderr.count("MOCK_STOP") == 1
for mode, code, needs_stop in (
    ("wrong_id", 1, False), ("init_error", 1, False), ("init_signal", 130, False),
    ("start_error", 1, True), ("timeout", 1, True), ("signal", 130, True),
    ("read_error", 1, True), ("stop_error", 1, True),
):
    result = run("test_cli_mock", ["range", "--frames", "1"], mode)
    assert result.returncode == code, (mode, result)
    assert ("MOCK_STOP" in result.stderr) == needs_stop, (mode, result)
    assert "MOCK_CLOSE" in result.stderr, (mode, result)
print("CLI mock: validation, probe, two frames, validity flags, failures, frame timeout and SIGINT cleanup OK")
