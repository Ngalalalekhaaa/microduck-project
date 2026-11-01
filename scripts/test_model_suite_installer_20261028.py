"""Installer failure-path tests; temporary files and mocked service checks only."""
import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from subprocess import CompletedProcess
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("suite_installer", Path(__file__).with_name("model_suite_installer_20261028.py"))
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)
PACK_SPEC = importlib.util.spec_from_file_location("suite_packager", Path(__file__).with_name("package_model_suite_20261028.py"))
packager = importlib.util.module_from_spec(PACK_SPEC)
PACK_SPEC.loader.exec_module(packager)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "target"
        self.source = self.root / "source"
        self.target.mkdir(); self.source.mkdir()
        (self.target / "robot.json").write_bytes(b'{"calibration":"unique local bytes"}\n')
        (self.target / "old.py").write_bytes(b"original\n")
        (self.source / "old.py").write_bytes(b"replacement\n")
        (self.source / "new.py").write_bytes(b"new\n")
        self.payload = {Path("old.py"): self.source / "old.py", Path("sub/new.py"): self.source / "new.py"}
        self.backup = self.target / "backups" / "test"

    def test_success_preserves_calibration_and_backs_up_overwritten_files(self):
        expected = (self.target / "robot.json").read_bytes()
        result = installer.install_transaction(self.target, self.payload, self.backup, postcheck=lambda: None)
        self.assertEqual(result["changed_files"], 2)
        self.assertEqual((self.target / "robot.json").read_bytes(), expected)
        self.assertEqual((self.backup / "files/old.py").read_bytes(), b"original\n")
        self.assertEqual((self.target / "old.py").read_bytes(), b"replacement\n")
        self.assertEqual(json.loads((self.backup / "transaction.json").read_text())["state"], "committed")

    def test_copy_failure_rolls_back_old_and_removes_new_files(self):
        def failing_copy(src, dst):
            installer.atomic_copy(src, dst)
            if dst.name == "new.py":
                raise OSError("injected disk failure after replace")
        with self.assertRaisesRegex(OSError, "injected"):
            installer.install_transaction(self.target, self.payload, self.backup, postcheck=lambda: None, copy_file=failing_copy)
        self.assertEqual((self.target / "old.py").read_bytes(), b"original\n")
        self.assertFalse((self.target / "sub").exists())
        self.assertEqual(json.loads((self.backup / "transaction.json").read_text())["state"], "rolled_back")
        self.assertEqual((self.target / "robot.json").read_bytes(), b'{"calibration":"unique local bytes"}\n')

    def test_post_install_validation_failure_rolls_back(self):
        def fail():
            raise RuntimeError("injected model validation failure")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            installer.install_transaction(self.target, self.payload, self.backup, postcheck=fail)
        self.assertEqual((self.target / "old.py").read_bytes(), b"original\n")
        self.assertFalse((self.target / "sub/new.py").exists())

    def test_rejects_active_service(self):
        with patch.object(Path, "is_dir", return_value=True), patch.object(installer.subprocess, "run", return_value=CompletedProcess([], 0, "LoadState=loaded\nActiveState=active\n", "")):
            with self.assertRaisesRegex(RuntimeError, "Stop service"):
                installer.assert_service_stopped("microduck-xbox.service")

    def test_accepts_inactive_service(self):
        with patch.object(Path, "is_dir", return_value=True), patch.object(installer.subprocess, "run", return_value=CompletedProcess([], 0, "LoadState=loaded\nActiveState=inactive\n", "")):
            self.assertEqual(installer.assert_service_stopped("microduck-xbox.service")["state"], "inactive")

    def test_missing_native_service_nonzero_return_is_allowed(self):
        with patch.object(Path, "is_dir", return_value=True), patch.object(installer.subprocess, "run", return_value=CompletedProcess([], 3, "LoadState=not-found\nActiveState=inactive\n", "Unit not found")):
            self.assertEqual(installer.assert_service_stopped("padd.service")["state"], "not_installed")

    def test_service_query_failure_and_restarting_states_are_refused(self):
        for result in (CompletedProcess([], 1, "", "Failed to connect to bus"),
                       CompletedProcess([], 0, "LoadState=loaded\nActiveState=activating\n", ""),
                       CompletedProcess([], 0, "LoadState=loaded\nActiveState=failed\n", ""),
                       CompletedProcess([], 0, "LoadState=not-found\nActiveState=active\n", "")):
            with patch.object(Path, "is_dir", return_value=True), patch.object(installer.subprocess, "run", return_value=result):
                with self.assertRaises(RuntimeError):
                    installer.assert_service_stopped("robotd.service")

    def test_default_services_include_native_and_web_controllers(self):
        self.assertEqual(set(installer.DEFAULT_SERVICES), {"microduck-xbox.service", "servo-web.service", "padd.service", "robotd.service"})

    def test_rejects_protected_config_and_symlink_destinations(self):
        with self.assertRaisesRegex(ValueError, "Protected"):
            installer.install_transaction(self.target, {Path("robot.json"): self.source / "new.py"}, self.backup, postcheck=lambda: None)
        (self.target / "sub").symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symbolic"):
            installer.install_transaction(self.target, self.payload, self.backup, postcheck=lambda: None)
        self.assertEqual((self.target / "old.py").read_bytes(), b"original\n")

    def test_rejects_traversal_and_hash_tampering(self):
        for path in ("../x", "/etc/default/microduck-xbox", "payload/../robot.json", "a//b", "a\\b"):
            with self.assertRaises(ValueError):
                installer.safe_relative(path)
        package = self.root / "package"
        (package / "payload/scripts").mkdir(parents=True)
        source = package / "payload/scripts/model_suite.py"
        source.write_bytes(b"print('ok')\n")
        manifest = {"format": "microduck-model-suite-v1", "files": {
            "payload/scripts/model_suite.py": {"sha256": installer.digest(source), "size": source.stat().st_size}}}
        (package / "MANIFEST.json").write_text(json.dumps(manifest))
        self.assertEqual(len(installer.verify_package(package)[0]), 1)
        source.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            installer.verify_package(package)

    def test_detects_motion_process_without_opening_device(self):
        proc = self.root / "proc"; (proc / "999999").mkdir(parents=True)
        (proc / "999999/cmdline").write_bytes(b"python\x00/home/user/deploy/scripts/xbox_autostart.py\x00")
        self.assertEqual(installer.motion_processes(proc)[0]["pid"], 999999)

    def test_offline_checks_are_not_misidentified_as_motion(self):
        proc = self.root / "proc"
        for pid, argv in enumerate((
            b"python\x00-m\x00pytest\x00tests/test_runtime.py\x00",
            b"python\x00scripts/model_suite.py\x00check\x00--config\x00robot.json\x00",
            b"python\x00-m\x00microduck_deploy\x00self-test\x00",
            b"python\x00-m\x00pytest\x00-k\x00robotd\x00",
        ), start=999990):
            (proc / str(pid)).mkdir(parents=True)
            (proc / str(pid) / "cmdline").write_bytes(argv)
        self.assertEqual(installer.motion_processes(proc), [])

    def test_native_and_web_manual_processes_are_refused(self):
        proc = self.root / "proc"
        for pid, argv in enumerate((
            b"/opt/robot/daemon/current/bin/padd\x00",
            b"/opt/robot/daemon/current/bin/robotd\x00--socket\x00/run/robotd.sock\x00",
            b"python\x00/home/user/servo_web_hd1910/server.py\x00",
        ), start=999980):
            (proc / str(pid)).mkdir(parents=True)
            (proc / str(pid) / "cmdline").write_bytes(argv)
        self.assertEqual(len(installer.motion_processes(proc)), 3)

    def test_occupied_serial_port_is_refused(self):
        with patch.object(installer, "assert_service_stopped", return_value={}), \
                patch.object(installer, "motion_processes", return_value=[]), \
                patch.object(Path, "exists", return_value=True), \
                patch.object(installer.shutil, "which", return_value="/usr/bin/fuser"), \
                patch.object(installer.subprocess, "run", return_value=CompletedProcess([], 0, "1234", "")):
            with self.assertRaisesRegex(RuntimeError, "Serial port occupied"):
                installer.assert_idle({"serial": {"port": "/dev/ttyUSB0"}}, ["microduck-xbox.service"])

    def test_package_selection_excludes_calibration_and_secrets(self):
        for name in packager.REQUIRED | {f"samples/suite_20261028/{i}.npz" for i in range(6)}:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test fixture")
        private = ("robot.json", "robot.json.bak.123", "servo_ids.user.json", "secret.key",
                   "private.pem", "credentials.json", "secrets.json", ".env",
                   "logs/session.json", "backups/old/robot.json", ".venv/x.py", "__pycache__/x.pyc")
        for name in private:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"must not enter archive")
        names = {p.as_posix() for p in packager.selected_files(self.source)}
        self.assertFalse(names & set(private))
        self.assertTrue(packager.REQUIRED <= names)

    def test_custom_bundle_name_controls_archive_root_manifest_and_instructions(self):
        name = "microduck-xbox-four-actions-20261028"
        output = self.root / (name + ".tar.gz")
        with redirect_stdout(io.StringIO()):
            packager.build(self.source, output, [Path("old.py")], name=name)
        with tarfile.open(output) as archive:
            self.assertTrue(all(p.name.startswith(name + "/") for p in archive.getmembers()))
            manifest = json.load(archive.extractfile(name + "/MANIFEST.json"))
            self.assertEqual(manifest["bundle"], name)
            instructions = archive.extractfile(name + "/INSTALL.txt").read().decode()
            self.assertIn(name + ".tar.gz.sha256", instructions)
            self.assertIn("cd ~/" + name, instructions)
            self.assertNotIn(packager.NAME, instructions)

    def test_unsafe_bundle_names_are_refused_and_default_is_unchanged(self):
        self.assertEqual(packager.validate_name(packager.NAME), "microduck-model-suite-20261028")
        for name in ("", "../bad", "/tmp/bad", "a/b", "a\\b", "-bad", "..", "a..b", "a\nb", "bad.", "x" * 97):
            with self.assertRaises(ValueError):
                packager.validate_name(name)


if __name__ == "__main__":
    unittest.main()
