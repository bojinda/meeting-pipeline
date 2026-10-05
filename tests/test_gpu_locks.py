"""Real util-linux flock tests; run on the Linux processing host or in WSL."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "bin" / "with-gpu-lock.sh"
LINUX_FLOCK = sys.platform.startswith("linux") and all(shutil.which(tool) for tool in ("bash", "flock", "setsid"))


@unittest.skipUnless(LINUX_FLOCK, "Requires Linux util-linux flock and setsid")
class GPULockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environment = dict(os.environ, AIHUB_GPU0_LOCK_FILE=str(self.root / "gpu0.lock"), AIHUB_GPU1_LOCK_FILE=str(self.root / "gpu1.lock"), AIHUB_GPU_LOCK_TIMEOUT="3")
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
        self.temporary.cleanup()

    def launch(self, resource, code, *, timeout=None, source_call=False, helper=HELPER, environment=None):
        env = dict(self.environment if environment is None else environment)
        if timeout is not None:
            env["AIHUB_GPU_LOCK_TIMEOUT"] = str(timeout)
        command = ["bash", str(helper), resource, "test workload", sys.executable, "-c", code]
        if source_call:
            command = ["bash", "-c", 'source "$1"; aihub_run_gpu_stage "$2" test "$3" -c "$4"', "stage", str(helper), resource, sys.executable, code]
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(process)
        return process

    def wait_file(self, path, process=None):
        deadline = time.monotonic() + 4
        while not path.exists() or path.stat().st_size == 0:
            if process is not None and process.poll() is not None:
                self.fail(f"Process exited before readiness: {process.communicate()}")
            if time.monotonic() > deadline:
                self.fail(f"No readiness marker: {path.name}")
            time.sleep(0.01)

    def holding_job(self, resource="gpu0", *, timeout=3, source_call=False, descendants=False):
        ready = self.root / f"ready-{len(self.processes)}"
        release = self.root / f"release-{len(self.processes)}"
        code = "import pathlib,time,os,subprocess; "
        if descendants:
            code += f"child=subprocess.Popen(['sleep','30']); pathlib.Path({str(ready.with_suffix('.child'))!r}).write_text(str(child.pid)); "
        code += f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); "
        code += f"\nwhile not pathlib.Path({str(release)!r}).exists(): time.sleep(0.01)"
        process = self.launch(resource, code, timeout=timeout, source_call=source_call)
        self.wait_file(ready, process)
        return process, ready, release

    def test_two_gpu0_jobs_serialize(self):
        self.assert_serialization("gpu0")

    def test_generic_fallback_paths_with_unset_or_empty_configuration(self):
        source = HELPER.read_text()
        fixture = self.root / "fallback-helper.sh"
        for resource in ("gpu0", "gpu1"):
            default = f"/tmp/aihub-{resource}.lock"
            self.assertIn(default, source)
            source = source.replace(default, str(self.root / f"aihub-{resource}.lock"))
        # Rebase only /tmp destinations in this fixture so testing the default
        # selection never acquires or creates shared installation lock files.
        fixture.write_text(source)
        for empty in (False, True):
            env = dict(self.environment)
            for variable in ("AIHUB_GPU0_LOCK_FILE", "AIHUB_GPU1_LOCK_FILE"):
                if empty:
                    env[variable] = ""
                else:
                    env.pop(variable, None)
            for resource in ("gpu0", "gpu1"):
                with self.subTest(resource=resource, empty=empty):
                    process = self.launch(resource, "pass", helper=fixture, environment=env)
                    _, logs = process.communicate(timeout=3)
                    self.assertEqual(process.returncode, 0, logs)
                    self.assertIn(str(self.root / f"aihub-{resource}.lock"), logs)
                    self.assertTrue((self.root / f"aihub-{resource}.lock").exists())

    def test_explicit_environment_paths_override_generic_defaults(self):
        env = dict(self.environment, AIHUB_GPU0_LOCK_FILE=str(self.root / "stable-resource-alpha.lock"), AIHUB_GPU1_LOCK_FILE=str(self.root / "stable-resource-beta.lock"))
        for resource, variable in (("gpu0", "AIHUB_GPU0_LOCK_FILE"), ("gpu1", "AIHUB_GPU1_LOCK_FILE")):
            with self.subTest(resource=resource):
                process = self.launch(resource, "pass", environment=env)
                _, logs = process.communicate(timeout=3)
                self.assertEqual(process.returncode, 0, logs)
                self.assertIn(env[variable], logs)
                self.assertNotIn(f"/tmp/aihub-{resource}.lock", logs)
                self.assertTrue(Path(env[variable]).exists())
                self.assertFalse((self.root / f"{resource}.lock").exists())

    def test_two_gpu1_jobs_serialize(self):
        self.assert_serialization("gpu1")

    def assert_serialization(self, resource):
        first, _, release = self.holding_job(resource)
        second_ready = self.root / "second-ready"
        second = self.launch(resource, f"from pathlib import Path; Path({str(second_ready)!r}).write_text('acquired')")
        time.sleep(0.15)
        self.assertFalse(second_ready.exists())
        self.assertIsNone(second.poll())
        release.touch()
        self.assertEqual(first.communicate(timeout=3)[0], "")
        _, logs = second.communicate(timeout=3)
        self.assertEqual(second.returncode, 0)
        self.assertTrue(second_ready.exists())
        self.assertIn("Waiting for", logs)
        self.assertIn("Acquired", logs)
        self.assertIn("Released", logs)

    def test_gpu0_and_gpu1_can_be_held_at_the_same_time(self):
        first, _, release0 = self.holding_job("gpu0")
        second, _, release1 = self.holding_job("gpu1")
        self.assertIsNone(first.poll())
        self.assertIsNone(second.poll())
        release0.touch()
        release1.touch()
        first.communicate(timeout=3)
        second.communicate(timeout=3)
        self.assertEqual((first.returncode, second.returncode), (0, 0))

    def test_timeout_does_not_run_workload_or_unlink_file(self):
        holder, _, release = self.holding_job()
        blocked = self.root / "must-not-run"
        contender = self.launch("gpu0", f"from pathlib import Path; Path({str(blocked)!r}).touch()", timeout=0.15)
        _, logs = contender.communicate(timeout=3)
        self.assertEqual(contender.returncode, 75)
        self.assertIn("TIMEOUT", logs)
        self.assertFalse(blocked.exists())
        self.assertTrue((self.root / "gpu0.lock").exists())
        release.touch()
        holder.communicate(timeout=3)

    def test_wait_timeout_does_not_limit_runtime_after_acquisition(self):
        process = self.launch("gpu0", "import time; time.sleep(0.35)", timeout=0.05)
        process.communicate(timeout=3)
        self.assertEqual(process.returncode, 0)

    def test_failure_preserves_exit_code_and_releases_persistent_lock(self):
        failure = self.launch("gpu0", "raise SystemExit(23)")
        failure.communicate(timeout=3)
        self.assertEqual(failure.returncode, 23)
        lock = self.root / "gpu0.lock"
        inode = lock.stat().st_ino
        again = self.launch("gpu0", "print('next')", timeout=0)
        stdout, _ = again.communicate(timeout=3)
        self.assertEqual(again.returncode, 0)
        self.assertEqual(stdout.strip(), "next")
        self.assertEqual(lock.stat().st_ino, inode)

    def test_sigint_and_sigterm_release_lock_and_stop_managed_children(self):
        for sig, expected in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
            with self.subTest(signal=sig):
                process, ready, _ = self.holding_job(descendants=True)
                workload_pid = int(ready.read_text())
                child_pid = int(ready.with_suffix(".child").read_text())
                process.send_signal(sig)
                process.communicate(timeout=8)
                self.assertEqual(process.returncode, expected)
                self.assertFalse(Path(f"/proc/{workload_pid}").exists())
                if Path(f"/proc/{child_pid}/stat").exists():
                    self.assertEqual(Path(f"/proc/{child_pid}/stat").read_text().split()[2], "Z")
                again = self.launch("gpu0", "pass", timeout=0)
                again.communicate(timeout=3)
                self.assertEqual(again.returncode, 0)

    def test_signals_while_waiting_do_not_run_workload(self):
        holder, _, release = self.holding_job()
        for sig, expected in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
            with self.subTest(signal=sig):
                blocked = self.root / f"blocked-{sig}"
                contender = self.launch("gpu0", f"from pathlib import Path; Path({str(blocked)!r}).touch()")
                time.sleep(0.1)
                contender.send_signal(sig)
                contender.communicate(timeout=3)
                self.assertEqual(contender.returncode, expected)
                self.assertFalse(blocked.exists())
        release.touch()
        holder.communicate(timeout=3)

    def test_sourced_stage_forwards_parent_signals_to_supervisor(self):
        for sig, expected in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
            with self.subTest(signal=sig):
                process, ready, _ = self.holding_job(source_call=True)
                workload_pid = int(ready.read_text())
                process.send_signal(sig)
                process.communicate(timeout=8)
                self.assertEqual(process.returncode, expected)
                self.assertFalse(Path(f"/proc/{workload_pid}").exists())
                again = self.launch("gpu0", "pass", timeout=0)
                again.communicate(timeout=3)
                self.assertEqual(again.returncode, 0)

    def test_lock_descriptor_is_not_inherited_by_workload(self):
        lock = self.root / "gpu0.lock"
        code = f"import os,pathlib; print(any(os.path.realpath(p)=={str(lock)!r} for p in pathlib.Path('/proc/self/fd').iterdir()))"
        process = self.launch("gpu0", code)
        stdout, _ = process.communicate(timeout=3)
        self.assertEqual(stdout.strip(), "False")

    def test_sigterm_forces_unresponsive_foreground_job_to_stop_before_release(self):
        ready = self.root / "unresponsive-ready"
        code = f"import signal,pathlib,time,os; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)"
        process = self.launch("gpu0", code)
        self.wait_file(ready, process)
        workload_pid = int(ready.read_text())
        process.terminate()
        process.communicate(timeout=8)
        self.assertEqual(process.returncode, 143)
        self.assertFalse(Path(f"/proc/{workload_pid}").exists())
        again = self.launch("gpu0", "pass", timeout=0)
        again.communicate(timeout=3)
        self.assertEqual(again.returncode, 0)

    def test_invalid_timeout_never_runs_workload(self):
        process = self.launch("gpu0", "print('must-not-run')", timeout="invalid")
        stdout, logs = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 64)
        self.assertEqual(stdout, "")
        self.assertIn("AIHUB_GPU_LOCK_TIMEOUT", logs)

    def test_sourced_stage_restores_caller_signal_traps(self):
        code = 'source "$1"; trap "echo original-int" INT; trap "echo original-term" TERM; aihub_run_gpu_stage gpu0 test true; trap -p INT TERM'
        result = subprocess.run(["bash", "-c", code, "test", str(HELPER)], env=self.environment, capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 0)
        self.assertIn("original-int", result.stdout)
        self.assertIn("original-term", result.stdout)


@unittest.skipUnless(LINUX_FLOCK, "Requires Linux util-linux flock and setsid")
class PipelineGPUStageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.binary = self.project / "bin"
        self.binary.mkdir(parents=True)
        (self.project / "config").mkdir()
        (self.project / "config" / ".env").write_text("DELETE_SOURCE_AUDIO_AFTER_TRANSCRIPTION=0\n")
        self.conda = self.root / "conda.sh"
        self.conda.write_text("conda() { :; }\n")
        self.events = self.root / "events.jsonl"
        self.environment = dict(os.environ, GPU_TEST_PROJECT=str(self.project), GPU_TEST_CONDA=str(self.conda), GPU_TEST_EVENTS=str(self.events), GPU_TEST_CONTROL=str(self.root), AIHUB_GPU0_LOCK_FILE=str(self.root / "gpu0.lock"), AIHUB_GPU1_LOCK_FILE=str(self.root / "gpu1.lock"), AIHUB_GPU_LOCK_TIMEOUT="3", HF_TOKEN="test-token", PATH=str(self.binary) + ":" + os.environ["PATH"])
        self.processes = []
        (self.binary / "with-gpu-lock.sh").write_text(HELPER.read_text())
        for profile in ("meeting", "lesson"):
            original = (ROOT / "bin" / f"postprocess-{profile}.sh").read_text()
            # Only relocate the existing machine-specific paths into fixtures;
            # stage calls, arguments, statuses, and command flow are unmodified.
            script = original.replace('BASE_DIR="$HOME/meeting-pipeline"', 'BASE_DIR="$GPU_TEST_PROJECT"')
            script = script.replace('source "$HOME/miniconda3/etc/profile.d/conda.sh"', 'source "$GPU_TEST_CONDA"')
            (self.binary / f"postprocess-{profile}.sh").write_text(script)
        worker = '''#!/usr/bin/env python3
import fcntl,json,os,pathlib,sys,time
name=pathlib.Path(sys.argv[0]).name
args=sys.argv[1:]
if name=='whisperx':
    recording=pathlib.Path(args[0]); profile='lesson' if 'lesson' in recording.name else 'meeting'; stage='whisper'; gpu=0
elif args and 'transcript_chunker.py' in args[0]:
    out=pathlib.Path(args[1]).parent/'chunks_out';out.mkdir();(out/'transcript_chunks.jsonl').write_text('{}\\n');sys.exit(0)
elif args and 'ollama_' in args[0]:
    profile='lesson' if 'lesson' in args[0] else 'meeting';stage='ollama';gpu=1
else: sys.exit(99)
def locked(path):
    with open(path,'a') as descriptor:
        try: fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(descriptor,fcntl.LOCK_UN);return False
        except BlockingIOError: return True
def log(event):
    data={'profile':profile,'stage':stage,'event':event,'gpu0_locked':locked(os.environ['AIHUB_GPU0_LOCK_FILE']),'gpu1_locked':locked(os.environ['AIHUB_GPU1_LOCK_FILE']),'args':args}
    with open(os.environ['GPU_TEST_EVENTS'],'a') as stream: stream.write(json.dumps(data)+'\\n')
log('start')
control=pathlib.Path(os.environ['GPU_TEST_CONTROL']);(control/(profile+'-'+stage+'-started')).touch()
gate=control/(profile+'-'+stage+'-hold')
while gate.exists(): time.sleep(.01)
time.sleep(.05)
if stage=='whisper':
    out=pathlib.Path(args[args.index('--output_dir')+1]);out.mkdir(parents=True,exist_ok=True);(out/'transcript.json').write_text('{"segments":[]}')
log('end')
sys.exit(int(os.environ.get('GPU_TEST_'+stage.upper()+'_EXIT','0')))
'''
        for name in ("whisperx", "python"):
            path = self.binary / name
            path.write_text(worker)
            path.chmod(0o755)

    def tearDown(self):
        for path in self.root.glob("*-hold"):
            path.unlink()
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=8)
        self.temporary.cleanup()

    def start_pipeline(self, profile):
        recording = self.root / f"{profile}-recording.wav"
        recording.touch()
        process = subprocess.Popen(["bash", str(self.binary / f"postprocess-{profile}.sh"), str(recording)], env=self.environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(process)
        return process

    def wait_stage(self, profile, stage):
        marker = self.root / f"{profile}-{stage}-started"
        deadline = time.monotonic() + 4
        while not marker.exists():
            if time.monotonic() > deadline:
                self.fail(f"Stage did not start: {profile}/{stage}")
            time.sleep(.01)

    def test_meeting_and_lesson_use_exact_corresponding_locks_and_device_index(self):
        for profile in ("meeting", "lesson"):
            process = self.start_pipeline(profile)
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stdout + stderr)
        events = [json.loads(line) for line in self.events.read_text().splitlines()]
        for event in events:
            if event["stage"] == "whisper":
                self.assertTrue(event["gpu0_locked"])
                self.assertFalse(event["gpu1_locked"])
                args = event["args"]
                self.assertEqual(args[args.index("--device_index") + 1], "0")
                self.assertEqual(args[args.index("--device") + 1], "cuda")
            else:
                self.assertFalse(event["gpu0_locked"])
                self.assertTrue(event["gpu1_locked"])
        self.assertEqual({event["profile"] for event in events}, {"meeting", "lesson"})

    def test_meeting_ollama_and_lesson_whisper_can_overlap(self):
        (self.root / "meeting-ollama-hold").touch()
        (self.root / "lesson-whisper-hold").touch()
        meeting = self.start_pipeline("meeting")
        self.wait_stage("meeting", "ollama")
        lesson = self.start_pipeline("lesson")
        self.wait_stage("lesson", "whisper")
        self.assertIsNone(meeting.poll())
        self.assertIsNone(lesson.poll())
        (self.root / "lesson-whisper-hold").unlink()
        time.sleep(.15)
        self.assertFalse((self.root / "lesson-ollama-started").exists())
        (self.root / "meeting-ollama-hold").unlink()
        meeting.communicate(timeout=5)
        lesson.communicate(timeout=5)
        self.assertEqual((meeting.returncode, lesson.returncode), (0, 0))

    def test_meeting_and_lesson_whisper_serialize_on_shared_gpu0(self):
        (self.root / "meeting-whisper-hold").touch()
        meeting = self.start_pipeline("meeting")
        self.wait_stage("meeting", "whisper")
        lesson = self.start_pipeline("lesson")
        time.sleep(.15)
        self.assertFalse((self.root / "lesson-whisper-started").exists())
        (self.root / "meeting-whisper-hold").unlink()
        meeting.communicate(timeout=5)
        lesson.communicate(timeout=5)
        self.assertEqual((meeting.returncode, lesson.returncode), (0, 0))

    def test_whisper_failure_releases_lock_and_preserves_pipeline_failure_behavior(self):
        self.environment["GPU_TEST_WHISPER_EXIT"] = "23"
        for profile in ("meeting", "lesson"):
            process = self.start_pipeline(profile)
            process.communicate(timeout=5)
            self.assertEqual(process.returncode, 1)
        lock = open(self.environment["AIHUB_GPU0_LOCK_FILE"], "a")
        try:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            lock.close()

    def test_ollama_failure_releases_lock_and_preserves_existing_exit_behavior(self):
        self.environment["GPU_TEST_OLLAMA_EXIT"] = "23"
        for profile, expected in (("meeting", 0), ("lesson", 1)):
            process = self.start_pipeline(profile)
            process.communicate(timeout=5)
            self.assertEqual(process.returncode, expected)
        lock = open(self.environment["AIHUB_GPU1_LOCK_FILE"], "a")
        try:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            lock.close()


if __name__ == "__main__":
    unittest.main()
