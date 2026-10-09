"""Exercise the meeting wrapper's audio policy independently of model output."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from gpu_admission_fixture import configure


ROOT = Path(__file__).resolve().parents[1]
LINUX_FLOCK = sys.platform.startswith("linux") and shutil.which("bash") and shutil.which("flock")


@unittest.skipUnless(LINUX_FLOCK, "Requires Linux bash and util-linux flock")
class MeetingRecordingCleanupTests(unittest.TestCase):
    def run_wrapper(self, directory, transcript, delete=True, whisper_exit=0):
        root = Path(directory)
        project = root / "project"
        binary = project / "bin"
        binary.mkdir(parents=True)
        (project / "config").mkdir()
        (project / "config" / ".env").write_text(f"DELETE_SOURCE_AUDIO_AFTER_TRANSCRIPTION={int(delete)}\n")
        conda = root / "conda.sh"
        conda.write_text("conda() { :; }\n")
        script = (ROOT / "bin" / "postprocess-meeting.sh").read_text()
        script = script.replace('BASE_DIR="$HOME/meeting-pipeline"', 'BASE_DIR="$CLEANUP_PROJECT"')
        script = script.replace('source "$HOME/miniconda3/etc/profile.d/conda.sh"', 'source "$CLEANUP_CONDA"')
        (binary / "postprocess-meeting.sh").write_text(script)
        (binary / "with-gpu-lock.sh").write_text((ROOT / "bin" / "with-gpu-lock.sh").read_text())
        (binary / "meeting_stage_inputs.py").write_text((ROOT / "bin/meeting_stage_inputs.py").read_text())
        worker = '''#!/usr/bin/env python3
import os,pathlib,sys
args=sys.argv[1:]
if pathlib.Path(sys.argv[0]).name == 'whisperx':
    out=pathlib.Path(args[args.index('--output_dir')+1])
    if os.environ['CLEANUP_TRANSCRIPT']:
        (out/'transcript.json').write_text(os.environ['CLEANUP_TRANSCRIPT'])
    sys.exit(int(os.environ['CLEANUP_WHISPER_EXIT']))
elif args and pathlib.Path(args[0]).name == 'transcript_chunker.py':
    out=pathlib.Path(args[1]).parent/'chunks_out'
    out.mkdir()
    (out/'transcript_chunks.jsonl').write_text(os.environ['CLEANUP_TRANSCRIPT']+'\\n')
elif args and pathlib.Path(args[0]).name == 'ollama_meeting_summary.py':
    pass
else:
    sys.exit(99)
'''
        for name in ("whisperx", "python"):
            path = binary / name
            path.write_text(worker)
            path.chmod(0o755)
        recording = root / "meeting-recording.wav"
        recording.write_bytes(b"synthetic recording")
        env = dict(os.environ, CLEANUP_PROJECT=str(project), CLEANUP_CONDA=str(conda), CLEANUP_TRANSCRIPT=transcript, CLEANUP_WHISPER_EXIT=str(whisper_exit), HF_TOKEN="test-token", AIHUB_GPU0_LOCK_FILE=str(root / "gpu0.lock"), AIHUB_GPU1_LOCK_FILE=str(root / "gpu1.lock"), AIHUB_GPU_LOCK_TIMEOUT="3", PATH=str(binary) + ":" + os.environ["PATH"])
        configure(root, env)
        result = subprocess.run(["bash", str(binary / "postprocess-meeting.sh"), str(recording)], env=env, capture_output=True, text=True, timeout=8)
        return result, recording, project / "meeting-transcripts" / recording.stem

    def test_spoken_redaction_does_not_override_configured_deletion(self):
        for text in ("Public business. Redact the following. Private text. End redaction. Public business.", "redact the following unclosed private text"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                transcript = json.dumps({"segments": [{"speaker": "SPEAKER_01", "text": text}]})
                result, recording, output = self.run_wrapper(directory, transcript)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(recording.exists())
                self.assertIn("Source audio deleted after transcription: yes", (output / "status.txt").read_text())
                self.assertEqual((output / "transcript.json").read_text(), transcript)
                self.assertEqual((output / "chunks_out" / "transcript_chunks.jsonl").read_text(), transcript + "\n")

    def test_uninspectable_transcript_does_not_override_normal_json_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            result, recording, output = self.run_wrapper(directory, "not valid JSON")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(recording.exists())
            self.assertEqual((output / "transcript.json").read_text(), "not valid JSON")

    def test_disabled_deletion_preserves_recording_with_redaction(self):
        with tempfile.TemporaryDirectory() as directory:
            result, recording, _ = self.run_wrapper(directory, '{"text":"redact the following private"}', delete=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(recording.read_bytes(), b"synthetic recording")

    def test_failed_transcription_preserves_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            result, recording, _ = self.run_wrapper(directory, '{"text":"redact the following private"}', whisper_exit=23)
            self.assertEqual(result.returncode, 1)
            self.assertTrue(recording.exists())

    def test_missing_transcript_json_preserves_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            _, recording, output = self.run_wrapper(directory, "")
            self.assertTrue(recording.exists())
            self.assertIn("skipped (no transcript JSON found)", (output / "status.txt").read_text())


if __name__ == "__main__":
    unittest.main()
