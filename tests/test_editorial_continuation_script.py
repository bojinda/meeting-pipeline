"""The review-only continuation script reserves a fresh private destination."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ignore/editorial-generation-review/continue-october.proposed.sh"


@unittest.skipUnless(sys.platform.startswith("linux") and SCRIPT.is_file(), "local proposed script and Linux required")
class ContinuationScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "bin").mkdir()
        (self.root / "ignore").mkdir()
        sealed = self.root / "ignore/october-editorial-verified-maps-20261010"
        sealed.mkdir()
        self.checkpoint = sealed / "editorial-map-checkpoint.private.json"
        self.checkpoint.write_text(json.dumps({"configuration": {"map_model": "qwen3.8:27b", "reduce_model": "qwen3.8:27b",
            "map_num_ctx": 32768, "reduce_num_ctx": 196608, "temperature": 0.2, "ollama_url": "http://127.0.0.1:11434",
            "keep_recap": True, "keep_alive": "30m"}, "inputs": {"transcript_index": {"path": str(self.root / "source/chunks_out/transcript_chunks.jsonl")},
            "approved_aliases": {"sha256": None}}}))
        self.original = self.root / "ignore/october-editorial-triage.bioHNv"
        self.original.mkdir()
        (self.original / "preserved.private").write_bytes(b"PRIVATE_ORIGINAL_EVIDENCE")
        self.output_parent = self.root / "ignore/october-editorial-continuation-20261010-01"
        # This launcher is a CPU fixture: no runner, Ollama or GPU operation.
        (self.root / "bin/summarize-existing-meeting.sh").write_text('''#!/usr/bin/env bash
set -euo pipefail
umask > "$BASE/observed-umask"
while [[ "$#" -gt 0 ]]; do
  if [[ "$1" == --meeting-notes-output-dir ]]; then output="$2"; break; fi
  shift
done
mkdir -- "$output"
printf 'fixture' > "$output/fixture.private"
''')
        self.script = self.root / "continue.sh"
        self.script.write_text(SCRIPT.read_text().replace("BASE=/home/bojinda/meeting-pipeline", "export BASE=" + shlex.quote(str(self.root)))
                               .replace("PYTHON=/home/bojinda/venvs/aihub-gpu-runner/bin/python", "PYTHON=" + shlex.quote(sys.executable)))
        self.env = dict(os.environ, OCTOBER_REVIEWED_CHECKPOINT_SHA256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())

    def tearDown(self):
        self.temp.cleanup()

    def run_script(self):
        return subprocess.run(["bash", str(self.script)], env=self.env, capture_output=True, text=True, timeout=10)

    def test_fresh_private_parent_files_and_preserved_originals(self):
        before = {p: p.read_bytes() for p in (self.checkpoint, self.original / "preserved.private")}
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "observed-umask").read_text().strip(), "0077")
        self.assertEqual(self.output_parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.output_parent / "editorial-output").stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.output_parent / "editorial-output/fixture.private").stat().st_mode & 0o777, 0o600)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        # A second invocation fails before the launcher and never overwrites.
        after = (self.output_parent / "editorial-output/fixture.private").read_bytes()
        second = self.run_script()
        self.assertEqual(second.returncode, 64)
        self.assertEqual((self.output_parent / "editorial-output/fixture.private").read_bytes(), after)

    def test_existing_parent_and_dangling_symlink_fail_without_launch(self):
        for kind in ("directory", "symlink"):
            with self.subTest(kind=kind):
                if kind == "directory": self.output_parent.mkdir()
                else: self.output_parent.symlink_to(self.root / "absent")
                result = self.run_script()
                self.assertEqual(result.returncode, 64)
                self.assertIn("fresh_private_output_parent_required", result.stderr)
                self.assertFalse((self.root / "observed-umask").exists())
                if kind == "directory": self.output_parent.rmdir()
                else: self.output_parent.unlink()
