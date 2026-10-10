"""A fresh native runner stage reuses completed maps after an empty register."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
stage_tests = None
if sys.platform.startswith("linux") and os.environ.get("MEETING_NATIVE_EDITORIAL_FIXTURE") == "1":
    import test_meeting_stage_identity as stage_tests


@unittest.skipUnless(sys.platform.startswith("linux") and os.environ.get("MEETING_NATIVE_EDITORIAL_FIXTURE") == "1", "native synthetic runner harness required")
class EditorialNativeStageTests(unittest.TestCase):
    def setUp(self):
        stage_tests.MeetingStageIdentityTests.setUp(self)

    def tearDown(self):
        stage_tests.MeetingStageIdentityTests.tearDown(self)

    def calls(self):
        return stage_tests.MeetingStageIdentityTests.calls(self)

    def launch(self, output, *, checkpoint=None, empty=False):
        environment = dict(self.environment, MEETING_FAKE_EDITORIAL_REGISTER_EMPTY="1" if empty else "0")
        tokenizer = self.root / "synthetic-tokenizer.json"
        tokenizer.write_text("Synthetic offline counter only")
        options = ["--meeting-notes", "--meeting-notes-output-dir", str(output), "--meeting-notes-tokenizer", str(tokenizer),
                   "--reduce-num-ctx", "98304", "--map-num-ctx", "32768", "--no-keep-recap"]
        if checkpoint:
            options += ["--meeting-notes-checkpoint", str(checkpoint), "--meeting-notes-checkpoint-sha256", hashlib.sha256(checkpoint.read_bytes()).hexdigest()]
        return subprocess.run(["bash", str(ROOT / "bin/summarize-existing-meeting.sh"), str(self.transcript), *options],
                              env=environment, capture_output=True, text=True, timeout=15)

    def test_failed_register_maps_continue_in_fresh_owned_stage_without_remapping(self):
        original = self.root / "failed-editorial"
        failed = self.launch(original, empty=True)
        self.assertEqual(failed.returncode, 1, failed.stderr)
        self.assertEqual(self.controller.snapshot()["owners"], {})
        before = {p.name: p.read_bytes() for p in original.iterdir() if p.is_file()}
        checkpoint = original / "editorial-map-checkpoint.private.json"
        record = json.loads(checkpoint.read_text())
        source_stage = self.controller.store.job(record["source_stage_id"])
        self.assertEqual(source_stage["state"], "failed")
        self.assertTrue(source_stage["cleanup_verified"])
        self.assertTrue(all(r["state"] == "completed" for r in source_stage["host_requests"]))
        first_calls = len(self.calls())
        continued = self.root / "continued-editorial"
        result = self.launch(continued, checkpoint=checkpoint)
        self.assertEqual(result.returncode, 0, result.stderr)
        later = self.calls()[first_calls:]
        generations = [r for r in later if (r.get("payload") or {}).get("prompt")]
        self.assertEqual(len(generations), 3)
        self.assertTrue(all(r["gpu0_free"] and r["gpu1_held"] for r in generations))
        self.assertFalse(any("Transcript chunk:" in r["payload"]["prompt"] for r in generations))
        self.assertEqual({p.name: p.read_bytes() for p in original.iterdir() if p.is_file()}, before)
        self.assertIn("Fixture model committee discussion", (continued / "meeting-notes-draft.md").read_text())
        self.assertEqual(self.controller.snapshot()["owners"], {})
        diagnostics = json.loads((continued / "editorial-generation.private.json").read_text())
        self.assertEqual([r["stage"] for r in diagnostics["requests"]], ["register", "notes", "detailed"])
        self.assertNotIn("PRIVATE_THINKING_SENTINEL", json.dumps(diagnostics))
