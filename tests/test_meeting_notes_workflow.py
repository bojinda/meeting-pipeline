"""Existing unattended wrappers and production notes, with synthetic GPU/model calls."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from meeting_postprocess import editorial as ed
from meeting_postprocess.publication import publication_payload


def fake_generate(**kwargs):
    """Child-process fixture; never contacts a model or GPU."""
    with open(os.environ["NOTES_TEST_CALLS"], "a") as stream:
        stream.write(json.dumps(kwargs, default=str) + "\n")
    prompt = kwargs["prompt"]
    if "Transcript chunk:" in prompt:
        return "## Topics\n" + prompt.split("Transcript chunk:\n", 1)[1]
    if "propose ONE private canonical" in prompt:
        answer = '{"items":[]}'
    elif "summary-reduction call" in prompt:
        evidence = json.loads(prompt.split("Untrusted input JSON:\n", 1)[1])["source_excerpts"]
        ref = next(r["id"] for r in evidence if r["section"] == ed.BUSINESS)
        block = {"text": "A participant reported that staffing remains unresolved.", "source_ids": [ref]}
        answer = json.dumps({"highlights": [], "previous_context": [], "issues": [{"heading": "Staffing", "paragraphs": [block]}],
                             "motions": [], "unresolved": [], "concerns": []})
        if os.environ.get("NOTES_TEST_EMPTY") == "1":
            answer = ""
    else:
        raise AssertionError("Notes-only run must not request detailed minutes or a separate recap")
    callback = kwargs.get("response_callback")
    if callback:
        callback({"done": True, "done_reason": "stop", "response_char_count": len(answer), "thinking_char_count": 10,
                  "num_predict": kwargs["num_predict"], "requested_thinking": "default" if kwargs["thinking"] is None else kwargs["thinking"]})
    return answer


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux unattended shell workflow")
class MeetingNotesWorkflowTests(unittest.TestCase):
    def setUp(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        self.project = self.home / "meeting-pipeline"
        self.binary = self.project / "bin"
        self.binary.mkdir(parents=True)
        for name in ("postprocess-meeting.sh", "summarize-existing-meeting.sh", "transcript_chunker.py", "ollama_meeting_summary.py"):
            shutil.copyfile(ROOT / "bin" / name, self.binary / name)
        (self.binary / "with-gpu-lock.sh").write_text('''aihub_run_gpu_stage() {
  printf '%s\n' "$1" >> "$NOTES_TEST_RESOURCES"
  shift 2
  "$@"
}
''')
        conda = self.home / "miniconda3/etc/profile.d/conda.sh"
        conda.parent.mkdir(parents=True)
        conda.write_text("conda() { :; }\n")
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer = self.home / ".cache/meeting-tokenizers/qwen3.5-27b/tokenizer.json"
        self.tokenizer.parent.mkdir(parents=True)
        tokenizer.save(str(self.tokenizer))
        config = self.project / "config/.env"
        config.parent.mkdir()
        config.write_text(f"HF_TOKEN=synthetic\nDELETE_SOURCE_AUDIO_AFTER_TRANSCRIPTION=0\nMEETING_KEEP_RECAP=1\n"
                          f"MEETING_SUMMARY_PYTHON={sys.executable}\nMEETING_SUMMARIES_ROOT={self.project}/production\n"
                          "OLLAMA_MAP_MODEL=legacy-model\nOLLAMA_REDUCE_MODEL=legacy-model\nOLLAMA_REDUCE_NUM_CTX=32768\n")
        whisper = self.binary / "whisperx"
        whisper.write_text('#!' + sys.executable + '''
import json, pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index('--output_dir') + 1])
segments = [
 {'speaker':'SPEAKER_00','text':"Okay guys, I think we will get started here.",'start':0,'end':2},
 {'speaker':'SPEAKER_01','text':"Staffing remains unresolved.",'start':2,'end':4},
 {'speaker':'SPEAKER_01','text':"Redact the following. PRIVATE_SECRET. End redaction.",'start':4,'end':6}]
(out / 'transcript.json').write_text(json.dumps({'segments':segments}))
''')
        whisper.chmod(0o755)
        python = self.binary / "python"
        python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        python.chmod(0o755)
        guard = self.root / "guard"
        guard.mkdir()
        (guard / "sitecustomize.py").write_text('''import pathlib, socket, sys
def forbidden(*a, **k): raise AssertionError("network/inference forbidden")
socket.socket.connect = forbidden
socket.create_connection = forbidden
if pathlib.Path(sys.argv[0]).name == 'ollama_meeting_summary.py':
    import ollama_session_summary as engine
    from test_meeting_notes_workflow import fake_generate
    engine.call_ollama = fake_generate
''')
        self.calls_file = self.root / "calls.jsonl"
        self.resources = self.root / "resources.txt"
        self.env = dict(os.environ, HOME=str(self.home), HF_TOKEN="synthetic", PYTHONDONTWRITEBYTECODE="1",
                        NOTES_TEST_CALLS=str(self.calls_file), NOTES_TEST_RESOURCES=str(self.resources),
                        PYTHONPATH=os.pathsep.join(map(str, (guard, ROOT / "bin", ROOT / "tests"))),
                        PATH=str(self.binary) + os.pathsep + os.environ["PATH"])
        self.recording = self.project / "meeting.wav"
        self.recording.write_bytes(b"synthetic recording")
        self.transcript = self.project / "meeting-transcripts/meeting"

    def tearDown(self):
        self.temp.cleanup()

    def process(self, empty=False):
        return subprocess.run(["bash", str(self.binary / "postprocess-meeting.sh"), str(self.recording)],
                              env={**self.env, "NOTES_TEST_EMPTY": str(int(empty))}, capture_output=True, text=True, timeout=15)

    def notes_dirs(self):
        return list((self.project / "ignore/meeting-notes").glob("*/notes"))

    def test_unattended_recording_to_private_notes_without_speaker_approval(self):
        result = self.process()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.notes_dirs()[0]
        document = (output / "meeting-notes-draft.md").read_text()
        self.assertIn("A participant reported", document)
        self.assertNotIn("SPEAKER_", document)
        self.assertFalse((self.transcript / "speaker_aliases.json").exists())
        self.assertFalse((self.transcript / "speaker_turn_corrections.json").exists())
        calls = [json.loads(line) for line in self.calls_file.read_text().splitlines()]
        reduced = [c for c in calls if "Transcript chunk:" not in c["prompt"]]
        self.assertEqual(len(reduced), 2)
        self.assertEqual([c["num_predict"] for c in reduced], [16384, 24576])
        self.assertTrue(all(c["num_ctx"] == 196608 and c["model"] == "qwen3.8:27b" and c["thinking"] is None for c in reduced))
        self.assertTrue(all(c["num_ctx"] == 16384 for c in calls if "Transcript chunk:" in c["prompt"]))
        self.assertNotIn("PRIVATE_SECRET", json.dumps(calls))
        self.assertEqual(self.resources.read_text().splitlines(), ["gpu0", "gpu0+gpu1"])
        self.assertFalse((output / "minutes-draft.md").exists())
        self.assertEqual(json.loads((output / ed.REVIEW).read_text())["status"], "review_hold")
        self.assertEqual((output / "meeting-notes-draft.md").stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, "hold"):
            publication_payload(output)
        status = (self.transcript / "status.txt").read_text()
        self.assertIn("Draft ready with review warnings", status)
        self.assertIn(str(output / "meeting-notes-draft.md"), status)
        self.assertIn("Summary runner exit status: 0", status)
        if os.environ.get("NOTES_WORKFLOW_DEMO_DIR"):
            demonstration = Path(os.environ["NOTES_WORKFLOW_DEMO_DIR"])
            shutil.copytree(output, demonstration)  # Preserve a synthetic review artifact, never overwrite.
            (demonstration / "status.txt").write_text(status)
        before = (output / "meeting-notes-draft.md").read_bytes()
        repeated = subprocess.run(["bash", str(self.binary / "summarize-existing-meeting.sh"), str(self.transcript),
                                   "--meeting-notes", "--dual-gpu-target", "meeting-dual"],
                                  env=self.env, capture_output=True, text=True, timeout=15)
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual(len(self.notes_dirs()), 2)
        self.assertEqual((output / "meeting-notes-draft.md").read_bytes(), before)

    def test_no_final_answer_reports_no_usable_draft_and_preserves_evidence(self):
        result = self.process(empty=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        output = self.notes_dirs()[0]
        self.assertFalse((output / "meeting-notes-draft.md").exists())
        self.assertTrue((output / ed.RESPONSE).exists())
        self.assertIn("No usable draft", (self.transcript / "status.txt").read_text())
        self.assertIn("Summary runner exit status: 1", (self.transcript / "status.txt").read_text())
        self.assertEqual(json.loads((output / ed.REVIEW).read_text())["status"], "review_hold")

    def test_manual_generation_overrides_are_preserved(self):
        self.assertEqual(self.process().returncode, 0)
        self.calls_file.unlink()
        output = self.project / "explicit-notes"
        result = subprocess.run(["bash", str(self.binary / "summarize-existing-meeting.sh"), str(self.transcript),
                                 "--meeting-notes", "--meeting-notes-output-dir", str(output),
                                 "--meeting-notes-tokenizer", str(self.tokenizer), "--reduce-num-ctx", "98304",
                                 "--map-num-ctx", "32768", "--reduce-model", "operator-model", "--meeting-notes-thinking", "disabled"],
                                env=self.env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        reduced = [json.loads(line) for line in self.calls_file.read_text().splitlines()
                   if "Transcript chunk:" not in json.loads(line)["prompt"]]
        self.assertEqual(len(reduced), 2)
        self.assertTrue(all(c["thinking"] is False and c["num_ctx"] == 98304 and c["model"] == "operator-model" for c in reduced))
        self.assertTrue((output / "meeting-notes-draft.md").exists())


if __name__ == "__main__":
    unittest.main()
