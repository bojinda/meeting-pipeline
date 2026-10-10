"""Operator CLI choices through normal notes generation, entirely synthetic/offline."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import suggest_meeting_speakers as cli
import ollama_session_summary as engine
from meeting_postprocess import editorial as ed, speaker_turns as turns
from meeting_postprocess.publication import publication_payload, strip_private_references

SOURCE = "\n".join([
    "[SPEAKER_00] Let's start the meeting.",
    "[SPEAKER_03] The schedule report is ready.",
    "[SPEAKER_03] The safety inspection remains incomplete.",
    "[SPEAKER_04] The equipment problem remains unresolved.",
    "[SPEAKER_03] The next report will cover staffing."])


def session(root, text=SOURCE):
    transcript = root / "session"
    index = transcript / "chunks_out/transcript_chunks.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"chunk_id": 1, "start_time": 0, "end_time": 60, "text": text}) + "\n", encoding="utf-8")
    return transcript, index


def command(root, *args):
    output = io.StringIO()
    with patch.dict(os.environ, {"MEETING_SUMMARIES_ROOT": str(root / "review")}, clear=True), \
            patch.object(sys, "argv", ["speakers", *map(str, args)]), \
            patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")), \
            patch.object(cli, "review_speakers", side_effect=AssertionError("inference forbidden")), \
            patch.object(cli, "review_turns", side_effect=AssertionError("inference forbidden")), \
            contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
        result = cli.main()
    return result, output.getvalue()


class SpeakerWorkflowTests(unittest.TestCase):
    def test_direct_name_command_changes_only_selected_label_without_json_editing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript, index = session(root)
            original = index.read_bytes()
            command(root, "set-alias", transcript, "--speaker-label", "SPEAKER_03", "--name", "Taylor Morgan")
            command(root, "set-alias", transcript, "--speaker-label", "SPEAKER_00", "--name", "Jordan Lee")
            command(root, "set-alias", transcript, "--speaker-label", "SPEAKER_03", "--name", "Taylor")
            aliases = transcript / "speaker_aliases.json"
            self.assertEqual(json.loads(aliases.read_text()), {"SPEAKER_03": "Taylor", "SPEAKER_00": "Jordan Lee"})
            before = aliases.read_bytes()
            for label, name in (("SPEAKER_99", "Casey"), ("not-a-label", "Casey"), ("SPEAKER_03", "<script>")):
                with self.subTest(label=label, name=name), self.assertRaises(SystemExit):
                    command(root, "set-alias", transcript, "--speaker-label", label, "--name", name)
                self.assertEqual(aliases.read_bytes(), before)
            with self.assertRaises(SystemExit):
                command(root, "set-alias", transcript, "--speaker-label", "SPEAKER_03")
            self.assertEqual(index.read_bytes(), original)
            self.assertFalse((transcript / turns.CORRECTIONS_FILE).exists())
            if os.name != "nt":
                self.assertEqual(aliases.stat().st_mode & 0o777, 0o600)

    def test_readable_inspection_keeps_redactions_and_existing_private_catalog(self):
        text = SOURCE + "\n[SPEAKER_05] Redact the following. PRIVATE_SECRET. End redaction. Visible report."
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript, _ = session(root, text)
            command(root, "set-alias", transcript, "--speaker-label", "SPEAKER_03", "--name", "Taylor Morgan")
            _, log = command(root, "inspect-turns", transcript)
            report = root / "review/session" / turns.INSPECTION_FILE
            readable = report.read_text(encoding="utf-8")
            catalog = json.loads((report.parent / turns.TURNS_FILE).read_text())
            self.assertIn("PRIVATE SPEAKER REVIEW", readable)
            self.assertIn("Current name: Taylor Morgan", readable)
            self.assertIn("Source label: SPEAKER_04 | Current name: SPEAKER_04", readable)
            self.assertIn("Time: 0–60", readable)
            for turn in catalog["turns"]:
                self.assertIn(turn["turn_id"], readable)
            self.assertNotIn("PRIVATE_SECRET", readable + log + json.dumps(catalog))
            self.assertNotIn("The schedule report is ready", log)
            self.assertEqual(strip_private_references(f"[private review]({turns.INSPECTION_FILE})"), "")
            with self.assertRaises(SystemExit):
                command(root, "set-alias", transcript, "--speaker-label", "SPEAKER_99", "--name", "Unknown")
            if os.name != "nt":
                self.assertEqual(report.stat().st_mode & 0o777, 0o600)

    def test_confirmed_names_and_exact_turn_override_reach_notes_while_unknowns_can_be_skipped(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(os.environ["SPEAKER_WORKFLOW_DEMO_DIR"]) if os.environ.get("SPEAKER_WORKFLOW_DEMO_DIR") else Path(tmp) / "demo"
            parent.mkdir(mode=0o700)
            tokenizer_path = parent / "synthetic-tokenizer.json"
            tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(tokenizer_path))
            for approve in (True, False):
                with self.subTest(approve=approve):
                    root = parent / ("confirmed" if approve else "skipped")
                    transcript, index = session(root)
                    original = index.read_bytes()
                    catalog = turns.turn_catalog(transcript)
                    target = catalog["turns"][2]["turn_id"]
                    if approve:
                        command(root, "set-alias", transcript, "--speaker-label", "SPEAKER_03", "--name", "Taylor Morgan")
                        command(root, "approve-turn", transcript, "--turn-id", target, "--name", "Casey")
                    if approve:
                        command(root, "inspect-turns", transcript)
                    # Advisory artifacts are deliberately not approval inputs.
                    (transcript / "speaker-suggestions.json").write_text(json.dumps({"suggestions": [
                        {"speaker_label": "SPEAKER_04", "suggested_name": "Unapproved Person"}]}))
                    calls, paragraphs = [], []
                    def model(**kwargs):
                        calls.append(kwargs)
                        prompt = kwargs["prompt"]
                        self.assertNotIn("Unapproved Person", prompt)
                        if "Transcript chunk:" in prompt:
                            return "## Topics\n" + prompt.split("Transcript chunk:\n", 1)[1]
                        if "propose ONE private canonical" in prompt:
                            return '{"items":[]}'
                        if "summary-reduction call" in prompt:
                            payload = json.loads(prompt.split("Untrusted input JSON:\n", 1)[1])
                            for row in payload["source_excerpts"]:
                                label, body = re.match(r"^\[([^\]]+)\]\s*(.*)$", row["text"]).groups()
                                if "start the meeting" in body:
                                    continue
                                person = "An unidentified participant" if label.startswith("SPEAKER_") else label
                                paragraphs.append({"text": f"{person} reported: {body}", "source_ids": [row["id"]]})
                            return json.dumps({"highlights": [], "previous_context": [], "issues": [{"heading": "Reports", "paragraphs": paragraphs}],
                                               "motions": [], "unresolved": [], "concerns": []})
                        return "# Detailed Minutes\n\n" + "\n\n".join(b["text"] for b in paragraphs)
                    output = root / "notes"
                    argv = ["summary", str(transcript), "--meeting-notes", "--meeting-notes-output-dir", str(output),
                            "--reduce-num-ctx", "196608", "--meeting-notes-tokenizer", str(tokenizer_path)]
                    with patch.dict(os.environ, {"MEETING_SUMMARIES_ROOT": str(root / "production"), "MEETING_KEEP_RECAP": "0"}, clear=True), \
                            patch.object(sys, "argv", argv), patch.object(engine, "call_ollama", side_effect=model), \
                            patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")), \
                            patch.object(engine, "review_turns", side_effect=AssertionError("speaker inference forbidden")), \
                            contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(engine.main(), 0)
                    document = (output / "meeting-notes-draft.md").read_text(encoding="utf-8")
                    prepared = (output / "meeting_sections.jsonl").read_text(encoding="utf-8")
                    if approve:
                        self.assertIn("[Taylor Morgan] The schedule report is ready.", prepared)
                        self.assertIn("[Casey] The safety inspection remains incomplete.", prepared)
                        self.assertIn("[Taylor Morgan] The next report will cover staffing.", prepared)
                        self.assertIn("Taylor Morgan reported: The schedule report is ready.", document)
                        self.assertIn("Casey reported: The safety inspection remains incomplete.", document)
                    else:
                        self.assertNotIn("Taylor Morgan", document + prepared)
                        self.assertNotIn("Casey", document + prepared)
                        self.assertFalse((transcript / "speaker_aliases.json").exists())
                        self.assertFalse((transcript / turns.CORRECTIONS_FILE).exists())
                    self.assertIn("An unidentified participant reported: The equipment problem remains unresolved.", document)
                    self.assertNotIn("SPEAKER_", document)
                    self.assertNotIn("Unapproved Person", document)
                    self.assertEqual(index.read_bytes(), original)
                    self.assertEqual(len(calls), len(engine.load_jsonl(output / "chunk_summaries.jsonl")) + 3)
                    self.assertEqual(json.loads((output / ed.REVIEW).read_text())["status"], "review_hold")
                    with self.assertRaisesRegex(ValueError, "hold"):
                        publication_payload(output)


if __name__ == "__main__":
    unittest.main()
