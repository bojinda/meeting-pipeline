from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import ollama_session_summary as engine
import suggest_meeting_speakers as command
from meeting_postprocess.publication import PRIVATE_MEETING_FILENAMES, PUBLIC_MEETING_FILENAMES, export_archive, export_documents, publication_payload, public_meeting_files
from meeting_postprocess.speaker_suggestions import PRIVATE_SUGGESTIONS_FILENAME, PRIVATE_ROSTER_FILENAME, build_speaker_suggestions, load_roster, write_private_json


def chunks(text, identifier=1, **metadata):
    return [{"chunk_id": identifier, "text": text, **metadata}]


def suggestions(text, **kwargs):
    report = build_speaker_suggestions(chunks(text), **kwargs)
    return {row["speaker_label"]: row for row in report["suggestions"]}


class TextSuggestionTests(unittest.TestCase):
    def test_explicit_self_identification_uses_only_spoken_name(self):
        for text in ("That's me, Taylor Morgan.", "My name is Taylor Morgan.", "I'm Taylor Morgan.", "This is Taylor Morgan speaking."):
            with self.subTest(text=text):
                row = suggestions("[SPEAKER_03] " + text)["SPEAKER_03"]
                self.assertEqual((row["suggested_name"], row["confidence"]), ("Taylor Morgan", "high"))
                self.assertIn("self_identification", row["evidence_types"])
                self.assertIn("Taylor Morgan", json.dumps(row["evidence"]))

    def test_introduction_requires_immediate_response_by_another_label(self):
        source = "[SPEAKER_00] I'd like to introduce Taylor Morgan.\n[SPEAKER_03] Good morning."
        row = suggestions(source)["SPEAKER_03"]
        self.assertEqual((row["suggested_name"], row["confidence"]), ("Taylor Morgan", "medium"))
        self.assertIn("introduction", row["evidence_types"])
        source = "[SPEAKER_00] I'd like to introduce Taylor Morgan.\n[SPEAKER_02] The report is missing.\n[SPEAKER_03] Good morning."
        self.assertIsNone(suggestions(source)["SPEAKER_03"]["suggested_name"])

    def test_name_extraction_stops_at_sentence_boundaries(self):
        row = suggestions("[SPEAKER_03] My name is Taylor Morgan. Casey Riley.")["SPEAKER_03"]
        self.assertEqual(row["suggested_name"], "Taylor Morgan")
        self.assertEqual(row["candidates"], ["Taylor Morgan"])
        self.assertNotIn("Casey Riley", json.dumps(row["evidence"]))

    def test_repeated_direct_address_is_stronger_than_single_address(self):
        source = "[SPEAKER_00] Taylor, can you give the report?\n[SPEAKER_03] Sure."
        self.assertEqual(suggestions(source)["SPEAKER_03"]["confidence"], "low")
        source += "\n[SPEAKER_00] Taylor, are you ready?\n[SPEAKER_03] Yes."
        row = suggestions(source)["SPEAKER_03"]
        self.assertEqual((row["suggested_name"], row["confidence"]), ("Taylor", "medium"))
        self.assertEqual(len(row["evidence"]), 2)

    def test_address_denials_and_mentions_are_not_identity_assignments(self):
        for text in ("[SPEAKER_00] Taylor, are you there?\n[SPEAKER_03] No, not me.", "[SPEAKER_00] Taylor filed the report.\n[SPEAKER_03] Yes."):
            with self.subTest(text=text):
                self.assertIsNone(suggestions(text)["SPEAKER_03"]["suggested_name"])

    def test_address_response_can_cross_source_chunks(self):
        source = chunks("[SPEAKER_00] Taylor, are you ready?", "2.1") + chunks("[SPEAKER_03] Yes.", "3.1")
        report = build_speaker_suggestions(source)
        row = next(row for row in report["suggestions"] if row["speaker_label"] == "SPEAKER_03")
        self.assertEqual(row["suggested_name"], "Taylor")

    def test_roster_alone_role_only_and_conversation_do_not_invent_names(self):
        roster = [{"name": "Taylor Morgan", "aliases": ["Taylor"], "role": "Chair"}]
        for text in ("I'm the chair.", "I'm really tired.", "The schedule is ready.", "This is The Report."):
            with self.subTest(text=text):
                row = suggestions("[SPEAKER_03] " + text, roster=roster)["SPEAKER_03"]
                self.assertEqual((row["suggested_name"], row["confidence"], row["candidates"]), (None, "unknown", []))

    def test_roster_canonical_spelling_requires_transcript_link_and_handles_case(self):
        roster = [{"name": "Taylor Morgan", "aliases": ["Taylor"], "role": "Chair"}]
        row = suggestions("[SPEAKER_03] My name is taylor.", roster=roster)["SPEAKER_03"]
        self.assertEqual(row["suggested_name"], "Taylor Morgan")
        self.assertIn("taylor", json.dumps(row["evidence"]))
        row = suggestions("[SPEAKER_03] My name is Casey Riley.", roster=roster)["SPEAKER_03"]
        self.assertEqual(row["suggested_name"], "Casey Riley")

    def test_conflicting_self_names_and_ambiguous_roster_variants_stay_unresolved(self):
        row = suggestions("[SPEAKER_03] My name is Taylor Morgan. My name is Casey Riley.")["SPEAKER_03"]
        self.assertEqual(set(row["candidates"]), {"Taylor Morgan", "Casey Riley"})
        self.assertIsNone(row["suggested_name"])
        self.assertTrue(row["ambiguity"])
        roster = [{"name": "Taylor Morgan", "aliases": [], "role": ""}, {"name": "Taylor Riley", "aliases": [], "role": ""}]
        row = suggestions("[SPEAKER_03] My name is Taylor.", roster=roster)["SPEAKER_03"]
        self.assertEqual(set(row["candidates"]), {"Taylor Morgan", "Taylor Riley"})
        self.assertIsNone(row["suggested_name"])

    def test_approved_aliases_are_authoritative_and_approved_elsewhere_is_flagged(self):
        approved = {"SPEAKER_03": "Taylor Morgan"}
        original = dict(approved)
        rows = suggestions("[SPEAKER_03] My name is Casey Riley.\n[SPEAKER_04] My name is Taylor Morgan.", approved=approved)
        self.assertNotIn("SPEAKER_03", rows)
        self.assertIsNone(rows["SPEAKER_04"]["suggested_name"])
        self.assertTrue(rows["SPEAKER_04"]["ambiguity"])
        self.assertEqual(approved, original)

    def test_role_evidence_is_secondary_and_never_raises_confidence(self):
        roster = [{"name": "Taylor Morgan", "aliases": ["Taylor"], "role": "Chair"}]
        row = suggestions("[SPEAKER_00] Taylor, can you report?\n[SPEAKER_03] Sure. As chair, I can report.", roster=roster)["SPEAKER_03"]
        self.assertEqual(row["confidence"], "low")
        self.assertIn("role_context", row["evidence_types"])

    def test_redacted_identity_and_raw_metadata_are_never_used(self):
        text = "[SPEAKER_03] Redact the following. My name is Casey Riley. End redaction. My name is Taylor Morgan."
        source = chunks(text, hidden_text="My name is Casey Riley.", speaker_span="Casey Riley")
        original = json.dumps(source)
        report = build_speaker_suggestions(source)
        self.assertNotIn("Casey Riley", json.dumps(report))
        self.assertEqual(report["suggestions"][0]["suggested_name"], "Taylor Morgan")
        self.assertEqual(json.dumps(source), original)
        self.assertEqual(build_speaker_suggestions(chunks("[SPEAKER_03] Redact the following. My name is Casey Riley."))["suggestions"], [])

    def test_address_links_across_redaction_gaps_are_not_inferred(self):
        source = chunks("[SPEAKER_00] Taylor, are you there? Redact the following.", "4.2") + chunks("[SPEAKER_04] Private interruption. End redaction.\n[SPEAKER_03] Yes.", "5.1")
        report = build_speaker_suggestions(source)
        row = next(row for row in report["suggestions"] if row["speaker_label"] == "SPEAKER_03")
        self.assertIsNone(row["suggested_name"])
        self.assertNotIn("Private interruption", json.dumps(report))

    def test_evidence_is_concise_and_keeps_the_actual_late_address(self):
        source = "[SPEAKER_00] " + "The budget is ready. " * 30 + "Taylor, can you report?\n[SPEAKER_03] Sure."
        row = suggestions(source)["SPEAKER_03"]
        self.assertIn("Taylor", row["evidence"][0]["excerpts"][0])
        self.assertTrue(all(len(excerpt) <= 240 for item in row["evidence"] for excerpt in item["excerpts"]))


class SpeakerPrivacyTests(unittest.TestCase):
    def test_review_roster_and_aliases_are_explicitly_private_in_all_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in PUBLIC_MEETING_FILENAMES:
                (root / name).write_text("# Public\nMeeting business.", encoding="utf-8")
            for name in (PRIVATE_SUGGESTIONS_FILENAME, PRIVATE_ROSTER_FILENAME, "speaker_aliases.json"):
                self.assertIn(name.casefold(), PRIVATE_MEETING_FILENAMES)
                (root / name).write_text("PRIVATE_IDENTITY_REVIEW", encoding="utf-8")
                with (root / "summary.md").open("a", encoding="utf-8") as stream:
                    stream.write(f"\n[Private review]({name})")
            payload = json.dumps(publication_payload(root))
            self.assertNotIn("PRIVATE_IDENTITY_REVIEW", payload)
            self.assertNotIn("speaker-", payload)
            self.assertNotIn("speaker_", payload)
            destination = root / "website"
            export_documents(root, destination)
            self.assertEqual({path.name for path in destination.iterdir()}, set(PUBLIC_MEETING_FILENAMES))
            archive = root / "public.zip"
            export_archive(root, archive)
            with zipfile.ZipFile(archive) as document:
                self.assertEqual(set(document.namelist()), set(PUBLIC_MEETING_FILENAMES))
                self.assertTrue(all(b"speaker" not in document.read(name) for name in document.namelist()))

    def test_hardlinked_private_review_cannot_be_exported_as_public_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (PRIVATE_SUGGESTIONS_FILENAME, PRIVATE_ROSTER_FILENAME):
                private = root / name
                private.write_text("PRIVATE_IDENTITY_REVIEW")
                public = root / "summary.md"
                os.link(private, public)
                self.assertEqual(public_meeting_files(root), [])
                public.unlink()

    def test_private_paths_are_denied_and_private_contents_are_never_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_directory = root / PRIVATE_SUGGESTIONS_FILENAME
            private_directory.mkdir()
            (private_directory / "summary.md").write_text("Private")
            self.assertEqual(public_meeting_files(private_directory), [])
            with self.assertRaises(ValueError):
                export_documents(root, root / PRIVATE_ROSTER_FILENAME)
            original = Path.read_text
            def read(path, *args, **kwargs):
                self.assertNotIn(path.name.casefold(), PRIVATE_MEETING_FILENAMES)
                return original(path, *args, **kwargs)
            (root / PRIVATE_ROSTER_FILENAME).write_text("Private")
            (root / "summary.md").write_text("Public")
            with patch.object(Path, "read_text", read):
                self.assertEqual(len(publication_payload(root)["documents"]), 1)

    @unittest.skipIf(os.name == "nt", "POSIX mode bits require Linux")
    def test_private_write_is_owner_only_even_when_replacing_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / PRIVATE_SUGGESTIONS_FILENAME
            path.write_text("Old review")
            path.chmod(0o644)
            write_private_json(path, {"suggestions": []})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(Path(directory).glob(".private-speaker-*.tmp")), [])

    def test_private_artifact_and_roster_are_gitignored_without_creating_them(self):
        git = shutil.which("git")
        if not git:
            self.skipTest("Git unavailable")
        for name in (PRIVATE_SUGGESTIONS_FILENAME, PRIVATE_ROSTER_FILENAME, "speaker_aliases.json", ".private-speaker-demo.tmp"):
            result = subprocess.run([git, "check-ignore", "--no-index", "custom-output/session/" + name], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


class SpeakerCommandTests(unittest.TestCase):
    def setup_meeting(self, root):
        transcript = root / "session"
        index = transcript / "chunks_out" / "transcript_chunks.jsonl"
        index.parent.mkdir(parents=True)
        index.write_text(json.dumps(chunks("[SPEAKER_03] That's me, Taylor Morgan.\n[SPEAKER_04] My name is Casey Riley.")[0]) + "\n")
        return transcript

    def run_command(self, root, argv):
        with patch.object(sys, "argv", ["speaker_review", *map(str, argv)]), patch.dict(os.environ, {"MEETING_SUMMARIES_ROOT": str(root / "outputs")}, clear=True), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return command.main()

    def test_standalone_stage_and_explicit_approval_merge_only_selected_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = self.setup_meeting(root)
            self.assertEqual(self.run_command(root, ["suggest", transcript]), 0)
            alias_file = transcript / "speaker_aliases.json"
            self.assertFalse(alias_file.exists())
            review = root / "outputs" / transcript.name / PRIVATE_SUGGESTIONS_FILENAME
            self.assertTrue(review.exists())
            self.assertEqual(self.run_command(root, ["approve", transcript, "--approve", "SPEAKER_03"]), 0)
            self.assertEqual(json.loads(alias_file.read_text()), {"SPEAKER_03": "Taylor Morgan"})

    def test_no_implicit_approval_cross_meeting_or_existing_alias_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = self.setup_meeting(root)
            self.run_command(root, ["suggest", transcript])
            review = root / "outputs" / transcript.name / PRIVATE_SUGGESTIONS_FILENAME
            with self.assertRaises(SystemExit):
                self.run_command(root, ["approve", transcript])
            self.assertFalse((transcript / "speaker_aliases.json").exists())
            other = root / "different-session"
            other.mkdir()
            with self.assertRaises(SystemExit):
                self.run_command(root, ["approve", other, "--suggestions", review, "--approve", "SPEAKER_03"])
            self.assertFalse((other / "speaker_aliases.json").exists())
            aliases = transcript / "speaker_aliases.json"
            original = '{"SPEAKER_03":"Morgan Riley"}'
            aliases.write_text(original)
            with self.assertRaises(SystemExit):
                self.run_command(root, ["approve", transcript, "--approve", "SPEAKER_04", "--approve", "SPEAKER_03"])
            self.assertEqual(aliases.read_text(), original)

    def test_null_or_unknown_suggestions_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = self.setup_meeting(root)
            self.run_command(root, ["suggest", transcript])
            review = root / "outputs" / transcript.name / PRIVATE_SUGGESTIONS_FILENAME
            data = json.loads(review.read_text())
            data["suggestions"][0]["suggested_name"] = None
            review.write_text(json.dumps(data))
            for label in ("SPEAKER_03", "SPEAKER_99"):
                with self.subTest(label=label), self.assertRaises(SystemExit):
                    self.run_command(root, ["approve", transcript, "--approve", label])
            self.assertFalse((transcript / "speaker_aliases.json").exists())

    def test_optional_roster_discovery_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(load_roster(root), [])
            path = root / PRIVATE_ROSTER_FILENAME
            path.write_text(json.dumps({"people": [{"name": "Taylor Morgan", "aliases": ["Taylor"], "role": "Chair"}]}))
            self.assertEqual(load_roster(root)[0]["name"], "Taylor Morgan")
            path.write_text('["Casey Riley"]')
            self.assertEqual(load_roster(root)[0]["name"], "Casey Riley")
            for invalid in ('{}', '{"people":[{"name":"Taylor","aliases":"not a list"}]}', '{"people":[{"name":"SPEAKER_03"}]}'):
                path.write_text(invalid)
                with self.assertRaises(ValueError):
                    load_roster(root)

    def test_approval_rejects_review_made_before_spoken_redaction_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = self.setup_meeting(root)
            self.run_command(root, ["suggest", transcript])
            index = transcript / "chunks_out" / "transcript_chunks.jsonl"
            index.write_text(json.dumps(chunks("[SPEAKER_03] Redact the following. That's me, Taylor Morgan. End redaction. Public business.")[0]) + "\n")
            with self.assertRaises(SystemExit):
                self.run_command(root, ["approve", transcript, "--approve", "SPEAKER_03"])
            self.assertFalse((transcript / "speaker_aliases.json").exists())


class SpeakerPipelineTests(unittest.TestCase):
    def run_pipeline(self, root, enabled=False, source=None, profile="meeting", invalid_roster=False, stale=False, llm=False, options=None):
        transcript = root / "session"
        index = transcript / "chunks_out" / "transcript_chunks.jsonl"
        index.parent.mkdir(parents=True)
        source = source if source is not None else "[SPEAKER_00] Let's get started.\n[SPEAKER_03] That's me, Taylor Morgan.\n[SPEAKER_04] Public business."
        original = json.dumps({"chunk_id": 1, "file_name": "001.md", "text": source}) + "\n"
        index.write_text(original)
        alias_file = transcript / "speaker_aliases.json"
        original_aliases = '{"SPEAKER_04":"Riley"}'
        alias_file.write_text(original_aliases)
        roster = transcript / PRIVATE_ROSTER_FILENAME
        roster.write_text("not JSON" if invalid_roster else '["Casey Morgan"]')
        output = root / "outputs" / transcript.name
        if stale:
            output.mkdir(parents=True)
            (output / PRIVATE_SUGGESTIONS_FILENAME).write_text("STALE_PRIVATE_IDENTITY")
        calls = []
        def response(**kwargs):
            calls.append(kwargs)
            return "# Model output\n- SPEAKER_03 offered comments."
        argv = ["summary", str(transcript), "--profile", profile] + (["--suggest-speakers"] if enabled else []) + (["--suggest-speakers-llm"] if llm else []) + (options or [])
        with patch.object(sys, "argv", argv), patch.dict(os.environ, {"MEETING_SUMMARIES_ROOT": str(root / "outputs"), "LESSON_SUMMARIES_ROOT": str(root / "outputs")}, clear=True), patch.object(engine, "call_ollama", side_effect=response), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = engine.main()
        self.assertEqual(index.read_text(), original)
        self.assertEqual(alias_file.read_text(), original_aliases)
        return status, calls, output

    def test_opt_in_stage_does_not_change_models_prompts_approved_aliases_or_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, base_calls, base_output = self.run_pipeline(root / "baseline")
            status, calls, output = self.run_pipeline(root / "suggested", enabled=True)
            self.assertEqual((baseline, status), (0, 0))
            self.assertFalse((base_output / PRIVATE_SUGGESTIONS_FILENAME).exists())
            self.assertEqual(calls, base_calls)
            report = json.loads((output / PRIVATE_SUGGESTIONS_FILENAME).read_text())
            row = next(row for row in report["suggestions"] if row["speaker_label"] == "SPEAKER_03")
            self.assertEqual(row["suggested_name"], "Taylor Morgan")
            self.assertNotIn("SPEAKER_04", [row["speaker_label"] for row in report["suggestions"]])
            self.assertNotIn("Casey Morgan", json.dumps(calls))
            self.assertNotIn(PRIVATE_SUGGESTIONS_FILENAME, json.dumps(calls))
            for name in PUBLIC_MEETING_FILENAMES:
                self.assertEqual((output / name).read_text(), (base_output / name).read_text())
                self.assertIn("SPEAKER_03", (output / name).read_text())
                self.assertNotIn("Taylor Morgan", (output / name).read_text())

    def test_redacted_self_identification_is_not_in_suggestions_or_model_calls(self):
        source = "[SPEAKER_00] Let's get started.\n[SPEAKER_03] Redact the following. My name is Casey Riley. End redaction. My name is Taylor Morgan."
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(Path(directory), enabled=True, source=source)
            self.assertEqual(status, 0)
            report = (output / PRIVATE_SUGGESTIONS_FILENAME).read_text()
            self.assertNotIn("Casey Riley", report)
            self.assertIn("Taylor Morgan", report)
            self.assertNotIn("Casey Riley", json.dumps(calls))

    def test_redaction_rerun_clears_stale_suggestions_when_stage_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(Path(directory), source="[SPEAKER_03] Public. Redact the following. Private identity. End redaction.", stale=True)
            self.assertEqual(status, 0)
            self.assertFalse((output / PRIVATE_SUGGESTIONS_FILENAME).exists())

    def test_all_redacted_meeting_writes_empty_review_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(Path(directory), enabled=True, source="[SPEAKER_03] Redact the following. My name is Taylor Morgan.", stale=True)
            self.assertEqual(status, 0)
            self.assertEqual(calls, [])
            self.assertEqual(json.loads((output / PRIVATE_SUGGESTIONS_FILENAME).read_text())["suggestions"], [])

    def test_disabled_stage_ignores_roster_errors_but_requested_stage_fails_before_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status, _, _ = self.run_pipeline(root / "disabled", invalid_roster=True)
            self.assertEqual(status, 0)
            status, calls, _ = self.run_pipeline(root / "enabled", enabled=True, invalid_roster=True)
            self.assertEqual(status, 2)
            self.assertEqual(calls, [])

    def test_lesson_defaults_remain_unchanged_and_suggestions_are_meeting_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status, calls, output = self.run_pipeline(root / "normal", profile="lesson", invalid_roster=True)
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 5)
            self.assertFalse((output / PRIVATE_SUGGESTIONS_FILENAME).exists())
            with self.assertRaises(SystemExit):
                self.run_pipeline(root / "requested", profile="lesson", enabled=True)


if __name__ == "__main__":
    unittest.main()
