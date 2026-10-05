from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import ollama_session_summary as engine
from meeting_postprocess.aliases import load_aliases
from meeting_postprocess.normalization import normalize_text, prepare_text
from meeting_postprocess.qa import check_minutes
from meeting_postprocess.rendering import insert_recap
from meeting_postprocess.sections import ADJOURNMENT, BUSINESS, PRE_MEETING, RECAP, prepare_chunks


def chunk(text: str, chunk_id: int = 1) -> dict:
    return {
        "chunk_id": chunk_id, "file_name": f"{chunk_id:03d}.md", "text": text,
        "speaker_span": "SPEAKER_00 -> SPEAKER_01", "chunk_type": "discussion",
        "start_time": 0.0, "end_time": 60.0,
    }


class NormalizationTests(unittest.TestCase):
    def test_artifacts_and_correct_unicode_on_same_line(self):
        source = "Mack Yard, Mackyard, Mac yard: Transport Canadaâ€™s hypodermical policy. Montréal’s café."
        expected = "Mac Yard, Mac Yard, Mac Yard: Transport Canada’s hypodermic policy. Montréal’s café."
        self.assertEqual(normalize_text(source), expected)
        self.assertEqual(normalize_text(expected), expected)

    def test_double_encoded_and_latin1_punctuation(self):
        for encoding in ("cp1252", "latin1"):
            with self.subTest(encoding=encoding):
                broken = "‘Canada’ — café".encode("utf-8").decode(encoding)
                self.assertEqual(normalize_text(broken), "‘Canada’ — café")
        broken = "Canada’s".encode("utf-8").decode("cp1252").encode("utf-8").decode("cp1252")
        self.assertEqual(normalize_text(broken), "Canada’s")

    def test_aliases_replace_whole_labels_only(self):
        self.assertEqual(
            prepare_text("[SPEAKER_01] SPEAKER_010, SPEAKER_02, XSPEAKER_01", {"SPEAKER_01": "Sam"}),
            "[Sam] SPEAKER_010, SPEAKER_02, XSPEAKER_01",
        )

    def test_alias_discovery_and_explicit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(load_aliases(root), {})
            (root / "speaker_aliases.json").write_text('{"SPEAKER_00": "  Renée  "}', encoding="utf-8-sig")
            self.assertEqual(load_aliases(root), {"SPEAKER_00": "Renée"})
            explicit = root / "alternate.json"
            explicit.write_text('{"SPEAKER_00": "Alex"}', encoding="utf-8")
            self.assertEqual(load_aliases(root, explicit), {"SPEAKER_00": "Alex"})
            with self.assertRaises(OSError):
                load_aliases(root, root / "missing.json")

    def test_invalid_aliases_fail_instead_of_guessing(self):
        invalid = [[], {"Alice": "Bob"}, {"SPEAKER_01": ""}, {"SPEAKER_01": 1},
                   {"SPEAKER_01": "SPEAKER_02"}, {"SPEAKER_01": "Alice\nBob"},
                   {"SPEAKER_01": "[Alice]"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "speaker_aliases.json"
            for data in invalid:
                with self.subTest(data=data):
                    path.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_aliases(root)


class SectionTests(unittest.TestCase):
    def test_all_four_sections_inside_one_merged_turn(self):
        text = "[SPEAKER_00] Good morning everyone. I call the meeting to order. Let's review the previous meeting. We approved the OLD plan. Moving on to new business. We approved the NEW plan. The meeting is adjourned."
        rows = prepare_chunks([chunk(text)], {"SPEAKER_00": "Alex"})
        self.assertEqual([row["meeting_section"] for row in rows], [PRE_MEETING, BUSINESS, RECAP, BUSINESS, ADJOURNMENT])
        self.assertIn("OLD plan", rows[2]["text"])
        self.assertIn("NEW plan", rows[3]["text"])
        self.assertTrue(all(row["source_chunk_id"] == 1 and row["start_time"] == 0.0 for row in rows))
        self.assertTrue(all("SPEAKER_00" not in row["text"] for row in rows))
        self.assertEqual(len({row["chunk_id"] for row in rows}), len(rows))
        # All text words survive, only the per-sentence speaker labels change.
        original_words = text.replace("[SPEAKER_00] ", "").split()
        prepared_words = " ".join(row["text"].replace("[Alex] ", "") for row in rows).split()
        self.assertEqual(prepared_words, original_words)

    def test_state_survives_source_chunk_boundaries(self):
        rows = prepare_chunks([
            chunk("[SPEAKER_00] At the last meeting, we discussed the old budget."),
            chunk("[SPEAKER_01] It was approved. Next item on the agenda: the new budget.", 2),
        ], {})
        self.assertEqual([r["meeting_section"] for r in rows], [RECAP, RECAP, BUSINESS])

    def test_ambiguous_discussion_and_casual_motion_stay_current(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] We should revisit the previous meeting's budget. I move to adjourn. If the meeting is adjourned, can we continue tomorrow?")], {})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["meeting_section"], BUSINESS)

    def test_approval_of_previous_minutes_is_current_business(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] Let's review the minutes from our last meeting. Approval of the previous minutes. New business: the schedule.")], {})
        self.assertEqual([r["meeting_section"] for r in rows], [RECAP, BUSINESS])

    def test_greetings_during_business_do_not_become_pre_meeting(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] The budget is approved.\n[SPEAKER_01] Hello, I joined late.")], {})
        self.assertEqual([r["meeting_section"] for r in rows], [BUSINESS])

    def test_unclear_recap_continuation_is_retained_as_current_business(self):
        rows = prepare_chunks([chunk("[SPEAKER_00] At the last meeting, the budget was approved.\n[SPEAKER_01] We need to reconsider the budget.\n[SPEAKER_00] The deadline is Friday.")], {})
        self.assertEqual([r["meeting_section"] for r in rows], [RECAP, BUSINESS])
        self.assertIn("deadline is Friday", rows[1]["text"])
        rows = prepare_chunks([chunk("[SPEAKER_00] Let's recap the previous meeting.\n[SPEAKER_01] An uncertain new topic.")], {})
        self.assertEqual([r["meeting_section"] for r in rows], [RECAP, BUSINESS])


class QATests(unittest.TestCase):
    def test_each_required_defect_is_flagged_with_a_line(self):
        findings = check_minutes("# Draft Minutes\n-SPEAKER_00 discussed Mackyard.\n- Transport Canadaâ€™s rules.\n- Moved by: Unknown; Seconded by: Unknown.")
        self.assertEqual({f.code for f in findings}, {"unresolved_speaker", "malformed_bullet", "mojibake", "mac_yard_spelling", "questionable_motion"})
        self.assertTrue(all(f.line >= 2 and f.excerpt for f in findings))

    def test_valid_markdown_and_explicit_motion_evidence(self):
        source = "[Alex] I move to approve the budget.\n[Sam] I second the motion."
        minutes = "# Draft Minutes\n\n- Mac Yard budget approved.\n  - Transport Canada’s policy.\n- **Moved by:** Alex; **Seconded by:** Sam.\n\n---\n\n*Emphasized paragraph*\n\n```text\n-not a markdown bullet\n```"
        self.assertEqual(check_minutes(minutes, source), [])

    def test_tentative_same_person_and_unsupported_attributions(self):
        source = "[Alex] We discussed the budget.\n[Sam] I was present."
        minutes = "- Moved by: Alex\n  Seconder: Alex\n\n- Possibly moved by Sam."
        findings = check_minutes(minutes, source)
        self.assertTrue(any("same person" in f.message for f in findings))
        self.assertTrue(any("no explicit role evidence" in f.message for f in findings))
        self.assertTrue(any("tentative" in f.message for f in findings))

    def test_explicit_named_roles_and_no_guess_from_attendance(self):
        source = "[Chair] Moved by Alex; seconded by Sam.\n[Sam] I second the question about attendance."
        self.assertEqual(check_minutes("- Mover: Alex; Seconder: Sam.", source), [])
        findings = check_minutes("- Seconder: Sam.", "[Sam] I second the question about attendance.")
        self.assertEqual(len(findings), 1)

    def test_malformed_bullets(self):
        for text in ("-bad bullet", "+bad bullet", "*bad bullet", "- - doubled", "• non-markdown", "1.missing space", "- "):
            with self.subTest(text=text):
                self.assertIn("malformed_bullet", {f.code for f in check_minutes(text)})

    def test_recap_heading_is_owned_by_pipeline(self):
        result = insert_recap("# Draft Minutes\n\n## Overview\nCurrent business.", "# A model title\n- Earlier business.")
        self.assertEqual(result.count("## Recap of Previous Meeting"), 1)
        self.assertNotIn("A model title", result)
        self.assertLess(result.index("Earlier business"), result.index("## Overview"))


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, root: Path, text: str, options: list[str] | None = None, aliases=None, profile="meeting", environment=None, default_profile=None):
        transcript = root / "session-123"
        chunk_dir = transcript / "chunks_out"
        chunk_dir.mkdir(parents=True)
        index = chunk_dir / "transcript_chunks.jsonl"
        original = json.dumps(chunk(text), ensure_ascii=False) + "\n"
        index.write_text(original, encoding="utf-8")
        if aliases is not None:
            (transcript / "speaker_aliases.json").write_text(json.dumps(aliases), encoding="utf-8")
        calls = []

        def fake_ollama(**kwargs):
            calls.append(kwargs)
            prompt = kwargs["prompt"]
            if "Transcript chunk:\n" in prompt:
                transcript_text = prompt.split("Transcript chunk:\n", 1)[1].split("\nWrite concise markdown", 1)[0]
                return "## Topics\n- " + transcript_text.strip()
            if "Summarize only the supplied historical recap" in prompt:
                return "# Unexpected model title\n- Previously approved OLD Mac yard plan."
            if "write formal draft minutes" in prompt:
                return "# Draft Minutes\n\n## Topics Discussed\n- Alex discussed Mackyard and Transport Canadaâ€™s hypodermical guidance.\n- SPEAKER_09 offered comments.\n-bad bullet\n- Moved by: Alex; Seconded by: Sam."
            return "# Other output\n- SPEAKER_00 Mack Yard."

        argv = ["summarizer", str(transcript)]
        if profile is not None:
            argv.extend(["--profile", profile])
        argv.extend(options or [])
        env = {"MEETING_SUMMARIES_ROOT": str(root / "outputs"), "LESSON_SUMMARIES_ROOT": str(root / "outputs"), "MEETING_KEEP_RECAP": "0"}
        env.update(environment or {})
        with patch.object(sys, "argv", argv), patch.dict(os.environ, env, clear=True), patch.object(engine, "call_ollama", side_effect=fake_ollama), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = engine.main(default_profile=default_profile)
        self.assertEqual(index.read_text(encoding="utf-8"), original)
        return status, calls, root / "outputs" / transcript.name

    def test_default_minutes_exclude_chatter_and_recap_and_run_qa(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(Path(directory), "[SPEAKER_00] Good morning. Let's review the previous meeting. OLD plan approved. Moving on to new business. NEW Mac yard plan discussed. The meeting is adjourned.", aliases={"SPEAKER_00": "Alex"})
            self.assertEqual(status, 0)
            minutes_prompt = next(c["prompt"] for c in calls if "write formal draft minutes" in c["prompt"])
            self.assertNotIn("OLD plan", minutes_prompt)
            self.assertNotIn("Good morning", minutes_prompt)
            self.assertIn("NEW Mac Yard plan", minutes_prompt)
            self.assertNotIn("[SPEAKER_00]", minutes_prompt)
            minutes = (output / "minutes-draft.md").read_text(encoding="utf-8")
            self.assertNotIn("Recap of Previous Meeting", minutes)
            self.assertIn("Mac Yard and Transport Canada’s hypodermic", minutes)
            report = json.loads((output / "minutes-qa.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "needs_review")
            self.assertEqual({f["code"] for f in report["findings"]}, {"unresolved_speaker", "malformed_bullet", "questionable_motion"})
            self.assertTrue((output / "minutes-qa.md").exists())
            sections = engine.load_jsonl(output / "meeting_sections.jsonl")
            self.assertEqual({r["meeting_section"] for r in sections}, {PRE_MEETING, RECAP, BUSINESS, ADJOURNMENT})

    def test_keep_recap_uses_separate_historical_reduction(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(Path(directory), "[SPEAKER_00] At the last meeting, OLD plan was approved. Moving on to new business. NEW plan discussed.", ["--keep-recap"])
            self.assertEqual(status, 0)
            recap_prompt = next(c["prompt"] for c in calls if "Summarize only the supplied historical recap" in c["prompt"])
            self.assertIn("OLD plan", recap_prompt)
            self.assertNotIn("NEW plan", recap_prompt)
            minutes = (output / "minutes-draft.md").read_text(encoding="utf-8")
            self.assertIn("## Recap of Previous Meeting\n\n- Previously approved OLD Mac Yard plan.", minutes)

    def test_action_items_use_same_current_sections_as_minutes(self):
        text = "[SPEAKER_00] Good morning everyone. Let's review the previous meeting. OLD plan approved. Moving on to new business. NEW plan discussed. The meeting is adjourned. Sam will circulate the notes tomorrow."
        for options in ([], ["--keep-recap"]):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                status, calls, _ = self.run_pipeline(Path(directory), text, options)
                self.assertEqual(status, 0)
                action_prompt = next(c["prompt"] for c in calls if "write an action-items document" in c["prompt"])
                minutes_prompt = next(c["prompt"] for c in calls if "write formal draft minutes" in c["prompt"])
                action_input = action_prompt.split("Chunk summaries:\n", 1)[1]
                self.assertEqual(action_input, minutes_prompt.split("Chunk summaries:\n", 1)[1])
                self.assertNotIn("OLD plan", action_input)
                self.assertNotIn("Good morning", action_input)
                self.assertNotIn(f"Meeting section: {RECAP}", action_input)
                self.assertNotIn(f"Meeting section: {PRE_MEETING}", action_input)
                self.assertIn(f"Meeting section: {BUSINESS}", action_input)
                self.assertIn(f"Meeting section: {ADJOURNMENT}", action_input)
                self.assertIn("NEW plan", action_input)
                self.assertIn("Sam will circulate the notes tomorrow", action_input)
                summary_prompt = next(c["prompt"] for c in calls if "write a concise executive summary" in c["prompt"])
                self.assertIn("OLD plan", summary_prompt)
                self.assertIn(f"Meeting section: {RECAP}", summary_prompt)
                self.assertIn("clearly label any previous-meeting recap as historical context", summary_prompt)

    def test_action_items_with_only_recap_have_no_historical_reduce_input(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Good morning. At the last meeting, Alex was assigned to circulate the OLD report.")
            self.assertEqual(status, 0)
            action_prompt = next(c["prompt"] for c in calls if "write an action-items document" in c["prompt"])
            action_input = action_prompt.split("Chunk summaries:\n", 1)[1]
            self.assertEqual(action_input.strip(), "No explicit current meeting business or adjournment was identified.")
            self.assertNotIn("OLD report", action_input)

    def test_keep_recap_without_recap_does_not_add_an_extra_call(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(Path(directory), "[SPEAKER_00] Current discussion of the budget.", ["--keep-recap"])
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 4)
            self.assertNotIn("Recap of Previous Meeting", (output / "minutes-draft.md").read_text(encoding="utf-8"))

    def test_lesson_profile_preserves_original_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(Path(directory), "[SPEAKER_00] Mackyard and Transport Canadaâ€™s rules.", ["--keep-recap"], aliases=[], profile="lesson")
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 5)
            self.assertIn("Mackyard", calls[0]["prompt"])
            self.assertFalse((output / "meeting_sections.jsonl").exists())
            self.assertFalse((output / "minutes-qa.json").exists())
            self.assertIn("SPEAKER_00 Mack Yard", (output / "lesson-notes.md").read_text(encoding="utf-8"))

    def assert_model_calls(self, calls, map_model, reduce_model, map_ctx, reduce_ctx):
        map_calls = [c for c in calls if "Transcript chunk:\n" in c["prompt"]]
        reduce_calls = [c for c in calls if c not in map_calls]
        self.assertTrue(map_calls)
        self.assertTrue(reduce_calls)
        for call in map_calls:
            self.assertEqual((call["model"], call["num_ctx"]), (map_model, map_ctx))
        for call in reduce_calls:
            self.assertEqual((call["model"], call["num_ctx"]), (reduce_model, reduce_ctx))

    def test_meeting_environment_defaults_override_shared_defaults_including_recap(self):
        environment = {
            "MEETING_MAP_MODEL": "qwen3.6:27b", "MEETING_REDUCE_MODEL": "qwen3.8:27b",
            "MEETING_MAP_NUM_CTX": "16384", "MEETING_REDUCE_NUM_CTX": "32768",
            "OLLAMA_MAP_MODEL": "shared-map", "OLLAMA_REDUCE_MODEL": "shared-reduce",
            "OLLAMA_MAP_NUM_CTX": "4096", "OLLAMA_REDUCE_NUM_CTX": "8192",
        }
        with tempfile.TemporaryDirectory() as directory:
            status, calls, output = self.run_pipeline(
                Path(directory), "[SPEAKER_00] At the last meeting, OLD plan was approved. Moving on to new business. NEW plan discussed.",
                ["--keep-recap"], environment=environment, profile=None, default_profile="meeting",
            )
            self.assertEqual(status, 0)
            self.assert_model_calls(calls, "qwen3.6:27b", "qwen3.8:27b", 16384, 32768)
            self.assertEqual(len([c for c in calls if "Transcript chunk:\n" not in c["prompt"]]), 4)
            self.assertIn("## Recap of Previous Meeting", (output / "minutes-draft.md").read_text(encoding="utf-8"))

    def test_meeting_cli_overrides_environment_even_invalid_context_defaults(self):
        environment = {
            "MEETING_MAP_MODEL": "env-map", "MEETING_REDUCE_MODEL": "env-reduce",
            "MEETING_MAP_NUM_CTX": "invalid", "MEETING_REDUCE_NUM_CTX": "invalid",
            "OLLAMA_MAP_NUM_CTX": "also-invalid", "OLLAMA_REDUCE_NUM_CTX": "also-invalid",
        }
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(
                Path(directory), "[SPEAKER_00] Current business.",
                ["--map-model", "custom-fast:latest", "--reduce-model", "custom-final:latest",
                 "--map-num-ctx", "8192", "--reduce-num-ctx", "49152"], environment=environment,
            )
            self.assertEqual(status, 0)
            self.assert_model_calls(calls, "custom-fast:latest", "custom-final:latest", 8192, 49152)

    def test_meeting_partial_overrides_fall_back_per_setting(self):
        environment = {
            "MEETING_REDUCE_MODEL": "qwen3.8:27b", "MEETING_REDUCE_NUM_CTX": "32768",
            "MEETING_MAP_MODEL": "", "MEETING_MAP_NUM_CTX": "",
            "OLLAMA_MAP_MODEL": "qwen2.5:32b", "OLLAMA_MAP_NUM_CTX": "16384",
        }
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Current business.", environment=environment)
            self.assertEqual(status, 0)
            self.assert_model_calls(calls, "qwen2.5:32b", "qwen3.8:27b", 16384, 32768)

    def test_meeting_without_overrides_preserves_shared_environment_defaults(self):
        environment = {
            "OLLAMA_MAP_MODEL": "existing-fast", "OLLAMA_REDUCE_MODEL": "existing-final",
            "OLLAMA_MAP_NUM_CTX": "4096", "OLLAMA_REDUCE_NUM_CTX": "8192",
        }
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Current business.", environment=environment)
            self.assertEqual(status, 0)
            self.assert_model_calls(calls, "existing-fast", "existing-final", 4096, 8192)

    def test_unconfigured_profiles_keep_existing_builtin_defaults(self):
        for profile in ("meeting", "lesson"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Current discussion.", profile=profile)
                self.assertEqual(status, 0)
                self.assert_model_calls(calls, "qwen2.5:32b", "qwen2.5:32b", None, None)

    def test_lesson_defaults_ignore_all_meeting_model_settings(self):
        environment = {
            "MEETING_MAP_MODEL": "qwen3.6:27b", "MEETING_REDUCE_MODEL": "qwen3.8:27b",
            "MEETING_MAP_NUM_CTX": "invalid", "MEETING_REDUCE_NUM_CTX": "invalid",
            "OLLAMA_MAP_MODEL": "lesson-map", "OLLAMA_REDUCE_MODEL": "lesson-reduce",
            "OLLAMA_MAP_NUM_CTX": "4096", "OLLAMA_REDUCE_NUM_CTX": "8192",
        }
        # Test both the lesson wrapper's default and an explicit lesson profile
        # overriding a meeting wrapper default; the selected profile must win.
        for profile, default_profile in ((None, "lesson"), ("lesson", "meeting")):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Lesson content.", profile=profile, default_profile=default_profile, environment=environment)
                self.assertEqual(status, 0)
                self.assert_model_calls(calls, "lesson-map", "lesson-reduce", 4096, 8192)
        meeting_only = {key: value for key, value in environment.items() if key.startswith("MEETING_")}
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Lesson content.", profile="lesson", environment=meeting_only)
            self.assertEqual(status, 0)
            self.assert_model_calls(calls, "qwen2.5:32b", "qwen2.5:32b", None, None)

    def test_explicit_meeting_profile_uses_meeting_defaults_from_lesson_wrapper(self):
        environment = {
            "MEETING_MAP_MODEL": "meeting-map", "MEETING_REDUCE_MODEL": "meeting-reduce",
            "MEETING_MAP_NUM_CTX": "16384", "MEETING_REDUCE_NUM_CTX": "32768",
        }
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Current business.", profile="meeting", default_profile="lesson", environment=environment)
            self.assertEqual(status, 0)
            self.assert_model_calls(calls, "meeting-map", "meeting-reduce", 16384, 32768)

    def test_invalid_selected_meeting_context_reports_configuration_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as error:
                self.run_pipeline(Path(directory), "[SPEAKER_00] Current business.", environment={"MEETING_MAP_NUM_CTX": "invalid"})
            self.assertEqual(error.exception.code, 2)

    def test_bad_alias_configuration_stops_before_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            status, calls, _ = self.run_pipeline(Path(directory), "[SPEAKER_00] Current business.", aliases={"SPEAKER_00": ""})
            self.assertEqual(status, 2)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
