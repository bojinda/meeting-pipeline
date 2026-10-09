"""Synthetic adaptive grouping and mocked boundary planning; never live models."""
import contextlib
import copy
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
import meeting_chunk_experiment as command
import ollama_session_summary as engine
from meeting_postprocess import chunking
from meeting_postprocess.sections import BUSINESS, RECAP, ADJOURNMENT, PRE_MEETING
from meeting_postprocess.publication import publication_payload, strip_private_references
from meeting_postprocess.speaker_turns import turn_catalog, approve_turn
from meeting_stage_inputs import input_files

PROMPTS = ROOT / "prompts/meeting"
SYSTEM = (PROMPTS / "chunk_system.txt").read_text().strip()


def row(number, text, section=BUSINESS):
    return {"chunk_id": number, "source_chunk_id": number, "meeting_section": section,
            "section_evidence": ["Synthetic source section"], "file_name": f"source-{number}.md",
            "start_time": number * 10, "end_time": number * 10 + 9, "text": text}


def settings(**updates):
    value = {"mode": "adaptive", "max_words": 10000, "map_context": 32768,
             "map_output": 4096, "map_model": "synthetic", "tokenizer_hash": None,
             "planner": False, "planner_context": 98304}
    value.update(updates)
    return value


def prompt(source):
    return engine.build_chunk_prompt(PROMPTS, source)


def response(merge_after):
    return {"done": True, "done_reason": "stop", "response": json.dumps({"merge_after": merge_after}),
            "prompt_eval_count": 2000, "eval_count": 20}


class GroupingTests(unittest.TestCase):
    def plan(self, rows, config=None, blocked=(), call=None, preflight=False):
        return chunking.make_plan(rows, {"synthetic": "source"}, config or settings(), chunking.counter(), prompt, SYSTEM,
                                  blocked, call, planner_preflight=preflight)

    def apply(self, plan, rows, blocked=()):
        return chunking.apply_plan(plan, rows, {"synthetic": "source"}, chunking.counter(), prompt, SYSTEM,
                                   "synthetic", plan["settings"]["map_context"], blocked)

    def test_connected_report_can_exceed_old_word_target_without_filling_other_reports(self):
        rows = [row(1, "[Taylor] " + "report detail " * 1300), row(2, "[Taylor] " + "report clarification " * 1300),
                row(3, "[Morgan] A separate short report.")]
        plan = self.plan(rows, settings(mode="medium", max_words=6000, map_context=98304))
        self.assertEqual(plan["groups"], [["P1", "P2"], ["P3"]])
        self.assertGreater(plan["budgets"][0]["words"], 5000)
        self.assertLess(plan["budgets"][1]["words"], 20)

    def test_question_answer_and_later_correction_remain_together(self):
        rows = [row(1, "[Taylor] How many relief jobs were restored?"),
                row(2, "[Morgan] One of four jobs will return.\n[Taylor] That qualifies the earlier restructuring report."),
                row(3, "[Morgan] Actually, the return is conditional on staffing.\n[Taylor] Understood.")]
        plan = self.plan(rows)
        self.assertEqual(plan["groups"], [["P1", "P2", "P3"]])
        group = self.apply(plan, rows)[0]
        self.assertEqual(group["text"], "\n".join(r["text"] for r in rows))
        self.assertEqual([p["portion_id"] for p in group["source_portions"]], ["P1", "P2", "P3"])
        self.assertEqual([{k: v for k, v in p.items() if k != "portion_id"} for p in group["source_portions"]], rows)

    def test_explicit_new_topic_and_new_officer_stop_deterministic_grouping(self):
        rows = [row(1, "[Taylor] The staffing report is complete."), row(2, "[Taylor] Next topic: convention arrangements."),
                row(3, "[Morgan] A short safety report.")]
        self.assertEqual(self.plan(rows)["groups"], [["P1"], ["P2"], ["P3"]])

    def test_sections_and_redaction_boundaries_remain_separate(self):
        rows = [row(1, "[Taylor] Earlier report.", RECAP), row(2, "[Taylor] Current report."),
                row(3, "[Taylor] Visible after redaction."), row(4, "[Chair] Meeting adjourned.", ADJOURNMENT)]
        plan = self.plan(rows, blocked={"3"})
        self.assertEqual(plan["groups"], [["P1"], ["P2"], ["P3"], ["P4"]])
        self.assertEqual([g["meeting_section"] for g in self.apply(plan, rows, {"3"})], [RECAP, BUSINESS, BUSINESS, ADJOURNMENT])

    def test_pre_meeting_never_enters_experimental_maps_or_planner(self):
        rows = [row(1, "[Taylor] PRIVATE_CHATTER_SENTINEL", PRE_MEETING), row(2, "[Taylor] The report.")]
        def call(text, *args):
            self.assertNotIn("PRIVATE_CHATTER_SENTINEL", text)
            return response([])
        plan = self.plan(rows, settings(planner=True), call=call)
        self.assertEqual(plan["coverage"], {"eligible_portions": 1, "covered_portions": 1, "complete": True})
        self.assertNotIn("PRIVATE_CHATTER_SENTINEL", self.apply(plan, rows)[0]["text"])

    def test_repeated_identical_utterances_keep_distinct_source_ids_and_timing(self):
        rows = [row(1, "[SPEAKER_03] I'll send the report."), row(2, "[SPEAKER_03] I'll send the report.")]
        group = self.apply(self.plan(rows), rows)[0]
        self.assertEqual(group["text"].count("I'll send the report."), 2)
        self.assertEqual(group["source_chunk_id"], [1, 2])
        self.assertEqual(group["start_time"], 10)
        self.assertEqual(group["end_time"], 29)

    def test_token_budget_overrides_word_ceiling_with_full_coverage(self):
        rows = [row(1, "[Taylor] " + "detail " * 450), row(2, "[Taylor] " + "detail " * 450)]
        single = chunking.measure(rows[:1], ["P1"], settings(), chunking.counter(), prompt, SYSTEM)
        config = settings(map_context=single["required_context"] + 10)
        plan = self.plan(rows, config)
        self.assertEqual(plan["groups"], [["P1"], ["P2"]])
        self.assertTrue(plan["coverage"]["complete"])
        self.assertTrue(all(b["required_context"] <= config["map_context"] for b in plan["budgets"]))
        self.assertEqual(plan["budgets"][0]["reserved_output"], 4096)
        self.assertEqual(plan["budgets"][0]["reserved_framing"], 1024)

    def test_indivisible_over_budget_source_fails_without_omitting_words(self):
        with self.assertRaisesRegex(chunking.ChunkingFailure, "indivisible"):
            self.plan([row(1, "[Taylor] " + "long report " * 11000)])

    def test_word_ceiling_and_baseline_groups_are_respected(self):
        rows = [row(1, "[Taylor] " + "detail " * 30), row(2, "[Taylor] " + "detail " * 30)]
        self.assertEqual(self.plan(rows, settings(max_words=50))["groups"], [["P1"], ["P2"]])
        self.assertEqual(self.plan(rows, settings(mode="baseline"))["groups"], [["P1"], ["P2"]])

    def test_optional_planner_makes_one_boundary_only_call(self):
        rows = [row(1, "[Taylor] Staffing report."), row(2, "[Morgan] A related follow-up."), row(3, "[Casey] Convention report.")]
        calls = []
        def call(*args):
            calls.append(args)
            self.assertEqual(set(args[2]["properties"]), {"merge_after"})
            self.assertIn("untrusted data", args[1])
            return response(["P1"])
        plan = self.plan(rows, settings(planner=True), call=call)
        self.assertEqual(len(calls), 1)
        self.assertEqual(plan["planner"]["status"], "validated")
        self.assertEqual(plan["groups"], [["P1", "P2"], ["P3"]])
        self.assertEqual(plan["planner"]["actual_output_tokens"], 20)

    def test_bad_boundaries_fall_back_without_discarding_any_portion(self):
        rows = [row(1, "[Taylor] First report."), row(2, "[Morgan] Second report."), row(3, "[Casey] Third report.")]
        for merges in (["P3"], ["P2", "P1"], ["P1", "P1"], ["P99"], [1], ["P01"], None):
            with self.subTest(merges=merges):
                plan = self.plan(rows, settings(planner=True), call=lambda *args: response(merges))
                self.assertEqual(plan["planner"]["status"], "deterministic_fallback")
                self.assertEqual(plan["groups"], [["P1"], ["P2"], ["P3"]])
                self.assertTrue(plan["coverage"]["complete"])
        # Converted/saved endpoint plans still enforce the original coverage rules.
        for ends in (["P1"], ["P3", "P1"], ["P1", "P1", "P3"], ["P99"], [], [1, "P3"]):
            with self.subTest(ends=ends), self.assertRaises(chunking.ChunkingFailure):
                chunking.validate_ends(ends, rows, settings(), chunking.counter(), prompt, SYSTEM, set())

    def test_planner_keeps_lengthy_report_questions_and_corrections_together(self):
        rows = [row(1, "[Taylor] Equipment report. " + "inspection detail " * 900),
                row(2, "[Morgan] Does that include the replacement equipment?"),
                row(3, "[Taylor] Yes, with a correction to the earlier count. " + "report clarification " * 500),
                row(4, "[Chair] Next topic: convention arrangements.")]
        calls = []
        def call(*args):
            calls.append(args)
            self.assertIn('return {"merge_after":["P2"]}', args[1])
            self.assertIn("NOT inherently topic boundaries", args[1])
            return response(["P1", "P2"])
        plan = self.plan(rows, settings(planner=True, map_context=98304), call=call)
        self.assertEqual(len(calls), 1)
        self.assertEqual(plan["groups"], [["P1", "P2", "P3"], ["P4"]])
        comparison = plan["planner"]["boundary_comparison"]
        self.assertEqual(comparison["removed_end_ids"], ["P1"])
        self.assertEqual(comparison["segmentation_change"], "coarser_than_deterministic")
        self.assertFalse(comparison["grouping_improvement_demonstrated"])

    def test_planner_preserves_short_discussion_at_genuine_topic_transition(self):
        rows = [row(1, "[Taylor] The equipment check is complete."),
                row(2, "[Taylor] Next topic: convention arrangements. Can two delegates attend?"),
                row(3, "[Morgan] Yes, two delegates can attend.")]
        plan = self.plan(rows, settings(planner=True), call=lambda *args: response(["P2"]))
        self.assertEqual(plan["groups"], [["P1"], ["P2", "P3"]])
        self.assertEqual(plan["planner"]["boundary_comparison"]["segmentation_change"], "same_as_deterministic")

    def test_merge_planner_keeps_new_report_after_recovery_discussion(self):
        rows = [row(1, "[Taylor] The member is recovering on modified duties. I helped with the forms. Any questions? Have we discussed the convention yet?"),
                row(2, "[Morgan] No, not yet.\n[Taylor] The convention was productive. Two delegates attended.")]
        plan = self.plan(rows, settings(planner=True), call=lambda *args: response([]))
        # An answer to a new-subject question does not extend the recovery report.
        self.assertEqual(plan["groups"], [["P1"], ["P2"]])
        self.assertEqual(plan["planner"]["merge_after"], [])

    def test_merge_planner_joins_wildfire_safety_clarification_across_speakers(self):
        rows = [row(1, "[Morgan] The wildfire moved toward the train. Crews were not warned and could not see the route. That was a safety concern."),
                row(2, "[Casey] But were they waiting for an opposing train?\n[Morgan] Possibly, but the smoke blocked visibility.\n[Casey] The recording showed a train passing them.")]
        calls = []
        def call(text, *args):
            calls.append(text)
            self.assertIn("wildfire moved toward the train", text)
            self.assertIn("waiting for an opposing train", text)
            return response(["P1"])
        plan = self.plan(rows, settings(planner=True), call=call)
        self.assertEqual(len(calls), 1)
        self.assertEqual(plan["ends"], ["P2"])
        self.assertEqual(plan["groups"], [["P1", "P2"]])
        self.assertEqual(self.apply(plan, rows)[0]["text"], "\n".join(r["text"] for r in rows))

    def test_merge_planner_sees_question_referring_to_earlier_report_passage(self):
        earlier = "Sixth-shift work still receives time and a half, including the stated exception."
        unrelated_tail = "The proposed job changes are still under review."
        rows = [row(1, "[Riley] " + earlier + "\n[Riley] Management meetings have not resolved the scheduling concerns.\n[Riley] " + unrelated_tail),
                row(2, "[Taylor] Just so I heard you correctly, are sixth shifts no longer paid at time and a half?\n[Riley] No, sixth shifts are still paid. The other cross-agreement cases are disputed.")]
        def call(text, system, *args):
            supplied = json.loads(text)["portions"]
            self.assertIn(earlier, supplied[0][3])
            self.assertTrue(supplied[0][3].endswith(unrelated_tail))
            self.assertIn("earlier part of the preceding officer report", system)
            self.assertIn("Same speaker does NOT imply same subject", system)
            return response(["P1"])
        plan = self.plan(rows, settings(planner=True), call=call)
        self.assertEqual(plan["groups"], [["P1", "P2"]])
        self.assertTrue(plan["coverage"]["complete"])

    def test_merge_chain_must_fit_as_a_whole_not_just_pairwise(self):
        rows = [row(i, f"[{speaker}] " + "detail " * 20) for i, speaker in enumerate(("Taylor", "Morgan", "Riley"), 1)]
        config = settings(planner=True, max_words=50)
        self.assertEqual(chunking.merge_endpoints(["P1"], rows, config, chunking.counter(), prompt, SYSTEM, set()), ["P2", "P3"])
        plan = self.plan(rows, config, call=lambda *args: response(["P1", "P2"]))
        self.assertEqual(plan["planner"]["status"], "deterministic_fallback")
        self.assertEqual(plan["planner"]["reason"], "group_over_budget")
        self.assertEqual(plan["groups"], [["P1"], ["P2"], ["P3"]])
        self.assertTrue(plan["coverage"]["complete"])

    def test_merge_chain_keeps_endpoint_plan_hash_and_legacy_compatibility(self):
        rows = [row(1, "[Taylor] Equipment report."), row(2, "[Taylor] More detail."),
                row(3, "[Taylor] A correction."), row(4, "[Morgan] Next topic: staffing.")]
        plan = self.plan(rows, settings(planner=True), call=lambda *args: response(["P1", "P2"]))
        self.assertEqual(plan["ends"], ["P3", "P4"])
        self.assertEqual(plan["source_hash"], chunking.digest(rows))
        self.assertEqual(plan["plan_hash"], chunking.digest({k: v for k, v in plan.items() if k != "plan_hash"}))
        self.assertEqual(plan["planner"]["response_contract"], "adjacent_merges_v1")
        legacy = copy.deepcopy(plan)
        legacy["planner"].pop("response_contract")
        legacy["planner"].pop("merge_after")
        legacy["plan_hash"] = chunking.digest({k: v for k, v in legacy.items() if k != "plan_hash"})
        self.assertEqual(self.apply(legacy, rows), self.apply(plan, rows))

    def test_singleton_planner_is_valid_without_demonstrating_grouping_improvement(self):
        rows = [row(1, "[Taylor] Equipment report begins."), row(2, "[Taylor] More equipment detail."),
                row(3, "[Taylor] A correction to the equipment count.")]
        plan = self.plan(rows, settings(planner=True), call=lambda *args: response([]))
        self.assertEqual(plan["planner"]["status"], "validated")
        self.assertEqual(plan["groups"], [["P1"], ["P2"], ["P3"]])
        self.assertTrue(plan["coverage"]["complete"])
        comparison = plan["planner"]["boundary_comparison"]
        self.assertTrue(comparison["structurally_valid"])
        self.assertEqual(comparison["deterministic_group_count"], 1)
        self.assertEqual(comparison["planner_group_count"], 3)
        self.assertEqual(comparison["singleton_group_count"], 3)
        self.assertEqual(comparison["added_end_ids"], ["P1", "P2"])
        self.assertEqual(comparison["segmentation_change"], "all_singletons_no_consolidation")
        self.assertFalse(comparison["grouping_improvement_demonstrated"])
        self.assertEqual(len(self.apply(plan, rows)), 3)

    def test_fewer_groups_do_not_certify_semantic_quality(self):
        rows = [row(1, "[Taylor] Equipment report."), row(2, "[Morgan] A separate convention report.")]
        plan = self.plan(rows, settings(planner=True), call=lambda *args: response(["P1"]))
        comparison = plan["planner"]["boundary_comparison"]
        self.assertTrue(comparison["structurally_valid"])
        self.assertEqual(comparison["deterministic_group_count"], 2)
        self.assertEqual(comparison["planner_group_count"], 1)
        self.assertFalse(comparison["grouping_improvement_demonstrated"])
        self.assertIn("source_review_required", comparison["quality_assessment"])

    def test_section_gap_and_over_budget_planner_groups_are_rejected(self):
        for rows, config, blocked in (
            ([row(1, "[Taylor] Historical.", RECAP), row(2, "[Morgan] Current.")], settings(planner=True), set()),
            ([row(1, "[Taylor] Before."), row(2, "[Morgan] After.")], settings(planner=True), {"2"}),
            ([row(1, "[Taylor] " + "detail " * 30), row(2, "[Morgan] " + "detail " * 30)], settings(planner=True, max_words=50), set()),
        ):
            plan = self.plan(rows, config, blocked, lambda *args: response(["P1"]))
            self.assertEqual(plan["planner"]["status"], "deterministic_fallback")
            self.assertEqual(plan["groups"], [["P1"], ["P2"]])

    def test_full_planner_budget_and_preflight_make_zero_calls_when_not_fitting(self):
        rows = [row(i, "[Taylor] " + "detail " * 200) for i in range(1, 12)]
        call = lambda *args: self.fail("No planner call permitted")
        plan = self.plan(rows, settings(planner=True, planner_context=2048), call=call)
        self.assertEqual(plan["planner"]["reason"], "planner_context_budget_exceeded")
        self.assertEqual(plan["planner"]["calls"], 0)
        self.assertTrue(plan["coverage"]["complete"])
        small = self.plan(rows[:1], settings(planner=True), call=call, preflight=True)
        self.assertEqual(small["planner"]["status"], "preflight_ready")
        self.assertEqual(small["planner"]["calls"], 0)

    def test_planner_failure_and_generation_limit_never_retry_or_accept_partial_json(self):
        rows = [row(1, "[Taylor] A report.")]
        for failure in (TimeoutError(), response([]) | {"response": "{"},
                        response([]) | {"done_reason": "length", "eval_count": 2048},
                        response([]) | {"done": False}, response([]) | {"response": '{"merge_after":[],"merge_after":[]}'},
                        response([]) | {"response": '{"merge_after":[],"summary":"unsupported"}'},
                        response([]) | {"response": '{"ends":["P1"]}'}):
            calls = []
            def call(*args):
                calls.append(args)
                if isinstance(failure, Exception):
                    raise failure
                return failure
            plan = self.plan(rows, settings(planner=True), call=call)
            self.assertEqual(len(calls), 1)
            self.assertEqual(plan["planner"]["status"], "deterministic_fallback")
            self.assertEqual(plan["groups"], [["P1"]])

    def test_changed_source_plan_config_and_prompt_fail_closed(self):
        rows = [row(1, "[Taylor] The report.")]
        plan = self.plan(rows)
        altered = copy.deepcopy(plan)
        altered["ends"] = ["P99"]
        for candidate, source in ((altered, rows), (plan, [row(1, "[Taylor] Changed report.")])):
            with self.assertRaises(chunking.ChunkingFailure):
                self.apply(candidate, source)
        with self.assertRaisesRegex(chunking.ChunkingFailure, "configuration"):
            chunking.apply_plan(plan, rows, {"synthetic": "source"}, chunking.counter(), prompt, SYSTEM, "other", 32768)
        with self.assertRaisesRegex(chunking.ChunkingFailure, "prompt"):
            chunking.apply_plan(plan, rows, {"synthetic": "source"}, chunking.counter(), lambda r: prompt(r) + "changed", SYSTEM, "synthetic", 32768)

    def test_experimental_map_generation_limit_is_not_accepted_as_complete_summary(self):
        data = {"response": "incomplete map", "done_reason": "length", "eval_count": 4096}
        with patch.object(engine, "managed_generate", return_value=data):
            with self.assertRaisesRegex(ValueError, "experimental_map_generation_limit"):
                engine.call_ollama("http://localhost:11434", "synthetic", "source", num_predict=4096)
            # Omitted experiment reserve preserves the established default API.
            self.assertEqual(engine.call_ollama("http://localhost:11434", "synthetic", "source"), "incomplete map")

    def test_saved_tokenizer_truncation_is_disabled(self):
        class Tokenizer:
            def no_truncation(self): self.truncation_disabled = True
            def no_padding(self): self.padding_disabled = True
        fake = type("Counter", (), {"tokenizer": Tokenizer()})()
        with patch.object(chunking, "TokenCounter", return_value=fake):
            self.assertIs(chunking.counter("local-only"), fake)
            self.assertTrue(fake.tokenizer.truncation_disabled)
            self.assertTrue(fake.tokenizer.padding_disabled)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "synthetic-session"
        (self.source / "chunks_out").mkdir(parents=True)
        dialogue = [
            "[Morgan] PRIVATE_CHATTER_SENTINEL.\n[Chair] Let's get started.\n[Chair] I'll start with last month's recap.",
            "[Chair] For the previous month, we reviewed staffing.\n[Chair] And then they were currently hiring at the yard.",
            "[Chair] We'll move on to this month.\n[SPEAKER_03] We can share the committee minutes with the health and safety committees.\n[SPEAKER_03] I'm going to do that after this meeting.",
            "[SPEAKER_03] Management directed the yardmasters to conduct tests of crews.\n[SPEAKER_03] Four relief jobs were removed.",
            "[SPEAKER_03] one will return, conditional on staffing.\n[SPEAKER_03] I'll try to file the grievance after reviewing the report.",
            "[Chair] Next topic: convention arrangements.\n[Morgan] I move to send two delegates to the convention.\n[Chair] Motion by Morgan, seconded by Riley. Carried.",
            "[Chair] Redact the following. SECRET_PLANNER_SENTINEL. End redaction.\n[Chair] Motion to adjourn.\n[Chair] Motion by Taylor, seconded by Casey. Carried.",
        ]
        self.index = self.source / "chunks_out/transcript_chunks.jsonl"
        self.index.write_text("".join(json.dumps({"chunk_id": i, "text": text, "start_time": i * 10, "end_time": i * 10 + 9}) + "\n" for i, text in enumerate(dialogue, 1)))
        self.environment = {"MEETING_MAP_MODEL": "synthetic", "MEETING_REDUCE_MODEL": "synthetic", "MEETING_MAP_NUM_CTX": "32768", "MEETING_REDUCE_NUM_CTX": "32768", "MEETING_SUMMARIES_ROOT": str(self.root / "production")}

    def tearDown(self): self.temp.cleanup()

    def plan(self):
        with patch.dict(os.environ, self.environment, clear=True), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(command.main([str(self.source), "--mode", "adaptive", "--output-dir", str(self.root / "plan")]), 0)
        return self.root / "plan/chunk-plan.json"

    def run_engine(self, plan=None, out=None):
        calls = []
        def fake(**kwargs):
            calls.append(kwargs)
            self.assertNotIn("SECRET_PLANNER_SENTINEL", kwargs["prompt"])
            if kwargs.get("usage_callback"):
                kwargs["usage_callback"]({"prompt_eval_count": 100, "eval_count": 40, "runtime_seconds": .01})
            if "Transcript chunk:\n" in kwargs["prompt"]:
                return "## Topics\n- " + kwargs["prompt"].split("Transcript chunk:\n", 1)[1].split("\nWrite concise markdown", 1)[0]
            return "# Generated document\nNo clear action items identified."
        args = ["summary", str(self.source), "--profile", "meeting"]
        if plan:
            args += ["--chunk-plan", str(plan), "--chunk-comparison-dir", str(out)]
        with patch.dict(os.environ, self.environment, clear=True), patch.object(sys, "argv", args), patch.object(engine, "call_ollama", side_effect=fake), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = engine.main()
        return status, calls

    def test_isolated_map_reduce_preserves_source_details_and_existing_outputs(self):
        before = self.index.read_bytes()
        production = self.root / "production"
        production.mkdir()
        sentinel = production / "minutes-draft.md"
        sentinel.write_text("ACCEPTED_OUTPUT_SENTINEL")
        plan_path = self.plan()
        status, calls = self.run_engine(plan_path, self.root / "comparison")
        self.assertEqual(status, 0)
        maps = [c for c in calls if "Transcript chunk:\n" in c["prompt"]]
        self.assertEqual(len(calls), len(maps) + 3)
        self.assertTrue(all(c["num_predict"] == 4096 for c in maps))
        self.assertTrue(all("PRIVATE_CHATTER_SENTINEL" not in c["prompt"] for c in calls))
        self.assertTrue(any("Four relief jobs" in c["prompt"] and "one will return" in c["prompt"] and "conditional on staffing" in c["prompt"] for c in maps))
        self.assertTrue(any("yardmasters to conduct tests of crews" in c["prompt"] for c in maps))
        self.assertTrue(any("send two delegates" in c["prompt"] and "seconded by Riley" in c["prompt"] and "Carried" in c["prompt"] for c in maps))
        for instruction in ("write an action-items document", "write formal draft minutes"):
            text = next(c["prompt"] for c in calls if instruction in c["prompt"])
            self.assertNotIn("currently hiring", text)
            self.assertIn("health and safety committees", text)
            self.assertIn("I'm going to do that after this meeting", text)
            self.assertIn("I'll try to file the grievance after reviewing the report", text)
        metadata = json.loads((self.root / "comparison/chunk-comparison.json").read_text())
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(len(metadata["usage"]), len(calls))
        self.assertTrue(metadata["coverage"]["complete"])
        self.assertEqual(self.index.read_bytes(), before)
        self.assertEqual(sentinel.read_text(), "ACCEPTED_OUTPUT_SENTINEL")

    def test_stale_index_or_modified_plan_makes_zero_model_calls(self):
        plan = self.plan()
        self.index.write_text(self.index.read_text().replace("Four relief jobs", "Five relief jobs"))
        status, calls = self.run_engine(plan, self.root / "comparison")
        self.assertEqual(status, 2)
        self.assertEqual(calls, [])
        self.assertFalse((self.root / "comparison/minutes-draft.md").exists())

    def test_preflight_never_writes_and_default_run_keeps_original_maps(self):
        before = self.index.read_bytes()
        with patch.dict(os.environ, self.environment, clear=True), patch.object(command, "managed_generate", side_effect=AssertionError("no inference")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(command.main([str(self.source), "--planner", "--preflight"]), 0)
        self.assertEqual(list(self.source.glob("chunk-plan.json")), [])
        status, calls = self.run_engine()
        self.assertEqual(status, 0)
        self.assertTrue(any("PRIVATE_CHATTER_SENTINEL" in c["prompt"] for c in calls if "Transcript chunk:\n" in c["prompt"]))
        self.assertTrue(all("num_predict" not in c for c in calls))
        self.assertEqual(self.index.read_bytes(), before)

    def test_approved_turn_correction_and_aliases_are_preserved(self):
        (self.source / "speaker_aliases.json").write_text(json.dumps({"SPEAKER_03": "Taylor"}))
        catalog = turn_catalog(self.source)
        turn = next(r for r in catalog["turns"] if "Four relief jobs" in r["text"])
        approve_turn(self.source, catalog, turn["turn_id"], "Casey")
        plan = self.plan()
        status, calls = self.run_engine(plan, self.root / "comparison")
        self.assertEqual(status, 0)
        text = "\n".join(c["prompt"] for c in calls if "Transcript chunk:\n" in c["prompt"])
        self.assertIn("[Casey] Four relief jobs", text)
        self.assertIn("[Taylor] We can share", text)

    def test_plan_and_metrics_are_private_and_never_exported(self):
        plan = self.plan()
        status, _ = self.run_engine(plan, self.root / "comparison")
        self.assertEqual(status, 0)
        documents = publication_payload(self.root / "comparison")["documents"]
        self.assertEqual({d["name"] for d in documents}, {"summary.md", "action-items.md", "minutes-draft.md"})
        self.assertEqual(strip_private_references("[private](chunk-plan.json)\nchunk-comparison.json\nSafe."), "\nSafe.")
        if os.name != "nt":
            self.assertEqual(plan.stat().st_mode & 0o777, 0o600)

    def test_existing_destination_and_lesson_never_accept_experiment(self):
        plan = self.plan()
        destination = self.root / "comparison"
        destination.mkdir()
        sentinel = destination / "minutes-draft.md"
        sentinel.write_text("ACCEPTED_OUTPUT_SENTINEL")
        for profile in ("meeting", "lesson"):
            args = ["summary", str(self.source), "--profile", profile, "--chunk-plan", str(plan), "--chunk-comparison-dir", str(destination)]
            with patch.dict(os.environ, self.environment, clear=True), patch.object(sys, "argv", args), patch.object(engine, "call_ollama", side_effect=AssertionError("no models")), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    engine.main()
        self.assertEqual(sentinel.read_text(), "ACCEPTED_OUTPUT_SENTINEL")

    def test_plan_command_keeps_private_redaction_out_of_mock_planner(self):
        calls = []
        def fake(url, payload, timeout):
            calls.append(payload)
            self.assertNotIn("SECRET_PLANNER_SENTINEL", payload["prompt"])
            self.assertNotIn("PRIVATE_CHATTER_SENTINEL", payload["prompt"])
            self.assertEqual(payload["options"]["num_ctx"], 98304)
            self.assertFalse(payload["think"])
            source = json.loads(payload["prompt"])
            self.assertTrue(source["portions"])
            return response([])
        with patch.dict(os.environ, self.environment, clear=True), patch.object(command, "managed_generate", side_effect=fake), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(command.main([str(self.source), "--planner", "--output-dir", str(self.root / "plan")]), 0)
        self.assertEqual(len(calls), 1)
        plan = json.loads((self.root / "plan/chunk-plan.json").read_text())
        self.assertEqual(plan["planner"]["status"], "validated")

    def test_source_and_plan_changes_have_distinct_comparison_identity(self):
        from aihub_gpu_runner.host_stage import command_identity
        plan = self.plan()
        cmd = [sys.executable, str(ROOT / "bin/ollama_meeting_summary.py"), str(self.source), "--chunk-plan", str(plan), "--chunk-comparison-dir", str(self.root / "comparison")]
        env = dict(os.environ, AIHUB_GPU_STAGE_INPUT_FILES=json.dumps(input_files(cmd)))
        first = command_identity(cmd, env)
        data = json.loads(plan.read_text())
        data["settings"]["max_words"] = 9000
        data["plan_hash"] = chunking.digest({k: v for k, v in data.items() if k != "plan_hash"})
        plan.write_text(json.dumps(data))
        self.assertNotEqual(first, command_identity(cmd, env))
        second = command_identity(cmd, env)
        self.index.write_text(self.index.read_text().replace("Four relief jobs", "Five relief jobs"))
        self.assertNotEqual(second, command_identity(cmd, env))


if __name__ == "__main__": unittest.main()
