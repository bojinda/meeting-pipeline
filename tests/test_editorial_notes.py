"""Synthetic, offline editorial reductions and publication holds."""
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
import ollama_session_summary as engine
from meeting_postprocess import editorial as ed
from meeting_postprocess.commitments import commitment_evidence
from meeting_postprocess.publication import export_documents, export_archive, publication_payload, strip_private_references
from meeting_postprocess.sections import BUSINESS, RECAP, PRE_MEETING, ADJOURNMENT


def portion(number, text, section=BUSINESS):
    return {"chunk_id": str(number), "source_chunk_id": str(number), "text": text,
            "meeting_section": section, "start_time": number * 10, "end_time": number * 10 + 9,
            "file_name": "synthetic-report.txt"}


def fixture():
    chunks = [portion(1, "[Morgan] I'll send PRIVATE_CHATTER about a vacation.", PRE_MEETING),
              portion(2, "[Taylor] At the last meeting, an earlier staffing plan was discussed.", RECAP),
              portion(3, "\n".join([
                  "[Morgan] I'll share committee minutes with health and safety committees after this meeting.",
                  "[Taylor] I think we should request another meeting.",
                  "[Chair] I'll bring Morgan's opportunity to speak first up at the next meeting.",
                  "[Riley] If the claim is denied, I'll try to file the report.",
                  "[Casey] I already sent the report.",
                  "[Morgan] I am grieving the attendance letters.",
                  "[Taylor] At management's meeting, Casey promised to review the report tomorrow.",
                  "[Morgan] Management directs efficiency tests, yardmasters administer them, operating crews are tested.",
                  "[Taylor] Approximately 70% of jobs were abolished at Mac Yard; relief assignments changed afterward.",
                  "[Taylor] Four jobs were cut; one will return as a relief job. Implementation remains unconfirmed.",
                  "[Riley] MRS journals still disagree with the consist; dangerous commodities may be out of sequence.",
              ])),
              portion(4, "[Chair] Motion by Taylor, seconded by Riley.\n[Morgan] We still need to discuss distribution.", ADJOURNMENT)]
    tasks = [
        ("undertaking", "Share committee minutes with health and safety committees after this meeting.", ["Morgan"], "3:L1", True),
        ("proposal", "Request another meeting.", [], "3:L2", False),
        ("undertaking", "Bring Morgan's opportunity to speak first up at the next meeting.", ["Chair"], "3:L3", True),
        ("undertaking", "If the claim is denied, try to file the report.", ["Riley"], "3:L4", False),
        ("completed", "Sent the report.", ["Casey"], "3:L5", False),
        ("ongoing", "Grieving the attendance letters.", ["Morgan"], "3:L6", False),
        ("external", "Review the report tomorrow.", ["Casey"], "3:L7", False),
        ("business", "Motion by Taylor, seconded by Riley.", [], "4:L1", False)]
    raw = {"items": [{"id": f"A{i}", "category": category, "task": task, "owners": owners,
                      "source_ids": [ref], "member_facing": public, "concerns": []}
                     for i, (category, task, owners, ref, public) in enumerate(tasks, 1)]}
    def block(text, *refs):
        return {"text": text, "source_ids": list(refs)}
    notes = {"highlights": [block("Committee minutes were to be circulated; MRS safety concerns remained open.", "3:L1", "3:L11")],
             "previous_context": [block("An earlier staffing plan was reviewed at the previous meeting.", "2:L1")],
             "issues": [
                 {"heading": "Efficiency testing", "paragraphs": [block("Management directs tests, yardmasters administer them, and operating crews are tested.", "3:L8")]},
                 {"heading": "Mac Yard staffing", "paragraphs": [block("An initial approximate 70% abolition was followed by changed relief assignments. Of four jobs cut, one was planned to return; implementation remained unconfirmed.", "3:L9", "3:L10")]},
                 {"heading": "MRS safety", "paragraphs": [block("MRS journals still differed from the consist, including dangerous-commodity sequencing.", "3:L11")]},
             ], "motions": [block("Motion to adjourn moved by Taylor, seconded by Riley.", "4:L1")],
             "unresolved": [block("Distribution remained under discussion after the motion.", "4:L2")], "concerns": ["policy_sensitive"]}
    return chunks, raw, notes


class EditorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer, models, pre_tokenizers
        cls.tokenizer_directory = tempfile.TemporaryDirectory()
        cls.tokenizer_path = Path(cls.tokenizer_directory.name) / "tokenizer.json"
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(cls.tokenizer_path))

    @classmethod
    def tearDownClass(cls):
        cls.tokenizer_directory.cleanup()

    def budget(self, context=196608):
        return ed.RequestBudget(self.tokenizer_path, context, "Synthetic editorial system")

    def setUp(self):
        self.chunks, self.raw, self.notes = fixture()
        self.records = ed.source_records(self.chunks)
        self.source = "\n".join(c["text"] for c in self.chunks if c["meeting_section"] in {BUSINESS, ADJOURNMENT})
        self.commitments = commitment_evidence(self.chunks, include_context=True)

    def register(self, raw=None):
        return ed.validate_register(raw or self.raw, self.records, self.commitments, self.source)

    def test_layout_order_shared_table_and_source_provenance(self):
        reg = self.register()
        notes = ed.validate_notes(self.notes, self.records, self.source)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            review = ed.finish(directory, reg, notes, "# Detailed Minutes\n## Action Items\n- Taylor: Invented task\n## Other\nRetained internal detail.", self.source, records=self.records)
            document = (directory / "meeting-notes-draft.md").read_text()
            headings = ["## Meeting Highlights", "## Previous-meeting context", "## Efficiency testing", "## Mac Yard staffing", "## MRS safety", "## Recorded motions and decisions", "## Recorded Undertakings", "## Unresolved Matters"]
            self.assertEqual([document.index(h) for h in headings], sorted(document.index(h) for h in headings))
            table = ed.undertaking_table(reg)
            for name in ("meeting-notes-draft.md", "action-items.md", "minutes-draft.md"):
                self.assertIn(table, (directory / name).read_text())
                self.assertNotIn("Invented task", (directory / name).read_text())
                self.assertNotIn("SPEAKER_", (directory / name).read_text())
            self.assertEqual(review["register_hash"], ed.digest(reg))
            self.assertEqual(reg["items"][0]["evidence"][0]["text"], self.chunks[2]["text"].splitlines()[0])
            self.assertEqual(reg["items"][0]["evidence"][0]["start_time"], 30)
            self.assertEqual(len(reg["categories"]), 6)
            provenance = json.loads((directory / ed.NOTES_EVIDENCE).read_text())
            self.assertEqual(provenance["sources"]["3:L10"]["text"], self.records["3:L10"]["text"])
            self.assertNotIn("PRIVATE_CHATTER", document)
            self.assertIn("Retained internal detail", (directory / "minutes-draft.md").read_text())
            if os.name != "nt":
                self.assertEqual((directory / ed.REGISTER).stat().st_mode & 0o777, 0o600)

    def test_categories_never_promote_proposals_outside_or_completed_work(self):
        reg = self.register()
        table = ed.undertaking_table(reg)
        self.assertNotIn("Request another meeting", table)
        self.assertNotIn("Review the report tomorrow", table)
        self.assertNotIn("Sent the report", table)
        self.assertNotIn("Grieving", table)
        wrong = copy.deepcopy(self.raw)
        wrong["items"][1].update(category="undertaking", owners=["Taylor"], member_facing=True)
        with self.assertRaisesRegex(ed.EditorialFailure, "unsupported_undertaking"):
            self.register(wrong)

    def test_source_qualifications_and_role_chronology_survive_rendering(self):
        reg = self.register()
        self.assertIn("If the claim is denied", reg["items"][3]["task"])
        wrong = copy.deepcopy(self.raw)
        wrong["items"][3]["task"] = "File the report."
        with self.assertRaisesRegex(ed.EditorialFailure, "lost_commitment_qualification"):
            self.register(wrong)
        text = ed.render_notes(ed.validate_notes(self.notes, self.records, self.source), reg)
        for phrase in ("Management directs tests", "yardmasters administer", "operating crews are tested", "one was planned to return", "implementation remained unconfirmed", "after the motion"):
            self.assertIn(phrase, text)
        self.assertNotIn("Carried", text)

    def test_no_invented_recipient_owner_or_owner_annotation(self):
        for change in ({"task": "Share committee minutes with Riley."},
                       {"owners": ["Morgan", "Taylor"]},
                       {"owners": ["Taylor (SPEAKER_03)"]}):
            wrong = copy.deepcopy(self.raw)
            wrong["items"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ed.EditorialFailure):
                self.register(wrong)

    def test_quoted_future_unaccepted_request_and_later_actor_cannot_support_task(self):
        for text, owner, task in (
            ("[Taylor] If they agree, I say I'll file the grievance.", "Taylor", "File the grievance."),
            ("[Chair] I'll ask Taylor to review the report.", "Taylor", "Review the report."),
            ("[Taylor] I'll send the package off, and if the office approves it, they will send it to Morgan.", "Taylor", "Send the package to Morgan."),
        ):
            chunks = [portion(1, text)]
            raw = {"items": [{"id": "A1", "category": "undertaking", "task": task, "owners": [owner], "source_ids": ["1:L1"], "member_facing": True, "concerns": []}]}
            with self.subTest(text=text), self.assertRaises(ed.EditorialFailure):
                ed.validate_register(raw, ed.source_records(chunks), commitment_evidence(chunks), text)

    def test_local_future_undertaking_keeps_original_wording_and_qualification(self):
        text = "[Taylor] If the claim is denied, I'm going to write the report."
        chunks = [portion(1, text)]
        raw = {"items": [{"id": "A1", "category": "undertaking", "task": "If the claim is denied, write the report.", "owners": ["Taylor"], "source_ids": ["1:L1"], "member_facing": True, "concerns": []}]}
        reg = ed.validate_register(raw, ed.source_records(chunks), [], text)
        self.assertEqual(reg["items"][0]["evidence"][0]["text"], text)
        raw["items"][0]["task"] = "Write the report."
        with self.assertRaisesRegex(ed.EditorialFailure, "lost_commitment_qualification"):
            ed.validate_register(raw, ed.source_records(chunks), [], text)

    def test_unresolved_owner_stays_private_and_triggers_hold(self):
        chunks = copy.deepcopy(self.chunks)
        chunks[2]["text"] = chunks[2]["text"].replace("[Morgan] I'll", "[SPEAKER_03] I'll")
        records = ed.source_records(chunks)
        raw = copy.deepcopy(self.raw)
        raw["items"][0]["owners"] = ["SPEAKER_03"]
        reg = ed.validate_register(raw, records, commitment_evidence(chunks, include_context=True), self.source)
        self.assertIn("SPEAKER_03", json.dumps(reg))
        self.assertNotIn("SPEAKER_03", ed.undertaking_table(reg))
        self.assertIn("Owner awaiting confirmation", ed.undertaking_table(reg))
        self.assertTrue(any(f["code"] == "unverified_owner" for f in reg["findings"]))

    def test_bad_references_recap_and_duplicate_items_fail_closed(self):
        for change in ({"source_ids": ["missing"]}, {"source_ids": ["2:L1"]},
                       {"source_ids": ["3:L1", "3:L1"]}, {"id": "A2"}, {"extra": 1}):
            raw = copy.deepcopy(self.raw)
            raw["items"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ed.EditorialFailure):
                self.register(raw)
        self.assertNotIn("1:L1", self.records)
        sent = ed.register_input(self.records, self.commitments, [])
        self.assertFalse(any(r["id"] == "2:L1" for r in sent))

    def test_confidentiality_and_unsupported_decorations_fail_closed(self):
        for content in ("Contact morgan@example.invalid.", "Call 416-555-0100.",
                        "Taylor was diagnosed yesterday.", "Medical history discussed.",
                        "Morgan received discipline.", "## Important Notes", "<script>bad</script>",
                        "SPEAKER_03 spoke.", "See 3:L1."):
            notes = copy.deepcopy(self.notes)
            notes["issues"][0]["paragraphs"][0]["text"] = content
            with self.subTest(content=content), self.assertRaises(ed.EditorialFailure):
                ed.validate_notes(notes, self.records, self.source)
        raw = copy.deepcopy(self.raw)
        raw["items"][0]["concerns"] = ["confidentiality"]
        with self.assertRaisesRegex(ed.EditorialFailure, "confidential_action"):
            self.register(raw)

    def test_duplicate_sections_unknown_blocks_and_recap_leaks_rejected(self):
        for change in ("duplicate", "recap", "unknown", "extra"):
            notes = copy.deepcopy(self.notes)
            if change == "duplicate":
                notes["issues"].append(notes["issues"][0])
            elif change == "recap":
                notes["issues"][0]["paragraphs"][0]["source_ids"] = ["2:L1"]
            elif change == "unknown":
                notes["motions"][0]["source_ids"] = ["invented"]
            else:
                notes["actions"] = []
            with self.subTest(change=change), self.assertRaises(ed.EditorialFailure):
                ed.validate_notes(notes, self.records, self.source)

    def test_private_artifacts_hold_all_export_paths_even_if_review_is_edited(self):
        for marker in (ed.REVIEW, ed.REGISTER, "meeting-notes-draft.md"):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "source"
                directory.mkdir()
                (directory / marker).write_text('{"status":"approved"}')
                (directory / "summary.md").write_text("Public candidate")
                with self.assertRaises(ValueError):
                    publication_payload(directory)
                with self.assertRaises(ValueError):
                    export_documents(directory, Path(tmp) / "public")
                with self.assertRaises(ValueError):
                    export_archive(directory, Path(tmp) / "public.zip")
                self.assertFalse((Path(tmp) / "public").exists())
                self.assertFalse((Path(tmp) / "public.zip").exists())
        for name in (ed.REGISTER, ed.REVIEW, ed.CHECKLIST, ed.RESPONSE, ed.REGISTER_MD, ed.NOTES_EVIDENCE):
            self.assertNotIn(name, strip_private_references(f"[private]({name})"))

    def test_omitted_commitment_and_conflict_require_private_review(self):
        raw = copy.deepcopy(self.raw)
        raw["items"] = raw["items"][1:]
        raw["items"][0]["concerns"] = ["source_conflict"]
        codes = {f["code"] for f in self.register(raw)["findings"]}
        self.assertIn("commitment_coverage_review", codes)
        self.assertIn("source_conflict", codes)

    def test_malformed_reduction_has_no_member_documents(self):
        for bad in ("", "{", '{"items":null}'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                calls = []
                def generate(prompt, **kwargs):
                    calls.append(prompt)
                    return bad
                with contextlib.redirect_stdout(io.StringIO()):
                    status = ed.run(Path(tmp), self.chunks, "summary", "current", "recap", self.commitments, [], {}, [], [], ROOT / "prompts/meeting", generate, False, budget=self.budget())
                self.assertEqual(status, 1)
                self.assertEqual(len(calls), 1)
                self.assertFalse((Path(tmp) / "meeting-notes-draft.md").exists())
                self.assertFalse((Path(tmp) / "action-items.md").exists())
                report = json.loads((Path(tmp) / ed.REVIEW).read_text())
                self.assertEqual(report["status"], "review_hold")
                self.assertNotIn("Morgan", json.dumps(report))

    def test_later_schema_failure_retains_private_register_without_partial_documents(self):
        calls = []
        def generate(prompt, **kwargs):
            calls.append(prompt)
            return json.dumps(self.raw) if len(calls) == 1 else '{"highlights":[]}'
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            status = ed.run(Path(tmp), self.chunks, "summary", "current", "recap", self.commitments, [], {}, [], [], ROOT / "prompts/meeting", generate, False, budget=self.budget())
            self.assertEqual(status, 1)
            self.assertEqual(len(calls), 2)
            self.assertTrue((Path(tmp) / ed.REGISTER).exists())
            for name in ("meeting-notes-draft.md", "action-items.md", "minutes-draft.md", "summary.md"):
                self.assertFalse((Path(tmp) / name).exists())
            self.assertEqual(json.loads((Path(tmp) / ed.REVIEW).read_text())["failure_category"], "invalid_notes_schema")

    def test_triaged_register_retains_all_proposals_and_actionable_findings(self):
        raw = copy.deepcopy(self.raw)
        raw["items"] = raw["items"][:3]
        raw["items"][0]["task"] = "Circulate committee minutes to health and safety committees after this meeting."
        raw["items"][1]["source_ids"] = ["fabricated"]
        raw["items"][2]["owners"] = ["Riley"]
        calls = []
        def generate(prompt, **kwargs):
            calls.append(prompt)
            return (json.dumps(raw), json.dumps(self.notes), "# Detailed Minutes\nSource-grounded fixture.")[len(calls) - 1]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            status = ed.run(directory, self.chunks, "summary", "current", "recap", self.commitments, [], {}, [], [],
                            ROOT / "prompts/meeting", generate, False, budget=self.budget())
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 3)
            response = json.loads((directory / ed.RESPONSE).read_text())
            self.assertEqual(json.loads(response["responses"]["register"]), raw)
            assessment = response["register_assessment"]
            self.assertEqual(assessment["proposed_register"], raw)
            self.assertEqual(assessment["sources"], self.records)
            self.assertEqual([r["outcome"] for r in assessment["outcomes"]], ["review_required", "hard_block", "hard_block"])
            checklist = (directory / ed.CHECKLIST).read_text()
            for item in raw["items"]:
                self.assertIn(item["id"], checklist)
                self.assertIn(item["task"], checklist)
            self.assertIn("action_support_review", checklist)
            for name in ("meeting-notes-draft.md", "action-items.md", "minutes-draft.md", "summary.md"):
                self.assertTrue((directory / name).exists())
            with self.assertRaises(ValueError):
                publication_payload(directory)

    def test_lesson_and_existing_output_destinations_reject_editorial_mode_before_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            for options in (["--profile", "lesson", "--meeting-notes", "--meeting-notes-output-dir", str(Path(tmp) / "new")],
                            ["--meeting-notes", "--meeting-notes-output-dir", tmp]):
                with self.subTest(options=options), patch.object(sys, "argv", ["summary", "/unused/transcript", *options]), patch.object(engine, "call_ollama") as mocked, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        engine.main()
                    mocked.assert_not_called()

    def test_structured_generation_limit_is_not_accepted_even_with_valid_json(self):
        with patch.object(engine, "managed_generate", return_value={"response": '{"items":[]}', "done_reason": "length"}):
            with self.assertRaisesRegex(ValueError, "editorial_generation_limit"):
                engine.call_ollama("http://unused", "synthetic", "synthetic", response_format="json")

    def test_existing_runner_uses_three_reductions_and_redacted_source_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "session"
            (transcript / "chunks_out").mkdir(parents=True)
            text = ("[Taylor] PRIVATE_CHATTER vacation.\n[Chair] Let's start the meeting.\n"
                    "[Taylor] Redact the following. PRIVATE_SECRET. End redaction.\n"
                    "[Taylor] I'll send the report to the committee.")
            source = {"chunk_id": 1, "file_name": "synthetic.txt", "text": text}
            index = transcript / "chunks_out/transcript_chunks.jsonl"
            index.write_text(json.dumps(source) + "\n")
            calls = []
            def generate(**kwargs):
                calls.append(kwargs)
                prompt = kwargs["prompt"]
                self.assertNotIn("PRIVATE_SECRET", prompt)
                if "Transcript chunk:" in prompt:
                    return "## Report\n" + prompt.split("Transcript chunk:\n", 1)[1]
                self.assertNotIn("PRIVATE_CHATTER", prompt)
                if "propose ONE private canonical" in prompt:
                    data = json.loads(prompt.split("Untrusted input JSON:\n")[1])
                    row = next(r for r in data["source_evidence"] if "I'll send" in r["text"])
                    return json.dumps({"items": [{"id": "A1", "category": "undertaking", "task": "Send the report to the committee.", "owners": ["Taylor"], "source_ids": [row["id"]], "member_facing": True, "concerns": []}]})
                if "summary-reduction call" in prompt:
                    data = json.loads(prompt.split("Untrusted input JSON:\n")[1])
                    ref = data["register"]["items"][0]["source_ids"]
                    block = {"text": "The committee report was discussed.", "source_ids": ref}
                    return json.dumps({"highlights": [block], "previous_context": [], "issues": [{"heading": "Committee reporting", "paragraphs": [block]}], "motions": [], "unresolved": [], "concerns": []})
                return "# Detailed Minutes\n## Topics\nCommittee reporting.\n## Action Items\n- Morgan: Invented task"
            output = root / "comparison"
            argv = ["summary", str(transcript), "--meeting-notes", "--meeting-notes-output-dir", str(output),
                    "--reduce-num-ctx", "196608", "--meeting-notes-tokenizer", str(self.tokenizer_path)]
            with patch.object(sys, "argv", argv), patch.dict(os.environ, {"MEETING_SUMMARIES_ROOT": str(root / "production"), "MEETING_KEEP_RECAP": "0"}, clear=True), patch.object(engine, "call_ollama", side_effect=generate), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(engine.main(), 0)
            reduced = [c for c in calls if "Transcript chunk:" not in c["prompt"]]
            self.assertEqual(len(reduced), 3)
            self.assertEqual([c.get("response_format") for c in reduced], ["json", "json", None])
            self.assertEqual([c["num_predict"] for c in reduced], [16384, 8192, 16384])
            self.assertTrue(all(r["fits"] for r in json.loads((output / ed.BUDGETS).read_text())["requests"]))
            table = ed.undertaking_table(json.loads((output / ed.REGISTER).read_text()))
            for name in ("meeting-notes-draft.md", "action-items.md", "minutes-draft.md"):
                result = (output / name).read_text()
                self.assertIn(table, result)
                self.assertNotIn("Invented task", result)
                self.assertNotIn("PRIVATE_SECRET", result)
            self.assertEqual(json.loads(index.read_text())["text"], text)
            self.assertFalse((root / "production").exists())

    def test_unrelated_valid_id_is_flagged_with_claim_and_evidence_for_review(self):
        notes = copy.deepcopy(self.notes)
        block = notes["issues"][0]["paragraphs"][0]
        block.update(text="Dangerous commodities remain out of sequence in MRS journals.", source_ids=["3:L5"])
        clean = ed.validate_notes(notes, self.records, self.source)
        self.assertEqual(clean["issues"][0]["paragraphs"][0]["support"]["status"], "uncertain_local_support")
        with tempfile.TemporaryDirectory() as tmp:
            review = ed.finish(Path(tmp), self.register(), clean, "# Minutes", self.source, records=self.records)
            entry = next(f for f in review["findings"] if f.get("notes_location") == "issues/0/paragraphs/0")
            self.assertEqual(entry["code"], "notes_support_review")
            self.assertEqual(entry["source_ids"], ["3:L5"])
            self.assertEqual(entry["claim"], block["text"])
            self.assertEqual(review["status"], "review_hold")

    def test_action_words_scattered_across_unrelated_evidence_are_uncertain(self):
        chunks = [portion(1, "[Taylor] We should review the staffing report."),
                  portion(2, "[Morgan] We should send committee minutes.")]
        records = ed.source_records(chunks)
        raw = {"items": [{"id": "A1", "category": "proposal", "task": "Send the staffing report.",
                          "owners": [], "source_ids": ["1:L1", "2:L1"], "member_facing": False, "concerns": []}]}
        register = ed.validate_register(raw, records, [], "\n".join(c["text"] for c in chunks))
        self.assertEqual(register["items"][0]["support"]["status"], "uncertain_local_support")
        self.assertIn("action_support_review", {f["code"] for f in register["findings"]})
        # A scattered proposal also cannot become a supported undertaking.
        raw["items"][0].update(category="undertaking", owners=["Morgan"])
        with self.assertRaisesRegex(ed.EditorialFailure, "unsupported_undertaking"):
            ed.validate_register(raw, records, [], "\n".join(c["text"] for c in chunks))

    def test_local_ranges_never_bridge_missing_lines_or_chunks(self):
        rows = [self.records["3:L1"], self.records["3:L11"], self.records["4:L1"]]
        self.assertTrue(all(len(r) == 1 for r in ed.local_ranges(rows)))
        self.assertTrue(all(len(r) <= ed.LOCAL_LINES and sum(len(q["text"]) for q in r) <= ed.LOCAL_CHARS
                            for r in ed.local_ranges(list(self.records.values()))))

    def test_compact_input_has_exact_deduplicated_excerpts_and_retains_private_evidence(self):
        register = self.register()
        before = ed.digest(register)
        payload = json.loads(ed.notes_prompt(ROOT / "prompts/meeting", self.records, "maps", register).split("Untrusted input JSON:\n")[1])
        self.assertEqual(ed.digest(register), before)
        self.assertTrue(all("evidence" not in i and len(i["source_ids"]) <= ed.LOCAL_LINES for i in payload["register"]["items"]))
        ids = [r["id"] for r in payload["source_excerpts"]]
        self.assertEqual(len(ids), len(set(ids)))
        for row in payload["source_excerpts"]:
            self.assertEqual(row["text"], self.records[row["id"]]["text"])

    def test_oversized_excerpt_is_an_explicit_gap_and_cannot_be_cited(self):
        chunks = [portion(1, "[Taylor] " + "words " * 500)]
        records = ed.source_records(chunks)
        payload = ed.source_excerpts(records)
        self.assertEqual(payload["source_excerpts"], [])
        self.assertEqual(payload["omitted_oversized_source_ids"], ["1:L1"])

    def test_tokenizer_disables_saved_truncation_and_padding_and_counts_reserves(self):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        tokenizer.enable_truncation(max_length=4)
        tokenizer.enable_padding(length=500)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            tokenizer.save(str(path))
            budget = ed.RequestBudget(path, 17000, "system")
            row = budget.measure("register", "one " * 100)
            self.assertGreater(row["input_tokens"], 4)
            self.assertLess(row["input_tokens"], 500)
            self.assertEqual(row["required_context"], row["input_tokens"] + 16384 + 1024)
            self.assertFalse(row["fits"])
            exact = ed.RequestBudget(path, row["required_context"], "system").measure("register", "one " * 100)
            self.assertTrue(exact["fits"])
            self.assertEqual(exact["headroom"], 0)

    def test_missing_invalid_tokenizer_or_unspecified_context_fails_without_byte_fallback(self):
        for tokenizer, context in ((None, 196608), (self.tokenizer_path, None), ("missing-tokenizer.json", 196608)):
            with self.subTest(tokenizer=tokenizer, context=context), self.assertRaises(ed.EditorialFailure):
                ed.RequestBudget(tokenizer, context, "system")

    def test_every_editorial_slot_fails_preflight_before_overbudget_generation(self):
        from unittest.mock import Mock
        for stage in ("register", "notes", "detailed", "recap"):
            budget = self.budget()
            measure = budget.measure
            def account(name, prompt, structured=False):
                row = measure(name, prompt, structured)
                if name == stage:
                    row.update(fits=False, headroom=-1)
                return row
            responses = iter([json.dumps(self.raw), json.dumps(self.notes), "# Minutes", "# Recap"])
            generate = Mock(side_effect=lambda *a, **k: next(responses))
            with tempfile.TemporaryDirectory() as tmp, patch.object(budget, "measure", side_effect=account), contextlib.redirect_stdout(io.StringIO()):
                result = ed.run(Path(tmp), self.chunks, "maps", "current", "recap", self.commitments, [], {}, [], [],
                                ROOT / "prompts/meeting", generate, True, budget=budget)
                self.assertEqual(result, 1)
                self.assertEqual(generate.call_count, list(ed.OUTPUT_TOKENS).index(stage))
                self.assertFalse((Path(tmp) / "meeting-notes-draft.md").exists())
                report = json.loads((Path(tmp) / ed.REVIEW).read_text())
                self.assertEqual(report["failure_category"], "editorial_" + stage + "_context_exceeded")
                self.assertFalse(json.loads((Path(tmp) / ed.BUDGETS).read_text())["requests"][-1]["fits"])

    def test_review_groups_prioritize_public_owners_conflicts_and_confidentiality(self):
        register = self.register()
        register["findings"] += [{"code": "unverified_owner", "item_id": "A1"},
                                 {"code": "uncertain_identity", "item_id": "A6"},
                                 {"code": "source_conflict", "item_id": "A1"},
                                 {"code": "confidentiality", "item_id": "A6"}]
        with tempfile.TemporaryDirectory() as tmp:
            result = ed.finish(Path(tmp), register, ed.validate_notes(self.notes, self.records, self.source), "# Minutes", self.source, records=self.records)
            groups = {g["id"]: g for g in result["review_groups"]}
            self.assertEqual(groups["member_undertaking_owners"]["priority"], 1)
            self.assertEqual(groups["private_background"]["priority"], 3)
            self.assertTrue(any(e.get("item_id") == "A6" for e in groups["private_background"]["entries"]))
            self.assertTrue(any(e["code"] == "source_conflict" for e in groups["substantive_conflicts"]["entries"]))
            self.assertTrue(any(e["code"] == "confidentiality" for e in groups["confidentiality"]["entries"]))
            owner = next(e for e in groups["member_undertaking_owners"]["entries"] if e["code"] == "owner_verification")
            self.assertIn("task", owner)
            self.assertIn("evidence_pointer", owner)
            self.assertEqual(json.loads((Path(tmp) / ed.NOTES_EVIDENCE).read_text())["sources"], self.records)
            self.assertEqual(result["status"], "review_hold")


if __name__ == "__main__":
    unittest.main()
