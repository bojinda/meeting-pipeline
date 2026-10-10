"""Full private October fixture regression; no inference or remote access.

The supplied private inputs remain ignored. Set OCTOBER_EDITORIAL_FIXTURE_ROOT
and OCTOBER_EDITORIAL_TOKENIZER when running outside the development checkout.
"""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from meeting_postprocess import editorial as ed
from preflight_editorial import preflight

FIXTURE = Path(os.environ.get("OCTOBER_EDITORIAL_FIXTURE_ROOT", ROOT / "ignore/editorial-implementation-review/october-offline-render"))
TOKENIZER = Path(os.environ.get("OCTOBER_EDITORIAL_TOKENIZER", ROOT / "ignore/tokenizers/qwen3.5-27b/tokenizer.json"))
SOURCE = Path(os.environ.get("OCTOBER_EDITORIAL_SOURCE", ROOT / "ignore/qwen38-output-review"))


@unittest.skipUnless((FIXTURE / ed.REGISTER).is_file() and TOKENIZER.is_file(), "private full October fixture and matching tokenizer required")
class FullFixtureEditorialTests(unittest.TestCase):
    def test_full_register_is_compact_without_mutating_complete_private_provenance(self):
        path = FIXTURE / ed.REGISTER
        original = path.read_bytes()
        register = json.loads(original)
        self.assertEqual(len(register["items"]), 35)
        self.assertGreater(len(original), 1_200_000)
        original_hash = ed.digest(register)
        compact = ed.model_register(register)
        self.assertEqual(ed.digest(register), original_hash)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(len(compact["items"]), 35)
        self.assertLess(len(json.dumps(compact).encode()), 25000)
        self.assertTrue(all(len(i["source_ids"]) <= 3 and "evidence" not in i and "private_review_detail" not in i for i in compact["items"]))
        self.assertTrue(all(i["evidence"] for i in register["items"]))

    def test_actual_tokenizer_full_fixture_prompts_fit_with_output_reserves(self):
        with patch("ollama_session_summary.call_ollama", side_effect=AssertionError("inference forbidden")), \
             patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
            report = preflight(SOURCE, FIXTURE / ed.REGISTER, TOKENIZER, 196608, True)
        self.assertEqual(report["inference_calls"], 0)
        self.assertEqual(report["register_items"], 35)
        self.assertEqual(report["source_excerpt_gaps"], [])
        self.assertEqual([r["stage"] for r in report["requests"]], ["register", "notes", "detailed", "recap"])
        self.assertTrue(report["all_fit"])
        for row in report["requests"]:
            self.assertEqual(row["method"], "configured_local_tokenizer")
            self.assertEqual(row["required_context"], row["input_tokens"] + row["reserved_output"] + 1024)
            self.assertEqual(row["headroom"], 196608 - row["required_context"])
        # Reproduce the original full-register pressure with the same tokenizer.
        register = json.loads((FIXTURE / ed.REGISTER).read_text())
        budget = ed.RequestBudget(TOKENIZER, 196608, "Editorial system")
        oversized = budget.measure("notes", json.dumps({"register": register}, ensure_ascii=False), True)
        self.assertFalse(oversized["fits"])
        self.assertEqual(len((FIXTURE / "meeting-notes-draft.md").read_text().split()), 1865)


if __name__ == "__main__":
    unittest.main()
