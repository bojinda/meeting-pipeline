"""Separate private concern bounds; frozen synthetic evidence only."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
from meeting_postprocess import editorial as ed
from meeting_postprocess.publication import publication_payload
from test_editorial_notes import fixture


class PrivateConcernLimitTests(unittest.TestCase):
    def setUp(self):
        self.chunks, self.raw, self.notes = fixture()
        self.records = ed.source_records(self.chunks)
        self.concern = self.notes['concerns'][0]

    def test_private_boundaries_preserve_full_text_and_flag_only_long_concerns(self):
        for size in (6000, 6001, 7673, 12000):
            with self.subTest(size=size):
                self.concern['text'] = 'x' * size
                before = copy.deepcopy(self.notes)
                clean = ed.validate_notes(self.notes, self.records, '')
                self.assertEqual(clean['concerns'], self.notes['concerns'])
                self.assertEqual(self.notes, before)
                flags = [f for f in ed.notes_concern_findings(clean) if f['code'] == 'long_private_concern_review']
                self.assertEqual(len(flags), int(size > 6000))
                if flags:
                    self.assertEqual(flags[0]['characters'], size)
                    self.assertEqual(flags[0]['source_ids'], self.concern['source_ids'])

    def test_excessive_malformed_private_concerns_still_fail(self):
        valid = copy.deepcopy(self.concern)
        for value in [None, [], 'uncertain_identity', {**valid, 'text': 'x' * 12001},
                      {**valid, 'text': ''}, {**valid, 'text': 42}, {**valid, 'category': []},
                      {**valid, 'category': 'invented'}, {**valid, 'extra': True}, {'text': 'private'}]:
            with self.subTest(value_type=type(value).__name__):
                notes = copy.deepcopy(self.notes); notes['concerns'] = [value]
                with self.assertRaises(ed.EditorialFailure):
                    ed.validate_notes(notes, self.records, '')

    def test_long_concerns_cannot_bypass_exact_reference_validation(self):
        self.concern['text'] = 'x' * 7673
        for refs in [['missing'], [], ['4:L2', '4:L2'], ['4:L99-L100'], ['1:L1']]:
            with self.subTest(refs=refs):
                self.concern['source_ids'] = refs
                with self.assertRaises(ed.EditorialFailure):
                    ed.validate_notes(self.notes, self.records, '')

    def test_public_limits_and_confidentiality_guards_are_unchanged(self):
        for text in ['x' * 6001, 'The medical history was discussed.']:
            self.notes['issues'][0]['paragraphs'][0]['text'] = text
            with self.assertRaises(ed.EditorialFailure):
                ed.validate_notes(self.notes, self.records, '')

    def test_long_sensitive_concern_stays_private_and_export_held(self):
        self.concern['text'] = 'Private medical history. ' + 'x' * 7650
        clean = ed.validate_notes(self.notes, self.records, '')
        register = ed.triage_register(ed.audit_register(self.raw, self.records, [], ''), self.records, [], '')
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            review = ed.finish(out, register, clean, None, '', records=self.records)
            self.assertEqual(review['status'], 'review_hold')
            self.assertIn('long_private_concern_review', [f['code'] for f in review['findings']])
            self.assertIn(self.concern['text'], (out / ed.TRACE).read_text(encoding='utf-8'))
            self.assertNotIn('Private medical history', (out / 'meeting-notes-draft.md').read_text(encoding='utf-8'))
            with self.assertRaisesRegex(ValueError, 'hold'):
                publication_payload(out)
