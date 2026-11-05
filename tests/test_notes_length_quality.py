"""Synthetic length diagnostics; no model-quality or semantic-coverage claims."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
from meeting_postprocess import editorial as ed
from meeting_postprocess.editorial_quality import review_quality
from test_editorial_notes import fixture


class NotesLengthQualityTests(unittest.TestCase):
    def setUp(self):
        self.chunks, _, self.notes = fixture()
        self.records = ed.source_records(self.chunks)
        self.ref = self.notes['issues'][0]['paragraphs'][0]['source_ids']

    def block(self, words):
        return {'text': 'Business ' * words, 'source_ids': list(self.ref)}

    def test_frozen_length_shape_explains_expansion_without_private_text(self):
        # November diagnostic counts only; every word here is synthetic.
        self.notes.update(highlights=[self.block(212)], previous_context=[self.block(122)],
                          motions=[self.block(9)], unresolved=[self.block(209)])
        sizes = [173,707,236,176,364,161,412,142,157,150,185,230,240,212]
        self.notes['issues'] = [{'heading': f'Issue {i}', 'paragraphs': [self.block(n)]} for i,n in enumerate(sizes)]
        self.notes['concerns'][0]['text'] = 'private ' * 1500
        before = copy.deepcopy(self.notes)
        metrics, flags = review_quality(self.notes, self.records, 'rendered ' * 4289)
        self.assertEqual(metrics['member_prose_words'], 4097)
        self.assertEqual(metrics['section_words']['issues'], 3545)
        self.assertEqual(metrics['document_words'], 4289)
        self.assertEqual(len(metrics['issues']), 14)
        self.assertEqual(sum(f['code']=='notes_issue_length_review' for f in flags), 3)
        self.assertEqual(sum(f['code']=='notes_section_length_review' for f in flags), 2)
        self.assertEqual(self.notes, before)
        self.assertFalse(metrics['automatic_revision'])

    def test_section_and_issue_thresholds_are_review_only(self):
        for words, paragraphs, expected in [(300,2,False),(301,2,True),(120,3,True)]:
            with self.subTest(words=words,paragraphs=paragraphs):
                self.notes['issues'] = [{'heading':'Business', 'paragraphs':[self.block(words-paragraphs+1)]+[self.block(1) for _ in range(paragraphs-1)]}]
                self.notes['highlights']=[self.block(120)]
                before=copy.deepcopy(self.notes)
                _, flags=review_quality(self.notes,self.records,'document ' * 2000)
                self.assertEqual(any(f['code']=='notes_issue_length_review' for f in flags),expected)
                self.assertFalse(any(f['code']=='notes_section_length_review' for f in flags))
                self.assertEqual(self.notes,before)
        self.notes['unresolved']=[self.block(121)]
        self.assertIn('notes_section_length_review',{f['code'] for f in review_quality(self.notes,self.records,'document ' * 2000)[1]})

    def test_repeated_heading_cue_preserves_distinct_support(self):
        self.notes['issues']=[{'heading':'Safety report','paragraphs':[self.block(20)]},
                              {'heading':'SAFETY: report!','paragraphs':[self.block(30)]}]
        before=copy.deepcopy(self.notes)
        flags=review_quality(self.notes,self.records,'document')[1]
        flag=next(f for f in flags if f['code']=='repeated_issue_heading_review')
        self.assertEqual(flag['notes_location'],'issues/1')
        self.assertEqual(flag['source_ids'],self.ref)
        self.assertEqual(self.notes,before)

    def test_shorter_prompt_preserves_all_exact_source_evidence_and_document_types(self):
        register={'items':[], 'findings':[]}
        prompt=ed.notes_prompt(ROOT/'prompts/meeting',self.records,'all synthetic maps',register)
        payload=json.loads(prompt.split('Untrusted input JSON:\n',1)[1])
        expected={k:v['text'] for k,v in self.records.items() if len(v['text'])<=ed.LOCAL_CHARS}
        self.assertEqual({row['id']:row['text'] for row in payload['source_excerpts']},expected)
        for phrase in ['2,400–2,750','one\nor two concise paragraphs','separately cited blocks',
                       '12,000 characters','more than 6,000','Detailed Minutes have no Meeting Notes word target',
                       'remove a material issue or strip qualifications']:
            self.assertIn(phrase,prompt)
        detailed=ed.detailed_prompt(ROOT/'prompts/meeting',self.records,'comprehensive evidence',register)
        self.assertNotIn('2,400–2,750',detailed)
        self.assertEqual(ed.OUTPUT_TOKENS['detailed'],32768)

    def test_within_target_does_not_approve_facts_identity_privacy_or_actions(self):
        text='The chair alleged false information and a three times difference since the audit started.'
        self.notes['issues'][0]['paragraphs'][0]['text']=text
        before=copy.deepcopy(self.notes)
        _,flags=review_quality(self.notes,self.records,'document ' * 2000)
        codes={f['code'] for f in flags}
        self.assertTrue({'role_attribution_review','allegation_status_review','numerical_basis_review','chronology_review'}<=codes)
        self.assertNotIn('length_review',codes)
        self.assertEqual(self.notes,before)
        self.notes['issues'][0]['paragraphs'][0]['text']='The medical history was discussed.'
        with self.assertRaises(ed.EditorialFailure):ed.validate_notes(self.notes,self.records,'')
        self.notes['issues'][0]['paragraphs'][0]=dict(text='Brief business.',source_ids=['missing:L1'])
        with self.assertRaises(ed.EditorialFailure):ed.validate_notes(self.notes,self.records,'')
