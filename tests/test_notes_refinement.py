"""Synthetic, source-linked refinement tests; no private transcripts or inference."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bin'))
from meeting_postprocess import editorial as ed, notes_refinement as nr
from meeting_postprocess.editorial_quality import blocks
from meeting_postprocess.publication import publication_payload
from test_editorial_notes import fixture


class Budget:
    def measure(self,*args):return {'fits':True,'reserved_output':49152}


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.out=Path(self.temp.name)
        chunks,raw,notes=fixture();self.records=ed.source_records(chunks)
        self.source='\n'.join(r['text'] for r in self.records.values())
        self.original=ed.validate_notes(notes,self.records,self.source)
        self.register=ed.triage_register(ed.audit_register(raw,self.records,[],self.source),self.records,[],self.source)
        ed.finish(self.out,self.register,self.original,'# Detailed Minutes\nComprehensive original.',self.source,records=self.records)
        ed.write_private_json(self.out/ed.RESPONSE,{'responses':{'notes':json.dumps(notes)},'unchanged':'original'})
        self.before={p.name:p.read_bytes() for p in self.out.iterdir() if p.is_file()}
        self.response={'notes':{k:v for k,v in nr.plain_notes(self.original).items() if k!='concerns'},'review_concerns':[],
                       'coverage':[{'original_block':loc,'revised_blocks':[loc],'disposition':'retained','explanation':'Preserved.'} for loc,b in blocks(self.original)]}
        self.calls=[]

    def generate(self,prompt,**kwargs):
        self.calls.append((prompt,kwargs));return json.dumps(self.response)

    def run_once(self,generate=None,budget=None):
        return nr.run_once(self.out,ROOT/'prompts/meeting',generate or self.generate,budget or Budget())

    def unchanged(self):
        for name,data in self.before.items():self.assertEqual((self.out/name).read_bytes(),data,name)

    def test_one_request_separate_versions_private_register_concerns_and_holds(self):
        result=self.run_once();self.assertEqual(result['result'],'candidate_requires_review')
        self.assertEqual(len(self.calls),1);self.assertTrue(self.calls[0][1]['structured'])
        self.assertEqual(self.calls[0][1]['num_predict'],49152)
        refined=self.out/nr.DIRECTORY
        self.assertEqual((refined/ed.REGISTER).read_bytes(),self.before[ed.REGISTER])
        self.assertEqual((refined/ed.REGISTER_MD).read_bytes(),self.before[ed.REGISTER_MD])
        self.assertEqual(json.loads((refined/ed.NOTES_EVIDENCE).read_text())['notes']['concerns'],self.original['concerns'])
        self.assertFalse((refined/'minutes-draft.md').exists());self.unchanged()
        self.assertFalse(result['candidate_selected'])
        self.assertIn('refinement_semantic_coverage_review',{f['code'] for f in result['findings']})
        for d in [self.out,refined]:
            with self.assertRaisesRegex(ValueError,'hold'):publication_payload(d)
        if sys.platform.startswith('linux'):
            self.assertEqual(refined.stat().st_mode&0o777,0o700)
            self.assertTrue(all(p.stat().st_mode&0o777==0o600 for p in refined.iterdir() if p.is_file()))
        self.assertEqual(self.run_once()['result'],'already_attempted');self.assertEqual(len(self.calls),1)

    def test_failed_response_has_no_retry_and_original_remains_valid(self):
        self.response['notes']['issues'][0]['paragraphs'][0]['text']='Private medical history.'
        result=self.run_once();self.assertEqual(result['result'],'failed');self.assertEqual(result['request_attempts'],1)
        self.assertEqual(result['failure_category'],'confidential_notes_content')
        self.assertTrue((self.out/nr.DIRECTORY/ed.RESPONSE).exists());self.unchanged()
        self.assertEqual(self.run_once()['result'],'already_attempted');self.assertEqual(len(self.calls),1)

    def test_transport_failure_preserves_original_and_no_retry(self):
        def fail(*a,**kw):self.calls.append(True);raise RuntimeError('private transport detail')
        result=self.run_once(fail);self.assertEqual(result['result'],'failed');self.assertEqual(len(self.calls),1)
        self.assertNotIn('private transport detail',json.dumps(result));self.unchanged()

    def test_context_failure_calls_no_model(self):
        class Small:
            def measure(self,*a):return {'fits':False,'reserved_output':49152}
        self.assertEqual(self.run_once(budget=Small())['request_attempts'],0);self.assertEqual(self.calls,[]);self.unchanged()

    def test_malformed_schema_reference_and_missing_coverage_rejected(self):
        cases=[]
        for mutation in [lambda r:r.update(extra=True),lambda r:r['notes'].update(concerns=[]),
                         lambda r:r.update(coverage=[]),lambda r:r['coverage'].append(copy.deepcopy(r['coverage'][0])),
                         lambda r:r['notes']['issues'][0]['paragraphs'][0].update(source_ids=['missing:L99']),
                         lambda r:r['coverage'][0].update(revised_blocks=['issues/99/paragraphs/0']),
                         lambda r:r['coverage'][0].update(revised_blocks=[]),
                         lambda r:r['notes']['issues'][0]['paragraphs'][0].update(text='x'*6001),
                         lambda r:r.update(review_concerns=[{'category':'invented','text':'private','source_ids':['missing']}])]:
            value=copy.deepcopy(self.response);mutation(value);cases.append(value)
        for raw in cases:
            with self.subTest(raw=str(type(raw))):
                with self.assertRaises(ed.EditorialFailure):nr.validate(raw,self.original,self.records,self.source)

    def test_lost_figures_and_qualifications_are_material_review_not_approval(self):
        old=copy.deepcopy(self.original);loc='issues/0/paragraphs/0'
        old['issues'][0]['paragraphs'][0]['text']='A participant alleged 25 disputed payments.'
        self.response['notes']['issues'][0]['paragraphs'][0]['text']='Payments were discussed.'
        _,findings=nr.validate(self.response,old,self.records,self.source)
        self.assertTrue({'refinement_figure_coverage_review','refinement_qualification_review'}<={f['code'] for f in findings})

    def test_combined_incidents_are_explicitly_reviewed(self):
        raw=copy.deepcopy(self.response)
        rows=[r for r in raw['coverage'] if r['original_block'].startswith('issues/')]
        extra=copy.deepcopy(self.original);extra['issues'][0]['paragraphs'].append(copy.deepcopy(extra['issues'][0]['paragraphs'][0]))
        raw['coverage'].append({'original_block':'issues/0/paragraphs/1','revised_blocks':['issues/0/paragraphs/0'],'disposition':'consolidated','explanation':'Related but distinct reports.'})
        _,flags=nr.validate(raw,extra,self.records,self.source)
        self.assertIn('refinement_combined_accounts_review',{f['code'] for f in flags})

    def test_private_withholding_is_accounted_and_original_concerns_preserved(self):
        raw=copy.deepcopy(self.response)
        # Remove a highlight, not its underlying issue; make the omission explicit.
        location='highlights/0';self.assertTrue(raw['notes']['highlights'])
        raw['notes']['highlights']=[]
        for row in raw['coverage']:
            if row['original_block'].startswith('highlights/'):
                row.update(revised_blocks=[],disposition='privacy_withheld',explanation='Identifying member detail withheld; systemic issue retained.')
        clean,flags=nr.validate(raw,self.original,self.records,self.source)
        self.assertEqual(clean['concerns'],self.original['concerns'])
        self.assertIn('refinement_material_change_review',{f['code'] for f in flags})

    def test_prompt_includes_exact_source_and_private_concerns_without_register_generation(self):
        prompt=nr.build_prompt(ROOT/'prompts/meeting',self.original,self.records,[])
        data=json.loads(prompt.split('Untrusted input JSON:\n')[1])
        self.assertEqual(data['original_notes']['concerns'],self.original['concerns'])
        self.assertEqual(set(r['id'] for r in data['original_blocks']),set(dict(blocks(self.original))))
        for row in data['source_excerpts']:self.assertEqual(row['text'],self.records[row['id']]['text'])
        self.assertIn('2,600–3,000',prompt);self.assertIn('separate train-handling incidents',prompt)
        self.assertIn('reported\ninterpretations',prompt);self.assertIn('Never infer a speaker',prompt)

    def test_complete_coverage_can_copy_existing_material_errors_and_still_requires_review(self):
        # Frozen synthetic counterexamples: no identities or meeting details.
        cases=[('In one case a train stalled, then a car derailed.', 'refinement_incident_relationship_review'),
               ('The Labour Code requires every request to be granted.', 'refinement_legal_interpretation_review'),
               ('The award is under review because the company provided false information.', 'refinement_allegation_causation_review'),
               ('The member in engineer training has a medical issue and is a single parent.', 'refinement_member_privacy_review')]
        for text,expected in cases:
            with self.subTest(expected=expected):
                original=copy.deepcopy(self.original);raw=copy.deepcopy(self.response)
                original['issues'][0]['paragraphs'][0]['text']=text
                raw['notes']['issues'][0]['paragraphs'][0]['text']=text
                _,flags=nr.validate(raw,original,self.records,self.source)
                self.assertIn(expected,{f['code'] for f in flags})
                self.assertIn('refinement_semantic_coverage_review',{f['code'] for f in flags})
