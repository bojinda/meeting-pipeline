"""Synthetic correction packets, per-edit guards, preservation and review binding."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'bin'))
from meeting_postprocess import editorial as ed,notes_correction as nc
from meeting_postprocess.editorial_quality import factual_risks
from meeting_postprocess.publication import publication_payload
from test_editorial_notes import fixture

class Budget:
    def measure(self,*a):return {'fits':True,'reserved_output':49152}

class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        chunks,raw,notes=fixture();self.records=ed.source_records(chunks);self.notes=ed.validate_notes(notes,self.records,'')
        register=ed.triage_register(ed.audit_register(raw,self.records,[],''),self.records,[],'')
        ed.finish(self.root,register,self.notes,'# Detailed Minutes\nComplete original business.','',records=self.records)
        self.original={p.name:p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        self.loc='issues/0/paragraphs/0';self.selections={self.loc:{'questions':['Check the attribution.']}}
        self.packets,_=nc.make_packets(self.notes,self.records,[],self.selections)
        self.row={'passage_id':self.loc,'replacement_text':'A participant reported that management directs tests, yardmasters administer them and operating crews are tested.',
                  'source_ids':['3:L8'],'explanation':'Attributes the report to a participant.','unresolved_questions':['Does the policy confirm this?'],
                  'evidence_insufficient':False,'support':[{'source_id':'3:L8','quote':'Management directs efficiency tests, yardmasters administer them, operating crews are tested.'}]}
    def unchanged(self):
        self.assertTrue(all((self.root/k).read_bytes()==v for k,v in self.original.items()))
    def response(self,rows=None):return json.dumps({'corrections':rows or [self.row]}).encode()

    def test_candidate_is_unverified_separate_and_keeps_all_private_material(self):
        out=self.root/'candidate';report=nc.render_version(self.root,out,self.response(),self.packets)
        self.assertEqual(report['applied_count'],1);self.assertFalse(report['candidate_selected'])
        self.assertEqual(report['mode'],'automated_unverified_candidate');self.unchanged()
        self.assertEqual((out/ed.REGISTER).read_bytes(),self.original[ed.REGISTER])
        self.assertEqual(json.loads((out/ed.NOTES_EVIDENCE).read_text())['notes']['concerns'],self.notes['concerns'])
        self.assertFalse((out/'minutes-draft.md').exists())
        self.assertIn(self.row['support'][0]['quote'],(out/'correction-comparison.private.md').read_text())
        with self.assertRaisesRegex(ValueError,'hold'):publication_payload(out)
        if sys.platform.startswith('linux'):
            self.assertEqual(out.stat().st_mode&0o777,0o700)
            self.assertTrue(all(p.stat().st_mode&0o777==0o600 for p in out.iterdir() if p.is_file()))

    def test_bad_quotes_invalid_refs_sensitive_prose_and_new_figures_rejected(self):
        for mutation in [lambda r:r.update(source_ids=['missing']),lambda r:r['support'][0].update(quote='fabricated supporting evidence'),
                         lambda r:r.update(replacement_text='Private medical history.'),lambda r:r.update(replacement_text='A participant reported 9876 tests.'),
                         lambda r:r.update(replacement_text='John Smith directed the tests.'),
                         lambda r:r.update(replacement_text='x'*6001),lambda r:r.update(extra=True),lambda r:r.update(source_ids=['3:L8','3:L8']),
                         lambda r:r.update(passage_id='issues/99/paragraphs/0')]:
            row=copy.deepcopy(self.row);mutation(row)
            self.assertEqual(nc.assess({'corrections':[row]},self.packets,self.notes,self.records)[0]['status'],'rejected')

    def test_invalid_proposal_does_not_erase_other_eligible_correction(self):
        report=nc.render_version(self.root,self.root/'partial',self.response([None,self.row]),self.packets)
        self.assertEqual(report['applied_count'],1);self.assertEqual(report['proposed_count'],2);self.unchanged()

    def test_insufficient_evidence_retains_original(self):
        row={**self.row,'evidence_insufficient':True,'replacement_text':None,'support':[],'source_ids':[]}
        report=nc.render_version(self.root,self.root/'insufficient',self.response([row]),self.packets)
        self.assertEqual(report['applied_count'],0);self.assertEqual(report['assessment'][0]['status'],'insufficient_evidence')
        self.assertTrue((self.root/'insufficient/meeting-notes-draft.md').read_bytes().endswith(self.original['meeting-notes-draft.md']))

    def test_independent_rejection_overrides_valid_ids_and_quotes(self):
        response=self.response();decision={'response_sha256':nc.sha(response),'evidence_sha256':nc.sha(self.original[ed.NOTES_EVIDENCE]),'packets_sha256':ed.digest(self.packets),
            'decisions':[{'passage_id':self.loc,'decision':'reject','reason':'The quoted words do not support the proposed relationship.'}]}
        report=nc.render_version(self.root,self.root/'reviewed',response,self.packets,decisions=decision)
        self.assertEqual(report['applied_count'],0);self.unchanged()
        decision['response_sha256']='0'*64
        with self.assertRaisesRegex(ed.EditorialFailure,'binding'):nc.render_version(self.root,self.root/'mismatch',response,self.packets,decisions=decision)

    def test_independent_acceptance_cannot_bypass_hard_validation(self):
        row={**self.row,'replacement_text':'Private medical history.'};response=self.response([row])
        decision={'response_sha256':nc.sha(response),'evidence_sha256':nc.sha(self.original[ed.NOTES_EVIDENCE]),'packets_sha256':ed.digest(self.packets),
            'decisions':[{'passage_id':self.loc,'decision':'accept','reason':'Attempted unsafe acceptance.'}]}
        with self.assertRaisesRegex(ed.EditorialFailure,'cannot_accept'):nc.render_version(self.root,self.root/'invalid',response,self.packets,decisions=decision)

    def test_duplicate_targets_rejected_not_last_write_wins(self):
        assessed=nc.assess({'corrections':[self.row,self.row]},self.packets,self.notes,self.records)
        self.assertTrue(all(r['status']=='rejected' for r in assessed))

    def test_copied_wording_is_not_counted_as_a_factual_correction(self):
        row={**self.row,'replacement_text':self.notes['issues'][0]['paragraphs'][0]['text'],
             'explanation':'Claims to have qualified the prose, but copied it.'}
        report=nc.render_version(self.root,self.root/'copied',self.response([row]),self.packets)
        self.assertEqual(report['unchanged_count'],1);self.assertEqual(report['applied_count'],0)
        self.assertEqual(report['assessment'][0]['status'],'unchanged');self.unchanged()

    def test_review_packet_hash_and_malformed_decisions_fail_closed(self):
        response=self.response();decision={'response_sha256':nc.sha(response),'evidence_sha256':nc.sha(self.original[ed.NOTES_EVIDENCE]),
            'packets_sha256':'0'*64,'decisions':[]}
        with self.assertRaisesRegex(ed.EditorialFailure,'binding'):
            nc.render_version(self.root,self.root/'changed-packets',response,self.packets,decisions=decision)
        decision['packets_sha256']=ed.digest(self.packets)
        for loc in (['malformed'],'missing'):
            decision['decisions']=[{'passage_id':loc,'decision':'accept','reason':'Invalid target.'}]
            with self.assertRaisesRegex(ed.EditorialFailure,'invalid_correction_decision'):
                nc.render_version(self.root,self.root/'bad-decisions',response,self.packets,decisions=decision)

    def test_one_request_and_no_retry_after_failure_or_success(self):
        calls=[]
        def generate(*a,**kw):calls.append(True);return self.response().decode()
        result=nc.run_once(self.root,ROOT/'prompts/meeting',generate,Budget(),selections=self.selections)
        self.assertEqual(result['result'],'unverified_candidate');self.assertEqual(len(calls),1);self.unchanged()
        self.assertEqual(nc.run_once(self.root,ROOT/'prompts/meeting',generate,Budget())['result'],'already_attempted')
        self.assertEqual(len(calls),1)

    def test_context_failure_and_empty_targets_make_no_request(self):
        class Small:
            def measure(self,*a):return {'fits':False,'reserved_output':49152}
        def forbidden(*a,**kw):raise AssertionError('must not call')
        result=nc.run_once(self.root,ROOT/'prompts/meeting',forbidden,Small(),selections=self.selections)
        self.assertEqual(result['request_attempts'],0);self.assertEqual(result['failure_category'],'correction_context_exceeded');self.unchanged()

    def test_qa_targets_relationship_certainty_privacy_and_terminology_without_rewriting(self):
        cases=[('A train stalled and a car derailed.','incident_relationship_review'),('The company acted because a member complained.','causal_support_review'),
               ('The Labour Code requires this.','legal_interpretation_review'),('The Code of Conduct prohibits family supervision.','policy_scope_review'),
               ('A complaint was filed.','claim_certainty_review'),('A member with a medical issue is a single parent.','member_identifiability_review'),
               ('The chemical was benzene.','substance_identity_review'),('The ABC system applies.','terminology_support_review')]
        for text,code in cases:self.assertIn(code,{c for c,_ in factual_risks(text,'Uncertain testimony.')})

    def test_distinct_report_outcomes_and_legacy_copied_text(self):
        variants=[(dict(self.row,disposition='edit'),'meaningful_edit_candidate'),
                  (dict(self.row,disposition='no_change',replacement_text=self.notes['issues'][0]['paragraphs'][0]['text']),'justified_no_change'),
                  (dict(self.row,disposition='insufficient_evidence',evidence_insufficient=True,replacement_text=None,source_ids=[],support=[]),'insufficient_evidence'),
                  (dict(self.row,source_ids=['missing']),'rejected'),
                  (dict(self.row,replacement_text=self.notes['issues'][0]['paragraphs'][0]['text']),'unexplained_unchanged')]
        for i,(row,outcome) in enumerate(variants):
            with self.subTest(outcome=outcome):
                report=nc.render_version(self.root,self.root/str(i),self.response([row]),self.packets)
                self.assertEqual(report['assessment'][0]['outcome'],outcome)
                self.assertEqual(report['outcome_counts'][outcome],1)
                self.assertFalse(report['automated_checks_certify_semantic_accuracy'])
                self.assertEqual(report['applied_count'],int(outcome=='meaningful_edit_candidate'))
                self.unchanged()

    def test_no_change_and_insufficient_evidence_cannot_hide_bad_references(self):
        for row in (dict(self.row,disposition='no_change'),
                    dict(self.row,disposition='insufficient_evidence',evidence_insufficient=True,replacement_text=None,source_ids=['missing']),
                    dict(self.row,disposition='no_change',replacement_text=self.notes['issues'][0]['paragraphs'][0]['text'],support=[])):
            self.assertEqual(nc.assess({'corrections':[row]},self.packets,self.notes,self.records)[0]['status'],'rejected')
