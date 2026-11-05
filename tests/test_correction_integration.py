"""Held factual-correction integration, using only synthetic offline replies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'bin'))
from meeting_postprocess import editorial as ed,notes_correction as nc
from meeting_postprocess.ollama_response import GenerationFailure
from meeting_stage_inputs import input_files
from test_editorial_notes import fixture


class Budget:
    measurements=[]
    def measure(self,*a):return {'fits':True,'reserved_output':49152}


class CorrectionIntegrationTests(unittest.TestCase):
    def test_one_final_answer_request_after_original_documents_and_safe_failure(self):
        chunks,register,notes=fixture()
        notes['issues'][0]['paragraphs'][0]['text']='A participant alleged that management directs tests, yardmasters administer them and operating crews are tested.'
        row={'passage_id':'issues/0/paragraphs/0','replacement_text':'A participant reported that management directs tests, yardmasters administer them and operating crews are tested.',
             'source_ids':['3:L8'],'explanation':'Attributes the account.','unresolved_questions':['Confirm the policy.'],'evidence_insufficient':False,
             'support':[{'source_id':'3:L8','quote':'Management directs efficiency tests, yardmasters administer them, operating crews are tested.'}]}
        detailed='# Detailed Minutes\n'+'Comprehensive business. '*1600
        for notes_only,correct,failure in [(True,False,None),(False,False,None),(True,True,None),(False,True,None),
                                          (False,True,'malformed'),(False,True,'ollama_thinking_only_answer')]:
            with self.subTest(notes_only=notes_only,correct=correct,failure=failure),tempfile.TemporaryDirectory() as td:
                calls=[]
                def generate(prompt,**kwargs):
                    calls.append((prompt,kwargs))
                    if len(calls)==1:return json.dumps(register)
                    if len(calls)==2:return json.dumps(notes)
                    if not notes_only and len(calls)==3:return detailed
                    self.assertIs(kwargs['thinking'],False)
                    if failure=='malformed':return '{malformed'
                    if failure:raise GenerationFailure(failure)
                    return json.dumps({'corrections':[row]})
                out=Path(td)
                status=ed.run(out,chunks,'maps','current','',[],[],{},[],[],ROOT/'prompts/meeting',generate,False,
                              budget=Budget(),notes_only=notes_only,correct=correct)
                self.assertEqual(status,0);self.assertEqual(len(calls),2+int(not notes_only)+int(correct))
                if not notes_only:self.assertIn(detailed.rstrip(),(out/'minutes-draft.md').read_text())
                self.assertEqual((out/nc.DIRECTORY).exists(),correct)
                self.assertIn('alleged',(out/'meeting-notes-draft.md').read_text())
                raw=json.loads((out/ed.RESPONSE).read_text())['responses']
                self.assertEqual(raw['notes'],json.dumps(notes));self.assertNotIn('correction',raw)
                self.assertEqual(json.loads((out/ed.REVIEW).read_text())['status'],'review_hold')
                if correct:
                    report=json.loads((out/nc.DIRECTORY/nc.REPORT).read_text())
                    self.assertEqual(report['request_attempts'],1);self.assertTrue(report['original_files_unchanged'])
                    self.assertEqual(report['result'],'failed' if failure else 'unverified_candidate')
                    if not failure:self.assertIn('Unverified correction candidate',(out/nc.DIRECTORY/'candidate/meeting-notes-draft.md').read_text())
                    self.assertEqual(nc.run_once(out,ROOT/'prompts/meeting',generate,Budget())['result'],'already_attempted')
                    self.assertEqual(len(calls),2+int(not notes_only)+1)

    def test_empty_target_set_skips_inference_and_storage_failure_keeps_original(self):
        chunks,register,notes=fixture()
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);records=ed.source_records(chunks)
            reg=ed.triage_register(ed.audit_register(register,records,[],''),records,[],'')
            ed.finish(out,reg,ed.validate_notes(notes,records,''),None,'',records=records)
            def forbidden(*a,**k):raise AssertionError('No request expected')
            result=nc.run_once(out,ROOT/'prompts/meeting',forbidden,Budget(),selections={})
            self.assertEqual(result['result'],'no_questionable_passages');self.assertEqual(result['request_attempts'],0)
        with tempfile.TemporaryDirectory() as td,patch.object(nc,'run_once',side_effect=PermissionError('synthetic')):
            replies=iter([json.dumps(register),json.dumps(notes)])
            status=ed.run(Path(td),chunks,'maps','current','',[],[],{},[],[],ROOT/'prompts/meeting',lambda *a,**k:next(replies),False,
                          budget=Budget(),notes_only=True,correct=True)
            self.assertEqual(status,0);self.assertTrue((Path(td)/'meeting-notes-draft.md').is_file())

    def test_review_tool_is_offline_and_rejects_unknown_decisions(self):
        from test_notes_correction import CorrectionTests
        case=CorrectionTests();case.setUp();self.addCleanup(case.doCleanups)
        response=case.response();correction=case.root/'saved';correction.mkdir()
        (correction/nc.RESPONSE).write_bytes(response);(correction/nc.PACKETS).write_text(json.dumps({'packets':case.packets}))
        decision={'response_sha256':nc.sha(response),'evidence_sha256':nc.sha(case.original[ed.NOTES_EVIDENCE]),'packets_sha256':ed.digest(case.packets),
                  'decisions':[{'passage_id':case.loc,'decision':'accept','reason':'Synthetic evidence directly attributes the account.'}]}
        path=case.root/'decisions.private.json';path.write_text(json.dumps(decision))
        out=case.root/'reviewed'
        guard=case.root/'guard';guard.mkdir()
        (guard/'sitecustomize.py').write_text("import socket\ndef denied(*a,**k): raise AssertionError('No network allowed')\nsocket.socket.connect=denied\nsocket.create_connection=denied\n")
        p=subprocess.run([sys.executable,str(ROOT/'bin/apply_notes_corrections.py'),str(case.root),str(correction),str(path),str(out)],
                         env=dict(os.environ,PYTHONPATH=str(guard)),capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout)['applied_count'],1)
        self.assertFalse(json.loads(p.stdout)['candidate_selected']);case.unchanged()
        decision['decisions'][0]['passage_id']='issues/999/paragraphs/0'
        with self.assertRaisesRegex(ed.EditorialFailure,'invalid_correction_decision'):
            nc.render_version(case.root,case.root/'unknown',response,case.packets,decisions=decision)

    def test_correction_artifacts_enforce_hold_and_private_link_stripping(self):
        from meeting_postprocess.publication import publication_payload,strip_private_references
        for name in (nc.RESPONSE,nc.PACKETS,nc.REPORT,'correction-comparison.private.md'):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);(root/name).write_text('{}');(root/'summary.md').write_text('Private candidate')
                with self.assertRaisesRegex(ValueError,'hold'):publication_payload(root)
                self.assertNotIn(name,strip_private_references('Review [source]('+name+').'))

    def test_source_binding_and_cli_scope(self):
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{'MEETING_NOTES_TOKENIZER':str(ROOT/'tests/tokenizer.synthetic.json')}):
            specs=input_files([sys.executable,str(ROOT/'bin/ollama_meeting_summary.py'),td,'--meeting-notes','--meeting-notes-correct'])
        for role in ['editorial_code_notes_correction','editorial_code_editorial_quality','editorial_prompt_notes_correction_prompt']:
            self.assertTrue(specs[role]['required'])
        self.assertTrue(specs['editorial_prompt_notes_correction_prompt']['snapshot']);self.assertLessEqual(len(specs),64)
        import ollama_session_summary as engine
        with patch.object(sys,'argv',['summary','/synthetic/transcript','--meeting-notes-correct']):
            with self.assertRaises(SystemExit):engine.main()

    @unittest.skipUnless(sys.platform.startswith('linux'),'Bash wrapper behavior')
    def test_wrapper_automatic_default_and_explicit_disable(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'bin').mkdir();(root/'config').mkdir();(root/'input/chunks_out').mkdir(parents=True)
            (root/'input/chunks_out/transcript_chunks.jsonl').write_text('{}\n')
            (root/'bin/summarize-existing-meeting.sh').write_bytes((ROOT/'bin/summarize-existing-meeting.sh').read_bytes())
            (root/'bin/with-gpu-lock.sh').write_text('aihub_run_gpu_stage() { printf "%s\\n" "$@"; }\n')
            for config,cli,expected in [(None,[],True),('0',[],False),('1',[],True),('1',['--no-meeting-notes-correct'],False),('0',['--meeting-notes-correct'],True)]:
                (root/'config/.env').write_text('' if config is None else 'MEETING_NOTES_CORRECTION='+config+'\n')
                p=subprocess.run(['bash',str(root/'bin/summarize-existing-meeting.sh'),str(root/'input'),'--meeting-notes',*cli],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr);self.assertEqual('--meeting-notes-correct' in p.stdout.splitlines(),expected)
                self.assertIn('--meeting-notes-only',p.stdout)
            p=subprocess.run(['bash',str(root/'bin/summarize-existing-meeting.sh'),str(root/'input')],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);self.assertNotIn('--meeting-notes-correct',p.stdout)
            (root/'config/.env').write_text('MEETING_NOTES_CORRECTION=invalid\n')
            p=subprocess.run(['bash',str(root/'bin/summarize-existing-meeting.sh'),str(root/'input'),'--meeting-notes'],capture_output=True,text=True)
            self.assertEqual(p.returncode,64)

    @unittest.skipUnless(sys.platform.startswith('linux'),'Full unattended shell workflow')
    def test_unattended_processing_adds_one_correction_without_repeating_stages(self):
        from test_meeting_notes_workflow import MeetingNotesWorkflowTests
        case=MeetingNotesWorkflowTests();case.setUp();self.addCleanup(case.tearDown)
        config=case.project/'config/.env';config.write_text(config.read_text().replace('MEETING_NOTES_CORRECTION=0\n',''))
        case.env['NOTES_TEST_CORRECTION']='1'
        p=case.process();self.assertEqual(p.returncode,0,p.stderr)
        out=case.notes_dirs()[0];calls=[json.loads(line) for line in case.calls_file.read_text().splitlines()]
        reduced=[c for c in calls if 'Transcript chunk:' not in c['prompt']]
        self.assertEqual(len(reduced),4)
        self.assertEqual(sum('propose ONE private canonical' in c['prompt'] for c in calls),1)
        self.assertEqual(sum('summary-reduction call' in c['prompt'] for c in calls),1)
        self.assertEqual(sum('Perform one targeted factual-correction pass' in c['prompt'] for c in calls),1)
        self.assertIs(reduced[-1]['thinking'],False)
        self.assertTrue(all(c['thinking'] is None for c in reduced[:-1]))
        self.assertEqual(case.resources.read_text().splitlines(),['gpu0','gpu0+gpu1'])
        self.assertTrue(case.recording.is_file());self.assertTrue((out/'minutes-draft.md').is_file())
        candidate=out/nc.DIRECTORY/'candidate'
        self.assertEqual((out/ed.REGISTER).read_bytes(),(candidate/ed.REGISTER).read_bytes())
        self.assertIn('alleged',(out/'meeting-notes-draft.md').read_text())
        self.assertIn('Unverified correction candidate',(candidate/'meeting-notes-draft.md').read_text())
        self.assertNotIn('alleged',(candidate/'meeting-notes-draft.md').read_text())
        self.assertNotIn('PRIVATE_SECRET',json.dumps(calls))
