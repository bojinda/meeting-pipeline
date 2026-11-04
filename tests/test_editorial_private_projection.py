"""Private-sensitive notes projection; raw evidence and hard guards stay intact."""
import sys,pathlib,unittest,copy,json,tempfile,contextlib,io
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bin'))
from meeting_postprocess import editorial as ed
from tests.test_editorial_notes import fixture
P=ROOT
class PrivacyProjectionTests(unittest.TestCase):
 def test_private_paragraph_preserved_and_removed_from_view(self):
  chunks,raw,notes=fixture();before=copy.deepcopy(notes);sensitive={'text':'The participant discussed a medical history.','source_ids':['3:L1']};notes['issues'][0]['paragraphs'].append(sensitive);original=copy.deepcopy(notes);selected,changes=ed.triage_notes(notes);self.assertEqual(notes,original);self.assertNotIn(sensitive,selected['issues'][0]['paragraphs']);self.assertEqual(selected['concerns'][-1]['text'],sensitive['text']);self.assertEqual(changes[0]['code'],'notes_confidential_block_withheld');ed.validate_notes(selected,ed.source_records(chunks),'')
 def test_direct_guard_remains_blocking(self):
  with self.assertRaises(ed.EditorialFailure):ed._plain('The medical history was discussed.',identity_review=True)
 def test_all_private_business_still_refuses(self):
  chunks,raw,notes=fixture();notes['issues']=[{'heading':'Private','paragraphs':[{'text':'The medical history was discussed.','source_ids':['3:L1']}]}];selected,_=ed.triage_notes(notes)
  with self.assertRaisesRegex(ed.EditorialFailure,'confidential_notes_content'):ed.validate_notes(selected,ed.source_records(chunks),'')
 def test_private_unknown_references_still_block(self):
  chunks,raw,notes=fixture();notes['issues'][0]['paragraphs'].append({'text':'A medical history was discussed.','source_ids':['invented']});selected,_=ed.triage_notes(notes)
  with self.assertRaises(ed.EditorialFailure):ed.validate_notes(selected,ed.source_records(chunks),'')
 def test_full_held_generation_has_no_sensitive_notes_prose(self):
  chunks,raw,notes=fixture();notes['issues'][0]['paragraphs'].append({'text':'The medical history was discussed.','source_ids':['3:L1']});calls=[]
  def generate(prompt,**kwargs):calls.append(prompt);return [json.dumps(raw),json.dumps(notes),'# Detailed Minutes\nDiscussion recorded.'][len(calls)-1]
  class Budget:
   measurements=[]
   def measure(self,*a):return {'fits':True,'reserved_output':100}
  with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
   D=pathlib.Path(d);status=ed.run(D,chunks,'summary','current','',[],[],{},[],[],P/'prompts/meeting',generate,False,budget=Budget());self.assertEqual(status,0);self.assertNotIn('medical history',(D/'meeting-notes-draft.md').read_text());self.assertIn('medical history',(D/ed.NOTES_EVIDENCE).read_text());self.assertIn('medical history',(D/ed.RESPONSE).read_text());self.assertEqual(json.loads((D/ed.REVIEW).read_text())['status'],'review_hold')

if __name__=="__main__":unittest.main()
