"""Exact source-range parsing and held full-document rendering; CPU fixtures only."""
import sys,pathlib,unittest,copy,json,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bin'))
from meeting_postprocess import editorial as ed
from tests.test_editorial_notes import fixture
class RangeTests(unittest.TestCase):
 def test_exact_expansion_with_split_chunk(self):
  records={f'16.2:L{i}':{} for i in range(1,5)};before=copy.deepcopy(records);self.assertEqual(ed.expand_notes_references(['16.2:L1-L3','16.2:L4'],records),list(records));self.assertEqual(before,records)
 def test_missing_malformed_ambiguous_and_unbounded_fail(self):
  records={'1:L1':{},'1:L2':{},'1:L4':{}}
  for refs in [[],None,['1:L1-L4'],['1:L2-L1'],['1:L1-2:L2'],['1:L0-L2'],['1:L1-L999999999'],['missing'],['1:L1','1:L1-L2'],['1:L1-L2','1:L2-L2']]:
   with self.subTest(refs=refs),self.assertRaises(ed.EditorialFailure):ed.expand_notes_references(refs,records)
 def test_notes_log_expansion_and_keep_holds(self):
  chunks,raw,notes=fixture();records=ed.source_records(chunks);notes['issues'][1]['paragraphs'][0]['source_ids']=['3:L9-L10'];changes=[];clean=ed.validate_notes(notes,records,'\n'.join(c['text'] for c in chunks),reference_changes=changes);self.assertEqual(clean['issues'][1]['paragraphs'][0]['source_ids'],['3:L9','3:L10']);self.assertEqual(changes[0]['code'],'notes_reference_range_expanded')
 def test_register_does_not_accept_compact_ranges(self):
  chunks,raw,notes=fixture();records=ed.source_records(chunks)
  with self.assertRaises(ed.EditorialFailure):ed._refs(['3:L1-L2'],records)
 def test_sensitive_notes_still_block(self):
  chunks,raw,notes=fixture();records=ed.source_records(chunks);notes['highlights'][0]={'text':'Review the employee medical history.','source_ids':['3:L1-L2']}
  with self.assertRaises(ed.EditorialFailure):ed.validate_notes(notes,records,'')
class FullRangeWorkflowTests(unittest.TestCase):
 def test_full_documents_and_private_hold_from_ranged_notes(self):
  chunks,raw,notes=fixture();notes['issues'][1]['paragraphs'][0]['source_ids']=['3:L9-L10'];calls=[]
  def generate(prompt,**kwargs):
   calls.append(prompt)
   return [json.dumps(raw),json.dumps(notes),'# Detailed Minutes\nThe staffing plan and safety issues were discussed.'][len(calls)-1]
  class Budget:
   measurements=[]
   def measure(self,*args):return {'fits':True,'reserved_output':100}
  with tempfile.TemporaryDirectory() as td:
   directory=pathlib.Path(td)
   status=ed.run(directory,chunks,'summary','current','',[],[],{},[],[],ROOT/'prompts/meeting',generate,False,budget=Budget())
   self.assertEqual(status,0);self.assertEqual(len(calls),3)
   for name in ['summary.md','action-items.md','minutes-draft.md','minutes-qa.json','minutes-qa.md',ed.REVIEW,ed.REGISTER,ed.REGISTER_MD]:self.assertTrue((directory/name).stat().st_size>0,name)
   self.assertEqual(json.loads((directory/ed.REVIEW).read_text())['status'],'review_hold')
   evidence=json.loads((directory/ed.NOTES_EVIDENCE).read_text());self.assertTrue(any(x['code']=='notes_reference_range_expanded' for x in evidence['notes']['reference_findings']))
if __name__=='__main__':unittest.main()