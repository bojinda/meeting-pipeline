#!/usr/bin/env python3
"""Complete missing reductions from explicitly bound, completed editorial replies.

Run under with-gpu-lock.sh in a new stage. --preflight makes no network calls
and deliberately stops before the missing detailed response; it is not success.
"""
import argparse, hashlib, json, os
from pathlib import Path
import ollama_session_summary as engine
from meeting_postprocess import editorial as ed, editorial_checkpoint as cp
from meeting_postprocess.gpu_admission import approved_input
from meeting_postprocess.commitments import commitment_evidence, source_context_evidence
from meeting_postprocess.qa import Finding
from aihub_gpu_runner.config import admission_from_config
from aihub_gpu_runner.core import encode

def sha(data): return hashlib.sha256(data).hexdigest()
def need(ok, message):
    if not ok: raise ed.EditorialFailure(message)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    def selected(role,path):
        if args.preflight:return Path(path)
        managed,snapshot=approved_input(role,path)
        need(managed and snapshot is not None,'completion_snapshot_missing')
        return Path(snapshot)
    manifest=json.loads(selected('completion_manifest',args.manifest).read_text())
    need(not args.output.exists(),'fresh_completion_output_required')
    data={}
    for role,item in manifest['files'].items():
        raw=selected(role,item['path']).read_bytes()
        need(sha(raw)==item['sha256'],'completion_source_changed')
        need(sha(Path(item['path']).read_bytes())==item['sha256'],'completion_original_changed')
        data[role]=raw
    record=json.loads(data['editorial_checkpoint']);saved=json.loads(data['completion_response'])
    chunks=[json.loads(x) for x in data['editorial_sections'].decode().splitlines() if x.strip()]
    maps=[json.loads(x) for x in data['editorial_maps'].decode().splitlines() if x.strip()]
    records=ed.source_records(chunks)
    need(ed.digest(records)==saved['source_hash'] and saved['register_assessment']['sources']==records,'completion_source_binding_mismatch')
    need(cp.digest(chunks)==record['prepared_source_hash'],'completion_preparation_changed')
    need(cp.code_identity(engine)==record['map_code'],'completion_map_code_changed')
    cp.validate_maps(chunks,record['map_sources'],maps)
    need(sha(data['editorial_maps'])==record['maps_file_sha256'] and sha(data['editorial_sections'])==record['sections_file_sha256'],'completion_maps_changed')
    need(set(saved['responses'])=={'register','notes'},'only_missing_detail_completion_supported')
    _,targets,admission=admission_from_config(os.environ['AIHUB_GPU_RUNNER_CONFIG'])
    original=admission.store.job(manifest['source_stage'])
    need(sha(encode(original))==manifest['source_job_sha256'],'completion_source_job_changed')
    need(original['state']=='failed' and original.get('cleanup_verified') is True and len(original.get('host_requests',[]))==2 and all(r['state']=='completed' for r in original['host_requests']),'completion_source_work_unresolved')
    need(not any(o['job_id']==manifest['source_stage'] for o in admission.snapshot()['owners'].values()),'completion_source_still_owned')
    if not args.preflight:cp.require_fresh_stage(record,sha(data['editorial_checkpoint']))
    config=record['configuration'];tokenizer=selected('editorial_tokenizer',manifest['tokenizer'])
    need(sha(tokenizer.read_bytes())==config['tokenizer_sha256'],'completion_tokenizer_changed')
    prompts=cp.StagePrompts(Path(__file__).resolve().parents[1]/'prompts/meeting')
    system=(prompts/'reduce_system.txt').read_text().strip()
    diagnostics=json.loads(data['completion_diagnostics'])['requests']
    need([r['stage'] for r in diagnostics]==['register','notes'],'completion_diagnostics_incomplete')
    source='\n'.join(c['text'] for c in chunks if c['meeting_section'] in {ed.BUSINESS,ed.ADJOURNMENT})
    blocked=set(record['blocked_context_source_ids']);calls=[]
    def generate(prompt,*,structured=False,num_predict,response_callback=None):
        phase=['register','notes','detailed','recap'][len(calls)];calls.append(phase)
        selected_system=system+('\nReturn only the requested JSON; no Markdown wrapper.' if structured else '')
        if phase in {'register','notes'}:
            index=['register','notes'].index(phase)
            payload={'model':config['reduce_model'],'prompt':prompt,'system':selected_system,'stream':False,'keep_alive':config['keep_alive'],'options':{'temperature':config['temperature'],'num_ctx':config['reduce_num_ctx'],'num_predict':num_predict},'format':'json'}
            need(sha(encode(payload))==original['host_requests'][index]['request_hash'],'completion_request_binding_mismatch')
            info=diagnostics[index]
            need(info.get('done') is True and not info.get('failure_category') and not info.get('empty_final_answer'),'completion_reply_unverified')
            if response_callback:response_callback({**{k:v for k,v in info.items() if k!='stage'},'reused_completed_response_from_stage':manifest['source_stage']})
            print('[completion] Reused verified completed '+phase+' response; no inference',flush=True)
            return saved['responses'][phase]
        if args.preflight:raise ed.EditorialFailure('preflight_missing_detail_not_executed')
        return engine.call_ollama(config['ollama_url'],config['reduce_model'],prompt,system=selected_system,keep_alive=config['keep_alive'],temperature=config['temperature'],num_ctx=config['reduce_num_ctx'],num_predict=num_predict,response_callback=response_callback)
    args.output.mkdir(mode=0o700)
    for role,name in [('editorial_sections','meeting_sections.jsonl'),('editorial_maps','chunk_summaries.jsonl'),('editorial_checkpoint',cp.FILE)]:
        with (args.output/name).open('xb') as f:f.write(data[role])
    status=ed.run(args.output,chunks,engine.build_reduce_input([m for m in maps if m['meeting_section'] in {ed.RECAP,ed.BUSINESS,ed.ADJOURNMENT}]),engine.build_reduce_input([m for m in maps if m['meeting_section'] in {ed.BUSINESS,ed.ADJOURNMENT}]),engine.build_reduce_input([m for m in maps if m['meeting_section']==ed.RECAP]),commitment_evidence(chunks,include_context=True,blocked_context_source_ids=blocked),source_context_evidence(chunks,blocked),record['aliases'],record['approved_passages'],[Finding(**w) for w in record['warnings']],prompts,generate,config['keep_recap'],budget=ed.RequestBudget(str(tokenizer),config['reduce_num_ctx'],system),notes_only=False)
    ed.write_private_json(args.output/'completion-provenance.private.json',{'source_stage':manifest['source_stage'],'source_manifest_sha256':sha(args.manifest.read_bytes()),'reused_responses':['register','notes'],'new_inference_stages':[p for p in calls if p not in {'register','notes'}] if not args.preflight else [],'preflight':args.preflight,'result':status})
    return status

if __name__=='__main__':raise SystemExit(main())
