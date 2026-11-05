"""One targeted correction request; independent edit decisions and held versions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from . import editorial as ed
from .editorial_quality import blocks, factual_risks, _literal

DIRECTORY = 'notes-correction'
RESPONSE = 'correction-response.private.json'
REPORT = 'correction-review.private.json'
PACKETS = 'correction-packets.private.json'
FIELDS = 'passage_id replacement_text source_ids explanation unresolved_questions evidence_insufficient support'
MAX_PACKETS = 24


def sha(data): return hashlib.sha256(data).hexdigest()


def plain_notes(notes):
    def block(b): return {k:deepcopy(b[k]) for k in ('text','source_ids')}
    return {**{k:[block(b) for b in notes[k]] for k in ('highlights','previous_context','motions','unresolved')},
            'issues':[{'heading':i['heading'],'paragraphs':[block(b) for b in i['paragraphs']]} for i in notes['issues']],
            'concerns':deepcopy(notes['concerns'])}


def make_packets(notes, records, findings, selections=None):
    """Whole cited turns plus six neighboring lines; omissions remain explicit."""
    candidates=[];lookup=dict(blocks(notes))
    if selections is None:
        selections={}
        for loc,b in lookup.items():
            src='\n'.join(records[k]['text'] for k in b['source_ids'])
            risks=[instruction for _,instruction in factual_risks(b['text'],src)]
            risks += [f.get('instruction',f['code']) for f in findings if f.get('notes_location')==loc and
                      f['code'] in {'allegation_status_review','role_attribution_review','numerical_basis_review','chronology_review','confidentiality'}]
            if risks:selections[loc]={'questions':list(dict.fromkeys(risks))}
    for loc,selection in selections.items():
        if loc not in lookup:raise ed.EditorialFailure('correction_unknown_target')
        b=lookup[loc];context=set()
        for key in b['source_ids']:
            chunk,line=key.rsplit(':L',1)
            context.update(f'{chunk}:L{n}' for n in range(max(1,int(line)-6),int(line)+7))
        context.update(selection.get('extra_source_ids',[]))
        available=[key for key,row in records.items() if key in context and len(row['text'])<=ed.LOCAL_CHARS]
        candidates.append({'passage_id':loc,'original_text':b['text'],'original_source_ids':b['source_ids'],
                           'context_source_ids':available,'questions':selection['questions'],
                           'omitted_source_ids':[key for key,row in records.items() if key in context and len(row['text'])>ed.LOCAL_CHARS]})
    return candidates[:MAX_PACKETS], [r['passage_id'] for r in candidates[MAX_PACKETS:]]


def build_prompt(prompts, packets, records, concerns):
    ids={key for packet in packets for key in packet['context_source_ids']}
    payload={'packets':packets,'source_excerpts':[{k:r[k] for k in ('id','section','text')} for key,r in records.items() if key in ids],
             'private_concerns':[c for c in concerns if set(c['source_ids'])&ids],
             'policy':'Private concerns and speaker labels are unverified evidence, not approved identities. No register or publication approval.'}
    return (prompts/'notes_correction_prompt.txt').read_text(encoding='utf-8')+'\n\nUntrusted input JSON:\n'+json.dumps(payload,ensure_ascii=False,separators=(',',':'))


def assess(raw, packets, original, records):
    """Per-edit hard guards; valid IDs/quotes do NOT certify semantic accuracy."""
    if not ed._object(raw,'corrections') or not ed._list(raw['corrections']):
        raise ed.EditorialFailure('invalid_correction_response')
    targets={p['passage_id']:p for p in packets};old=dict(blocks(original))
    counts={}
    for row in raw['corrections']:
        if isinstance(row,dict) and isinstance(row.get('passage_id'),str):
            counts[row['passage_id']]=counts.get(row['passage_id'],0)+1
    results=[]
    for index,row in enumerate(raw['corrections']):
        result={'proposal_index':index,'status':'rejected','reason':'invalid_correction_schema'}
        if isinstance(row,dict) and isinstance(row.get('passage_id'),str):result['passage_id']=row['passage_id']
        try:
            if (not (ed._object(row,FIELDS) or ed._object(row,FIELDS+' disposition')) or row.get('passage_id') not in targets
                    or counts.get(row['passage_id'])!=1 or type(row['evidence_insufficient']) is not bool
                    or not ed._text(row['explanation'],limit=3000) or not ed._list(row['unresolved_questions'])
                    or any(not ed._text(q,limit=2000) for q in row['unresolved_questions'])
                    or not ed._list(row['source_ids']) or not ed._list(row['support'])):
                raise ed.EditorialFailure('invalid_correction_schema')
            disposition=row.get('disposition','insufficient_evidence' if row['evidence_insufficient'] else 'edit')
            if disposition not in ('edit','no_change','insufficient_evidence') or (disposition=='insufficient_evidence')!=row['evidence_insufficient']:
                raise ed.EditorialFailure('invalid_correction_disposition')
            if row['evidence_insufficient']:
                if row['replacement_text'] is not None:raise ed.EditorialFailure('insufficient_evidence_replacement')
            else:ed._plain(row['replacement_text'],identity_review=True)
            refs=row['source_ids'];allowed=set(targets[row['passage_id']]['context_source_ids'])
            if (not refs and not row['evidence_insufficient'] or any(not isinstance(k,str) or k not in allowed for k in refs)
                    or len(set(refs))!=len(refs)):
                raise ed.EditorialFailure('invalid_correction_references')
            quoted=[]
            for item in row['support']:
                if (not ed._object(item,'source_id quote') or item['source_id'] not in refs
                        or not ed._text(item['quote'],limit=ed.LOCAL_CHARS)
                        or len(item['quote'])<12 or item['quote'] not in records[item['source_id']]['text']):
                    raise ed.EditorialFailure('correction_quote_not_exact')
                quoted.append(item['source_id'])
            if set(quoted)!=set(refs):raise ed.EditorialFailure('correction_unquoted_reference')
            if row['evidence_insufficient']:
                result.update(status='insufficient_evidence',reason='retain_original_with_review_flag')
                results.append(result);continue
            if disposition=='no_change' and row['replacement_text']!=old[row['passage_id']]['text']:
                raise ed.EditorialFailure('no_change_contains_edit')
            source='\n'.join(records[k]['text'] for k in refs)
            # Reject new numerical assertions absent even lexically from the
            # supplied evidence. Spelled-number equivalence needs source review.
            numbers=lambda s:set(re.findall(r'\b\d+(?:[.,:/–-]\d+)*%?',s))
            invented=numbers(row['replacement_text'])-numbers(old[row['passage_id']]['text'])-numbers(source)
            if invented:raise ed.EditorialFailure('correction_new_unsupported_figure')
            for name in re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',row['replacement_text']):
                if name not in old[row['passage_id']]['text'] and name not in source:
                    raise ed.EditorialFailure('correction_new_unsupported_name_or_title')
            candidate=plain_notes(original);replace_block(candidate,row['passage_id'],row['replacement_text'],refs)
            ed.validate_notes(candidate,records,'\n'.join(r['text'] for r in records.values()))
            risk=[{'code':code,'instruction':instruction} for code,instruction in factual_risks(row['replacement_text'],source)]
            if row['replacement_text']==old[row['passage_id']]['text']:
                result.update(status='justified_no_change_unverified' if disposition=='no_change' else 'unchanged',
                              reason='model_source_rationale_requires_review' if disposition=='no_change' else 'no_text_change_review_remains',risk_findings=risk)
            elif re.findall(r'\w+',row['replacement_text'].casefold())==re.findall(r'\w+',old[row['passage_id']]['text'].casefold()):
                result.update(status='non_substantive_change',reason='formatting_only_review_remains',risk_findings=risk)
            else:
                result.update(status='eligible_unverified',reason='independent_semantic_review_required',risk_findings=risk)
        except (KeyError,TypeError,ValueError) as exc:
            result['reason']=str(exc) if isinstance(exc,ed.EditorialFailure) else 'invalid_correction_schema'
        results.append(result)
    present={r.get('passage_id') for r in results}
    for loc in targets.keys()-present:results.append({'passage_id':loc,'status':'missing','reason':'model_omitted_target'})
    return results


def replace_block(notes, location, text, refs):
    parts=location.split('/');parent=notes
    for part in parts[:-1]:parent=parent[int(part)] if isinstance(parent,list) else parent[part]
    parent[int(parts[-1])]={'text':text,'source_ids':deepcopy(refs)}


def render_version(parent, output, response, packets, *, decisions=None):
    """Apply eligible edits to an unverified candidate, or explicit reviewed edits.

    decisions is bound to response and original evidence hashes; it authorizes
    private text changes only, never actions, identities or publication.
    """
    parent,output=Path(parent),Path(output)
    if output.exists() or output.is_symlink():raise ed.EditorialFailure('correction_fresh_output_required')
    evidence_bytes=(parent/ed.NOTES_EVIDENCE).read_bytes();evidence=json.loads(evidence_bytes)
    original,records=evidence['notes'],evidence['sources'];register_bytes=(parent/ed.REGISTER).read_bytes();register=json.loads(register_bytes)
    if evidence['source_hash']!=ed.digest(records) or register['source_hash']!=evidence['source_hash']:
        raise ed.EditorialFailure('correction_source_binding_mismatch')
    raw=json.loads(response);assessment=assess(raw,packets,original,records)
    selected={};mode='automated_unverified_candidate'
    if decisions is not None:
        if (not ed._object(decisions,'response_sha256 evidence_sha256 packets_sha256 decisions') or decisions['response_sha256']!=sha(response)
                or decisions['evidence_sha256']!=sha(evidence_bytes) or decisions['packets_sha256']!=ed.digest(packets)
                or not ed._list(decisions['decisions'])):
            raise ed.EditorialFailure('correction_review_binding_mismatch')
        for decision in decisions['decisions']:
            if (not ed._object(decision,'passage_id decision reason') or decision['decision'] not in ('accept','reject')
                    or not ed._text(decision['reason'],limit=6000) or not isinstance(decision['passage_id'],str)
                    or decision['passage_id'] not in {p['passage_id'] for p in packets}
                    or decision['passage_id'] in selected):
                raise ed.EditorialFailure('invalid_correction_decision')
            selected[decision['passage_id']]=decision
        mode='independently_reviewed_private_edits'
    notes=plain_notes(original);applied=[]
    for item in assessment:
        loc=item.get('passage_id');decision=selected.get(loc)
        if decision:item['independent_review']=decision
        if item['status']=='eligible_unverified' and (decisions is None or decision and decision['decision']=='accept'):
            row=raw['corrections'][item['proposal_index']]
            replace_block(notes,loc,row['replacement_text'],row['source_ids']);applied.append(loc)
        elif decision and decision['decision']=='accept':
            raise ed.EditorialFailure('cannot_accept_invalid_correction')
    # Report model proposals separately from independent private edit decisions.
    categories={'eligible_unverified':'meaningful_edit_candidate','justified_no_change_unverified':'justified_no_change',
                'insufficient_evidence':'insufficient_evidence','rejected':'rejected','unchanged':'unexplained_unchanged',
                'non_substantive_change':'non_substantive_change','missing':'missing'}
    for item in assessment:
        item['category']=categories[item['status']]
        item['outcome']='rejected' if item.get('independent_review',{}).get('decision')=='reject' else item['category']
    outcome_counts={key:sum(a['outcome']==key for a in assessment) for key in dict.fromkeys(categories.values())}
    source='\n'.join(r['text'] for r in records.values() if r['section'] in {ed.BUSINESS,ed.ADJOURNMENT})
    clean=ed.validate_notes(notes,records,source)
    output.mkdir(mode=0o700)
    review=ed.finish(output,register,clean,None,source,records=records)
    # Visible private status follows the version when opened independently.
    label = ('Unverified correction candidate. Independent semantic and privacy review required.'
             if decisions is None else 'Private correction review copy. Final human approval remains required.')
    banner = '> ' + label + ' Publication hold; original draft remains selected.\n\n'
    for name in ('meeting-notes-draft.md','summary.md'):
        path=output/name
        path.write_text(banner+path.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
    review['notes_word_count']=len((output/'meeting-notes-draft.md').read_text(encoding='utf-8').split())
    review['notes_quality']['document_words']=review['notes_word_count']
    for name in (ed.REGISTER,ed.REGISTER_MD):
        if (parent/name).exists():(output/name).write_bytes((parent/name).read_bytes())
    report={'status':'review_hold','mode':mode,'candidate_selected':False,'response_sha256':sha(response),'evidence_sha256':sha(evidence_bytes),
            'packets_sha256':ed.digest(packets),'unchanged_count':sum(a['status']=='unchanged' for a in assessment),
            'outcome_counts':outcome_counts,'automated_checks_certify_semantic_accuracy':False,
            'proposed_count':len(raw['corrections']),'applied_passages':applied,'applied_count':len(applied),
            'unapplied_count':len(raw['corrections'])-len(applied),'assessment':assessment,'word_count':review['notes_word_count'],
            'semantic_accuracy':'Independent source review required; valid references and exact quotes are not proof.',
            'original_concerns_preserved':clean['concerns']==original['concerns']}
    ed.write_private_json(output/REPORT,report);(output/RESPONSE).write_bytes(response)
    ed.write_private_json(output/PACKETS,packets)
    review['findings'].append({'code':'correction_semantic_review_required','instruction':report['semantic_accuracy'],'evidence_pointer':REPORT})
    review['correction_mode']=mode;review['candidate_selected']=False;ed.write_private_json(output/ed.REVIEW,review)
    lines=['# Private factual-correction comparison','','Review hold. Unverified proposals; no distribution or action approval.','']
    lookup=dict(blocks(original))
    for item in assessment:
        loc=item.get('passage_id','unknown');lines += ['## '+loc,'', 'Outcome: '+item['outcome']+'; status: '+item['status']+'; '+item['reason'],'']
        if loc in lookup:lines+=['### Original','',_literal(lookup[loc]['text']),'']
        if 'proposal_index' not in item:continue
        row=raw['corrections'][item['proposal_index']]
        if not isinstance(row,dict):
            lines+=['### Invalid proposal (not applied)','',_literal(row),''];continue
        lines+=['### Proposed replacement','',_literal(row.get('replacement_text')),'',
                '### Correction explanation and unresolved questions','',_literal(row.get('explanation')),'',_literal(row.get('unresolved_questions')),'']
        if item.get('independent_review'):lines+=['### Independent edit decision','',_literal(item['independent_review']),'']
        lines+=['### Exact source evidence','']
        for key in row['source_ids'] if isinstance(row.get('source_ids'),list) else []:
            if isinstance(key,str) and key in records:lines += [_literal(key+'\n'+records[key]['text']),'']
    ed.private_text(output/'correction-comparison.private.md','\n'.join(lines))
    with (output/ed.CHECKLIST).open('a',encoding='utf-8') as stream:
        stream.write('\n- [ ] Review [correction comparison](correction-comparison.private.md) and [edit decisions]('+REPORT+'). Valid source IDs/quotes do not prove semantic accuracy. The original remains selected.\n')
    for p in output.iterdir():
        if p.is_file():p.chmod(0o600)
    return report


def run_once(directory, prompts, generate, budget, *, selections=None):
    """One request at most, never retries, rewrites originals or publishes."""
    parent=Path(directory);out=parent/DIRECTORY
    if out.exists() or out.is_symlink():return {'status':'review_hold','result':'already_attempted','request_attempts':0}
    protected={p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()}
    out.mkdir(mode=0o700);result={'status':'review_hold','result':'failed','request_attempts':0,'candidate_selected':False}
    ed.write_private_json(out/ed.REVIEW,{'status':'review_hold','phase':'correction_incomplete'})
    try:
        evidence=json.loads(protected[ed.NOTES_EVIDENCE]);prior=json.loads(protected[ed.REVIEW])
        if prior['status']!='review_hold' or evidence['source_hash']!=ed.digest(evidence['sources']):
            raise ed.EditorialFailure('correction_original_binding_mismatch')
        packets,unselected=make_packets(evidence['notes'],evidence['sources'],prior['findings'],selections)
        ed.write_private_json(out/PACKETS,{'packets':packets,'unselected_passages':unselected})
        if not packets:result['result']='no_questionable_passages';return result
        prompt=build_prompt(prompts,packets,evidence['sources'],evidence['notes']['concerns'])
        ed.private_text(out/'request.private.txt',prompt);measured=budget.measure('notes',prompt,True)
        ed.write_private_json(out/ed.BUDGETS,{'requests':[measured]})
        if not measured['fits']:raise ed.EditorialFailure('correction_context_exceeded')
        def callback(info):ed.write_private_json(out/ed.DIAGNOSTICS,{'requests':[{'stage':'correction',**info}]})
        result['request_attempts']=1
        answer=generate(prompt,structured=True,num_predict=measured['reserved_output'],response_callback=callback,thinking=False)
        if not isinstance(answer,str):raise ed.EditorialFailure('ollama_invalid_response')
        raw=answer.encode('utf-8');(out/RESPONSE).write_bytes(raw)
        if not answer.strip():raise ed.EditorialFailure('ollama_empty_final_answer')
        report=render_version(parent,out/'candidate',raw,packets)
        result.update(result='unverified_candidate',proposed_count=report['proposed_count'],applied_count=report['applied_count'],unchanged_count=report['unchanged_count'],
                      word_count=report['word_count'],unselected_passages=unselected)
    except Exception as exc:
        from .ollama_response import GenerationFailure
        result['failure_category']=str(exc) if isinstance(exc,(ed.EditorialFailure,GenerationFailure)) else 'correction_invalid_response_or_internal_failure'
    finally:
        result['original_files_unchanged']=all((parent/k).read_bytes()==v for k,v in protected.items())
        if not result['original_files_unchanged']:result.update(result='failed',failure_category='correction_original_drift')
        if result['result']=='failed' and not (out/ed.DIAGNOSTICS).exists():
            ed.write_private_json(out/ed.DIAGNOSTICS,{'requests':[{'stage':'correction','failure_category':result.get('failure_category'),
                                  'request_attempts':result['request_attempts'],'backend_metadata_available':False}]})
        ed.write_private_json(out/REPORT,result)
        for p in out.iterdir():
            if p.is_file():p.chmod(0o600)
    return result
