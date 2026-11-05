"""One optional source-bound editorial request; never replaces the first draft."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from . import editorial as ed
from .editorial_quality import blocks

PUBLIC = ('highlights', 'previous_context', 'issues', 'motions', 'unresolved')
DIRECTORY = 'notes-refinement'
REPORT = 'refinement-review.private.json'
QUALIFIED = re.compile(r'\b(?:report\w*|alleg\w*|unconfirm\w*|uncertain\w*|participant\w*|interpret\w*|propos\w*|tentative\w*|hearsay|suspect\w*|may|might|could)\b', re.I)
NUMBERS = re.compile(r'\b\d+(?:[.,:/–-]\d+)*%?')


def plain_notes(notes):
    """Strip validation annotations, never rewrite original prose/evidence."""
    def block(b): return {k: deepcopy(b[k]) for k in ('text', 'source_ids')}
    return {**{k: [block(b) for b in notes[k]] for k in PUBLIC if k != 'issues'},
            'issues': [{'heading': i['heading'], 'paragraphs': [block(b) for b in i['paragraphs']]} for i in notes['issues']],
            'concerns': deepcopy(notes['concerns'])}


def supplied_records(notes, records):
    # Exact cited lines and nearby context help distinguish separate incidents.
    wanted = {key for _, b in blocks(notes) for key in b['source_ids']}
    wanted.update(key for c in notes['concerns'] for key in c['source_ids'])
    for key in list(wanted):
        chunk, line = key.rsplit(':L', 1)
        wanted.update(f'{chunk}:L{n}' for n in range(max(1, int(line)-3), int(line)+4))
    return {k: r for k,r in records.items() if k in wanted and len(r['text']) <= ed.LOCAL_CHARS}


def build_prompt(prompts, notes, records, findings):
    selected = supplied_records(notes, records)
    payload = {'original_notes': plain_notes(notes),
               'original_blocks': [{'id': loc, **{k:b[k] for k in ('text','source_ids')}} for loc,b in blocks(notes)],
               'source_excerpts': [{k:r[k] for k in ('id','section','text')} for r in selected.values()],
               'omitted_oversized_source_ids': [k for k in records if len(records[k]['text']) > ed.LOCAL_CHARS],
               'validation_concerns': findings,
               'action_policy': 'All register proposals remain private and unapproved. No action-register generation or assignment approval.'}
    return (prompts / 'notes_refinement_prompt.txt').read_text(encoding='utf-8') + '\n\nUntrusted input JSON:\n' + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))


def validate(raw, original, records, source):
    if not ed._object(raw, 'notes coverage review_concerns') or not ed._object(raw['notes'], ' '.join(PUBLIC)):
        raise ed.EditorialFailure('invalid_refinement_schema')
    if not ed._list(raw['review_concerns']) or not ed._list(raw['coverage']):
        raise ed.EditorialFailure('invalid_refinement_schema')
    changes = []
    proposed = {**raw['notes'], 'concerns': deepcopy(original['concerns']) + raw['review_concerns']}
    # Confidential prose hard-fails this optional candidate. Keep the valid original.
    clean = ed.validate_notes(proposed, supplied_records(original, records), source, reference_changes=changes)
    before, after = dict(blocks(original)), dict(blocks(clean))
    seen, used, findings = set(), set(), []
    for row in raw['coverage']:
        if (not ed._object(row, 'original_block revised_blocks disposition explanation')
                or not isinstance(row['original_block'], str) or row['original_block'] not in before
                or row['original_block'] in seen or not ed._list(row['revised_blocks'])
                or any(not isinstance(k,str) or k not in after for k in row['revised_blocks'])
                or len(set(row['revised_blocks'])) != len(row['revised_blocks'])
                or row['disposition'] not in ('retained','consolidated','privacy_withheld','source_correction')
                or not ed._text(row['explanation'], limit=2000)):
            raise ed.EditorialFailure('invalid_refinement_coverage')
        seen.add(row['original_block']); used.update(row['revised_blocks'])
        if not row['revised_blocks'] and row['disposition'] not in ('privacy_withheld','source_correction'):
            raise ed.EditorialFailure('unaccounted_refinement_omission')
        old = before[row['original_block']]
        new = ' '.join(after[k]['text'] for k in row['revised_blocks'])
        def flag(code, **extra):
            findings.append({'code':code, **deepcopy(row), 'source_ids':old['source_ids'], **extra})
        if row['disposition'] in ('privacy_withheld','source_correction'):
            flag('refinement_material_change_review')
        lost = sorted(set(NUMBERS.findall(old['text'])) - set(NUMBERS.findall(new)))
        if lost: flag('refinement_figure_coverage_review', missing_figures=lost)
        if QUALIFIED.search(old['text']) and not QUALIFIED.search(new):
            flag('refinement_qualification_review')
        cited = {ref for k in row['revised_blocks'] for ref in after[k]['source_ids']}
        if cited and not cited.intersection(old['source_ids']):
            flag('refinement_source_change_review')
    if seen != set(before) or used != set(after):
        raise ed.EditorialFailure('incomplete_refinement_coverage')
    for location,block in after.items():
        origins = [r['original_block'] for r in raw['coverage'] if location in r['revised_blocks']]
        if len(origins) > 1:
            findings.append({'code':'refinement_combined_accounts_review','notes_location':location,'original_blocks':origins,
                             'instruction':'Check that distinct incidents, actors, chronology, disagreements and qualifications were not silently merged.'})
        evidence = ' '.join(records[k]['text'] for k in block['source_ids'])
        new_numbers = sorted(set(NUMBERS.findall(block['text'])) - set(NUMBERS.findall(evidence)))
        if new_numbers:
            findings.append({'code':'refinement_figure_source_review','notes_location':location,'figures':new_numbers,
                             'instruction':'Verify figures and their units/bases against exact source; numeric matching alone cannot prove support.'})
    findings.append({'code':'refinement_semantic_coverage_review','instruction':'Coverage mappings are model claims, not proof. Compare every original major issue, figure, outcome, hazard, allegation, legal interpretation and action status with the source before acceptance. No identities or actions are approved.'})
    return clean, findings


def run_once(original_directory, prompts, generate, budget):
    """Default callers opt in. One attempted request, private result, no retry."""
    parent = Path(original_directory)
    out = parent / DIRECTORY
    if out.exists() or out.is_symlink():
        return {'status':'review_hold','result':'already_attempted','candidate_selected':False}
    # Freeze bytes before creating any refinement artifacts. Original source is read-only.
    protected = {p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()}
    out.mkdir(mode=0o700)
    result = {'status':'review_hold','result':'failed','candidate_selected':False,'request_attempts':0,
              'original_hashes':{k:hashlib.sha256(v).hexdigest() for k,v in protected.items()}}
    ed.write_private_json(out / ed.REVIEW, {'status':'review_hold','phase':'refinement_incomplete'})
    try:
        evidence = json.loads(protected[ed.NOTES_EVIDENCE]); original = evidence['notes']; records = evidence['sources']
        register = json.loads(protected[ed.REGISTER]); prior_review = json.loads(protected[ed.REVIEW])
        if (evidence['source_hash'] != ed.digest(records) or register['source_hash'] != evidence['source_hash']
                or prior_review['status'] != 'review_hold' or prior_review['source_hash'] != evidence['source_hash']):
            raise ed.EditorialFailure('refinement_original_binding_mismatch')
        source = '\n'.join(r['text'] for r in records.values() if r['section'] in {ed.BUSINESS,ed.ADJOURNMENT})
        ed.validate_notes(plain_notes(original), records, source)
        prompt = build_prompt(prompts,original,records,prior_review['findings'])
        measured = budget.measure('notes',prompt,True)  # Same bounded Qwen output capacity; no token cutoff target.
        ed.write_private_json(out / ed.BUDGETS, {'requests':[measured]})
        ed.private_text(out / 'request.private.txt',prompt)
        if not measured['fits']: raise ed.EditorialFailure('refinement_context_exceeded')
        def callback(info): ed.write_private_json(out / ed.DIAGNOSTICS, {'requests':[{'stage':'refinement',**info}]})
        result['request_attempts'] = 1
        answer = generate(prompt,structured=True,num_predict=measured['reserved_output'],response_callback=callback)
        ed.write_private_json(out / ed.RESPONSE, {'source_hash':evidence['source_hash'],'responses':{'refinement':answer}})
        clean, findings = validate(json.loads(answer),original,records,source)
        review = ed.finish(out,register,clean,None,source,records=records)
        # Copy the original canonical register byte-for-byte, including its private audit.
        for name in (ed.REGISTER,ed.REGISTER_MD):
            if name in protected: (out/name).write_bytes(protected[name])
        ed.write_private_json(out / 'original-concerns.private.json', original['concerns'])
        result.update(result='candidate_requires_review',original_words=prior_review['notes_word_count'],
                      refined_words=review['notes_word_count'],coverage=json.loads(answer)['coverage'],findings=findings,
                      original_concerns_preserved=clean['concerns'][:len(original['concerns'])]==original['concerns'])
        review['findings'].extend(findings);review['refinement_candidate_selected']=False
        ed.write_private_json(out / ed.REVIEW,review)
        with (out / ed.CHECKLIST).open('a',encoding='utf-8') as stream:
            stream.write('\n## Refinement comparison — private review hold\n\nOriginal draft remains selected. Review every coverage disposition and discrepancy in [private comparison]('+REPORT+').\n')
            for f in findings:stream.write('- [ ] '+f['code']+' '+f.get('original_block',f.get('notes_location',''))+'\n')
    except Exception as exc:
        from .ollama_response import GenerationFailure
        result['failure_category'] = str(exc) if isinstance(exc,(ed.EditorialFailure,GenerationFailure)) else 'refinement_invalid_response_or_internal_failure'
        ed.write_private_json(out / ed.REVIEW, {'status':'review_hold','phase':'refinement_failed','failure_category':result['failure_category']})
    finally:
        result['original_files_unchanged'] = all((parent/name).read_bytes()==data for name,data in protected.items())
        if not result['original_files_unchanged']:
            result.update(result='failed',failure_category='refinement_original_drift')
        ed.write_private_json(out / REPORT,result)
        for path in out.iterdir():
            if path.is_file(): path.chmod(0o600)
    return result
