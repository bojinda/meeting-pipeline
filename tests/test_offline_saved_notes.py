"""Frozen synthetic saved-generation completion and fail-closed bindings."""
import copy
import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
import render_saved_notes as offline
from meeting_postprocess import editorial as ed, editorial_checkpoint as cp
from test_editorial_notes import fixture


class OfflineSavedNotesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.source = self.root / 'source'; self.source.mkdir()
        chunks, raw, notes = fixture(); notes['concerns'][0]['text'] = 'Private review. ' + 'x' * 7658
        records = ed.source_records(chunks); source = '\n'.join(c['text'] for c in chunks if c['meeting_section'] in {ed.BUSINESS, ed.ADJOURNMENT})
        maps = [{**c, 'summary': 'Synthetic map'} for c in chunks]
        sections = '\n'.join(json.dumps(c) for c in chunks).encode(); map_data = '\n'.join(json.dumps(c) for c in maps).encode()
        checkpoint = dict(source_stage_id='synthetic-original', sections_file_sha256=offline.sha(sections), maps_file_sha256=offline.sha(map_data),
                          prepared_source_hash=cp.digest(chunks), maps_hash=cp.digest(maps), map_code=cp.code_identity(offline.engine),
                          map_sources=chunks, blocked_context_source_ids=[], warnings=[])
        saved = dict(source_hash=ed.digest(records), register_assessment=ed.audit_register(raw, records, [], source), responses={'register': json.dumps(raw), 'notes': json.dumps(notes)})
        files = {'meeting_sections.jsonl': sections, 'chunk_summaries.jsonl': map_data, cp.FILE: json.dumps(checkpoint).encode(),
                 ed.RESPONSE: json.dumps(saved).encode(), ed.DIAGNOSTICS: json.dumps({'requests': [dict(stage=s, done=True, done_reason='stop') for s in ['register', 'notes']]}).encode()}
        for name, data in files.items(): (self.source / name).write_bytes(data)
        self.job = self.root / 'job.json'; self.job.write_text(json.dumps(dict(request_id='synthetic-original', state='failed', cleanup_verified=True, host_requests=[{'state': 'completed'}])))
        self.owners = self.root / 'owners.json'; self.owners.write_text('{"owners":{}}')
        self.manifest = self.root / 'manifest.json'
        self.data = dict(source_directory=str(self.source), files={n: offline.sha(d) for n,d in files.items()},
                         source_job_path=str(self.job), source_job_sha256=offline.sha(self.job.read_bytes()), ownership_path=str(self.owners))
        self.write_manifest(); self.out = self.root / 'output'

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.data)); self.digest = offline.sha(self.manifest.read_bytes())

    def test_offline_render_retains_original_bytes_and_long_concern_with_hold(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')), patch.object(offline.engine, 'call_ollama', side_effect=AssertionError('inference forbidden')):
            review = offline.render(self.manifest, self.digest, self.out)
        self.assertEqual(review['status'], 'review_hold')
        self.assertIn('long_private_concern_review', {f['code'] for f in review['findings']})
        for name in ['meeting-notes-draft.md','summary.md',ed.REGISTER,ed.REGISTER_MD,ed.TRACE,ed.REVIEW,ed.CHECKLIST]:
            self.assertTrue((self.out/name).is_file(), name)
        self.assertEqual((self.out/ed.RESPONSE).read_bytes(), before[ed.RESPONSE])
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.source.iterdir()})
        self.assertFalse((self.out/'minutes-draft.md').exists())

    def test_tampered_manifest_or_source_fails_before_render(self):
        with self.assertRaisesRegex(ed.EditorialFailure, 'offline_manifest_changed'):
            offline.render(self.manifest, '0'*64, self.out)
        (self.source/ed.RESPONSE).write_text('{}')
        with self.assertRaisesRegex(ed.EditorialFailure, 'offline_source_changed'):
            offline.render(self.manifest, self.digest, self.out)
        self.assertFalse(self.out.exists())

    def test_unsettled_or_owned_job_fails_before_render(self):
        self.owners.write_text('{"owners":{"gpu1":{"job_id":"synthetic-original"}}}')
        with self.assertRaisesRegex(ed.EditorialFailure, 'offline_source_still_owned'):
            offline.render(self.manifest,self.digest,self.out)
        self.owners.write_text('{"owners":{}}')
        j=json.loads(self.job.read_text());j['cleanup_verified']=False;self.job.write_text(json.dumps(j))
        self.data['source_job_sha256']=offline.sha(self.job.read_bytes());self.write_manifest()
        with self.assertRaisesRegex(ed.EditorialFailure, 'offline_source_job_unsettled'):
            offline.render(self.manifest,self.digest,self.out)
        self.assertFalse(self.out.exists())

    def test_existing_destination_is_preserved(self):
        self.out.mkdir(); sentinel=self.out/'preserve';sentinel.write_bytes(b'evidence')
        with self.assertRaisesRegex(ed.EditorialFailure, 'offline_fresh_destination_required'):
            offline.render(self.manifest,self.digest,self.out)
        self.assertEqual(sentinel.read_bytes(), b'evidence')
