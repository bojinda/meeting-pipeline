"""Approved-input identity and historical summary entry point; synthetic Linux I/O."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
from meeting_stage_inputs import input_files
from meeting_postprocess.speaker_turns import turn_catalog, approve_turn, CORRECTIONS_FILE
from aihub_gpu_runner.host_stage import command_identity, capture_input_files, HostStage
from aihub_gpu_runner.config import load_config
from aihub_gpu_runner.core import NativeLock
from gpu_admission_fixture import configure


@unittest.skipUnless(sys.platform.startswith('linux'), 'Native Linux runner-managed meeting stages')
class MeetingStageIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='meeting-gates-')
        self.root = Path(self.temp.name)
        self.transcript = self.root / 'historical session'
        (self.transcript / 'chunks_out').mkdir(parents=True)
        self.index = self.transcript / 'chunks_out/transcript_chunks.jsonl'
        text = ('[SPEAKER_00] Okay guys, I think we will get started here.\n'
                '[SPEAKER_00] I will start with the recap last month.\n'
                '[SPEAKER_01] Last month we reviewed HISTORICAL_ONLY_SENTINEL.\n'
                '[SPEAKER_00] Moving right along.\n'
                '[SPEAKER_01] CURRENT_PUBLIC_SENTINEL needs discussion.\n'
                '[SPEAKER_00] Redact the following.\n'
                '[SPEAKER_01] PRIVATE_REDACTION_SENTINEL must be withheld.\n'
                '[SPEAKER_00] End redaction.\n'
                '[SPEAKER_01] Visible closing statement.')
        self.index.write_text(json.dumps({'chunk_id': 1, 'text': text, 'start_time': 0, 'end_time': 120}) + '\n')
        self.aliases = self.transcript / 'speaker_aliases.json'
        self.aliases.write_text(json.dumps({'SPEAKER_00': 'Chair', 'SPEAKER_01': 'Morgan'}))
        self.environment = dict(os.environ, AIHUB_GPU_LOCK_TIMEOUT='1',
            OLLAMA_URL='http://127.0.0.1:1', MEETING_MAP_MODEL='synthetic', MEETING_REDUCE_MODEL='synthetic',
            MEETING_SUMMARIES_ROOT=str(self.root / 'comparison-output'), MEETING_KEEP_RECAP='1')
        self.controller = configure(self.root, self.environment)
        self.output = self.root / 'comparison-output' / self.transcript.name
        self.log = self.root / 'requests.synthetic.jsonl'
        self.environment.update(MEETING_FAKE_OLLAMA_LOG=str(self.log),
                                MEETING_TEST_GPU0=str(self.root / 'gpu0.lock'),
                                MEETING_TEST_GPU1=str(self.root / 'gpu1.lock'))
        self.settings = self.root / 'meeting.synthetic.env'
        self.settings.write_text('MEETING_KEEP_RECAP=1\nDELETE_SOURCE_AUDIO_AFTER_TRANSCRIPTION=1\n')
        self.environment['MEETING_CONFIG_FILE'] = str(self.settings)
        self.processes = []
    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=8)
        self.temp.cleanup()
    def command(self, aliases=None):
        command = [sys.executable, str(ROOT / 'bin/ollama_meeting_summary.py'), str(self.transcript)]
        if aliases is not None:
            command += ['--speaker-aliases', str(aliases)]
        return command
    def env_for(self, command):
        return dict(self.environment, AIHUB_GPU_STAGE_INPUT_FILES=json.dumps(input_files(command)))
    def identity(self, aliases=None):
        command = self.command(aliases)
        return command_identity(command, self.env_for(command))
    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
    def historical(self, options=(), stage_id=None):
        environment = dict(self.environment)
        if stage_id is not None:
            environment['AIHUB_GPU_STAGE_ID'] = stage_id
        outcome = subprocess.run(['bash', str(ROOT / 'bin/summarize-existing-meeting.sh'), str(self.transcript), *options],
                                 env=environment, capture_output=True, text=True, timeout=12)
        return outcome
    def test_default_alias_edit_add_remove_changes_identity(self):
        first = self.identity()
        self.aliases.write_text(json.dumps({'SPEAKER_00': 'Chair', 'SPEAKER_01': 'Riley'}))
        self.assertNotEqual(self.identity(), first)
        edited = self.identity()
        self.aliases.unlink()
        self.assertNotEqual(self.identity(), edited)
        absent = self.identity()
        self.aliases.write_text('{}')
        self.assertNotEqual(self.identity(), absent)
    def test_explicit_alias_override_binds_selected_file_not_unused_default(self):
        explicit = self.root / 'approved explicit aliases.json'
        explicit.write_text(json.dumps({'SPEAKER_01': 'Taylor'}))
        first = self.identity(explicit)
        self.aliases.write_text(json.dumps({'SPEAKER_01': 'Unselected'}))
        self.assertEqual(self.identity(explicit), first)
        explicit.write_text(json.dumps({'SPEAKER_01': 'Jordan'}))
        self.assertNotEqual(self.identity(explicit), first)
    def test_explicit_missing_alias_fails_closed_without_substitution(self):
        with self.assertRaises(ValueError):
            self.identity(self.root / 'missing-approved.json')
        self.assertFalse((self.root / 'missing-approved.json').exists())
    def test_approved_turn_edit_and_removal_change_identity(self):
        catalog = turn_catalog(self.transcript)
        selected = next(turn for turn in catalog['turns'] if 'CURRENT_PUBLIC_SENTINEL' in turn['text'])
        first = self.identity()
        approve_turn(self.transcript, catalog, selected['turn_id'], 'Taylor')
        second = self.identity()
        self.assertNotEqual(first, second)
        document = json.loads((self.transcript / CORRECTIONS_FILE).read_text())
        document['corrections'][selected['turn_id']]['name'] = 'Jordan'
        (self.transcript / CORRECTIONS_FILE).write_text(json.dumps(document))
        self.assertNotEqual(self.identity(), second)
        third = self.identity()
        (self.transcript / CORRECTIONS_FILE).unlink()
        self.assertNotEqual(self.identity(), third)
    def test_advisory_suggestions_never_replace_missing_approved_inputs(self):
        self.aliases.unlink()
        first = self.identity()
        (self.transcript / 'speaker-suggestions.json').write_text('{"proposal":"NOT_APPROVED_SENTINEL"}')
        self.assertEqual(self.identity(), first)
        files = capture_input_files(self.env_for(self.command()))
        self.assertIsNone(files['approved_aliases']['data'])
        self.assertIsNone(files['approved_turn_corrections']['data'])
    def test_historical_command_no_recording_or_transcription_and_same_id_reconnect(self):
        self.assertEqual(list(self.transcript.glob('*.wav')), [])
        before = self.index.read_bytes()
        first = self.historical(['--keep-recap'])
        self.assertEqual(first.returncode, 0, first.stderr)
        model_calls = [call for call in self.calls() if call.get('payload')]
        self.assertTrue(model_calls)
        self.assertTrue(all(call['gpu0_free'] and call['gpu1_held'] for call in self.calls()))
        count = len(model_calls)
        second = self.historical(['--keep-recap'])
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(len([call for call in self.calls() if call.get('payload')]), count)
        self.assertEqual(self.index.read_bytes(), before)
        self.assertEqual(list(self.transcript.glob('*.wav')), [])
        self.assertIn('Recap of Previous Meeting', (self.output / 'minutes-draft.md').read_text())
        prompts = json.dumps([call.get('payload') for call in self.calls()])
        self.assertNotIn('PRIVATE_REDACTION_SENTINEL', prompts)
        self.assertIn('HISTORICAL_ONLY_SENTINEL', prompts)
        self.assertIn('CURRENT_PUBLIC_SENTINEL', prompts)
        self.assertEqual(self.controller.snapshot()['owners'], {})
    def test_changed_default_alias_never_reuses_completed_generation(self):
        self.assertEqual(self.historical().returncode, 0)
        count = len(self.calls())
        self.aliases.write_text(json.dumps({'SPEAKER_00': 'Chair', 'SPEAKER_01': 'Riley'}))
        second = self.historical()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertGreater(len(self.calls()), count)
        later = json.dumps(self.calls()[count:])
        self.assertIn('[Riley]', later)
        self.assertNotIn('[Morgan]', later)
    def test_same_explicit_stage_id_with_changed_approval_conflicts(self):
        self.assertEqual(self.historical(stage_id='fixed-meeting-stage').returncode, 0)
        count = len(self.calls())
        self.aliases.write_text(json.dumps({'SPEAKER_01': 'Riley'}))
        result = self.historical(stage_id='fixed-meeting-stage')
        self.assertEqual(result.returncode, 70)
        self.assertEqual(len(self.calls()), count)
    def test_changed_turn_correction_never_reuses_completed_generation(self):
        catalog = turn_catalog(self.transcript)
        turn = next(turn for turn in catalog['turns'] if 'CURRENT_PUBLIC_SENTINEL' in turn['text'])
        approve_turn(self.transcript, catalog, turn['turn_id'], 'Taylor')
        first = self.historical()
        self.assertEqual(first.returncode, 0, first.stderr)
        count = len(self.calls())
        approve_turn(self.transcript, catalog, turn['turn_id'], 'Jordan')
        result = self.historical()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(len(self.calls()), count)
        self.assertIn('[Jordan]', json.dumps(self.calls()[count:]))
    def test_snapshots_are_private_and_original_edits_cannot_change_active_inputs(self):
        env = self.env_for(self.command())
        inputs = capture_input_files(env)
        _, targets, *_ = load_config(self.environment['AIHUB_GPU_RUNNER_CONFIG'])
        stage = HostStage(self.controller, 'gpu1', 'snapshot-case', 'snapshot-hash',
                          ollama_target=targets['ollama'], input_files=inputs)
        marker = self.root / 'snapshot.readiness'
        # Child edits the original after launch, then reports its immutable approved snapshot.
        child = ('import os,json,pathlib; snapshots=json.loads(os.environ["AIHUB_GPU_STAGE_INPUT_SNAPSHOTS"]); '
                 f'pathlib.Path({str(self.aliases)!r}).write_text("{{}}" ); '
                 f'pathlib.Path({str(marker)!r}).write_bytes(pathlib.Path(snapshots["approved_aliases"]).read_bytes())')
        code = stage.run([sys.executable, '-c', child], env, .2)
        self.assertEqual(code, 70)  # Changed approvals cannot become a cached success.
        self.assertEqual(json.loads(marker.read_text())['SPEAKER_01'], 'Morgan')
        self.assertEqual(self.controller.store.job('snapshot-case')['state'], 'failed')
        snapshot = self.controller.store.path('inputs/snapshot-case/approved_aliases.json')
        self.assertEqual(stat.S_IMODE(snapshot.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(snapshot.parent.stat().st_mode), 0o700)
        self.assertEqual(self.controller.snapshot()['owners'], {})
    def test_changed_input_while_waiting_never_launches_old_payload(self):
        lease = self.controller.try_acquire('held', 'hash', ('gpu1',), 'ollama')
        environment = dict(self.environment, AIHUB_GPU_STAGE_ID='wait-edit')
        process = subprocess.Popen(['bash', str(ROOT / 'bin/summarize-existing-meeting.sh'), str(self.transcript)],
                                   env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(process)
        deadline = time.monotonic() + 3
        while self.controller.store.read('jobs/wait-edit.json') is None and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertIsNotNone(self.controller.store.read('jobs/wait-edit.json'))
        self.aliases.write_text('{}')
        lease.release({'completion_verified': True, 'cleanup_verified': True})
        lease.close()
        process.communicate(timeout=6)
        self.assertEqual(process.returncode, 70)
        self.assertEqual(self.calls(), [])
        self.assertEqual(self.controller.snapshot()['owners'], {})
    def test_missing_index_and_unsupported_whole_mode_never_run_models(self):
        result = self.historical(['--synthesis-mode', 'whole'])
        self.assertEqual(result.returncode, 64)
        self.index.unlink()
        result = self.historical()
        self.assertEqual(result.returncode, 66)
        self.assertEqual(self.calls(), [])
    def test_explicit_comparison_output_override_preserves_existing_outputs(self):
        protected = self.root / 'existing accepted summaries'
        protected.mkdir()
        sentinel = protected / 'summary.md'
        sentinel.write_text('PREVIOUS_ACCEPTED_OUTPUT')
        self.settings.write_text('MEETING_SUMMARIES_ROOT=' + json.dumps(str(protected)) + '\nMEETING_KEEP_RECAP=1\n')
        result = self.historical()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sentinel.read_text(), 'PREVIOUS_ACCEPTED_OUTPUT')
        self.assertTrue((self.output / 'summary.md').exists())

if __name__ == '__main__':
    unittest.main()
