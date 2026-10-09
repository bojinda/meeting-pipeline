"""R2 shared runner/meeting tests: native temporary locks and mock backends only."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from gpu_admission_fixture import configure, recover_synthetic

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'bin/with-gpu-lock.sh'


class MockBackend:
    def __init__(self, entered=None, finish=None, uncertain=False):
        self.entered, self.finish, self.uncertain, self.calls = entered, finish, uncertain, 0
    def validate(self, operation, payload):
        if operation != 'generate' or payload != {'model': 'synthetic', 'prompt': 'neutral'}:
            raise ValueError('synthetic_request_only')
    def execute(self, operation, payload, deadline, progress):
        from aihub_gpu_runner.backends import Execution, Uncertain
        self.calls += 1
        if self.entered:
            self.entered.set()
        if self.finish and not self.finish.wait(5):
            raise AssertionError('synthetic backend fixture timed out')
        if self.uncertain:
            raise Uncertain('synthetic_lost_reply')
        return Execution({'response': 'synthetic', 'done': True}, {})
    def cleanup(self, execution, deadline):
        return {'completion_verified': True, 'cleanup_verified': True, 'policy': 'synthetic'}


class MockTransport:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
    def json(self, method, route, payload=None, timeout=5):
        self.calls.append((method, route, payload))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


@unittest.skipUnless(sys.platform.startswith('linux'), 'Requires native Linux process/lock semantics')
class MeetingAdmissionTests(unittest.TestCase):
    def setUp(self):
        from aihub_gpu_runner.config import load_config
        self.temp = tempfile.TemporaryDirectory(prefix='r2-meeting-')
        self.root = Path(self.temp.name)
        self.environment = dict(os.environ, AIHUB_GPU_LOCK_TIMEOUT='1')
        self.controller = configure(self.root, self.environment)
        _, self.targets, *_ = load_config(self.environment['AIHUB_GPU_RUNNER_CONFIG'])
        self.processes, self.runners, self.finish_events = [], [], []
    def tearDown(self):
        for event in self.finish_events:
            event.set()
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.communicate(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)
        for service in self.runners:
            service.close()
        self.temp.cleanup()
    def launch(self, resource, code, request_id, timeout=1):
        env = dict(self.environment, AIHUB_GPU_STAGE_ID=request_id, AIHUB_GPU_LOCK_TIMEOUT=str(timeout))
        process = subprocess.Popen(['bash', str(HELPER), resource, 'synthetic stage', sys.executable, '-c', code],
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(process)
        return process
    def wait_file(self, marker, process):
        deadline = time.monotonic() + 4
        while not marker.exists():
            if process.poll() is not None:
                self.fail('Synthetic helper exited: ' + repr(process.communicate()))
            if time.monotonic() > deadline:
                self.fail('Synthetic helper readiness deadline exceeded')
            time.sleep(.01)
    def hold(self, resource, job_id):
        ready, release = self.root / (job_id + '.ready'), self.root / (job_id + '.release')
        code = f'import os,pathlib,time;pathlib.Path({str(ready)!r}).write_text(str(os.getpid()));\nwhile not pathlib.Path({str(release)!r}).exists():time.sleep(.01)'
        process = self.launch(resource, code, job_id)
        self.wait_file(ready, process)
        return process, ready, release
    def service(self, gpu0=None, gpu1=None):
        from aihub_gpu_runner.backends import Target
        from aihub_gpu_runner.core import Runner
        targets = dict(self.targets, wallpaper=Target('wallpaper', 'comfy', ('gpu0',), 'http://127.0.0.1:1', acquisition_timeout=.5))
        # Resource admission is real; every backend effect and cleanup is synthetic.
        service = Runner(self.controller, targets,
                         {'ollama': gpu1 or MockBackend(), 'wallpaper': gpu0 or MockBackend()}, poll_interval=.005)
        self.runners.append(service)
        return service
    def submit(self, service, job_id, target='ollama'):
        return service.submit(job_id, target, 'generate', {'model': 'synthetic', 'prompt': 'neutral'})
    def borrowed_client(self, responses):
        from aihub_gpu_runner.host_stage import HostStage, HostOllamaClient
        from aihub_gpu_runner.core import timestamp
        lease = self.controller.try_acquire('owned-summary', 'synthetic-hash', ('gpu1',), 'ollama')
        self.addCleanup(lease.close)
        self.controller.store.put_job({'request_id': 'owned-summary', 'request_hash': 'synthetic-hash',
            'host_stage': True, 'host_requests': [], 'target': 'ollama', 'resources': ['gpu1'],
            'state': 'running', 'created_at': timestamp(), 'lease_id': lease.owner['lease_id']})
        transport = MockTransport(responses)
        client = HostOllamaClient(self.controller, self.targets['ollama'], 'owned-summary',
                                 lease.owner['lease_id'], transport)
        stage = HostStage(self.controller, 'gpu1', 'owned-summary', 'synthetic-hash',
                          ollama_target=self.targets['ollama'], transport_factory=lambda _: transport)
        stage.lease = lease
        return client, stage, transport
    def test_host_gpu0_blocks_runner_wallpaper_but_allows_gpu1(self):
        process, _, release = self.hold('gpu0', 'host-gpu0')
        service = self.service()
        self.submit(service, 'wallpaper-wait', 'wallpaper')
        self.submit(service, 'independent-summary')
        self.assertEqual(service.wait('independent-summary', 2)['state'], 'success')
        self.assertEqual(service.status('wallpaper-wait')['state'], 'waiting')
        release.touch()
        process.communicate(timeout=3)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(service.wait('wallpaper-wait', 2)['state'], 'success')
    def test_runner_gpu1_blocks_helper_until_verified_handoff(self):
        entered, finish = threading.Event(), threading.Event()
        self.finish_events.append(finish)
        service = self.service(gpu1=MockBackend(entered, finish))
        self.submit(service, 'runner-owns-gpu1')
        self.assertTrue(entered.wait(2))
        marker = self.root / 'helper-started'
        process = self.launch('gpu1', f'from pathlib import Path;Path({str(marker)!r}).touch()', 'helper-waits')
        time.sleep(.15)
        self.assertFalse(marker.exists())
        finish.set()
        self.assertEqual(service.wait('runner-owns-gpu1', 2)['state'], 'success')
        process.communicate(timeout=3)
        self.assertEqual(process.returncode, 0)
        self.assertTrue(marker.exists())
    def test_runner_uncertainty_blocks_helper_even_when_flock_is_free(self):
        from aihub_gpu_runner.core import NativeLock
        service = self.service(gpu1=MockBackend(uncertain=True))
        self.submit(service, 'uncertain-api-job')
        self.assertEqual(service.wait('uncertain-api-job', 2)['state'], 'reconciling')
        physical = NativeLock(self.controller.lock_paths['gpu1'])
        self.assertTrue(physical.acquire(0))
        physical.close()
        process = self.launch('gpu1', "print('must-not-run')", 'blocked-host', 0)
        stdout, _ = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 75)
        self.assertEqual(stdout, '')
    def test_helper_interruption_blocks_runner_until_exact_recovery(self):
        from aihub_gpu_runner.core import NativeLock
        process, _, _ = self.hold('gpu1', 'interrupted-host')
        process.terminate()
        process.communicate(timeout=8)
        self.assertEqual(process.returncode, 143)
        owner = self.controller.snapshot()['owners']['gpu1']
        physical = NativeLock(self.controller.lock_paths['gpu1'])
        self.assertTrue(physical.acquire(0))
        physical.close()
        service = self.service()
        self.submit(service, 'blocked-by-host')
        self.assertEqual(service.wait('blocked-by-host', 2)['state'], 'failed')
        self.assertEqual(service.backends['ollama'].calls, 0)
        evidence = dict(owner, approval_ref='SYNTHETIC', operator_approved_backend_recovery=True,
                        backend_quiescent=True, cleanup_verified=True)
        with self.assertRaises(ValueError):
            self.controller.recover('interrupted-host', dict(evidence, lease_id='stale'))
        self.controller.recover('interrupted-host', evidence)
        self.submit(service, 'after-reviewed-recovery')
        self.assertEqual(service.wait('after-reviewed-recovery', 2)['state'], 'success')
    def test_sigkill_of_helper_retains_owner_and_retry_never_replays(self):
        process, ready, _ = self.hold('gpu1', 'killed-host')
        child_pid = int(ready.read_text())
        process.kill()
        # Only the synthetic orphan's group is terminated; SIGKILL cannot be trapped.
        os.killpg(child_pid, signal.SIGTERM)
        process.communicate(timeout=5)
        self.assertEqual(process.returncode, -9)
        self.assertIn('gpu1', self.controller.snapshot()['owners'])
        again = self.launch('gpu1', "print('must-not-replay')", 'killed-host', 0)
        stdout, _ = again.communicate(timeout=3)
        self.assertEqual(again.returncode, 70)
        self.assertEqual(stdout, '')
        self.assertIn('gpu1', self.controller.snapshot()['owners'])
    def test_duplicate_completed_stage_attaches_and_changed_payload_conflicts(self):
        count = self.root / 'executions'
        code = f'from pathlib import Path;p=Path({str(count)!r});p.write_text(p.read_text()+"x" if p.exists() else "x")'
        for _ in range(2):
            process = self.launch('gpu1', code, 'stable-stage')
            process.communicate(timeout=3)
            self.assertEqual(process.returncode, 0)
        self.assertEqual(count.read_text(), 'x')
        conflict = self.launch('gpu1', "print('must-not-run')", 'stable-stage')
        stdout, _ = conflict.communicate(timeout=3)
        self.assertEqual(conflict.returncode, 70)
        self.assertEqual(stdout, '')
    def test_concurrent_duplicate_stage_has_only_one_supervisor(self):
        process, _, release = self.hold('gpu1', 'same-id')
        old = self.controller.store.job('same-id')
        # Identical invocation via same CLI arguments, not a new command hash.
        contender = subprocess.Popen(process.args, env=dict(self.environment, AIHUB_GPU_STAGE_ID='same-id'),
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(contender)
        time.sleep(.15)
        self.assertIsNone(contender.poll())
        self.assertEqual(self.controller.store.job('same-id')['lease_id'], old['lease_id'])
        release.touch()
        process.communicate(timeout=3)
        contender.communicate(timeout=3)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(contender.returncode, 0)
    def test_api_restart_does_not_rewrite_active_host_stage(self):
        process, _, release = self.hold('gpu1', 'independent-supervisor')
        service = self.service()
        self.assertEqual(self.controller.store.job('independent-supervisor')['state'], 'running')
        service.close()
        self.runners.remove(service)
        self.service()
        self.assertEqual(self.controller.store.job('independent-supervisor')['state'], 'running')
        release.touch()
        process.communicate(timeout=3)
        self.assertEqual(process.returncode, 0)
    def test_owned_multi_request_summary_preserves_payload_and_cleans_once_at_stage_end(self):
        client, stage, transport = self.borrowed_client([
            {'done': True, 'response': 'map result'}, {'done': True, 'response': 'reduce result'},
            {'done': True}, {'models': []}])
        payload = {'model': 'synthetic', 'prompt': 'PRIVATE_SYNTHETIC_PROMPT', 'system': 'unchanged system',
                   'stream': False, 'keep_alive': '30m', 'options': {'num_ctx': 16384, 'temperature': .2}}
        self.assertEqual(client.generate('http://127.0.0.1:1', payload)['response'], 'map result')
        self.assertEqual(client.generate('http://127.0.0.1:1', payload)['response'], 'reduce result')
        self.assertEqual(transport.calls[0][2], payload)
        self.assertEqual(len(transport.calls), 2)
        self.assertNotIn('PRIVATE_SYNTHETIC_PROMPT', json.dumps(self.controller.store.job('owned-summary')))
        self.assertEqual(stage._cleanup()['request_count'], 2)
        self.assertEqual(len([call for call in transport.calls if call[2] and call[2].get('keep_alive') == 0]), 1)
    def test_swallowed_ambiguous_request_blocks_cleanup_and_later_submissions(self):
        from aihub_gpu_runner.core import RecoveryRequired
        client, stage, transport = self.borrowed_client([OSError('SYNTHETIC_PRIVATE_TRANSPORT_ERROR')])
        payload = {'model': 'synthetic', 'prompt': 'neutral'}
        with self.assertRaises(Exception):
            client.generate('http://127.0.0.1:1', payload)
        with self.assertRaises(RecoveryRequired):
            client.generate('http://127.0.0.1:1', payload)
        with self.assertRaises(RecoveryRequired):
            stage._cleanup()
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(self.controller.store.job('owned-summary')['host_requests'][0]['state'], 'submit_intent')
    def test_distinct_map_reduce_models_unload_before_empty_residency_check(self):
        from dataclasses import replace
        client, stage, transport = self.borrowed_client([
            {'done': True, 'response': 'map'}, {'done': True, 'response': 'reduce'},
            {'done': True}, {'done': True}, {'models': []}])
        target = replace(self.targets['ollama'], models=('synthetic', 'synthetic-reduce'))
        stage.target = target
        client.target = client.backend.target = target
        for model in target.models:
            client.generate('http://127.0.0.1:1', {'model': model, 'prompt': 'neutral'})
        self.assertEqual(stage._cleanup()['request_count'], 2)
        self.assertEqual([call[2]['model'] for call in transport.calls
                          if call[2] and call[2].get('keep_alive') == 0], list(target.models))
        self.assertEqual([call[1] for call in transport.calls[-3:]], ['/api/generate', '/api/generate', '/api/ps'])
    def test_configured_summary_without_owned_context_has_no_direct_http_fallback(self):
        spec = importlib.util.spec_from_file_location('summary_admission_test', ROOT / 'bin/ollama_session_summary.py')
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(ROOT / 'bin'))
        try:
            spec.loader.exec_module(module)
            with patch.dict(os.environ, self.environment), patch('urllib.request.urlopen') as direct:
                with self.assertRaises(KeyError):
                    module.call_ollama('http://127.0.0.1:1', 'synthetic', 'neutral')
                direct.assert_not_called()
        finally:
            sys.path.pop(0)
    def test_gpu0_missing_calibration_never_launches(self):
        config = Path(self.environment['AIHUB_GPU_RUNNER_CONFIG'])
        settings = json.loads(config.read_text())
        settings['host_stages']['whisperx']['idle_memory_mb'] = None
        settings['state_dir'] = str(self.root / 'fresh-uncalibrated-state')
        config.write_text(json.dumps(settings))
        process = self.launch('gpu0', "print('must-not-run')", 'uncalibrated')
        stdout, _ = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 64)
        self.assertEqual(stdout, '')
    def test_gpu0_cleanup_failure_retains_owner_and_distinct_state(self):
        from aihub_gpu_runner.host_stage import HostStage
        from aihub_gpu_runner.backends import CleanupPending
        from aihub_gpu_runner.core import NativeLock
        samples = iter([0, 100, 100, 100])
        stage = HostStage(self.controller, 'gpu0', 'memory-cleanup-failed', 'hash',
                          host_policy={'idle_memory_mb': 0, 'cleanup_samples': 2, 'cleanup_timeout': .02},
                          probe=lambda: next(samples), poll_interval=.01)
        with self.assertRaises(CleanupPending):
            stage.run([sys.executable, '-c', 'pass'], self.environment, .1)
        self.assertEqual(self.controller.store.job(stage.job_id)['state'], 'cleanup_pending')
        physical = NativeLock(self.controller.lock_paths['gpu0'])
        self.assertTrue(physical.acquire(0))
        physical.close()
        self.assertIsNone(self.controller.try_acquire('blocked-memory', 'hash', ('gpu0',), 'wallpaper'))
    def test_request_completion_storage_fault_after_effect_blocks_handoff(self):
        from aihub_gpu_runner.core import RecoveryRequired
        client, stage, transport = self.borrowed_client([{'done': True, 'response': 'synthetic'}])
        original = self.controller.store.put_job
        def fail_after_effect(job):
            original(job)
            if job['host_requests'] and job['host_requests'][-1]['state'] == 'completed':
                raise OSError('SYNTHETIC STORAGE FAULT')
        with patch.object(self.controller.store, 'put_job', side_effect=fail_after_effect):
            with self.assertRaises(OSError):
                client.generate('http://127.0.0.1:1', {'model': 'synthetic', 'prompt': 'neutral'})
        with self.assertRaises(RecoveryRequired):
            client.generate('http://127.0.0.1:1', {'model': 'synthetic', 'prompt': 'neutral'})
        with self.assertRaises(RecoveryRequired):
            stage._cleanup()
        self.assertEqual(len(transport.calls), 1)
    def test_inactive_physical_owner_cannot_borrow_durable_context(self):
        from aihub_gpu_runner.core import RecoveryRequired
        client, stage, transport = self.borrowed_client([{'done': True, 'response': 'synthetic'}])
        stage.lease.close()
        with self.assertRaises(RecoveryRequired):
            client.generate('http://127.0.0.1:1', {'model': 'synthetic', 'prompt': 'neutral'})
        self.assertEqual(transport.calls, [])
    def test_recovery_cli_records_failed_stage_and_never_replays_old_id(self):
        process, _, _ = self.hold('gpu1', 'cli-recovery')
        process.terminate()
        process.communicate(timeout=8)
        owner = self.controller.snapshot()['owners']['gpu1']
        evidence = self.root / 'recovery.synthetic.json'
        evidence.write_text(json.dumps(dict(owner, approval_ref='SYNTHETIC TEST ONLY',
            operator_approved_backend_recovery=True, backend_quiescent=True, cleanup_verified=True)))
        result = subprocess.run([sys.executable, '-m', 'aihub_gpu_runner', '--config',
            self.environment['AIHUB_GPU_RUNNER_CONFIG'], 'recover', '--job-id', 'cli-recovery',
            '--evidence-file', str(evidence)], env=self.environment, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.controller.store.job('cli-recovery')['host_exit_code'], 70)
        self.assertEqual(self.controller.snapshot()['owners'], {})
        again = subprocess.Popen(process.args, env=dict(self.environment, AIHUB_GPU_STAGE_ID='cli-recovery'),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(again)
        again.communicate(timeout=3)
        self.assertEqual(again.returncode, 70)

if __name__ == '__main__':
    unittest.main()
