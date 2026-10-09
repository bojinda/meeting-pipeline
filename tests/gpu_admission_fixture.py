"""Synthetic shared journal, calibrated mock probe and native temporary locks."""
import json
import os
from pathlib import Path
import sys


def configure(root, environment):
    from aihub_gpu_runner.config import admission_from_config
    root = Path(root)
    lock_paths = {resource: root / (resource + '.lock') for resource in ('gpu0', 'gpu1')}
    for path in lock_paths.values():
        path.touch()
    config = root / 'runner.synthetic.json'
    config.write_text(json.dumps({'state_dir': str(root / 'private-runner-state'),
        'lock_paths': {resource: str(path) for resource, path in lock_paths.items()},
        'host_stages': {'whisperx': {'idle_memory_mb': 0, 'cleanup_samples': 2, 'cleanup_timeout': 1}},
        'targets': {'ollama': {'kind': 'ollama', 'resources': ['gpu1'],
                             'base_url': 'http://127.0.0.1:1', 'models': ['synthetic'],
                             'acquisition_timeout': .2}}}))
    _, _, controller = admission_from_config(config)
    controller.bootstrap({'approval_ref': 'SYNTHETIC TEST ONLY',
        'bootstrap_id': controller.snapshot()['bootstrap_id'],
        'backend_quiescent': True, 'cleanup_verified': True})
    binary = root / 'probe-fixture'
    binary.mkdir()
    probe = binary / 'nvidia-smi'
    probe.write_text('#!' + sys.executable + '\nprint(0)\n')
    probe.chmod(0o755)
    environment.update(AIHUB_GPU_RUNNER_CONFIG=str(config), AIHUB_GPU_RUNNER_PYTHON=sys.executable,
                       AIHUB_GPU_OLLAMA_TARGET='ollama',
                       PATH=str(binary) + os.pathsep + environment['PATH'])
    return controller


def recover_synthetic(controller):
    for job_id in {owner['job_id'] for owner in controller.snapshot()['owners'].values()}:
        owner = next(owner for owner in controller.snapshot()['owners'].values() if owner['job_id'] == job_id)
        controller.recover(job_id, dict(owner, approval_ref='SYNTHETIC TEST RECOVERY ONLY',
            backend_quiescent=True, cleanup_verified=True, operator_approved_backend_recovery=True))
