"""Transport-only bridge for an enclosing shared runner host-stage lease."""
import os
import json
from pathlib import Path


def approved_input(role, default_path):
    """Select a private runner snapshot, preserving an explicitly missing input."""
    value = os.environ.get('AIHUB_GPU_STAGE_INPUT_SNAPSHOTS')
    if value is None:
        return False, default_path
    snapshots = json.loads(value)
    if not isinstance(snapshots, dict) or role not in snapshots:
        raise ValueError('approved_stage_snapshot_missing')
    path = snapshots[role]
    if path is not None and not isinstance(path, str):
        raise ValueError('invalid_approved_stage_snapshot')
    return True, None if path is None else Path(path)


def managed_generate(url, payload, timeout=3600):
    # Legacy CPU/mock use remains importable without installing the runner.
    # A configured/inherited coordinated context must never fall back to HTTP.
    if not any(os.environ.get(key) for key in
               ('AIHUB_GPU_RUNNER_CONFIG', 'AIHUB_GPU_HOST_JOB_ID', 'AIHUB_GPU_HOST_LEASE_ID')):
        return None
    from aihub_gpu_runner.host_stage import generate_in_host_stage
    return generate_in_host_stage(url, payload, timeout)
