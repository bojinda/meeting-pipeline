"""Opt-in child transport fixture: real local admission, no network or inference."""
import json
import os
from pathlib import Path


def configure_transport(root, environment):
    guard = Path(root) / "transport-guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text("from runner_transport_fixture import install\ninstall()\n", encoding="utf-8")
    repository = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = os.pathsep.join([str(guard), str(repository / "tests"), str(repository / "bin"), environment.get("PYTHONPATH", "")])
    environment["MEETING_TRANSPORT_FIXTURE_ROOT"] = str(root)


def install():
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic fixture prohibits network/inference")
    socket.socket.connect = forbidden
    socket.socket.connect_ex = forbidden
    socket.create_connection = forbidden
    from aihub_gpu_runner.backends import HTTPTransport
    from aihub_gpu_runner.core import NativeLock, encode
    root = Path(os.environ["MEETING_TRANSPORT_FIXTURE_ROOT"]).resolve()
    config = Path(os.environ["AIHUB_GPU_RUNNER_CONFIG"]).resolve()
    if not config.is_relative_to(root) or config.name != "runner.synthetic.json":
        raise RuntimeError("synthetic_configuration_required")

    def request(self, method, route, payload=None, params=None, timeout=5):
        locks = []
        for key in ("MEETING_TEST_GPU0", "MEETING_TEST_GPU1"):
            path = Path(os.environ[key]).resolve()
            if not path.is_relative_to(root):
                raise RuntimeError("synthetic_lock_required")
            lock = NativeLock(str(path))
            locks.append(lock.acquire(0))
            lock.close()
        with open(os.environ["MEETING_FAKE_OLLAMA_LOG"], "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"route": route, "payload": payload, "gpu0_free": locks[0], "gpu1_held": not locks[1]}) + "\n")
        if route == "/api/ps":
            value = {"models": []}
        elif route == "/api/generate" and payload.get("keep_alive") == 0:
            value = {"done": True}
        elif route == "/api/generate" and payload.get("prompt"):
            prompt = payload["prompt"]
            if "Transcript chunk:\n" in prompt:
                response = "## Topics\n- " + prompt.split("Transcript chunk:\n", 1)[1]
            elif "propose ONE private canonical" in prompt:
                response = "" if os.environ.get("MEETING_FAKE_EDITORIAL_REGISTER_EMPTY") == "1" else '{"items":[]}'
            elif "summary-reduction call" in prompt:
                evidence = json.loads(prompt.split("Untrusted input JSON:\n", 1)[1])["source_excerpts"]
                ref = next(row["id"] for row in evidence if row["section"] == "current_meeting_business")
                response = json.dumps({"highlights": [], "previous_context": [], "motions": [], "unresolved": [], "concerns": [],
                    "issues": [{"heading": "Committee discussion", "paragraphs": [{"text": "Fixture model committee discussion.", "source_ids": [ref]}]}]})
            elif "Summarize only the supplied historical recap" in prompt:
                response = "- HISTORICAL_ONLY_SENTINEL reviewed previously."
            else:
                response = "# Generated Output\n\n## Overview\n- CURRENT_PUBLIC_SENTINEL discussed."
            value = {"done": True, "response": response, "thinking": "PRIVATE_THINKING_SENTINEL",
                     "done_reason": "stop", "eval_count": 20, "prompt_eval_count": 100}
        else:
            raise AssertionError("unexpected synthetic transport route")
        return encode(value), "application/json"
    HTTPTransport._request = request
