#!/usr/bin/env python3
"""Internal private-pipe worker launched under the existing GPU1 supervisor."""
import json
import sys
from meeting_postprocess.speaker_review import _http_call, failure_diagnostics


def main():
    try:
        request = json.load(sys.stdin)
        if request.get("operation") == "turn_review":
            from meeting_postprocess.speaker_turn_review import run_two_pass
            response = run_two_pass(request["report"], request["turns"], request["aliases"], request["roster"], model=request["model"], ollama_url=request["ollama_url"], options=request["options"], call=_http_call)
            print(json.dumps(response))
            return 0
        response = _http_call(request)
        print(json.dumps(response))
        return 0
    except Exception as exc:
        # Never echo prompts or model/HTTP error bodies into public logs.
        print("[speaker-review] Local inference unavailable", file=sys.stderr)
        print(json.dumps({"failure": failure_diagnostics(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
