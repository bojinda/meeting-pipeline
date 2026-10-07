#!/usr/bin/env python3
"""Internal private-pipe worker launched under the existing GPU1 supervisor."""
import json
import sys
from meeting_postprocess.speaker_review import _http_call


def main():
    try:
        request = json.load(sys.stdin)
        response = _http_call(request)
        print(json.dumps({"response": response}))
        return 0
    except Exception:
        # Never echo prompts or model/HTTP error bodies into public logs.
        print("[speaker-review] Local inference unavailable", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
