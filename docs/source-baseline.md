# Source baseline and offline verification

The reconciliation branch starts at `0982cb0be20cf8abb5b24f54fdadd2d78c574fcd`
on `feature/minutes-postprocessing`. It retains the shared deployed repairs and
the Windows private output-location development. It does not deploy the branch,
change installed runner code, move historical outputs or authorize distribution.

## Deliberately separate outputs

| Output | Purpose and review rule |
| --- | --- |
| Meeting Notes | Shorter member-facing draft, normally 1,800–2,400 words for substantial meetings; justified exceptions require review. |
| Detailed Minutes | Comprehensive internal draft; no Meeting Notes word target. |
| Summary | Concise highlights; no expansion to the notes target. |
| Action Register | Private source-backed proposals and owners; explicit human approval remains required. |
| QA and Editorial Review | Private accuracy, uncertainty, attribution, confidentiality and source checks. |

`notes-source-review.private.md` pairs each rendered notes paragraph and private
concern with its exact cited source text, original chunk identity and containing
chunk timestamps. Start with its link in `operator-review.private.md`. The JSON
provenance and original model responses remain authoritative evidence. Chunk
bounds are not exact line timings; source IDs and lexical matches do not prove
that a claim is accurate. The readable source view is private and export-blocked.

Length warnings include word counts and a safe revision strategy. Preserve the
original and revise a separate copy; consolidate repetitions and irrelevant
historical recap, then compare coverage and every changed claim with the source.
Never truncate, pad, remove material union business or relax confidentiality to
meet a count. Short meetings do not need padding. Numerical bases, role claims,
allegations and chronology receive heuristic review cues, never automatic fact
correction or approval. A warning can be justified after human review; absence
of a warning is not verification. Raw responses and source bytes are retained.

## Production contracts

The automated meeting wrapper requests full output with
`--meeting-notes --no-meeting-notes-only`. Manual `--meeting-notes` runs retain
their notes-only default. Lesson summaries use `AIHUB_GPU_RUNNER_PYTHON`, ensuring
the child interpreter can import the installed runner SDK. The existing runner
owns admission, snapshots, source identity, physical locks and cleanup. No new
physical lock paths or separate runner installation are introduced here.

The example retention flag is `DELETE_SOURCE_AUDIO_AFTER_TRANSCRIPTION=0`.
Existing production settings remain in the private installed configuration.
The output root defaults to `MEETING_SUMMARIES_ROOT`; an explicit
`MEETING_NOTES_ROOT` still wins. Each run reserves a new private session directory,
with no `notes/` subdirectory. Existing runs, legacy summaries and transcripts
are protected from overwrite.

`continue-editorial-responses.py` is a narrow source-bound recovery entry point.
It requires a reviewed manifest, exact saved source/map/response hashes, original
completed request hashes, verified cleanup, no remaining source-stage owner and
a fresh runner-admitted stage. It only reuses completed register/notes replies
when their reconstructed requests match exactly; changed prompts fail closed.
It is not an automatic retry or a way to reuse stale approvals. `--preflight`
creates a new private diagnostic destination and intentionally fails at the first
missing response without inference; it is not a successful completion. Preserve
that destination and use a different fresh destination for any later run.

## Reproduce tests

Use a fresh Linux checkout with Python 3.11+ and Git. Source files are UTF-8 with
LF line endings, enforced by `.gitattributes`. Do not run these tests from either
active production checkout. Install `requirements-test.txt` into an isolated
environment and provide the existing separately managed `aihub-gpu-runner` SDK.
The reconciliation was qualified against `0.1.0+reliability.20261101.2`; this is a
compatibility statement, not an instruction to upgrade or reinstall production.

```sh
python -m pip install -r requirements-test.txt
MEETING_NATIVE_EDITORIAL_FIXTURE=1 python -B -m unittest discover -s tests -v
find bin -name '*.sh' -exec bash -n {} \;
git diff --check
```

Native admission tests use temporary journals/locks and synthetic transport.
The transport fixture blocks real network calls in its child processes; it is
installed only by the tests. Tests never need real transcripts or recordings.
Optional saved-response, historical local-script and private-identifier tests
remain opt-in and may skip in a clean checkout. Do not add private fixtures to
Git to make those optional tests run.

Keep `.env` and credential backups, all recording/transcript/document outputs,
private provenance, journals, recovery receipts, generated packages and separate
`runner-*` work directories outside version control. Stage explicit reviewed
paths. Review the focused branch before pushing; do not overwrite the remote
feature branch or replace active checkouts as a synchronization shortcut.
