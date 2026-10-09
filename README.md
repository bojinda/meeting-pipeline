# Meeting Pipeline

A self-hosted transcription and summarization pipeline built around **WhisperX**, **speaker-aware chunking**, and **Ollama**.

It is designed to be modular and self-hosted: audio logging, transcription, summarization, and automation can be used together or independently.

This pipeline can be used in two main modes:

- **Meeting mode** for structured meeting summaries
- **Lesson mode / study companion mode** for course videos, lectures, and certification study material

---

## Features

- Remote or automated audio logging from another host
- Diarized transcription with WhisperX
- Speaker-aware transcript chunking
- Map/reduce-style summarization with Ollama
- Meeting-oriented outputs such as:
  - `summary.md`
  - `action-items.md`
  - `minutes-draft.md`
- Lesson-oriented outputs such as:
  - `lesson-notes.md`
  - `key-terms.md`
  - `quiz.md`
  - `flashcards.md`
  - `study-guide.md`
- `.env`-driven configuration
- Example `systemd` units for unattended automation
- Optional Home Assistant controls
- Optional post-transcription source-audio deletion for temporary audio logging workflows

---

## Pipeline Overview

The typical flow is:

1. Start temporary audio logging
2. Forward logged audio to a WhisperX server
3. Run WhisperX transcription with diarization
4. Chunk the transcript into speaker-aware sections
5. Summarize chunks with Ollama using the selected output profile
6. Write final outputs to disk
7. Delete the logged audio immediately after successful transcription

---

## Architecture

A common setup looks like this:

### Audio source host
This can be:
- a Windows machine logging meeting/system audio
- another host that can stream audio to the processing server

### Processing host
Runs:
- audio logging listener
- WhisperX
- transcript chunker
- Ollama summarization
- optional systemd automation

### Optional control layer
You can trigger the pipeline via:
- manual wrapper scripts
- Home Assistant buttons/scripts
- webhook automation when appropriate

The recommended design is to keep the audio/transcription/summarization logic on the processing host and use thin wrappers or webhooks for remote triggers.

---

## Control Options

This project supports multiple control styles:

- **Manual start/stop** via wrapper scripts
- **Home Assistant buttons/scripts** for dashboard control
- **Webhook-driven automation** where appropriate

For lessons, manual start/stop is often sufficient.  
For meetings, manual control or external automation can be used depending on the environment.

---

## Profiles

The pipeline is easiest to think of as one shared engine with different output profiles.

### Meeting profile
Best for:
- union meetings
- internal meetings
- committee-style discussions
- formal or semi-formal spoken sessions

Typical outputs:
- `summary.md`
- `action-items.md`
- `minutes-draft.md`

### Lesson / study companion profile
Best for:
- certification videos
- technical course lectures
- conference talks
- educational videos

Typical outputs:
- `lesson-notes.md`
- `key-terms.md`
- `quiz.md`
- `flashcards.md`
- `study-guide.md`

The audio logging, transcription, and chunking stages can stay mostly the same; the main difference is the summarization prompt and output files.

---

## Project Structure

```text
meeting-pipeline/
├── bin/
│   ├── postprocess-meeting.sh
│   ├── transcript_chunker.py
│   ├── ollama_meeting_summary.py
│   ├── ha-start-meeting.sh
│   └── ha-stop-meeting.sh
├── config/
│   └── .env
├── systemd/
│   ├── meeting-capture.service
│   ├── meeting-postprocess.path
│   └── meeting-postprocess.service
├── meeting-recordings/
├── meeting-transcripts/
├── meeting-summaries/
├── lesson-transcripts/
└── lesson-summaries/
```

You do not need to use every directory at once. The processing and output roots can be customized through `.env`.

---

## Requirements

At a minimum, you will need:

- a Linux processing host
- Python environment for WhisperX
- Ollama reachable over HTTP
- a GPU strongly recommended for WhisperX and summarization
- a Hugging Face read token for diarization-enabled WhisperX use

Optional:
- Home Assistant for UI control
- a remote Windows audio source host
- meeting-start webhooks or join-detection automation

---

## Outputs

### Meeting transcripts
Stored under:

```text
meeting-transcripts/<session-folder>/
```

These typically include:
- WhisperX transcript outputs
- status/log files
- chunked transcript files under `chunks_out/`

### Meeting summaries
Stored under:

```text
meeting-summaries/<session-folder>/
```

These typically include:
- `summary.md`
- `action-items.md`
- `minutes-draft.md`
- `chunk_summaries.jsonl`

### Lesson transcripts
Stored under:

```text
lesson-transcripts/<lesson-folder>/
```

### Lesson summaries
Stored under:

```text
lesson-summaries/<lesson-folder>/
```

These can include:
- `lesson-notes.md`
- `key-terms.md`
- `quiz.md`
- `flashcards.md`
- `study-guide.md`
- `chunk_summaries.jsonl`

---

## Configuration

Copy the example environment file and edit it for your setup:

```bash
cp config/env.example config/.env
```

Important settings include:

### WhisperX
- `HF_TOKEN`
- `WHISPERX_MODEL`
- `WHISPERX_BATCH_SIZE`
- `WHISPERX_COMPUTE_TYPE`
- `WHISPERX_DEVICE`
- `WHISPERX_DEVICE_INDEX` (default `0`, passed explicitly to WhisperX)
- `WHISPERX_LANGUAGE`

### Chunking
- `TRANSCRIPT_CHUNK_TARGET_WORDS`
- `TRANSCRIPT_CHUNK_MAX_WORDS`

### Ollama
- `OLLAMA_URL`
- `OLLAMA_MAP_MODEL`
- `OLLAMA_REDUCE_MODEL`
- `OLLAMA_KEEP_ALIVE`
- `OLLAMA_TEMPERATURE`
- `OLLAMA_MAP_NUM_CTX`
- `OLLAMA_REDUCE_NUM_CTX`

Optional meeting-only overrides: `MEETING_MAP_MODEL`, `MEETING_REDUCE_MODEL`,
`MEETING_MAP_NUM_CTX`, and `MEETING_REDUCE_NUM_CTX`. In meeting mode, precedence
is CLI flag → nonempty `MEETING_*` variable → nonempty `OLLAMA_*` variable →
existing built-in default (`qwen2.5:32b` for models; Ollama's default context
when no context is configured). Lesson mode ignores these meeting overrides.

### Output paths
- `MEETING_SUMMARIES_ROOT`
- `LESSON_SUMMARIES_ROOT`
- `MEETING_TRANSCRIPTS_ROOT`
- `LESSON_TRANSCRIPTS_ROOT`

### Optional remote control
- `WINDOWS_HOST`
- `WINDOWS_USER`
- `WINDOWS_SSH_KEY`
- `WINDOWS_START_TASK`
- `WINDOWS_STOP_SCRIPT`
- `MEETING_CONTROL_LOG`

### Source-audio cleanup
Suggested settings:

- `DELETE_SOURCE_AUDIO_AFTER_TRANSCRIPTION=1`
- `DELETE_SOURCE_AUDIO_ONLY_ON_SUCCESS=1`

Recommended behavior:
- delete source audio only after WhisperX has successfully produced the transcript outputs you need
- keep the transcript, chunked transcript, and summary artifacts
- do **not** delete logged audio before successful transcription completes

---

## Systemd Automation

Example `systemd` units are provided in the `systemd/` directory:

- `meeting-capture.service.example`
- `meeting-postprocess.service.example`
- `meeting-postprocess.path.example`

Copy and adapt them for your machine:

```bash
sudo cp systemd/meeting-capture.service.example /etc/systemd/system/meeting-capture.service
sudo cp systemd/meeting-postprocess.service.example /etc/systemd/system/meeting-postprocess.service
sudo cp systemd/meeting-postprocess.path.example /etc/systemd/system/meeting-postprocess.path
```

Edit the copied units for your environment, including:

- `User=`
- `WorkingDirectory=`
- `ExecStart=`
- any host-specific paths

Then reload systemd:

```bash
sudo systemctl daemon-reload
```

---

## Typical Workflow

### Manual testing
A common testing path is:

1. Start audio logging
2. Stop audio logging
3. Let the postprocess pipeline run automatically
4. Inspect:
   - transcript folder
   - status/log files
   - summary output folder

### Production / unattended mode
A typical unattended setup uses:

- `meeting-capture.service`
- `meeting-postprocess.path`
- `meeting-postprocess.service`

with external triggers from:
- Home Assistant
- wrapper scripts
- webhook automation

---

## Home Assistant Integration

Home Assistant is optional, but works well as a control layer.

Recommended pattern:
- use `shell_command` to SSH into the processing host
- call:
  - `ha-start-meeting.sh`
  - `ha-stop-meeting.sh`
- expose those via two HA scripts or dashboard buttons

Suggested controls:
- **Start Audio Logging**
- **Stop Audio Logging**

This works well for both meetings and lesson capture, especially when manual control is sufficient.

---

## Cooperative GPU Resources

The following logical two-GPU configuration is an example:

| Logical resource | Pipeline stage | Default shared lock file |
| --- | --- | --- |
| GPU 0 | WhisperX | `/tmp/aihub-gpu0.lock` |
| GPU 1 | Ollama, all map/reduce calls | `/tmp/aihub-gpu1.lock` |

These are logical cooperative resource locks. Installations may use UUID-derived
or otherwise stable paths, but every workload sharing a physical GPU must use
the same configured path/inode. A lock filename has no special relationship to
CUDA and does not assign a workload to a device. Explicit `AIHUB_GPU0_LOCK_FILE`
and `AIHUB_GPU1_LOCK_FILE` environment settings override the generic defaults,
so existing installations keep their configured resource paths. Other workloads
can adopt those same configured locks. The pipeline does not set GPU visibility
variables or reconfigure the Ollama server.

Configure the shared resources in `config/.env`:

```env
AIHUB_GPU0_LOCK_FILE=/tmp/aihub-gpu0.lock
AIHUB_GPU1_LOCK_FILE=/tmp/aihub-gpu1.lock
AIHUB_GPU_LOCK_TIMEOUT=3600
WHISPERX_DEVICE=cuda
WHISPERX_DEVICE_INDEX=0
```

Both postprocess wrappers use the canonical runner's shared durable admission
and the original physical GPU locks: GPU0 for foreground WhisperX, CPU chunking
outside reservations, then GPU1 for the entire normal map/reduce and recap stage.
GPU0 transcription and GPU1 summarization may overlap; same-resource jobs serialize.
WhisperX keeps its existing device index. Device placement and backend cleanup
still require approved live calibration before deployment.

`bin/with-gpu-lock.sh` requires `AIHUB_GPU_RUNNER_CONFIG`, a Python 3.11+ runner
interpreter (`AIHUB_GPU_RUNNER_PYTHON`) and the canonical SDK in the child summary
environment. Its trusted config must share the runner's journal, scope and
original physical lock paths; the `/tmp` names above are examples, not permission
to replace production locks. Missing config/package or mismatched locks fails
closed. No standalone flock or direct HTTP fallback is used in managed stages.
`AIHUB_GPU_LOCK_TIMEOUT` limits admission waiting, not runtime; timeout returns 75.
The helper retains signal/exit handling and the HA start/stop interface.

The admission check, native physical acquisition and durable claim are atomic
under the existing runner contract. Do not nest a second GPU acquisition inside
a stage. A known terminal result and verified backend cleanup are required for
release. Interruption, lost replies or cleanup failure retain durable ownership
even if the OS descriptor closes. Use exact operator-approved recovery; never
delete the journal, unlink locks or bypass the runner to clear a hold. Legacy
unwrapped clients must remain excluded.

For the supported meeting command vector (Python interpreter, summary script,
transcript directory, then options), the helper binds selected approved aliases,
approved turn corrections and the chunk index into stage identity. Private alias
and correction snapshots are passed to the summarizer. Missing optional files
remain missing; an explicitly selected missing alias file fails before admission.
A changed approved input produces a new automatic ID or conflicts with a reused
explicit `AIHUB_GPU_STAGE_ID`. Inputs changed while waiting prevent launch; changes
during execution fail the stage after verified cleanup rather than cache success.
Hold other semantic inputs, prompts and code stable during a stage; use a new
reviewed ID for their intentional changes. No approval or speaker substitute is
created by this mechanism.

### Historical meeting summarization without recordings

The supported entry point below consumes an existing, nonempty
`chunks_out/transcript_chunks.jsonl`. It does not run WhisperX, rechunk audio,
require a recording or change the audio-deletion policy. By default it acquires GPU1
through the shared runner and preserves map/reduce, recap and redactions.
This command is a future live-use template, not authorization to run inference:

```bash
MEETING_CONFIG_FILE="/approved/private/meeting.env" \
MEETING_SUMMARIES_ROOT="/approved/private/new-comparison-output" \
bash bin/summarize-existing-meeting.sh "/approved/existing/transcript-directory" --keep-recap
```

The launcher sources the selected config (default `config/.env`). An explicitly
supplied `MEETING_SUMMARIES_ROOT` wins over that file so comparison runs need not
overwrite accepted output. Choose a fresh private destination and preserve existing
outputs. Keep the configured normal models; no model upgrade is part of this work.
Supported optional flags are `--keep-recap`, `--no-keep-recap`, and two-argument
`--speaker-aliases`, `--map-model`, `--reduce-model`, `--map-num-ctx`,
`--reduce-num-ctx`, `--keep-alive`, `--temperature`, `--ollama-url`.
Whole-meeting modes, speaker suggestions and other flags are rejected. Omit
`--speaker-aliases` unless an existing approved file is selected. Default approved
files are the existing `speaker_aliases.json` and `speaker_turn_corrections.json`
inside that transcript directory. Missing approvals remain unresolved.

An explicit `--dual-gpu-target NAME` selects the configured session-only dual
target and `gpu0+gpu1`. It requires matching `AIHUB_GPU_DUAL_APPROVAL` and the
operator-approved mode policy; no automatic mode selection or context/model change
occurs. The same map/reduce/recap session holds both resources until model unload,
backend readiness and original GPU1 placement/cleanup are verified. Uncertain
switching/work retains both durable owners; reconnect never blindly replays.
Routine historical commands and existing HA/transcription interfaces stay unchanged.
See the canonical runner's dual session/deployment guide before any live use.

Same ID/hash reconnects to prior completion without rerunning generation; it does
not recreate deleted output files. Uncertain/interrupted work is never replayed
automatically. Preserve private stage/client state for reconnect and recovery.

With the canonical SDK available, run synthetic native Linux tests (no GPU jobs):

```bash
python3 -B -m unittest discover -s tests -p test_meeting_stage_identity.py -v
python3 -B -m unittest discover -s tests -p test_gpu_admission.py -v
```

## Notes on Ollama Usage

The summarizer is designed around a map/reduce pattern:

- **Map step:** summarize transcript chunks
- **Reduce step:** combine chunk summaries into final outputs

Recommended context split:
- map/chunk summarization: smaller context
- reduce/final synthesis: larger context

Example:

```env
OLLAMA_MAP_NUM_CTX=16384
OLLAMA_REDUCE_NUM_CTX=32768
```

If `ollama ps` shows the larger context after a run, that is usually expected because the reduce step was the last model state loaded.

### Recommended local models

For this meeting profile, use [`qwen3.8:27b`](https://ollama.com/library/qwen3.8)
for reduce/final minutes. Use [`qwen3.6:27b`](https://ollama.com/library/qwen3.6)
or your existing `qwen2.5:32b` for map/chunk summaries.
[`qwen2.5-coder`](https://ollama.com/library/qwen2.5-coder) models are not recommended for meeting
minutes; use a general-purpose model for this prose workflow.

Start with a reduce-only upgrade: keep your current map model and change the
reduce model to `qwen3.8:27b`. Review the resulting minutes before trying
`qwen3.8:27b` for both stages. Models remain configurable; no Qwen3 model name
is required by the summarizer.

Recommended command with the newer map model and an optional historical recap:

```bash
bash bin/with-gpu-lock.sh gpu1 "Ollama meeting summaries (GPU1)" \
  python bin/ollama_meeting_summary.py meeting-transcripts/session-123 \
  --map-model qwen3.6:27b --reduce-model qwen3.8:27b \
  --map-num-ctx 16384 --reduce-num-ctx 32768 --keep-recap
```

For a reduce-only upgrade, replace `--map-model qwen3.6:27b` with
`--map-model qwen2.5:32b` or your existing fast model. Omit `--keep-recap` when
you do not want the recap, or use `--no-keep-recap` to override an enabled
`MEETING_KEEP_RECAP` setting.

For unattended meeting runs, uncomment these optional settings in `config/.env`:

```env
MEETING_MAP_MODEL=qwen3.6:27b
MEETING_REDUCE_MODEL=qwen3.8:27b
MEETING_MAP_NUM_CTX=16384
MEETING_REDUCE_NUM_CTX=32768
```

The postprocess wrapper exports `config/.env`. For direct Python invocation,
export the variables yourself or use the explicit CLI command above. CLI flags
always win, including when `--profile meeting` is passed to the shared engine.

Home Assistant's existing `shell_command` calls over SSH use the same
`bin/ha-start-meeting.sh`, `bin/ha-stop-meeting.sh`, `bin/ha-start-lesson.sh`, and
`bin/ha-stop-lesson.sh` interfaces. The meeting stop script sends the remote
stop command; model selection occurs in the subsequent `postprocess-meeting.sh`
step, which already exports `config/.env` and calls the meeting summarizer
without model flags. Enable `MEETING_*` there for meeting runs. Lesson start/stop
behavior and lesson model defaults remain unchanged.

---

## Meeting Minutes Post-processing

Meeting mode applies speaker aliases and known text repairs before map-stage
summarization, and again to generated text. Raw transcripts and chunk indexes
are preserved. Lesson mode continues to use its existing flow. These helpers
use the Python standard library and require no additional packages.

### Speaker names

Place `speaker_aliases.json` in the individual meeting's transcript directory,
alongside `chunks_out/`. Use only identities you have verified:

```json
{
  "SPEAKER_00": "Morgan",
  "SPEAKER_01": "Riley"
}
```

See `config/speaker_aliases.example.json` for a sample. The file is optional;
unmapped labels remain available for review. Invalid mappings stop generation
before any Ollama calls. After updating names, rerun the summarizer:

```bash
python bin/ollama_meeting_summary.py meeting-transcripts/session-123
```

To select another mapping file, pass `--speaker-aliases /path/to/names.json`.
The mapping is a flat JSON object with plain, single-line names.

### Optional speaker suggestions (private, advisory only)

Meeting mode can suggest names from **non-redacted text evidence**. Enable it
explicitly with `--suggest-speakers`; normal processing and Home Assistant
commands retain their defaults. There are no voiceprints, speaker embeddings,
biometric matching or cross-meeting identity stores. Heuristic mode makes no
additional model calls. Only the explicit local LLM option adds bounded discovery
and verification calls (at most two per invocation).
Suggestions never modify or replace approved aliases and are never supplied to
the map/reduce models. Approval remains the operator's decision.

Operator workflow:

1. Process the meeting with the optional suggestion stage:
   ```bash
   python bin/ollama_meeting_summary.py meeting-transcripts/session-123 --suggest-speakers
   ```
   Or generate suggestions without running Ollama:
   ```bash
   python bin/suggest_meeting_speakers.py suggest meeting-transcripts/session-123
   ```
2. Optionally review conversational context using local Ollama, without rerunning
   WhisperX or the meeting map/reduce stages:
   ```bash
   python bin/suggest_meeting_speakers.py suggest meeting-transcripts/session-123 --llm
   ```
   Or add it to normal meeting processing explicitly:
   ```bash
   python bin/ollama_meeting_summary.py meeting-transcripts/session-123 --suggest-speakers --suggest-speakers-llm
   ```
3. Inspect `meeting-summaries/session-123/speaker-suggestions.json` privately.
   Each unresolved visible `SPEAKER_XX` has a suggested name or `null`, a
   confidence label, evidence excerpts/types/IDs, conflicting candidates, and
   `origin` (`heuristic`, `llm`, or `both`). In the default two-pass mode,
   `llm_review` separately records source-turn assignments, both passes,
   coverage, validation issues, and diagnostics; `speaker-turns.json` contains
   the corresponding redacted source text and effective approved identities.
4. Verify identities, then approve/edit the existing per-meeting
   `speaker_aliases.json`. The optional helper merges **only** explicitly selected,
   unambiguous suggestions, never all suggestions:
   ```bash
   python bin/suggest_meeting_speakers.py approve meeting-transcripts/session-123 --approve SPEAKER_03
   ```
   Multiple `--approve` flags select multiple labels. It rejects a review from
   another meeting or a changed visible transcript and cannot overwrite a
   different existing approved alias.
   This label-wide helper is for heuristic or explicit legacy reviews only;
   two-pass reviews require the exact-turn approval workflow below. The helper
   rechecks current source relationships and model evidence. Editing
   JSON confidence/candidate/ambiguity fields cannot bypass that verification.
   Low-confidence, ambiguous, or null suggestions require manual verification
   and alias editing. Use the same `--speaker-roster` when reviewing/approving
   with an explicitly selected roster.
5. Rerun postprocessing using approved aliases:
   ```bash
   python bin/ollama_meeting_summary.py meeting-transcripts/session-123
   ```

Both commands use `MEETING_SUMMARIES_ROOT` when set. Standalone `suggest` also
accepts `--output-dir`; `approve --suggestions /private/path/speaker-suggestions.json`
selects a review there. Existing `--speaker-aliases` behavior remains supported.

#### Source turns and explicit corrections

A diarization label can represent multiple people. Use an exact-turn correction
instead of assigning one global alias to a mixed label:

```bash
python bin/suggest_meeting_speakers.py inspect-turns meeting-transcripts/session-123
python bin/suggest_meeting_speakers.py suggest meeting-transcripts/session-123 --llm
# Inspect the private JSON files, then copy the exact T... ID you verified:
python bin/suggest_meeting_speakers.py approve-turn meeting-transcripts/session-123 --turn-id T... --name "Casey"
python bin/suggest_meeting_speakers.py turn-conflicts meeting-transcripts/session-123
python bin/ollama_meeting_summary.py meeting-transcripts/session-123
# To undo only that correction:
python bin/suggest_meeting_speakers.py remove-turn meeting-transcripts/session-123 --turn-id T...
```

`approve-turn` requires an operator-supplied ID **and** name on every invocation;
it never imports model assignments. Inspecting, suggesting, or editing a review
artifact cannot approve a correction. It writes private
`speaker_turn_corrections.json` beside the source index, leaving global aliases
and source files unchanged. The correction takes precedence over a global alias
only for that bound turn during meeting postprocessing. Other turns retain their
existing alias behavior. Conflicts are written to private
`speaker-turn-conflicts.json`; inspection supports `--turn-id` and both inspection
commands support `--output-dir`. Nothing prints transcript excerpts to logs.

Turn IDs hash original source text, speaker, and source coordinates before
redaction. Existing unambiguous WhisperX turn metadata supplies precise timestamps
when available; historical indexes otherwise use source chunk/line coordinates
and containing chunk timestamps. No WhisperX rerun is required. Repeating a review
against the same source produces the same IDs; aliases do not change them.
Changed bindings, fully redacted turns, and corrections copied from another
meeting are rejected. Adding/changing source metadata or moving a session can
require new approvals. `remove-turn` can remove a stale entry. A merged ASR line
remains one source unit: this feature does not split speakers inside that line.

Turn corrections are a direct human decision, independent of a model's confidence.
Conflicting or unverified model assignments have `suggested_name: null`, retain
their leads privately, and cannot be promoted through label-wide `approve`.
Even a grounded, independently verified assignment remains advisory and requires
an explicit `approve-turn`. There is no automatic correction or bulk approval.

An optional per-meeting `speaker_roster.private.json` can supply preferred
spellings, name variants, and roles:

```json
{"people": [{"name": "Taylor Morgan", "aliases": ["Taylor"], "role": "Chair"}]}
```

A list of names is also accepted. `--speaker-roster /private/path/speaker_roster.private.json`
selects another private roster. **A roster or role alone never establishes an
identity.** Only an evidenced name can match a roster candidate. Self-identification
is high confidence; an introduction with an immediate response or repeated
address/response patterns is medium; a single address/response is low. Conflicting
names, ambiguous roster variants, or a name approved for another label produce
`null` for human review. Role context is secondary and cannot raise confidence.
These are heuristic review labels, not statistical probabilities.

Discourse/courtesy prefixes are parsed before direct-address detection, so
ordinary openings such as “Pardon, could you...” or “But, can you...” are not
names. A bare question such as “Taylor?” requires a private roster match or an
earlier independent identity cue. Adjacent same-speaker lines such as “Yeah, that's me.” followed immediately
by “Taylor Morgan.” can support identity; intervening speakers, unrelated text,
or redaction gaps prevent that joining. Strong self-identification does not
establish identity for every occurrence of a diarization label. Conflicting
address/self-identification evidence is retained with `suggested_name: null`;
there is no automatic speaker splitting or correction.
The standalone command reads `chunks_out/transcript_chunks.jsonl`. Its chunker
can join multiple ASR segments into one line, unlike a turn-per-line text export.
A fragmented self-identification within a merged exchange is retained as
uncertain source attribution and cannot be approved through the helper, even
without a competing name. Compare the session index when checking production
candidate counts; the text export may have different grouping.

Local LLM review uses the existing `OLLAMA_URL` and meeting reduce-model default.
`--speaker-suggestion-model MODEL` overrides only this review's model;
standalone `--reduce-model` and `--ollama-url` also select settings explicitly.
The default two-pass review has its **own 98,304-token context**, independent of
meeting map/reduce settings. Set `SPEAKER_REVIEW_NUM_CTX` or the higher-priority
`--speaker-review-num-ctx` flag. `--reduce-num-ctx` does not control this mode.
Source remains authoritative: names, labels, relationship
types, and evidence IDs must be grounded in the supplied redacted snippets.
Roster-only guesses, fabricated IDs, and invented names are rejected. A model
candidate whose name appears in cited transcript evidence but whose relationship
cannot be independently grounded is kept privately as an unverified assignment
or verification lead (legacy: `unverified_leads`), with evidence IDs and no
approval eligibility. This includes possible relationships
and unrelated mentions needing human interpretation; neither becomes a verified
candidate. Fresh approval checks repeat grounding from the current source.
Heuristic/model disagreement or explicit model uncertainty remains
for manual review; a model cannot choose a winner in a diarization collision or
replace an approved alias. Transcript text is untrusted data, never instructions.

Discovery receives the full redacted meeting when it fits its input budget.
Independent verification sees proposed names/turn IDs without discovery confidence,
their cited context, neighboring turns, and competing identity evidence elsewhere
in the meeting. It uses a smaller context when possible. Each pass is one request,
temperature zero, with a 600-second HTTP timeout and no automatic retry.
If discovery fails or finds no candidates, verification is skipped.

Input budgeting includes instructions, JSON schema, roster, a 1,024-token framing
reserve, and a structured-output reserve (up to 16,384 tokens for discovery and
8,192 for verification). Configure a **matching local** `tokenizer.json` with
`SPEAKER_REVIEW_TOKENIZER` or `--speaker-review-tokenizer` to count model tokens;
this requires the optional Python `tokenizers` package in the processing environment.
No tokenizer/model is downloaded. Without one, the conservative UTF-8 byte upper
bound may select windows even when the model's actual token count would fit.
The private report records the counting method and actual Ollama token counts
when returned. A configured but unavailable tokenizer fails safely.

If the meeting does not fit, discovery plans contiguous source-turn windows with
up to two overlapping turns, processes **one selected window per invocation**,
and records incomplete coverage plus all included/omitted/oversized turn IDs.
Whole source turns are never silently truncated. Select another zero-based window
with `--speaker-review-window N` (or `SPEAKER_REVIEW_WINDOW`); use a distinct
`--output-dir` for each review to retain earlier window results. An individually
oversized turn is reported uncovered. Verification that cannot fit its full
evidence set is reported incomplete rather than dropping conflicting context.
For example:

```bash
python bin/suggest_meeting_speakers.py suggest meeting-transcripts/session-123 --llm --speaker-review-num-ctx 98304 --speaker-review-tokenizer /private/models/matching-tokenizer/tokenizer.json --speaker-review-window 0 --output-dir /private/reviews/session-123/window-0
```

For compatibility, explicitly select `--speaker-review-mode legacy` to use the
previous single-call label reviewer. Only that mode retains the 32,768-token
maximum, 18,000-character sampled evidence limit, meeting reduce-context default,
and 120-second timeout. The default two-pass mode has neither of those old caps.
The request supplies an explicit [Ollama JSON schema](https://docs.ollama.com/capabilities/structured-outputs)
and validates the generated JSON locally. Speaker review disables model thinking
by default; set `SPEAKER_REVIEW_THINK=true` to request it, or `default` to omit
that option for a model that requires its own default. This setting affects only
speaker review and never adds a retry or another inference call. Public AI endpoints,
redirects, environment HTTP proxies, and cloud-tagged models are not used;
select an installed local model. No live model is required
by the tests. Unavailable Ollama, timeout, or invalid JSON leaves safe heuristic
results and an incomplete/unavailable status; normal meeting processing continues.
Private per-pass `llm_review.passes[].diagnostics` (or legacy
`llm_review.diagnostics`) records a failure category, generated-output
character count, `done_reason` when supplied by Ollama, token/thinking-length
counts when available, and validation error categories. It distinguishes empty
or malformed generated JSON, invalid schema, generation token limits, HTTP/model/
transport failures, and grounding failures. Rejected candidates include a
zero-based index, offending field, rejection category and valid target turn ID;
these diagnostics never contain rejected names or transcript excerpts.
A name-validation rejection can be quarantined only when remaining candidates
are source-grounded and demonstrably independent of its labels, evidence and
conflicts. Shared labels, overlapping or linked evidence, unresolved conflicts,
and unverifiable dependencies remain fail-closed. Verification may proceed for
independent candidates, but any rejection keeps the overall review incomplete.
Suggestions remain advisory and require explicit operator approval.
Raw malformed responses, thinking
text, and error bodies are discarded; neither model responses nor transcript
evidence are printed in public logs. Earlier reports with only
`invalid_json_or_schema` lack enough metadata to identify the original cause.

Standalone LLM review uses `bin/with-gpu-lock.sh` for GPU1. The helper exposes a
managed-lock ownership marker to its child so a speaker review inside an already
locked meeting stage does not acquire GPU1 again. The existing lock paths,
timeouts, signal/exit behavior, persistent files, and physical assignments stay
the same. Heuristic-only review needs no GPU lock. For direct full processing,
the existing cooperative wrapper can cover all model calls:
```bash
bash bin/with-gpu-lock.sh gpu1 "Meeting summaries" python bin/ollama_meeting_summary.py meeting-transcripts/session-123 --suggest-speakers --suggest-speakers-llm
```

Spoken redaction is applied before evidence lookup, including in the standalone
helper. Address/response pairs involving redaction-affected source chunks are
omitted conservatively; visible self-identifications can still be reviewed.
A redaction-bearing rerun clears stale suggestion artifacts even when suggestions
are disabled. Source transcripts and recordings are not edited by this stage.

Suggestions, rosters, aliases, turn catalogs, turn corrections, conflict reports,
and private temporary writes are gitignored.
Suggestion/approval writes use atomic replacement and owner-only `0600`
permissions on POSIX; protect the private directory with filesystem ACLs on
Windows. Custom roster/alias filenames must also remain private and gitignored.
Speaker review files and links to them are explicitly denied by the shared
publication/export boundary; the three public document names are unchanged.

### Experimental whole-meeting synthesis (Checkpoint B)

Map/reduce remains the default. `--synthesis-mode whole` is an explicit meeting-only
experiment against an existing `chunks_out/transcript_chunks.jsonl` index. It
requires no recording, WhisperX rerun, or diarization changes. Run speaker review
and approve aliases/turn corrections separately first; unapproved suggestions are
never identities in this mode.

Use a **new** destination outside the source directory and production summary
root. The runner rejects existing destinations rather than overwriting a previous
experiment. Example operator commands (run only after reviewing this checkpoint):

```bash
python bin/ollama_meeting_summary.py meeting-transcripts/session-123 --synthesis-mode whole --synthesis-model qwen3.8:27b --synthesis-num-ctx 98304 --synthesis-tokenizer /private/models/matching-tokenizer/tokenizer.json --experiment-output-dir ignore/experiments/session-123-whole --keep-recap
# Compare the unchanged production map/reduce result with the experiment:
diff -u meeting-summaries/session-123/minutes-draft.md ignore/experiments/session-123-whole/minutes-draft.md
diff -u meeting-summaries/session-123/action-items.md ignore/experiments/session-123-whole/action-items.md
```

The whole-meeting context defaults independently to **98,304 tokens**. Optional
`MEETING_SYNTHESIS_NUM_CTX`, `MEETING_SYNTHESIS_MODEL`, and
`MEETING_SYNTHESIS_TOKENIZER` supply defaults; the corresponding CLI flags win.
The model otherwise inherits the selected meeting reduce model. The tokenizer
otherwise uses `SPEAKER_REVIEW_TOKENIZER` when configured. A matching local
Qwen `tokenizer.json` and the optional `tokenizers` Python package enable model
counts without downloads. Without that tokenizer, conservative UTF-8 byte
budgeting may cause a fallback even when actual Qwen tokens would fit.
`MEETING_SYNTHESIS_THINK=false` is the review default; `true` or `default` explicitly
requests thinking or the model default. Map/reduce and lesson settings remain
independent. The existing Ollama GPU1/Q8 configuration is used unchanged.

The experiment applies the same redaction, normalization, classifier, approved
global aliases and exact-turn corrections as meeting processing. It preserves
chronological classified source records, retains historical recap separately,
and excludes pre-meeting chatter from **both** whole-mode model prompts and all
three public documents. Source indexes remain immutable. Containing chunk times
are identified as source coordinates, not invented precise sentence timestamps.

Two principal model stages are used:

1. Scan the entire meeting, then extract substantive evidence in priority order:
   motions, explicit actions, decisions, outstanding issues, health/safety,
   qualifications/disagreements, historical recap, then other discussion topics.
   A topic represents a meaningful subject supported by relevant records, rather
   than one item per sentence/turn. The `meeting-evidence-v2` response contains
   `id`, `kind`, canonical `section`, `primary_record_id`,
   `supporting_record_ids`, `owners`, `mover`, `seconder` and `outcome`.
   It does **not** return duplicate statements/quotation bodies or free-text topic
   titles. Python hydrates exact text and quotations from the immutable originals
   before applying the established source validation. Named motion roles and outcomes need explicit source evidence;
   a meeting ending is not evidence that a motion carried. Actions need supported
   undertakings/assignments for every owner, with existing outreach/proposal
   guards and actor/recipient boundaries. Explicit collective undertakings with no
   named owner stay unassigned and add a QA finding; their announcing speaker is
   not silently made responsible. Historical actions stay historical.
2. Organize the validated evidence into structured document plans. Deterministic
   rendering produces the existing Markdown filenames from those evidence IDs,
   so this pass cannot add an ungrounded prose claim or identity. Every validated
   item is retained in the appropriate documents. Summary recap is explicitly
   historical; minutes recap is controlled by `--keep-recap`; action items always
   exclude recap. Source IDs appear only in private artifacts.

Whole-mode source encoding uses short **run-local** record numbers instead of
repeating hash IDs. The model receives compact JSON arrays grouped by section,
with one shared speaker table: each row is `[record_number, speaker_index, text]`.
`B`, `R`, and `A` identify current business, historical recap, and adjournment.
Every utterance stays in its own row, in order; identical utterances and different
speakers are never collapsed. Redaction-affected source turns and turns following
removed content are marked in `redaction_gaps`. These marks are conservative:
continuity within or across such turns must not be assumed.

`whole-source.json` retains complete original classified records, timing/provenance,
source-turn IDs, original line coordinates, and the exact `compact_to_original`
map plus a binding digest. Model citations use canonical decimal JSON strings
(e.g. `"1"`) in primary/supporting reference fields. The validator resolves them
to original record IDs and restores **original record bodies**, never model
paraphrases. Full quotations and original provenance remain in private evidence;
unsupported category/role/owner interpretations are still rejected. Unknown/noncanonical IDs, missing rows,
collisions, changed speaker/section boundaries, and changed quotations fail closed.
No content, qualification, ownership rule or output reserve is removed to fit.

A supported **read-only preflight** prepares the same source and performs the same
accounting as the actual experiment, without an output destination, Ollama calls,
GPU locks, or writes to transcripts/processing outputs:

```bash
python -B bin/ollama_meeting_summary.py meeting-transcripts/session-123 --synthesis-mode whole --synthesis-preflight --synthesis-model qwen3.8:27b --synthesis-num-ctx 98304 --synthesis-tokenizer /private/models/matching-tokenizer/tokenizer.json
```

Its JSON report contains source coverage and excluded-section counts, the old
verbose and new compact source costs, transcript text, record IDs, section/speaker
identifiers, JSON fields/structure, instructions/schema, the full generation and
framing reserves, required context, signed remaining headroom, and selected mode.
The source breakdown uses **ordered marginal ablations** (text, IDs, sections,
speakers), leaving JSON structure and empty-placeholder costs as the residual.
This accounts for BPE token-boundary effects; components are order-dependent but
sum to the exact serialized-source count. Plain transcript tokenization is also
reported separately and is not added a second time to the budget. Saved tokenizer
truncation/padding is disabled for the in-memory whole-mode counter, never by
editing the tokenizer file. If the matching tokenizer is unavailable, byte-bound
counts are explicitly labelled; they are not Qwen token measurements.

A preflight selecting fallback does **not** run map/reduce. An actual experiment
still falls back safely if the complete compact source cannot fit. Preflight
checks extraction capacity; document-stage capacity is checked after evidence
extraction because that generated evidence does not exist during preflight.
Preflight can be run before deciding whether to approve another live experiment.

Evidence output has an explicit contract distinct from compact source encoding:
`item.id` is unique and matches `^E[0-9]{1,6}$` (e.g. `E1`, `E15`). Source
citations in v2 are canonical decimal strings such as `primary_record_id="1"`;
`supporting_record_ids` carries additional relevant evidence without copied text.
The primary record selects the actual source assertion/undertaking, not a filler
turn. Supporting quotations are restored verbatim, preserving each speaker;
they may include necessary conditions and disagreement. There is no arbitrary
item/support-reference cap in v2: bounded generation remains authoritative and
a limit hit always fails closed. Exact duplicate topic groups are reported and
rejected rather than silently discarded. Legacy statement/quotation responses
remain supported by offline validation for earlier retained artifacts only.
Output `item.section` uses the full names `current_meeting_business`,
`previous_meeting_recap`, or `adjournment`; input B/R/A codes are **not** valid
output section values. Schema and validator share the ID pattern and section
contract. Invalid formats/codes are rejected, never renamed or reinterpreted.

Structural diagnostics retain item indexes and fixed field/category names only:
`missing_or_extra_fields`, `invalid_evidence_id_format`, `duplicate_evidence_id`,
`invalid_kind`, `invalid_section`, `invalid_statement`, `invalid_owners`, and
`invalid_role_fields` (plus quotation diagnostics). An item can contribute several
categories; aggregate counts count distinct rejected items per category rather
than inflating counts for multiple bad role fields. `whole-run.json` and
`whole-evidence.json` include those counts and a distinct rejected-item count.
Neither diagnostic records nor console summaries include field values, quotes,
meeting names or raw model output. Structurally valid items still undergo all
exact quotation/reference, section, motion, decision and owner checks.

Optional **private response retention** avoids repeating inference to diagnose a
schema failure. Add `--synthesis-retain-response` to an explicitly reviewed whole
experiment command. This writes owner-only, atomic `whole-model-response.json`
inside its private isolated output, including the exact evidence-stage response
buffer, safe generation metadata and opaque source-binding digests. It is
explicitly gitignored and denied by publication/export/link filtering. Thinking
text and HTTP error bodies are not retained. Retention is off by default; it
adds no model call or retry and does not accept malformed/truncated JSON.

Validate a retained response offline against the **same** unchanged source and
approved identity configuration, without a tokenizer, GPU lock, model call,
public documents, alias edits, or response edits:

```bash
python -B bin/ollama_meeting_summary.py meeting-transcripts/session-123 --synthesis-mode whole --synthesis-validate-response ignore/experiments/session-123-whole/whole-model-response.json
```

The command reports accepted/rejected counts and safe indexed categories only.
It returns 0 for passed validation, 1 for rejected evidence, and 2 for invalid
binding/response/generation. Source binding covers the meeting location, complete
original transcript index, approved aliases/corrections, prepared records and
compact mapping. Changed or foreign source, changed approvals and accidental
response edits are rejected. A token-limit response remains unusable even if its
JSON happens to parse. Saved schema/prompt digests are provenance: offline checks
use the current strict validator so future validator fixes do not require another
model call. Offline success is diagnostic only and never applies the response to
meeting documents. Prior runs without retention cannot recover their discarded
raw responses through this helper.

The output remains a draft for operator review. Existing action filtering, motion
cleanup, formatting, normalization, private-reference removal and deterministic
QA are reused. Unresolved speakers and ambiguous roles remain flagged. Clear
source commitments and formal motion cues omitted by extraction add private QA
findings instead of invented replacement tasks. For actions only, Python may
restore the existing commitment parser's exact own-task span before a later
third-party step, while retaining the full original record in private quotations.
That span preserves its conditions and qualifications; it cannot reassign the
undertaking to the later actor. Full input coverage means all eligible text was supplied;
it does **not** certify perfect semantic recall. Compare evidence, QA and documents.
This experiment tests completeness/source accuracy before allowing free-form
paraphrased synthesis.

`whole-run.json` separates `context_coverage` (eligible records submitted) from
`semantic_evidence_coverage`. It records accepted counts by category,
duplicate/rejected counts, the allotted generation budget, estimated response
tokens and actual Ollama `eval_count`, plus private omitted commitment/motion
record IDs. Known source checks cover clear commitments and formal motion cues;
passing them never certifies complete semantic recall of every substantive issue,
condition, disagreement or topic. Missing evidence is disclosed as needing
operator review. If generation is truncated, accepted counts stay zero and
rejection/duplicate counts remain unknown rather than parsing partial JSON.
There are still exactly two principal inference stages, no retries and no
automatic speaker assignment. The 98,304 context, Q8 server configuration and
16,384 extraction reserve are unchanged.

Context planning reserves the full schema, instructions, a 1,024-token framing
allowance, up to 16,384 extraction tokens and up to 8,192 document-plan tokens.
The second stage uses a smaller context when possible. Neither stage truncates
records or drops qualifications/conflicting context to fit. If the complete
source or document-plan input cannot fit, the established map/reduce path runs
under the same GPU1 lock, writing only to the experiment's isolated destination.
`whole-run.json` records the explicit fallback reason. Invalid JSON, unsupported
claims/references, token-limit responses or transport failures stop safely;
there are no automatic retries or partial JSON acceptance.

The existing `with-gpu-lock.sh gpu1` supervisor covers both stages and any
fallback. A stage already holding that configured lock is reused. Lock paths,
timeouts, signal handling, device assignments, HA interfaces and audio deletion
are unchanged. No voice embeddings, automatic aliases, or publication are added.

Each experiment keeps these **private, gitignored, owner-only JSON artifacts**:

- `whole-source.json`: eligible redacted records, source coordinates/turn IDs,
  compact-to-original ID mapping and binding digest.
- `whole-evidence.json`: validated evidence and safe rejection categories.
- `whole-plan.json`: validated document plans referencing evidence IDs.
- `whole-model-response.json`: optional source-bound evidence response retained
  only with `--synthesis-retain-response`, including failed JSON/schema responses
  for private offline diagnosis.
- `whole-run.json`: requested/actual mode, coverage and excluded-section counts,
  configured context/tokenizer, actual Ollama input/output token counts when
  available, per-call and total runtime, fallback/failure reason, and QA counts.
  It also records the preflight token breakdown for the identical extraction
  input. Missing provider token counts remain `null`, never guessed as actual usage.

Existing `meeting_sections.jsonl`, `chunk_summaries.jsonl`, `minutes-qa.md` and
`minutes-qa.json` remain available. Whole mode has an empty chunk-summary file
because it did not map chunks; a fallback writes the established chunk summaries.
Private evidence/metrics filenames and links are explicitly denied at the shared
publication/export boundary. Its public three-document allowlist is unchanged.
Custom experiment locations must remain private and gitignored.

Files are prepared in a private staging directory and the completed directory is
renamed into place only after validation/rendering/QA succeed. Failed runs retain
private diagnostics with no `summary.md`, `minutes-draft.md` or `action-items.md`
at the destination. No real meeting trial or change to the default mode is part
of this checkpoint.

### Normalization and meeting sections

Known repairs include `Mack Yard`, `Mackyard`, and `Mac yard` → `Mac Yard`,
`Transport Canadaâ€™s` → `Transport Canada’s`, and `hypodermical` → `hypodermic`.
Encoding repairs target known sequences and preserve correctly encoded accents.

A stateful rule classifier labels transcript portions as `pre_meeting_chatter`,
`previous_meeting_recap`, `current_meeting_business`, or `adjournment`. It can
split a source chunk at section boundaries while retaining all text for the
map stage. When a formal start is recognized anywhere in the transcript, all
preceding conversation is pre-meeting chatter, including personal discussion,
clothing sizes, and invitations or informal tasks. The classifier recognizes
informal openings such as “Okay, yeah, I guess we'll get started there,”
“Okay guys, all set here,” and “Okay, everybody all set?” Explicit chair starts
with a recap/agenda/minutes are recognized, including natural readiness/start
constructions such as “start off with a recap of the last meeting” joined to the
preceding readiness phrase
without punctuation. Bare openings such as “Let's get started” are recognized,
while task continuations such as “Let's get started on the next project” or
“Let's get started with the equipment check” do not create meeting boundaries.
Boundary matching tolerates ordinary leading discourse/hesitation fillers
(`okay`, `yeah`, `so`, `well`, `uh`, `um`, `oh`) and up to three such fillers
between a readiness phrase and its opening agenda/recap construction. This
changes cue detection only; stored transcript wording is retained.

A request for previous notes or an intended recap marks recap setup, but does
not automatically make subsequent discussion historical. An actual historical
report begins the recap. Unmarked interruptions while awaiting that report, or
explicit current/recent discussion, stay current business. A retrospective cue
such as “So, yeah, the last meeting, Morgan reported...” begins/resumes historical
recap. Transitions such as “Moving right along,” a request to volunteer to go
first, or “I can go first” return to current business. An actual adjournment
statement marks the close.
A standalone explicit “Motion to adjourn” marks adjournment. Requests for a
motion, casual proposals, questions, and hypothetical future motions remain
current business. Approval of previous
minutes also remains current business. For recordings without a recognizable
formal start, ambiguous material retains the current-business fallback. Once
a recap is established, unmarked continuations retain its historical context;
current/recent discussion or an explicit current-business boundary can interrupt it.

Review `meeting_sections.jsonl` in the meeting's summary directory for the
normalized text, source chunk ID, section label, and classification evidence.
Time ranges refer to the containing source chunk, not precise section boundaries.
The classifier is heuristic: implicit transitions and mixed historical/current
statements still need human review.

`action-items.md` and `minutes-draft.md` use the same reduce input containing
only current-business and adjournment summaries. Recap and pre-meeting chatter
are excluded from that input. `summary.md` receives only historical recap,
current-business, and adjournment summaries; pre-meeting chatter is excluded
before all three public reduce calls. Recap in `summary.md` must be clearly
labelled historical context. To include a separate historical recap in draft
minutes, pass:

```bash
python bin/ollama_meeting_summary.py meeting-transcripts/session-123 --keep-recap
```

This summarizes recap portions separately and inserts them under the exact heading
**Recap of Previous Meeting**. Previous decisions and assignments stay in that
section. No recap section is added if none was identified. For unattended runs,
set `MEETING_KEEP_RECAP=1` in `config/.env`; `--no-keep-recap` overrides it.
An additional reduce call is made only when an identified recap is included.
The existing conservative rules for motions, decisions, and action items remain.

### Spoken redaction and private review

In **meeting mode**, say the exact commands:

- **“redact the following”** to start excluding content.
- **“end redaction”** to resume normal meeting content.

Matching is case-insensitive and accepts punctuation around the phrases and
whitespace between their exact words. Commands may span transcript/chunk
boundaries. Similar wording such as “redact following” or “stop redacting” does
not trigger redaction. The command phrases themselves are omitted from normal
meeting content. Redaction continues across speaker changes, turns, sentences,
and chunks, while text before the start and after the end is preserved.

Redaction happens **before** aliases, normalization, section classification,
and every Ollama map/reduce call. Excluded text is absent from the model prompts,
`chunk_summaries.jsonl`, `meeting_sections.jsonl`, summaries, action items,
minutes, historical recap, and QA excerpts. Existing classification, model
defaults, motion/action cleanup, lesson processing, and Home Assistant command
interfaces remain in place.

Each meeting writes a **private** `redactions.json` in its private summary
directory. It contains the actual excluded transcript wording with original
speaker labels, sequential redaction numbers, starting/ending source chunk IDs,
containing chunk timestamps (or `null`), encountered speakers, closure status,
and `explicit_spoken_redaction` or `unclosed_at_eof` reason. Its text is neither
normalized nor replaced with speaker aliases. Do not publish this directory or
the original transcript/chunks wholesale. Original transcript files are never
rewritten. When `DELETE_SOURCE_AUDIO_AFTER_TRANSCRIPTION=1`, successful meeting
transcription with a transcript JSON deletes the source recording through the
normal cleanup guard, including meetings with spoken redaction. Redaction does
not override audio retention settings or require a recording inspection. The
excluded transcript text remains available privately in `redactions.json`.

Repeated starts keep exclusion active; unmatched ends preserve surrounding text;
an unclosed span excludes everything through EOF. Each adds a QA warning that
describes the control problem without quoting excluded content. Source-redaction
warnings have no final-document line number and always have empty excerpts.
If all content is excluded, the pipeline writes empty/no-content documents with
no Ollama calls. On a redaction-bearing rerun, earlier managed summaries/debug/QA
outputs are removed before regeneration so a failure cannot expose stale text.

The private record is written atomically with owner-only `0600` permissions on
POSIX systems where supported. On Windows, protect the private output directory
with its filesystem ACL; POSIX permission bits do not provide the same ownership
control. `redactions.json` and private temporary writes are ignored by Git even
outside the default summary directory.

### Safe meeting publication/export

The repository has no deployed website publisher or meeting document sync.
The controller/extension exchange lesson control metadata, and the existing
lesson sync copies only named recording/session metadata sidecars. Those paths
do not discover meeting artifacts and remain unchanged.

Use the shared meeting exporter for website/member-facing files or archives:

```bash
python bin/export_meeting.py meeting-summaries/session-123 website/session-123
python bin/export_meeting.py meeting-summaries/session-123 meeting-public.zip --archive
```

`meeting_postprocess.publication.publication_payload()` uses the same selection
for API integrations. Only `summary.md`, `action-items.md`, and `minutes-draft.md`
are allowlisted. `redactions.json` is explicitly denied by name in addition to
the allowlist; filtering does not depend on its extension. No recursive copying
occurs, and symlinks/hardlinks to the private record are rejected. Private-record
links/references are removed from generated meeting Markdown and export payloads.
The private record, raw transcripts/audio, debug JSONL, QA files, and unlisted
attachments are never included in these exports. Future publishers must use
this shared boundary instead of copying the private processing directory.

### Final QA

Final meeting Markdown omits chunk IDs, source chunk labels, and source filenames.
These remain in `chunk_summaries.jsonl` and `meeting_sections.jsonl` for review.
Prompts prohibit provenance labels, and a presentation cleanup removes common
citations such as `(Chunk 2.2)` if a model still emits them.

Action extraction distinguishes future assignments from completed outreach.
An explicitly assigned or volunteered future contact task can remain; a report
that recipients were already contacted does not assign them new response tasks.
A source-based guard removes such unsupported recipient
bullets while retaining explicit later assignments or commitments. Completed
contacts may still be described under Topics Discussed.

Collective suggestions such as “I think we should...”, “we probably need to...”,
and “I think we have to...” stay proposals/tentative next steps until explicit
agreement, assignment, scheduling, or commitment establishes a task. Merely
addressing someone by name before a collective proposal does not assign them
responsibility. Prompts and a source-based action
guard enforce this in map summaries, action items, and the minutes' Action Items
section. Explicit “I'll...” commitments remain, including promises to approve,
follow up, arrange team participation, contact others, or try to prepare/share
information. A promise to try retains its qualification.

Meeting action-items and minutes reductions also receive a small supplemental
set of source-backed first-person future undertakings from current-business and
adjournment sections. This can recover tasks omitted by map summaries without
adding model calls. Matching accepts “I'll”, “I will”, “I'm going to”, and “I'm
gonna”, including a “when/once” lead-in and qualified “I'll try to” tasks.
Evidence keeps the prepared speaker identity and wording; it excludes recap,
pre-meeting chatter, conversational promises, capability, jokes, and
hypotheticals. At most 32 distinct quotations of up to 1,000 characters are
supplied. They are evidence for the model, not automatically published action
items; existing source guards still validate generated assignments.

These deterministic action guards apply to bullets, numbered items, and ordinary
Markdown tables in Action Items. Table headers identify the owner/task columns,
including `# | Assigned To | Action`, `# | Action Item | Responsible`, and
`Owner | Task`. Owner annotations such as `Taylor (SPEAKER_03)` or
`SPEAKER_04 (Committee Chair)` are supported for matching actual source commitments;
an annotation itself is not commitment evidence. Every distinct owner of a
joint action must have explicit source support. A name and its speaker
annotation are alternate identities for one owner; another person's commitment
cannot supply support. Unsupported joint actions are removed as a whole.
Retained rows and table
headers/separators stay intact. If every action row is removed, the section is
replaced with `None noted.`; no empty header/separator-only table remains.

Named adjournment announcements take precedence over the announcing speaker's
identity. For example, a chair saying “motion by Taylor, seconded by Casey”
identifies Taylor and Casey, even if the chair has an unresolved speaker label.
When the source explicitly says the motion carried, the record is “Motion to
adjourn moved by Taylor, seconded by Casey. Carried.” Named source evidence is
also supplied to the reduce stage and QA. Corrections require one unambiguous
adjournment announcement; conflicting records remain for review. An outcome is
never inferred solely from the presence of a mover and seconder.
Common bold Markdown motion headings and adjacent outcome/closure sentences
are corrected together. An invented “Carried”, “Passed”, or equivalent outcome
is removed from that motion record when the source does not support it;
explicit supported outcomes are preserved, and other motions are unaffected.
Spoken filler following an announced name is excluded from identity evidence:
“I'll say a motion by Casey, seconded by Taylor there” supports Casey as mover
and Taylor as seconder. QA does not require either named person to speak the role.

Each successful meeting run writes `minutes-qa.md` and `minutes-qa.json` beside
`minutes-draft.md`. QA flags unresolved speaker labels, malformed Markdown list
markers, remaining mojibake, inconsistent Mac Yard spelling, and questionable
mover/seconder attributions. Findings identify final-document line numbers.
Motion checks flag missing/unknown roles, identical mover/seconder names,
tentative wording, and names without explicit role evidence in the source.
They do not prove that a motion was valid or match an attribution to a particular
motion; review flagged lines against the transcript.

QA findings are advisory: drafts and all summary outputs are retained, and a
successful generation still exits successfully when review is needed.

Public examples and regression dialogues use synthetic participants. The
behavior-oriented fixtures cover natural starts, pre-meeting chatter, recap
interruption/resumption, proposals versus commitments, table action items, and
named role announcements. Runtime matching uses linguistic/structural patterns
and supplied aliases; no participant name is hard-coded as a rule. The
repository regression checks enforce behavior-oriented fixture/test names,
synthetic example participants, and structural runtime matching.

For an optional private local identifier audit, maintain an untracked
`tests/private-identifiers.txt` yourself. Put one identifier per line; blank
lines and lines beginning with `#` are ignored. Matching is case-insensitive.
When present, the test scans public prompts, documentation, production code and
comments, configuration examples, tests, and fixtures, reporting only file/line
locations for matches. When absent, this audit is skipped; normal public CI does
not need the file. It is ignored by Git and is not supplied by the repository.

Run the regression tests without Ollama:

```bash
python -m unittest discover -s tests -v
```

---

## Notes on Lesson / Study Use

For lessons, the pipeline generally does not need different audio capture logic.

What changes is the output profile.

Intended lesson behavior:
- do not treat the transcript like meeting minutes
- extract concepts, definitions, examples, and exam-relevant points
- generate review materials such as quizzes and flashcards
- keep lesson outputs separate from meeting outputs

This makes the same core pipeline useful as a **study companion** for technical courses, school, tutorial videos, etc.

---

## Known Limitations

- Summary quality depends heavily on transcript quality
- Audio-only lesson notes may miss visual slides, diagrams, or on-screen commands
- Sarcastic statements may be taken literally 
- Commentary videos/podcasts can still be summarized, but outputs may not resemble true meeting minutes
- Large Ollama models may occasionally require a restart or retry under VRAM pressure
- Remote audio capture and automation are environment-specific and may need local adaptation

---

## Troubleshooting

See:

- `TROUBLESHOOTING.md`

That file can cover common issues including:

- WhisperX installation/runtime problems
- disk space problems
- chunking behavior
- Ollama GPU issues
- Windows stop-script path issues
- source-audio deletion safety checks

---

## Suggested Next Steps

Once the core pipeline is working, useful additions include:

- Home Assistant buttons/automations
- completion notifications
- processing status sensors
- website publishing (for example via a CMS)
- improved warm-up/retry behavior for Ollama
- richer metadata storage
- lesson-specific prompt/output profiles
- optional immediate source-audio deletion after transcription

---

## Safety / Privacy Notes

This project is intended for self-hosted use.

Use:
- `env.example`
- `*.example` service files
- sanitized docs

---

## License

AGPLv3


## Optional adaptive meeting-chunk experiment

Map/reduce remains the production default. These private comparisons reuse an
existing transcript index; they do not run WhisperX, change models or apply
unapproved identities. Approved turn corrections and aliases, redaction and
section classification run before planning. Pre-meeting portions are excluded
from experimental map inputs; historical recap stays separate from current
business and adjournment.

`bin/plan-meeting-chunks.sh` groups adjacent classified source portions without
rewriting text. Medium mode has a 6,000-word ceiling (configurable, for example
5,500); adaptive mode has an initial 10,000-word ceiling. These are ceilings,
not fill targets. Deterministic grouping extends a continuing single-speaker
report or a clearly connected question/answer/clarification. It stops at a new
report/topic, section boundary or source chunk affected by redaction. Ambiguous
continuity stays separate. The optional planner can recognize wider relationships.
Exceptionally long reports may be split only at existing source-portion boundaries.
If even one indivisible portion exceeds the budget, planning fails explicitly;
no words are omitted. The normal baseline workflow remains available unchanged.

Use a matching **local** tokenizer for the selected models (`--tokenizer`, or
`MEETING_CHUNK_TOKENIZER`). Saved tokenizer truncation/padding is disabled for
counting. Without one, counts use a conservative UTF-8 byte upper bound, labelled
as such rather than reported as actual model tokens. Full map prompts, system
instructions, a configurable generation reserve (default 4,096 tokens), and 1,024
framing tokens must fit the explicitly configured map context. Token safety
always overrides word ceilings. An experimental map response that exhausts its
output reserve fails instead of being accepted as a complete summary. Context must be known from configuration or
`--map-num-ctx`; this experiment never selects a model or context automatically.

Example offline preparation, with a new private destination for each plan:

```bash
session="/private/completed-session"
tokenizer="/private/matching-Qwen/tokenizer.json"
comparison_root="$(mktemp -d "$PWD/ignore/chunk-comparison.XXXXXX")"

# Zero inference calls and no files written. Inspect token method and budgets.
bash bin/plan-meeting-chunks.sh "$session" --mode medium \
  --max-words 5500 --map-num-ctx 32768 --tokenizer "$tokenizer" --preflight

# Write a private, source-bound deterministic plan only; no public documents.
bash bin/plan-meeting-chunks.sh "$session" --mode medium \
  --max-words 5500 --map-num-ctx 32768 --tokenizer "$tokenizer" \
  --output-dir "$comparison_root/medium-plan"
```

Repeat with `--mode adaptive` to compare the 10,000-word ceiling. Do not increase
context merely to meet a word target. The selected map/reduce models remain the
existing configured models; optional CLI model/context overrides retain their
usual precedence. For a future boundary-planner trial, first run:

```bash
bash bin/plan-meeting-chunks.sh "$session" --mode adaptive \
  --map-num-ctx 32768 --tokenizer "$tokenizer" --planner --preflight
```

The planner preflight measures the **entire eligible source**, instructions,
JSON schema, framing and its complete output allowance. Planner context defaults
to 98,304 independently of map/reduce; `--planner-num-ctx` may lower it, never
exceed it. There is no 192K or dual-GPU switch. Only after reviewing the preflight,
add `--planner --output-dir "$comparison_root/topic-plan"` to a planning command.
That explicit request uses one boundary-only JSON inference under the existing
GPU1 runner. It returns ordered inclusive source-end IDs, not summaries or
extraction records. Invalid, incomplete, missing, repeated, out-of-order,
section-crossing, redaction-crossing or over-budget boundaries fall back to
validated deterministic larger groups with a recorded reason. There are no
retries or partial JSON acceptance.

After review, run an isolated comparison through the existing historical runner:

```bash
bash bin/summarize-existing-meeting.sh "$session" --keep-recap \
  --map-num-ctx 32768 --chunk-tokenizer "$tokenizer" \
  --chunk-plan "$comparison_root/medium-plan/chunk-plan.json" \
  --chunk-comparison-dir "$comparison_root/medium-output"
```

The comparison destination must be new and outside transcript/production output
directories. Plan and source hashes, approved identity inputs, prompt settings
and the tokenizer binding are validated again before map inference; the runner
captures the private plan in its comparison identity. A changed source or plan
cannot reuse incompatible completed results. Map output retains the existing
six-section structure, with narrow guidance to preserve questions/answers,
actor/object roles, commitments, motions, numbers and later qualifications.
Reduce, action guards, motion cleanup and QA continue to use the original
classified source. There is no additional extraction layer.

`chunk-plan.json` and `chunk-comparison.json` are private, gitignored, securely
written artifacts excluded from public exports and links. They record full
source-portion coverage, budgets, fallback reasons, hashes, runtime and actual
model token usage where Ollama supplies it. Grouped chunk summaries also retain
exact private source-portion provenance and containing timestamps. Coverage
means every eligible source portion occurs exactly once in chronological order;
it does not certify that an LLM retained every substantive detail.

For historical review, compare baseline/medium/adaptive outputs against source
for recap separation, committee-minutes sharing commitments and recipients,
efficiency-testing actor/object roles, Mac Yard restructuring chronology and
later numerical qualifications, convention motion roles/outcomes, and action
completeness/ownership/qualification. Compare `minutes-qa` findings as well.
No live comparison is necessary to run the synthetic offline tests.
