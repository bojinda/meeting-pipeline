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

Both `postprocess-meeting.sh` and `postprocess-lesson.sh` acquire GPU 0 immediately
before running WhisperX and release it when that command finishes. They perform
CPU chunking without a GPU lock, then acquire GPU 1 immediately before invoking
the corresponding summarizer and hold it for the whole map/reduce process,
including optional recap generation. Both profiles use the exact same resource
paths. A GPU 0 WhisperX job can run concurrently with a GPU 1 summarization job;
two jobs targeting the same physical GPU serialize. WhisperX explicitly receives
`--device_index "$WHISPERX_DEVICE_INDEX"`; keep its CUDA index consistent with
the intended device placement and CUDA-visible device order.

`bin/with-gpu-lock.sh` uses util-linux [flock](https://man7.org/linux/man-pages/man1/flock.1.html),
with `setsid` from util-linux and GNU `env --default-signal` from coreutils for
process/signal handling. If occupied, it logs the resource/path and waits up to
`AIHUB_GPU_LOCK_TIMEOUT` seconds (default `3600`). This limits only acquisition
waiting, never runtime after acquisition. The helper returns `75` on lock
timeout and otherwise preserves the workload's exit code. Existing pipeline
status/exit handling and Home Assistant start/stop interfaces are retained.

Lock files persist and are never unlinked on release. Closing the supervisor's
descriptor releases the lock on success or failure. SIGINT/SIGTERM stop the
managed foreground process group before release; unresponsive processes receive
SIGKILL after a five-second interruption grace period. The descriptor is not
inherited by workloads. Sourced stage calls forward parent signals and restore
caller signal traps after ordinary completion.

Reuse the helper for another foreground workload:

```bash
bash bin/with-gpu-lock.sh gpu0 "SongGen (GPU0)" python /path/to/songgen.py
bash bin/with-gpu-lock.sh gpu1 "Ollama meeting summaries (GPU1)" \
  python bin/ollama_meeting_summary.py meeting-transcripts/session-123 --keep-recap
```

It also accepts an explicit lock path instead of `gpu0`/`gpu1`. From a Bash
script, source the helper and call `aihub_run_gpu_stage gpu0 LABEL COMMAND...`
to forward signals from the calling script. Supply a foreground command, not a
daemon-start request. Do not nest acquisition of the same resource. Direct
WhisperX/Python invocations bypass scheduling unless wrapped with this helper.
Locks coordinate cooperating callers; they do not stop existing unwrapped
services or change Ollama's model keep-alive policy. All adopters must share the
same local lock-file paths/inodes and have permission to open them; do not remove
or rotate these files while jobs are running.

Run the real lock tests on Linux (no GPU, WhisperX, or Ollama required):

```bash
python3 -m unittest discover -s tests -p test_gpu_locks.py -v
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
additional model calls. Only the explicit local LLM option adds a bounded call.
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
   `origin` (`heuristic`, `llm`, or `both`). The private `llm_review` field records
   status, sampled evidence, structured response, and validation issues.
4. Verify identities, then approve/edit the existing per-meeting
   `speaker_aliases.json`. The optional helper merges **only** explicitly selected,
   unambiguous suggestions, never all suggestions:
   ```bash
   python bin/suggest_meeting_speakers.py approve meeting-transcripts/session-123 --approve SPEAKER_03
   ```
   Multiple `--approve` flags select multiple labels. It rejects a review from
   another meeting or a changed visible transcript and cannot overwrite a
   different existing approved alias.
   The helper rechecks current source relationships and model evidence. Editing
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

Local LLM review uses the existing `OLLAMA_URL` and meeting reduce-model/context
defaults. `--speaker-suggestion-model MODEL` overrides only this review's model;
standalone `--reduce-model`, `--reduce-num-ctx`, and `--ollama-url` also select
settings explicitly. Source remains authoritative: names, labels, relationship
types, and evidence IDs must be grounded in the supplied redacted snippets.
Roster-only guesses, fabricated IDs, and invented names are rejected. A model
candidate whose name appears in cited transcript evidence but whose relationship
cannot be independently grounded is kept privately under `unverified_leads`,
with evidence IDs and `approvable: false`. This includes possible relationships
and unrelated mentions needing human interpretation; neither becomes a verified
candidate. Fresh approval checks repeat grounding from the current source.
Heuristic/model disagreement or explicit model uncertainty remains
for manual review; a model cannot choose a winner in a diarization collision or
replace an approved alias. Transcript text is untrusted data, never instructions.

The stage makes at most one request per meeting, samples introductions, addresses,
roles and representative turns across the entire meeting, and bounds the evidence
packet to at most 60 windows/18,000 characters. Review context is bounded to
4,096–32,768 tokens; generation is bounded to one quarter of that context
(at most 8,192 tokens), temperature zero, with a 120-second HTTP timeout.
The request supplies an explicit [Ollama JSON schema](https://docs.ollama.com/capabilities/structured-outputs)
and validates the generated JSON locally. Speaker review disables model thinking
by default; set `SPEAKER_REVIEW_THINK=true` to request it, or `default` to omit
that option for a model that requires its own default. This setting affects only
speaker review and never adds a retry or another inference call. Public AI endpoints,
redirects, environment HTTP proxies, and cloud-tagged models are not used;
select an installed local model. No live model is required
by the tests. Unavailable Ollama, timeout, or invalid JSON leaves safe heuristic
results and an incomplete/unavailable status; normal meeting processing continues.
Private `llm_review.diagnostics` records a failure category, generated-output
character count, `done_reason` when supplied by Ollama, token/thinking-length
counts when available, and validation error categories. It distinguishes empty
or malformed generated JSON, invalid schema, generation token limits, HTTP/model/
transport failures, and grounding failures. Raw malformed responses, thinking
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

Suggestions, rosters, aliases, and private temporary writes are gitignored.
Suggestion/approval writes use atomic replacement and owner-only `0600`
permissions on POSIX; protect the private directory with filesystem ACLs on
Windows. Custom roster/alias filenames must also remain private and gitignored.
Speaker review files and links to them are explicitly denied by the shared
publication/export boundary; the three public document names are unchanged.

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
