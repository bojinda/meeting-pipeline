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
