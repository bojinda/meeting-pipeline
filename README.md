# Meeting Pipeline

A self-hosted meeting recording, transcription, chunking, and summarization pipeline built around **WhisperX**, **speaker-aware chunking**, and **Ollama**.

It is designed to be modular and self-hosted: capture, transcription, summarization, and automation can be used together or independently.

## Features

- Remote or automated meeting audio capture
- Diarized transcription with WhisperX
- Speaker-aware transcript chunking
- Map/reduce-style summarization with Ollama
- Markdown outputs for:
  - `summary.md`
  - `action-items.md`
  - `minutes-draft.md`
- `.env`-driven configuration
- Example `systemd` units for unattended automation
- Optional Home Assistant controls
- Optional webhook-driven automation when a meeting starts

## Pipeline Overview

The typical flow is:

1. Start meeting audio capture
2. Record incoming audio to WAV
3. Run WhisperX transcription with diarization
4. Chunk the transcript into speaker-aware sections
5. Summarize chunks with Ollama
6. Write summary outputs to disk

## Architecture

A common setup looks like this:

### Audio source host
This can be:
- a Windows machine capturing meeting/system audio
- another host that can stream audio to the processing server

### Processing host
Runs:
- audio capture listener
- WhisperX
- transcript chunker
- Ollama summarization
- optional systemd automation

### Optional control layer
You can trigger the pipeline via:
- manual wrapper scripts
- Home Assistant buttons/scripts
- webhook automation when a meeting starts

## Control Options

This project supports multiple control styles:

- **Manual start/stop** via wrapper scripts
- **Home Assistant buttons/scripts** for dashboard control
- **Webhook-driven automation** for automatic capture start

The recommended design is to keep the capture/transcription/summarization logic on the processing host and use thin wrappers or webhooks for remote triggers.

## Project Structure

```
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
└── meeting-summaries/
```

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

## Outputs

For each processed recording, the pipeline generates:

### Transcripts
Stored under:

```text
meeting-transcripts/<meeting-folder>/
```

These typically include:
- WhisperX transcript outputs
- status/log files
- chunked transcript files under `chunks_out/`

### Summaries
Stored under:

```text
meeting-summaries/<meeting-folder>/
```

These include:
- `summary.md`
- `action-items.md`
- `minutes-draft.md`
- `chunk_summaries.jsonl`

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

### Optional remote control
- `WINDOWS_HOST`
- `WINDOWS_USER`
- `WINDOWS_SSH_KEY`
- `WINDOWS_START_TASK`
- `WINDOWS_STOP_SCRIPT`
- `MEETING_CONTROL_LOG`

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

## Typical Workflow

### Manual testing
A common testing path is:

1. Start audio capture
2. Stop audio capture
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

## Home Assistant Integration

Home Assistant is optional, but works well as a control layer.

Recommended pattern:
- use `shell_command` to SSH into the processing host
- call:
  - `ha-start-meeting.sh`
  - `ha-stop-meeting.sh`
- expose those via two HA scripts or dashboard buttons

Suggested controls:
- **Start Meeting Capture**
- **Stop Meeting Capture**

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

## Known Limitations

- Summary quality depends heavily on transcript quality
- Commentary videos/podcasts can still be summarized, but outputs may not resemble true meeting minutes
- Large Ollama models may occasionally require a restart or retry under VRAM pressure
- Remote audio capture and automation are environment-specific and may need local adaptation

## Troubleshooting

See:

- `TROUBLESHOOTING.md`

That file covers common issues including:

- WhisperX installation/runtime problems
- disk space problems
- chunking behavior
- Ollama GPU issues
- Windows stop-script path issues

## Suggested Next Steps

Once the core pipeline is working, useful additions include:

- home assistant buttons/automations
- completion notifications
- processing status sensors
- website publishing (for example via a CMS)
- improved warm-up/retry behavior for Ollama
- richer meeting metadata storage

## Safety / Privacy Notes

This project is intended for self-hosted use.

Use:
- `env.example`
- `*.example` service files
- sanitized docs

## License

AGPLv3
