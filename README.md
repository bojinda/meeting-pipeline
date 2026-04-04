# Meeting Pipeline

A self-hosted meeting recording, transcription, chunking, and summarization pipeline built around **FFmpeg**, **WhisperX**, **Ollama**, and optional **Home Assistant** controls.

This project is designed for long-form audio such as meetings, conventions, discussions, or livestream-style recordings. It can automatically:

- capture audio from a remote Windows machine
- record the stream on Linux
- transcribe audio with speaker diarization using WhisperX
- split transcripts into speaker-aware chunks
- summarize chunks with Ollama using a map/reduce workflow
- generate:
  - `summary.md`
  - `action-items.md`
  - `minutes-draft.md`

## Features

- **Remote audio capture** from a Windows source machine
- **Automatic post-processing** after recording finishes
- **WhisperX diarized transcription**
- **Speaker-aware chunking** that avoids splitting mid-turn
- **Ollama summarization** with separate map/reduce context sizes
- **Markdown outputs** for review and editing
- **Optional Home Assistant control layer** for start/stop buttons
- **Environment-driven configuration** via `.env`

## High-Level Flow

1. Start audio transmission on the source machine
2. Record the incoming stream on the Linux processing machine
3. Trigger automatic post-processing when recording completes
4. Run WhisperX transcription with diarization
5. Convert transcript JSON into speaker-aware chunks
6. Summarize chunks with Ollama
7. Write outputs to a meeting-specific summary folder

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
├── meeting-recordings/
├── meeting-transcripts/
└── meeting-summaries/
```

## Outputs

For each processed recording, the pipeline produces:

### Transcript output
Located under:

```
meeting-transcripts/<meeting-name>/
```

Typical files:

- raw WhisperX transcript files (`.txt`, `.srt`, `.vtt`, `.json`, etc.)
- `status.txt`
- `whisperx.log`
- `chunks_out/`

### Summary output
Located under:

``` 
meeting-summaries/<meeting-name>/
```

Files:

- `chunk_summaries.jsonl`
- `summary.md`
- `action-items.md`
- `minutes-draft.md`

## Requirements

### Linux processing machine

- Python environment for WhisperX
- Ollama
- FFmpeg
- systemd
- enough disk space for recordings and transcripts
- NVIDIA GPU strongly recommended for WhisperX and larger Ollama models

### Windows source machine (optional but supported)

- FFmpeg
- VB-Audio Virtual Cable or another audio routing method
- OpenSSH server enabled
- a start mechanism (for example a scheduled task)
- a stop PowerShell script for ending the sender cleanly

### Optional

- Home Assistant for remote start/stop controls

## Configuration

Copy `env.example` to your real config file and adjust values for your environment.

```bash
cp env.example config/.env
```

Important configuration groups include:

### WhisperX
- Hugging Face token
- model name
- batch size
- compute type
- device
- optional speaker count hints

### Chunking
- target words per chunk
- max words per chunk

### Ollama
- base URL
- map model
- reduce model
- keep alive
- temperature
- separate context settings for map and reduce steps

### Windows / control wrappers
- source host IP
- source username
- SSH key path
- start task name
- stop script path
- control log path

## Recommended Ollama Strategy

For long transcripts, a map/reduce setup works better than sending the full text in one prompt.

Recommended pattern:

- **Map step**: summarize each transcript chunk
- **Reduce step**: combine chunk summaries into final markdown outputs

Example:

- `OLLAMA_MAP_MODEL=qwen2.5:32b`
- `OLLAMA_REDUCE_MODEL=qwen2.5:32b`
- `OLLAMA_MAP_NUM_CTX=16384`
- `OLLAMA_REDUCE_NUM_CTX=32768`

This allows chunk summarization with a smaller working context while keeping the final synthesis step larger.

## Home Assistant Integration

The project can be controlled from Home Assistant using:

- `ha-start-meeting.sh`
- `ha-stop-meeting.sh`

These wrappers can be called from HA `shell_command` entries and exposed as dashboard buttons or scripts.

## Notes on GPU Use

If you are running multiple GPUs, it is a good idea to explicitly pin Ollama to the intended card. In testing, constraining Ollama to a single visible GPU improved stability and made runner behavior easier to reason about.

## Known Limitations

- Summary quality depends heavily on transcript quality
- Random YouTube or podcast audio will not summarize like a formal meeting
- Large Ollama models may occasionally need a clean reload or restart under VRAM pressure
- Very long meetings may still require prompt and chunk-size tuning

## Next Steps

Once the core pipeline is working, potential useful additions include:

- Home Assistant dashboard buttons / automation
- completion notifications
- status sensors
- publishing summaries to a CMS or website
- additional cleanup / review tooling

## License

AGPLv3
