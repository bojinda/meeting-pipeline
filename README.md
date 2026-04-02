# meeting-pipeline

Windows -> Linux meeting audio capture and post-processing pipeline.

## Goal

Send meeting audio live from a Windows note-taker machine to a Linux VM/server, record one master WAV on Linux, then post-process it into transcripts, speaker-separated output, and meeting summaries.

## Current flow

1. Windows joins the meeting as a silent note-taker.
2. Windows routes meeting audio to VB-CABLE.
3. FFmpeg on Windows sends raw PCM audio over TCP to Linux.
4. Linux listens on port 4000 and records a single WAV file.
5. After recording ends, Linux runs post-processing.

## Repo structure

- `bin/record-meeting.sh`  
  Starts the Linux listener and writes one WAV per meeting.
- `bin/postprocess-meeting.sh`  
  Placeholder for transcript / diarization / summary steps.
- `systemd/meeting-capture.service`  
  Systemd unit for always-on capture service.
- `config/.env.example`  
  Example environment variables for paths, ports, and models.

## Windows sender command

Replace `LINUX_IP` with the Linux server IP.

```powershell
ffmpeg -f dshow -i audio="CABLE Output (VB-Audio Virtual Cable)" `
  -ac 1 -ar 16000 -c:a pcm_s16le -f s16le `
  tcp://LINUX_IP:4000