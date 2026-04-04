# TROUBLESHOOTING

This document covers common issues encountered while building and testing the meeting pipeline.

It focuses on:

- WhisperX setup and runtime problems
- transcript chunking issues
- Ollama model loading and GPU behavior
- Windows audio control issues
- Home Assistant integration issues

---

## 1. WhisperX problems

### Problem: WhisperX installs but the system runs out of disk space

Symptoms:
- `No space left on device`
- Conda/pip installs fail
- Hugging Face model downloads fail partway through

What to check:

```bash
df -h
du -sh ~/.cache/pip ~/miniconda3/pkgs 2>/dev/null
```

What to do:
- expand the VM disk in Proxmox
- grow the Linux partition/filesystem
- clean old pip and conda caches if needed

Useful commands:

```bash
python -m pip cache purge
conda clean -a -y
```

---

### Problem: WhisperX works, but GPU is not being used

Symptoms:
- runs very slowly
- logs suggest CPU-only processing

Test:

```bash
python -c "import whisperx, torch; print('whisperx ok'); print(torch.cuda.is_available())"
```

Expected:
- `whisperx ok`
- `True`

Also confirm your `.env` values:

```env
WHISPERX_COMPUTE_TYPE=float16
WHISPERX_DEVICE=cuda
```

---

### Problem: WhisperX language detection wastes time or produces odd results

Fix:
- pin the language in `.env`

```env
WHISPERX_LANGUAGE=en
```

And ensure `postprocess-meeting.sh` includes:

```bash
--language "$WHISPERX_LANGUAGE"
```

---

### Problem: Hugging Face diarization/token issues

Symptoms:
- diarization fails
- pyannote model cannot load
- WhisperX works but speaker diarization does not

Check:
- Hugging Face token exists in `.env`
- account accepted the required model terms

```env
HF_TOKEN=your_huggingface_read_token_here
```

---

## 2. Automatic post-processing problems

### Problem: recordings finish but transcription never starts

Check the systemd path/service:

```bash
sudo systemctl status meeting-postprocess.path --no-pager
sudo systemctl status meeting-postprocess.service --no-pager
sudo journalctl -u meeting-postprocess.service -n 100 --no-pager
```

Common causes:
- service file missing
- path unit active but service not installed
- wrong file paths inside the unit

---

### Problem: long meetings get killed during post-processing

Fix:
- set systemd timeout to infinity

In `meeting-postprocess.service`:

```ini
[Service]
TimeoutStartSec=infinity
```

Then reload:

```bash
sudo systemctl daemon-reload
sudo systemctl restart meeting-postprocess.path
```

---

## 3. Chunking problems

### Problem: chunks feel cut off or awkward

The chunker works best when:
- chunks begin on a new speaker
- chunks end after the previous speaker finishes
- very large single-speaker blocks are split safely
- tiny tail chunks are avoided

Tune in `.env`:

```env
TRANSCRIPT_CHUNK_TARGET_WORDS=1400
TRANSCRIPT_CHUNK_MAX_WORDS=2200
```

If chunks feel too large:
- reduce `TRANSCRIPT_CHUNK_TARGET_WORDS`

If chunks feel too fragmented:
- increase `TRANSCRIPT_CHUNK_TARGET_WORDS`

---

### Problem: chunk timestamps are not exact for oversized fallback splits

This can happen when a single long speaker turn is split into multiple subchunks.

This is usually acceptable if:
- the goal is summarization
- chunk ordering matters more than exact timestamps

If you need more precise timing later, the oversized-turn split logic can be refined further.

---

## 4. Ollama problems

### Problem: model loads, but Ollama seems to stall or hang

First, check whether it is actually waiting on generation:

```bash
docker exec ollama ollama ps
docker logs --since 2m ollama
```

If the Python summarizer appears stuck, add temporary progress prints:
- which chunk is being summarized
- which model is being called
- whether reduce stage has started

This often reveals that the process is just working silently, not frozen.

---

### Problem: `ollama ps` shows 32768 context even though map step should be 16384

This is expected.

The summarizer uses:
- map step: `OLLAMA_MAP_NUM_CTX`
- reduce step: `OLLAMA_REDUCE_NUM_CTX`

But `ollama ps` usually shows the context of the **currently loaded runner**, which is often the **last reduce-stage model state**.

So seeing `32768` after a run does not mean the map step ignored `16384`.

Recommended config:

```env
OLLAMA_MAP_NUM_CTX=16384
OLLAMA_REDUCE_NUM_CTX=32768
```

---

### Problem: Ollama sometimes falls back to CPU or behaves inconsistently on GPU

Symptoms:
- mixed CPU/GPU behavior
- `failed to initialize CUDA`
- runs succeed only after `docker restart ollama`

Likely causes:
- VRAM pressure
- stale model state
- model eviction/reload instability inside Ollama

What helped:
- using smaller context for chunk summarization
- keeping final synthesis at larger context
- forcing Ollama to see only one GPU
- restarting the container when it gets into a bad state

Useful commands:

```bash
docker exec ollama ollama ps
docker logs --since 2m ollama
docker restart ollama
```

---

### Problem: Ollama keeps seeing multiple GPUs when only one should be used

If the container still sees both GPUs, check the container runtime setup.

Verify inside the container:

```bash
docker exec ollama nvidia-smi -L
docker exec ollama env | grep -E 'NVIDIA_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|OLLAMA_LLM_LIBRARY'
```

Final working state:
- only one GPU exposed inside the container
- Ollama logs show only one inference GPU
- model loads happen on that single visible GPU

Note:
- once only one GPU is visible, Ollama may refer to it as `CUDA0` even if it was GPU 1 on the host
- that renumbering is normal

---

### Problem: `qwen2.5:32b` is unstable but `14b` is fine

This is expected on tighter VRAM budgets.

What we observed:
- `14b` fits comfortably and can stay fully on GPU
- `32b` often uses mixed CPU/GPU placement
- `32b` is much more sensitive to stale model loads and VRAM accounting

If stability matters more than peak quality:
- use `14b` for map step
- use `32b` only for reduce step

If maximum quality matters and time is acceptable:
- use `32b` for both, but expect more frequent need for restart/retry logic

---

## 5. Summary quality problems

### Problem: summaries invent action items or decisions

Symptoms:
- discussion topics become fake tasks
- vague conversation turns into formal decisions
- outputs feel too “meeting-like” for what was actually said

Fix:
- make prompts stricter and more conservative

Rules that helped:
- only include action items if explicitly assigned or agreed
- only include decisions if clearly made
- prefer `None noted` instead of inference
- do not turn general discussion into formal outputs

---

### Problem: random YouTube tests produce weird meeting summaries

That is normal.

This system is designed for:
- formal meetings
- semi-formal discussions
- convention/committee/business style transcripts

If you feed:
- commentary videos
- movie discussions
- podcasts
- reaction videos

the output can still be coherent, but it may not look like true meeting minutes.

Use those tests mainly for:
- chunking
- GPU load behavior
- pipeline stability
- formatting sanity checks

---

## 6. Windows audio sender problems

### Problem: Windows sender starts, but Linux records nothing

Test the Windows FFmpeg command directly over SSH.

Example pattern:

```bash
ssh -i ~/.ssh/id_ed25519_windows user@windows-host 'powershell -NoProfile -Command "& ffmpeg -f dshow -i \"audio=CABLE Output (VB-Audio Virtual Cable)\" -t 10 -ac 1 -ar 16000 -c:a pcm_s16le -f s16le tcp://LINUX_IP:4000"'
```

If Linux sees the incoming stream, the transport path is fine.

Then the problem is likely in:
- scheduled task setup
- quoting inside the wrapper script
- wrong device name on Windows

---

### Problem: stop script path gets mangled when sourced from `.env`

Symptoms:
- path becomes `C:meeting-audiostop-meeting-audio.ps1`
- PowerShell says file does not exist

Cause:
- Bash strips backslashes when sourcing unquoted `.env` values

Fix:
- use forward slashes in `.env`

```env
WINDOWS_STOP_SCRIPT=C:/meeting-audio/stop-meeting-audio.ps1
```

This is the simplest and most reliable solution.

---

### Problem: PowerShell windows pop up visibly when starting audio

This can happen depending on how the scheduled task or PowerShell action is registered.

A hidden PowerShell task action is cleaner than directly launching a visible shell window.

---

## 7. Home Assistant integration problems

### Problem: HA shell command does nothing

Check:
- SSH key exists under `/config/.ssh`
- public key is in `ai-hub` `authorized_keys`
- `known_hosts` contains the target host
- the SSH command works manually from HA terminal/add-on

Example manual test from HA:

```bash
ssh -i /config/.ssh/id_ed25519_aihub \
  -o UserKnownHostsFile=/config/.ssh/known_hosts \
  user@ai-hub '/home/user/meeting-pipeline/bin/ha-start-meeting.sh'
```

---

### Problem: changes to `shell_command` do not take effect

Home Assistant requires a restart after changing `shell_command`.

Restart HA after editing `configuration.yaml`.

---

## 8. File/output location problems

### Problem: summaries are saved inside transcript folders instead of the dedicated summary folder

The summarizer should write to:

```
~/meeting-pipeline/meeting-summaries/<meeting-folder>/
```

This is controlled by:

```env
MEETING_SUMMARIES_ROOT=${HOME}/meeting-pipeline/meeting-summaries
```

If files are still landing in the transcript directory, check `ollama_meeting_summary.py` and verify it builds:

- `summary_root`
- `summaries_dir = summary_root / transcript_dir.name`

---

## 9. Quick sanity checklist

If the full pipeline is not behaving, check these in order:

### Recording
```bash
ls -lt ~/meeting-pipeline/meeting-recordings | head
cat ~/meeting-pipeline/meeting-recordings/last_recording.txt
```

### Postprocess
```bash
sudo systemctl status meeting-postprocess.path --no-pager
sudo systemctl status meeting-postprocess.service --no-pager
```

### Transcript
```bash
LATEST_DIR="$(ls -dt ~/meeting-pipeline/meeting-transcripts/* | head -n 1)"
cat "$LATEST_DIR/status.txt"
tail -100 "$LATEST_DIR/whisperx.log"
```

### Chunks
```bash
find "$LATEST_DIR/chunks_out" -maxdepth 2 -type f | sort
```

### Summaries
```bash
SUMMARY_DIR="${HOME}/meeting-pipeline/meeting-summaries/$(basename "$LATEST_DIR")"
find "$SUMMARY_DIR" -maxdepth 1 -type f | sort
```

### Ollama
```bash
docker exec ollama ollama ps
docker logs --since 2m ollama
```

### Control wrappers
```bash
tail -50 ~/meeting-pipeline/meeting-control.log
```

---

## 10. When in doubt

If the workflow fails unexpectedly:

1. restart Ollama if GPU behavior looks wrong
2. test the summarizer manually on the latest transcript folder
3. confirm the wrappers still work manually
4. confirm the transcript JSON exists before chunking
5. confirm the chunk JSONL exists before summarization

That usually isolates the problem quickly.