#!/usr/bin/env bash
# Source for aihub_run_gpu_stage, or run: bash with-gpu-lock.sh gpu0 LABEL CMD...

aihub_run_gpu_stage() {
  local aihub_supervisor_pid aihub_stage_status aihub_saved_int aihub_saved_term
  aihub_saved_int="$(trap -p INT)"
  aihub_saved_term="$(trap -p TERM)"
  # Reset async-shell SIGINT inheritance so the supervisor can handle it.
  env --default-signal=INT,TERM bash "${BASH_SOURCE[0]}" "$@" &
  aihub_supervisor_pid=$!
  trap 'kill -TERM "$aihub_supervisor_pid" 2>/dev/null || :; wait "$aihub_supervisor_pid" 2>/dev/null || :; exit 130' INT
  trap 'kill -TERM "$aihub_supervisor_pid" 2>/dev/null || :; wait "$aihub_supervisor_pid" 2>/dev/null || :; exit 143' TERM
  if wait "$aihub_supervisor_pid"; then
    aihub_stage_status=0
  else
    aihub_stage_status=$?
  fi
  if [ -n "$aihub_saved_int" ]; then eval "$aihub_saved_int"; else trap - INT; fi
  if [ -n "$aihub_saved_term" ]; then eval "$aihub_saved_term"; else trap - TERM; fi
  return "$aihub_stage_status"
}

_aihub_gpu_lock_main() {
  if [ "$#" -lt 3 ]; then
    echo "Usage: bash with-gpu-lock.sh gpu0|gpu1|CONFIGURED_LOCK LABEL COMMAND [ARG...]" >&2
    exit 64
  fi
  if [ -z "${AIHUB_GPU_RUNNER_CONFIG:-}" ]; then
    echo "[gpu-lock] ERROR: AIHUB_GPU_RUNNER_CONFIG is required; no standalone flock fallback" >&2
    exit 64
  fi
  local aihub_script="${4:-}"
  case "${aihub_script##*/}" in
    ollama_meeting_summary.py|ollama_session_summary.py)
      local aihub_inputs_program
      aihub_inputs_program="$(dirname "${BASH_SOURCE[0]}")/meeting_stage_inputs.py"
      if ! AIHUB_GPU_STAGE_INPUT_FILES="$("${AIHUB_GPU_RUNNER_PYTHON:-python3}" "$aihub_inputs_program" "${@:3}")"; then
        echo "[gpu-lock] ERROR: approved meeting input declaration failed" >&2
        exit 64
      fi
      export AIHUB_GPU_STAGE_INPUT_FILES
      ;;
  esac
  # The runner package atomically claims the journal and physical lock, owns
  # the foreground process group, and retains uncertain ownership on exit.
  # Install startup signal exit codes before importing the runner. Its host
  # supervisor replaces these handlers once it can persist interrupted work.
  exec "${AIHUB_GPU_RUNNER_PYTHON:-python3}" -c 'import signal,sys,runpy; signal.signal(signal.SIGINT,lambda n,f:sys.exit(128+n)); signal.signal(signal.SIGTERM,lambda n,f:sys.exit(128+n)); runpy.run_module("aihub_gpu_runner.host_stage",run_name="__main__")' "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  _aihub_gpu_lock_main "$@"
fi
