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
    echo "Usage: bash with-gpu-lock.sh gpu0|gpu1|LOCK_FILE LABEL COMMAND [ARG...]" >&2
    exit 64
  fi
  local aihub_resource="$1" aihub_label="$2" aihub_lock_file
  shift 2
  case "$aihub_resource" in
    gpu0) aihub_lock_file="${AIHUB_GPU0_LOCK_FILE:-/tmp/aihub-gpu0.lock}" ;;
    gpu1) aihub_lock_file="${AIHUB_GPU1_LOCK_FILE:-/tmp/aihub-gpu1.lock}" ;;
    *) aihub_lock_file="$aihub_resource" ;;
  esac
  local aihub_timeout="${AIHUB_GPU_LOCK_TIMEOUT:-3600}"
  if [[ ! "$aihub_timeout" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[gpu-lock] ERROR: AIHUB_GPU_LOCK_TIMEOUT must be nonnegative seconds" >&2
    exit 64
  fi
  if ! command -v flock >/dev/null || ! command -v setsid >/dev/null; then
    echo "[gpu-lock] ERROR: util-linux flock and setsid are required" >&2
    exit 69
  fi
  # Disable job control so setsid can use the child PID as its process-group ID.
  set +m
  local aihub_fd="" aihub_wait_pid="" aihub_job_pid="" aihub_killer_pid=""
  local aihub_acquired=0 aihub_interrupted=0 aihub_status=0

  _aihub_gpu_cleanup() {
    local aihub_exit_status=$?
    trap - EXIT
    trap '' INT TERM
    if [ -n "$aihub_wait_pid" ]; then
      kill -TERM "$aihub_wait_pid" 2>/dev/null || :
      wait "$aihub_wait_pid" 2>/dev/null || :
    fi
    if [ -n "$aihub_job_pid" ] && [ "$aihub_interrupted" = "1" ]; then
      echo "[gpu-lock] Stopping $aihub_label before releasing $aihub_lock_file" >&2
      kill -TERM -- "-$aihub_job_pid" 2>/dev/null || kill -TERM "$aihub_job_pid" 2>/dev/null || :
      # Give foreground workloads time to clean up, then stop unresponsive
      # descendants before releasing the physical resource. This grace applies
      # only to interruption, never to normal workload runtime.
      (
        exec {aihub_fd}>&-
        sleep 5
        kill -KILL -- "-$aihub_job_pid" 2>/dev/null || kill -KILL "$aihub_job_pid" 2>/dev/null || :
      ) >/dev/null 2>&1 &
      aihub_killer_pid=$!
      wait "$aihub_job_pid" 2>/dev/null || :
      if kill -0 -- "-$aihub_job_pid" 2>/dev/null; then
        wait "$aihub_killer_pid" 2>/dev/null || :
      else
        kill -TERM "$aihub_killer_pid" 2>/dev/null || :
        wait "$aihub_killer_pid" 2>/dev/null || :
      fi
    fi
    if [ -n "$aihub_fd" ]; then exec {aihub_fd}>&-; fi
    if [ "$aihub_acquired" = "1" ]; then
      echo "[gpu-lock] Released $aihub_label: $aihub_lock_file" >&2
    fi
    exit "$aihub_exit_status"
  }
  trap _aihub_gpu_cleanup EXIT
  trap 'aihub_interrupted=1; exit 130' INT
  trap 'aihub_interrupted=1; exit 143' TERM
  # Append-open creates the persistent inode without truncating/unlinking it.
  if ! exec {aihub_fd}>>"$aihub_lock_file"; then
    echo "[gpu-lock] ERROR: Cannot open lock file $aihub_lock_file" >&2
    exit 73
  fi
  if flock -x -n -E 75 "$aihub_fd"; then
    aihub_acquired=1
  else
    aihub_status=$?
    if [ "$aihub_status" != "75" ]; then
      echo "[gpu-lock] ERROR: Lock acquisition failed for $aihub_lock_file" >&2
      exit "$aihub_status"
    fi
    echo "[gpu-lock] Waiting for $aihub_label: $aihub_lock_file (up to ${aihub_timeout}s)" >&2
    flock -x -w "$aihub_timeout" -E 75 "$aihub_fd" &
    aihub_wait_pid=$!
    if wait "$aihub_wait_pid"; then
      aihub_acquired=1
    else
      aihub_status=$?
    fi
    aihub_wait_pid=""
    if [ "$aihub_acquired" != "1" ]; then
      if [ "$aihub_status" = "75" ]; then
        echo "[gpu-lock] TIMEOUT waiting for $aihub_label: $aihub_lock_file after ${aihub_timeout}s" >&2
      else
        echo "[gpu-lock] ERROR waiting for $aihub_label: $aihub_lock_file" >&2
      fi
      exit "$aihub_status"
    fi
  fi
  echo "[gpu-lock] Acquired $aihub_label: $aihub_lock_file" >&2
  (
    # Only the supervisor holds the lock. Descendants cannot accidentally
    # retain its file descriptor after the managed command exits.
    exec {aihub_fd}>&-
    exec setsid --wait env --default-signal=INT,TERM -- "$@"
  ) &
  aihub_job_pid=$!
  if wait "$aihub_job_pid"; then aihub_status=0; else aihub_status=$?; fi
  aihub_job_pid=""
  exit "$aihub_status"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  _aihub_gpu_lock_main "$@"
fi
