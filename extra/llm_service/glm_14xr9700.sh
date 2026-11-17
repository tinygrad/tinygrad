#!/usr/bin/env bash
# Launch text-only GLM-5.3 on fourteen R9700 devices. PENDING is never accepted as power.
set -euo pipefail
ACTION=${1:---help}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/extra/llm_service/glm_kv_profiles.sh"
apply_glm_kv_profile || exit 2
STATE_DIR=${STATE_DIR:-$PWD/.llm/glm-c9}
GLM_WEIGHTS=${GLM_WEIGHTS:-/raid/weights}
GLM53_POWER_LIMIT=${GLM53_POWER_LIMIT:-PENDING_POWER_VERIFICATION}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$STATE_DIR/cache}
PIDFILE=$STATE_DIR/service.pid
LOGFILE=$STATE_DIR/service.log
MODEL=$GLM_WEIGHTS/GLM-5.3-UD-IQ4_XS/GLM-5.3-UD-IQ4_XS-00001-of-00009.gguf
SERVER_ENV=(
  HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTHONUNBUFFERED=1
  DEV=PCI+AMD VFIO=1 AM_POWER_LIMIT="$GLM53_POWER_LIMIT"
  SPEC_DRAFTS=2 SPEC_DRAFTS_BY_SIZE=1:2,2:2,4:3,8:3,16:3
  SEQ_GATHER_UNION=65536 P2P_AR_TIMEOUT_MS=20000
  SEQ_ROWS="${SEQ_ROWS:-268800}" SEQ_SLOTS="${SEQ_SLOTS:-32}" SEQ_SIZES="${SEQ_SIZES:-1,4,8,16}" SEQ_MAX_ACTIVE="${SEQ_MAX_ACTIVE:-32}"
  SEQ_STASH_GB="${SEQ_STASH_GB:-0}" SEQ_STASH_ENTRIES="${SEQ_STASH_ENTRIES:-128}" LOAD_DEFER=1
  SEQ_STICKY="${SEQ_STICKY:-30}"
  PREFILL_CHUNK="${PREFILL_CHUNK:-4032}" PREFILL_SIZES="${PREFILL_SIZES:-1344,2688}" PREFILL_BIG_ALONE="${PREFILL_BIG_ALONE:-2}"
  MODEL_ALIASES=zai-org/GLM-5.3 MODEL_OWNED_BY="${MODEL_OWNED_BY:-tinygrad}"
  VISION_MMPROJ="" XDG_CACHE_HOME="$XDG_CACHE_HOME"
)
for _name in API_KEY API_KEY_FILE KEEPALIVE ENC_CACHE GLM_KV FREQS_HOST P2P_GLM5_OFF AMD_SCRATCH_MAX AMD_SCRATCH_MIN SEQ_COPY_SCRATCH \
  SEQ_FP8 SEQ_FP8_IDX SEQ_KV4 SEQ_KV4_FAST \
  AM_PERF_LEVEL AM_CLOCKS CLOCK_FAST CLOCK_MANY CLOCK_PREFILL \
  SEQ_RESERVE SEQ_PROTECT SEQ_STASH_CTX AGING SEQ_STARVE SERVE_RENDER_CACHE \
  MEM_TRACE DEBUG_LOAD AMD_SCRATCH_LOG LIN_NAMES TELEMETRY_LOG SERVE_TRACE DISABLE_AMD_KERNELS; do
  if [[ -n ${!_name+x} ]]; then SERVER_ENV+=("$_name=${!_name}"); fi
done
COMMAND=(python3 -m tinygrad.llm --model "$MODEL" --shard 14 --max_context "${MAX_CONTEXT:-262144}" --serve "${PORT:-8000}")

check_python() {
  # (the server's Python prerequisites, before any GPU is opened: a missing one fails the model load after the cards were set up)
  python3 -c 'import numpy, jinja2' 2>/dev/null || {
    echo "missing Python prerequisites (numpy, jinja2): sudo apt-get install python3-numpy python3-jinja2 (an iris VM: tools/vm/setup.sh <vm>)" >&2; return 2; }
}
check_weights() {
  command -v sha256sum >/dev/null
  (cd "$(dirname -- "$MODEL")" && sha256sum -c "$ROOT/extra/llm_service/glm53-iq4xs.sha256")
}
if [[ $ACTION == --print-command ]]; then
  printf '%q ' "${SERVER_ENV[@]}" "${COMMAND[@]}"; printf '\n'
elif [[ $ACTION == start ]]; then
  [[ $GLM53_POWER_LIMIT =~ ^[0-9]+$ ]] && ((GLM53_POWER_LIMIT > 0)) || { echo 'PENDING: set GLM53_POWER_LIMIT from completed ts5 power verification; 210 W is not safe' >&2; exit 2; }
  check_python
  check_weights
  mkdir -p "$STATE_DIR"
  cd "$ROOT"
  if [[ ${AM_FIRST_START:-0} == 1 ]]; then SERVER_ENV+=(AM_VM_RESET=1); fi
  [[ ! -e $PIDFILE ]] || { echo "PID file already exists: $PIDFILE (check status first)" >&2; exit 2; }
  echo "start $(git rev-parse HEAD 2>/dev/null || cat DEPLOYED_COMMIT 2>/dev/null) model=$MODEL power=${GLM53_POWER_LIMIT}W" >> "$LOGFILE"
  setsid sudo -n env "${SERVER_ENV[@]}" "${COMMAND[@]}" >> "$LOGFILE" 2>&1 & child=$!
  echo "$child" > "$PIDFILE"
elif [[ $ACTION == stop ]]; then
  [[ -r $PIDFILE ]] || { echo 'GLM is not running (no PID file)' >&2; exit 1; }
  pid=$(cat "$PIDFILE")
  # (a start that died (a refused card, an error) leaves its PID file: nothing to stop, and the next start would refuse)
  if ! sudo -n kill -0 -- "-$pid" 2>/dev/null; then rm "$PIDFILE"; echo "GLM was not running (stale PID file of pid=$pid removed)"; exit 0; fi
  sudo -n kill -TERM -- "-$pid"
  for _ in {1..90}; do
    if ! sudo -n kill -0 "$pid" 2>/dev/null; then rm "$PIDFILE"; echo 'GLM stopped'; exit 0; fi
    sleep 1  # (not read -t: with stdin at its end (ssh, nohup) it returns at once and the 90 s were ~1 s)
  done
  echo "GLM pid=$pid did not exit in 90 s; inspect before taking further action" >&2; exit 1
elif [[ $ACTION == status ]]; then
  curl -fsS "http://127.0.0.1:${PORT:-8000}/status"
elif [[ $ACTION == check-weights ]]; then
  check_python
  check_weights
else
  cat >&2 <<'USAGE'
Usage: glm_14xr9700.sh start|stop|status|check-weights|--print-command
C9 is text-only: there is no GLM-5.3 vision tower, and VISION_MMPROJ is forced empty.
GLM53_POWER_LIMIT is also required for a future start; PENDING is rejected. The former 210 W default is unsafe.
AM_FIRST_START=1 enables the single controlled AM reset. No process or GPU is touched by --print-command.
USAGE
  [[ $ACTION == --help || $ACTION == -h ]] && exit 0
  exit 2
fi
