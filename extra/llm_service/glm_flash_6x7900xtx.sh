#!/usr/bin/env bash
# Launch GLM-5.3 Flash D34 on six 7900 XTX devices. Nothing runs for --print-command.
set -euo pipefail
ACTION=${1:---help}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
STATE_DIR=${STATE_DIR:-$PWD/.llm/glm-flash-c5}
GLM_FLASH_WEIGHTS=${GLM_FLASH_WEIGHTS:-/weights/GLM-5.3-Flash-UD-IQ3_XXS}
AM_POWER_LIMIT=${AM_POWER_LIMIT:-225}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$STATE_DIR/cache}
PIDFILE=$STATE_DIR/service.pid
LOGFILE=$STATE_DIR/service.log
MODEL=$GLM_FLASH_WEIGHTS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf
SERVER_ENV=(
  HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTHONUNBUFFERED=1
  DEV=AMD AM_POWER_LIMIT="$AM_POWER_LIMIT"
  MLA_SEQ_SHARD=1 WARMUP_PARALLEL=1 IQ2S_K256=1
  HOST_STASH=0 HOST_MIN=4096 SLOT_HOST=1 SLOT_CKPT=1 BATCH_SLOTS=8
  SEQ_WARMUP_ONLY= SEQ_IDX_SHARD=1 SEQ_KV4=1
  PREFILL_PARTS=4 LOAD_DEFER=1 WARMUP_JITS=all
  SEQ_ROWS="${SEQ_ROWS:-3916800}" SEQ_SLOTS="${SEQ_SLOTS:-18}" SEQ_MAX_ACTIVE="${SEQ_MAX_ACTIVE:-18}"
  SEQ_SIZES="${SEQ_SIZES:-1,2,4,8,12,13,18}" SEQ_STASH_GB="${SEQ_STASH_GB:-0}" SEQ_STASH_ENTRIES=128
  FLAG_AR_MAXN=32768 AMD_SCRATCH_MIN=576 AMD_SCRATCH_MAX=64 SEQ_COPY_SCRATCH=1
  PREFILL_CHUNK="${PREFILL_CHUNK:-768}" PREFILL_SIZES="${PREFILL_SIZES:-192,768}" PREFILL_MB1_ROWS=1200
  PREFILL_SLICE=1536 SEQ_STARVE_ALT=1 RING_MAXW=32768 RING_RS_LIKE_COPIES=0
  XDG_CACHE_HOME="$XDG_CACHE_HOME"
)
for _name in API_KEY API_KEY_FILE MODEL_ALIASES MODEL_OWNED_BY KEEPALIVE ENC_CACHE \
  SEQ_RESERVE SEQ_PROTECT SEQ_STICKY SEQ_STASH_CTX \
  MEM_TRACE DEBUG_LOAD AMD_SCRATCH_LOG KPROF KPROF_TOP LIN_NAMES TELEMETRY_LOG SERVE_TRACE DISABLE_AMD_KERNELS; do
  if [[ -n ${!_name+x} ]]; then SERVER_ENV+=("$_name=${!_name}"); fi
done
COMMAND=(python3 -m tinygrad.llm --model "$MODEL" --shard 6 --max_context 524288 --serve "${PORT:-8000}")

check_weights() {
  command -v sha256sum >/dev/null
  (cd "$GLM_FLASH_WEIGHTS" && sha256sum -c "$ROOT/extra/llm_service/glm-flash-iq3xxs.sha256")
}
if [[ $ACTION == --print-command ]]; then
  printf '%q ' "${SERVER_ENV[@]}" "${COMMAND[@]}"; printf '\n'
elif [[ $ACTION == start ]]; then
  [[ $AM_POWER_LIMIT =~ ^[0-9]+$ ]] && ((AM_POWER_LIMIT > 0)) || { echo 'AM_POWER_LIMIT must be a positive integer' >&2; exit 2; }
  ((AM_POWER_LIMIT <= 250)) || { echo 'Use a measured thermal limit of at most 250 W; D34 ran at 225 W' >&2; exit 2; }
  check_weights
  mkdir -p "$STATE_DIR"
  cd "$ROOT"
  if [[ ${AM_FIRST_START:-0} == 1 ]]; then SERVER_ENV+=(AM_VM_RESET=1); fi
  [[ ! -e $PIDFILE ]] || { echo "PID file already exists: $PIDFILE (check status first)" >&2; exit 2; }
  echo "start $(git rev-parse HEAD) model=$MODEL power=${AM_POWER_LIMIT}W" >> "$LOGFILE"
  setsid sudo -n env "${SERVER_ENV[@]}" "${COMMAND[@]}" >> "$LOGFILE" 2>&1 & child=$!
  echo "$child" > "$PIDFILE"
  echo "GLM Flash started pid=$child log=$LOGFILE"
elif [[ $ACTION == stop ]]; then
  [[ -r $PIDFILE ]] || { echo 'GLM Flash is not running (no PID file)' >&2; exit 1; }
  pid=$(cat "$PIDFILE")
  kill -TERM -- "-$pid"
  for _ in {1..90}; do
    if ! kill -0 "$pid" 2>/dev/null; then rm "$PIDFILE"; echo 'GLM Flash stopped'; exit 0; fi
    read -r -t 1 _ || true
  done
  echo "GLM Flash pid=$pid did not exit in 90 s; inspect before taking further action" >&2; exit 1
elif [[ $ACTION == status ]]; then
  curl -fsS "http://127.0.0.1:${PORT:-8000}/status"
elif [[ $ACTION == check-weights ]]; then
  check_weights
else
  cat >&2 <<'USAGE'
Usage: glm_flash_6x7900xtx.sh start|stop|status|check-weights|--print-command
D34 ran at 225 W/card with 18 seats of 262144 context (524288 server context).
AM_FIRST_START=1 enables the single controlled AM reset. No process or GPU is touched by --print-command.
USAGE
  [[ $ACTION == --help || $ACTION == -h ]] && exit 0
  exit 2
fi
