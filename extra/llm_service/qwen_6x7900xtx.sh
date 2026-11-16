#!/usr/bin/env bash
# Launch Qwen V3.1 service on six 7900 XTX devices. Nothing runs for --print-command.
set -euo pipefail
ACTION=${1:---help}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
STATE_DIR=${STATE_DIR:-$PWD/.llm/qwen-c2}
QWEN_WEIGHTS=${QWEN_WEIGHTS:-/weights/Qwen3.8-Flash-Next-UD-IQ4_XS-V31}
AM_POWER_LIMIT=${AM_POWER_LIMIT:-250}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$STATE_DIR/cache}
PIDFILE=$STATE_DIR/service.pid
LOGFILE=$STATE_DIR/service.log
MODEL=$QWEN_WEIGHTS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
MMPROJ=${VISION_MMPROJ:-$QWEN_WEIGHTS/mmproj/mmproj-F16.gguf}
# a picture cap above 1,024 rows keeps its tower's bigger buffers on every card: it is paid in resident cache rows (the stash keeps
# the rest). Measured pairs (Qwen: g2 serve, w-gpf Nov 15, results/w-gpf/*/publisher/): 8192 -> 3299328; any other cap needs SEQ_ROWS set by hand.
case "${VISION_MAX_TOKENS:-1024}" in
  1024) ;;
  8192) SEQ_ROWS="${SEQ_ROWS:-3299328}" ;;
  *) [[ -n ${SEQ_ROWS:-} ]] || { echo "VISION_MAX_TOKENS=$VISION_MAX_TOKENS needs SEQ_ROWS (measured: 8192 -> 3299328)" >&2; exit 2; } ;;
esac
SERVER_ENV=(
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTHONUNBUFFERED=1 HOME="$HOME"
  DEV=AMD AM_MEC_PIPES=2 AM_IDLE_MS=1000 AM_POWER_LIMIT="$AM_POWER_LIMIT"
  SPEC_DRAFTS="${SPEC_DRAFTS:-2}" QSA_KV_BITS4_LAYERS="${QSA_KV_BITS4_LAYERS:-39,43,47,48}"
  SEQ_ROWS="${SEQ_ROWS:-4310016}" SEQ_SLOTS="${SEQ_SLOTS:-64}" SEQ_SIZES="${SEQ_SIZES:-1,8,16,24,30}" SEQ_MAX_ACTIVE="${SEQ_MAX_ACTIVE:-30}"
  SEQ_STASH_GB="${SEQ_STASH_GB:-46}" SEQ_CKPT2="${SEQ_CKPT2:-1}"
  SEQ_BASE_ALIGN="${SEQ_BASE_ALIGN:-24}"
  PREFILL_CHUNK="${PREFILL_CHUNK:-2304}" PREFILL_SIZES="${PREFILL_SIZES:-1152}" PREFILL_PARTS="${PREFILL_PARTS:-2}"
  PREFILL_MB1_ROWS="${PREFILL_MB1_ROWS:-1200}" PREFILL_PK_CTX="${PREFILL_PK_CTX:-131072}"
  PARALLEL=40 TOP_K=20
  CLOCK_FAST=2000 CLOCK_MANY=1900 CLOCK_PREFILL=2000
  VISION_MMPROJ="$MMPROJ" VISION_SLICE_MS="${VISION_SLICE_MS:-200}"
  VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-1024}" VISION_CACHE_MB="${VISION_CACHE_MB:-1024}" VISION_URL_CACHE_MB="${VISION_URL_CACHE_MB:-512}"
  XDG_CACHE_HOME="$XDG_CACHE_HOME"
)
for _name in SESSION_END_TOKEN API_KEY API_KEY_FILE MODEL_ALIASES MODEL_OWNED_BY KEEPALIVE ENC_CACHE \
  SEQ_RESERVE SEQ_PROTECT SEQ_STICKY SEQ_STASH_CTX \
  SPEC_DRAFTS_BY_SIZE AGING SEQ_STARVE SERVE_RENDER_CACHE SEQ_CKPT_TAIL \
  VISION_BUCKETS VISION_MAX_BYTES VISION_MAX_TOKENS VISION_MAX_PIXELS VISION_MAX_BODY_MB VISION_DECODERS VISION_DECODE_WAIT \
  MEM_TRACE DEBUG_LOAD AMD_SCRATCH_LOG KPROF KPROF_TOP LIN_NAMES TELEMETRY_LOG SERVE_TRACE DISABLE_AMD_KERNELS; do
  if [[ -n ${!_name+x} ]]; then SERVER_ENV+=("$_name=${!_name}"); fi
done
COMMAND=(python3 -m tinygrad.llm --model "$MODEL" --shard 6 --max_context 262144 --serve "${PORT:-8000}")

check_weights() {
  [[ -s $MMPROJ ]] || { echo "VISION_MMPROJ is mandatory: no readable nonempty projector at $MMPROJ" >&2; return 2; }
  command -v sha256sum >/dev/null
  (cd "$QWEN_WEIGHTS" && sha256sum -c "$ROOT/extra/llm_service/qwen-v31.sha256")
  printf '%s %s\n' '1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980' "$MMPROJ" | sha256sum -c -
}
if [[ $ACTION == --print-command ]]; then
  printf '%q ' "${SERVER_ENV[@]}" "${COMMAND[@]}"; printf '\n'
elif [[ $ACTION == start ]]; then
  [[ $AM_POWER_LIMIT =~ ^[0-9]+$ ]] && ((AM_POWER_LIMIT > 0)) || { echo 'AM_POWER_LIMIT must be a positive integer' >&2; exit 2; }
  ((AM_POWER_LIMIT <= 250)) || { echo 'Use a measured thermal limit of at most 250 W; a hot host should use 225' >&2; exit 2; }
  check_weights
  mkdir -p "$STATE_DIR"
  cd "$ROOT"
  if [[ ${AM_FIRST_START:-0} == 1 ]]; then SERVER_ENV+=(AM_VM_RESET=1); fi
  [[ ! -e $PIDFILE ]] || { echo "PID file already exists: $PIDFILE (check status first)" >&2; exit 2; }
  echo "start $(git rev-parse HEAD) model=$MODEL power=${AM_POWER_LIMIT}W" >> "$LOGFILE"
  setsid env "${SERVER_ENV[@]}" "${COMMAND[@]}" >> "$LOGFILE" 2>&1 & child=$!
  echo "$child" > "$PIDFILE"
  echo "Qwen started pid=$child log=$LOGFILE"
elif [[ $ACTION == stop ]]; then
  [[ -r $PIDFILE ]] || { echo 'Qwen is not running (no PID file)' >&2; exit 1; }
  pid=$(cat "$PIDFILE")
  kill -TERM -- "-$pid"
  for _ in {1..90}; do
    if ! kill -0 "$pid" 2>/dev/null; then rm "$PIDFILE"; echo 'Qwen stopped'; exit 0; fi
    read -r -t 1 _ || true
  done
  echo "Qwen pid=$pid did not exit in 90 s; inspect before taking further action" >&2; exit 1
elif [[ $ACTION == status ]]; then
  curl -fsS "http://127.0.0.1:${PORT:-8000}/status"
elif [[ $ACTION == check-weights ]]; then
  check_weights
else
  cat >&2 <<'USAGE'
Usage: qwen_6x7900xtx.sh start|stop|status|check-weights|--print-command
Vision is mandatory. The projector path/hash is checked before any model process is started.
Defaults are the validated V3.1 six-card production profile. Overrides use the RUNNING.md contract.
AM_FIRST_START=1 enables the single controlled AM reset. No process or GPU is touched by --print-command.
USAGE
  [[ $ACTION == --help || $ACTION == -h ]] && exit 0
  exit 2
fi
