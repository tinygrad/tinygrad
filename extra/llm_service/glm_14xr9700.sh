#!/usr/bin/env bash
# Launch GLM-5.3 on fourteen R9700 devices. PENDING is never accepted as power or vision.
set -euo pipefail
ACTION=${1:---help}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
STATE_DIR=${STATE_DIR:-$PWD/.llm/glm-c9}
GLM_WEIGHTS=${GLM_WEIGHTS:-/raid/weights}
GLM53_POWER_LIMIT=${GLM53_POWER_LIMIT:-PENDING_POWER_VERIFICATION}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$STATE_DIR/cache}
PIDFILE=$STATE_DIR/service.pid
LOGFILE=$STATE_DIR/service.log
MODEL=$GLM_WEIGHTS/GLM-5.3-UD-IQ4_XS/GLM-5.3-UD-IQ4_XS-00001-of-00009.gguf
VISION_MMPROJ=${VISION_MMPROJ:-}
SERVER_ENV=(
  HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTHONUNBUFFERED=1
  DEV=PCI+AMD VFIO=1 AM_POWER_LIMIT="$GLM53_POWER_LIMIT"
  MLA_SEQ_SHARD=1 WARMUP_PARALLEL=1 IQ2S_K256=1
  HOST_STASH=0 HOST_MIN=4096 SLOT_HOST=1 SLOT_CKPT=1
  SPEC_DRAFTS=2 SPEC_DRAFTS_BY_SIZE=1:2,2:2,4:3,8:3,16:3 SEQ_WARMUP_ONLY=
  GLM_SEQ=1 GLM_ATTN_GRP=3 SEQ_GATHER_UNION=65536 P2P_AR_TIMEOUT_MS=20000
  SEQ_ROWS="${SEQ_ROWS:-268800}" SEQ_SLOTS="${SEQ_SLOTS:-32}" SEQ_SIZES="${SEQ_SIZES:-1,4,8,16}" SEQ_MAX_ACTIVE="${SEQ_MAX_ACTIVE:-32}"
  SEQ_STASH_GB="${SEQ_STASH_GB:-0}" SEQ_STASH_ENTRIES="${SEQ_STASH_ENTRIES:-128}" LOAD_DEFER=1
  PREFILL_CHUNK="${PREFILL_CHUNK:-4032}" PREFILL_SIZES="${PREFILL_SIZES:-1344,2688}" PREFILL_BIG_ALONE="${PREFILL_BIG_ALONE:-2}"
  MODEL_ALIASES=zai-org/GLM-5.3 MODEL_OWNED_BY="${MODEL_OWNED_BY:-tinygrad}"
  VISION_MMPROJ="$VISION_MMPROJ" XDG_CACHE_HOME="$XDG_CACHE_HOME"
)
for _name in API_KEY API_KEY_FILE KEEPALIVE ENC_CACHE GLM_KV FREQS_HOST P2P_GLM5_OFF AMD_SCRATCH_MAX AMD_SCRATCH_MIN SEQ_COPY_SCRATCH \
  AM_PERF_LEVEL AM_CLOCKS CLOCK_FAST CLOCK_MANY CLOCK_PREFILL \
  SEQ_RESERVE SEQ_PROTECT SEQ_STICKY SEQ_STASH_CTX \
  MEM_TRACE DEBUG_LOAD AMD_SCRATCH_LOG KPROF KPROF_TOP LIN_NAMES TELEMETRY_LOG SERVE_TRACE DISABLE_AMD_KERNELS; do
  if [[ -n ${!_name+x} ]]; then SERVER_ENV+=("$_name=${!_name}"); fi
done
COMMAND=(python3 -m tinygrad.llm --model "$MODEL" --shard 14 --max_context "${MAX_CONTEXT:-262144}" --serve "${PORT:-8000}")

check_weights() {
  command -v sha256sum >/dev/null
  (cd "$GLM_WEIGHTS" && sha256sum -c "$ROOT/extra/llm_service/glm53-iq4xs.sha256")
}
if [[ $ACTION == --print-command ]]; then
  [[ -n $VISION_MMPROJ ]] || { echo 'VISION_MMPROJ is mandatory: no text-only command is offered' >&2; exit 2; }
  echo 'PENDING: preview only; an approved GLM-5.3 projector and glm_dsa picture integration are required before any start' >&2
  printf '%q ' "${SERVER_ENV[@]}" "${COMMAND[@]}"; printf '\n'
elif [[ $ACTION == start ]]; then
  [[ $GLM53_POWER_LIMIT =~ ^[0-9]+$ ]] && ((GLM53_POWER_LIMIT > 0)) || { echo 'PENDING: set GLM53_POWER_LIMIT from completed ts5 power verification; 210 W is not safe' >&2; exit 2; }
  [[ -n $VISION_MMPROJ && -s $VISION_MMPROJ ]] || { echo 'VISION_MMPROJ is mandatory: provide the approved nonempty GLM-5.3 projector; text-only is not offered' >&2; exit 2; }
  echo 'PENDING: glm_dsa does not yet have an approved picture input path; C9 start remains blocked even with a projector file' >&2
  exit 2
elif [[ $ACTION == stop ]]; then
  [[ -r $PIDFILE ]] || { echo 'GLM is not running (no PID file)' >&2; exit 1; }
  pid=$(cat "$PIDFILE")
  kill -TERM -- "-$pid"
  for _ in {1..90}; do
    if ! kill -0 "$pid" 2>/dev/null; then rm "$PIDFILE"; echo 'GLM stopped'; exit 0; fi
    read -r -t 1 _ || true
  done
  echo "GLM pid=$pid did not exit in 90 s; inspect before taking further action" >&2; exit 1
elif [[ $ACTION == status ]]; then
  curl -fsS "http://127.0.0.1:${PORT:-8000}/status"
elif [[ $ACTION == check-weights ]]; then
  check_weights
else
  cat >&2 <<'USAGE'
Usage: glm_14xr9700.sh start|stop|status|check-weights|--print-command
PENDING: C9 lacks approved projector facts and glm_dsa picture integration. No text-only service or start is offered.
--print-command is only a configuration preview and still requires VISION_MMPROJ.
GLM53_POWER_LIMIT is also required for a future start; PENDING is rejected. The former 210 W default is unsafe.
AM_FIRST_START=1 enables the single controlled AM reset. No process or GPU is touched by --print-command.
USAGE
  [[ $ACTION == --help || $ACTION == -h ]] && exit 0
  exit 2
fi
