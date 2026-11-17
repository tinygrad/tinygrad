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
MMPROJ=${VISION_MMPROJ:-/home/runner/glmvis/mmproj-F16.gguf}
# a picture cap above 1,024 rows keeps its tower's bigger buffers on every card: it is paid in resident cache rows (the stash keeps
# the rest). Measured pairs (GLM: g3 serve + g2 NLL gate, card 0 0.15 GB free, w-gpf Nov 15, results/w-gpf/*/publisher/): 8000 -> 1505280; any other cap needs SEQ_ROWS set by hand.
case "${VISION_MAX_TOKENS:-1024}" in
  1024) ;;
  8000) SEQ_ROWS="${SEQ_ROWS:-1505280}" ;;
  *) [[ -n ${SEQ_ROWS:-} ]] || { echo "VISION_MAX_TOKENS=$VISION_MAX_TOKENS needs SEQ_ROWS (measured: 8000 -> 1505280)" >&2; exit 2; } ;;
esac
SERVER_ENV=(
  HOME="$HOME" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. PYTHONUNBUFFERED=1
  DEV=AMD AM_POWER_LIMIT="$AM_POWER_LIMIT"

  BATCH_SLOTS=8
  SEQ_IDX_SHARD=1 SEQ_KV4=1
  PREFILL_PARTS=4 LOAD_DEFER=1 WARMUP_JITS=all
  SEQ_ROWS="${SEQ_ROWS:-3916800}" SEQ_SLOTS="${SEQ_SLOTS:-18}" SEQ_MAX_ACTIVE="${SEQ_MAX_ACTIVE:-18}"
  SEQ_SIZES="${SEQ_SIZES:-1,2,4,8,12,13,18}" SEQ_STASH_GB="${SEQ_STASH_GB:-12}" SEQ_STASH_ENTRIES="${SEQ_STASH_ENTRIES:-64}"
  AMD_SCRATCH_MIN=576 AMD_SCRATCH_MAX=64 SEQ_COPY_SCRATCH=1
  PREFILL_CHUNK="${PREFILL_CHUNK:-768}" PREFILL_SIZES="${PREFILL_SIZES:-192,768}" PREFILL_MB1_ROWS=1200
  PREFILL_SLICE=1536 SEQ_STARVE_ALT=1
  VISION_SLICE_MS="${VISION_SLICE_MS:-200}" VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-2048}" VISION_CACHE_MB="${VISION_CACHE_MB:-2048}"
  VISION_URL_CACHE_MB="${VISION_URL_CACHE_MB:-512}" SERVE_RENDER_CACHE="${SERVE_RENDER_CACHE:-1}"
  XDG_CACHE_HOME="$XDG_CACHE_HOME"
)
for _name in API_KEY API_KEY_FILE MODEL_ALIASES MODEL_OWNED_BY KEEPALIVE ENC_CACHE \
  SEQ_RESERVE SEQ_PROTECT SEQ_STICKY SEQ_STASH_CTX AGING SEQ_STARVE \
  VISION_MAX_TOKENS VISION_MAX_BODY_MB \
  MEM_TRACE DEBUG_LOAD AMD_SCRATCH_LOG KPROF KPROF_TOP LIN_NAMES TELEMETRY_LOG SERVE_TRACE DISABLE_AMD_KERNELS; do
  if [[ -n ${!_name+x} ]]; then SERVER_ENV+=("$_name=${!_name}"); fi
done
# a parked conversation's states take ~57.9 MB a card before any 3,072-token chunk (results/w-gpf/stash/stash_sizes.py):
# a stash too small for its entries silently holds 0 chunks, so refuse it (the qualified stash profile is 12 GB / 64 entries)
if ! awk -v gb="${SEQ_STASH_GB:-12}" -v n="${SEQ_STASH_ENTRIES:-64}" 'BEGIN{exit !(gb == 0 || gb * 1000 >= n * 57.9 + 100)}'; then
  echo "SEQ_STASH_GB=${SEQ_STASH_GB} holds no chunk beside SEQ_STASH_ENTRIES=${SEQ_STASH_ENTRIES:-64} entries of 57.9 MB a card" >&2; exit 2
fi
COMMAND=(python3 -m tinygrad.llm --model "$MODEL" --shard 6 --max_context 524288 --serve "${PORT:-8000}")
SERVER_ENV+=(VISION_MMPROJ="$MMPROJ" VISION_BUCKETS="${VISION_BUCKETS:-320,576}")

check_python() {
  # (the server's Python prerequisites, before any GPU is opened: a missing one fails the model load after the cards were set up)
  python3 -c 'import numpy, jinja2, PIL' 2>/dev/null || {
    echo "missing Python prerequisites (numpy, jinja2, pillow >= 10.0.1): sudo apt-get install python3-numpy python3-jinja2 python3-pil (an iris VM: tools/vm/setup.sh <vm>)" >&2; return 2; }
  python3 -c 'import sys, PIL; sys.exit(tuple(int(x) for x in PIL.__version__.split(".")[:2]) < (10, 0))' || {
    echo "the vision tower needs pillow >= 10.0.1: sudo python3 -m pip install --break-system-packages 'pillow>=10.0.1'" >&2; return 2; }
}
check_weights() {
  [[ -s $MMPROJ ]] || { echo "VISION_MMPROJ is mandatory: no readable nonempty projector at $MMPROJ" >&2; return 2; }
  command -v sha256sum >/dev/null
  (cd "$GLM_FLASH_WEIGHTS" && sha256sum -c "$ROOT/extra/llm_service/glm-flash-iq3xxs.sha256")
  printf '%s %s\n' '922b809664d6a89cc1e40bfeaa63618000edc6651baf4082ce0718a784adf947' "$MMPROJ" | sha256sum -c -
}
if [[ $ACTION == --print-command ]]; then
  printf '%q ' "${SERVER_ENV[@]}" "${COMMAND[@]}"; printf '\n'
elif [[ $ACTION == start ]]; then
  [[ $AM_POWER_LIMIT =~ ^[0-9]+$ ]] && ((AM_POWER_LIMIT > 0)) || { echo 'AM_POWER_LIMIT must be a positive integer' >&2; exit 2; }
  ((AM_POWER_LIMIT <= 250)) || { echo 'Use a measured thermal limit of at most 250 W; D34 ran at 225 W' >&2; exit 2; }
  check_python
  check_weights
  mkdir -p "$STATE_DIR"
  cd "$ROOT"
  if [[ ${AM_FIRST_START:-0} == 1 ]]; then SERVER_ENV+=(AM_VM_RESET=1); fi
  [[ ! -e $PIDFILE ]] || { echo "PID file already exists: $PIDFILE (check status first)" >&2; exit 2; }
  echo "start $(git rev-parse HEAD 2>/dev/null || cat DEPLOYED_COMMIT 2>/dev/null) model=$MODEL power=${AM_POWER_LIMIT}W" >> "$LOGFILE"
  setsid sudo -n env "${SERVER_ENV[@]}" "${COMMAND[@]}" >> "$LOGFILE" 2>&1 & child=$!
  echo "$child" > "$PIDFILE"
  echo "GLM Flash started pid=$child log=$LOGFILE"
elif [[ $ACTION == stop ]]; then
  [[ -r $PIDFILE ]] || { echo 'GLM Flash is not running (no PID file)' >&2; exit 1; }
  pid=$(cat "$PIDFILE")
  # (a start that died (a refused card, an error) leaves its PID file: nothing to stop, and the next start would refuse)
  if ! sudo -n kill -0 -- "-$pid" 2>/dev/null; then rm "$PIDFILE"; echo "GLM Flash was not running (stale PID file of pid=$pid removed)"; exit 0; fi
  sudo -n kill -TERM -- "-$pid"
  for _ in {1..90}; do
    if ! sudo -n kill -0 "$pid" 2>/dev/null; then rm "$PIDFILE"; echo 'GLM Flash stopped'; exit 0; fi
    sleep 1  # (not read -t: with stdin at its end (ssh, nohup) it returns at once and the 90 s were ~1 s)
  done
  echo "GLM Flash pid=$pid did not exit in 90 s; inspect before taking further action" >&2; exit 1
elif [[ $ACTION == status ]]; then
  curl -fsS "http://127.0.0.1:${PORT:-8000}/status"
elif [[ $ACTION == check-weights ]]; then
  check_python
  check_weights
else
  cat >&2 <<'USAGE'
Usage: glm_flash_6x7900xtx.sh start|stop|status|check-weights|--print-command
D34 ran at 225 W/card with 18 seats of 262144 context (524288 server context).
Vision is mandatory, never text-only. Default projector: /home/runner/glmvis/mmproj-F16.gguf.
Another installation path is accepted only for the byte-identical checked projector.
The benchmark/default leaves other picture limits unset; only the named G3 profile forwards its six extra knobs.
AM_FIRST_START=1 enables the single controlled AM reset. No process or GPU is touched by --print-command.
USAGE
  [[ $ACTION == --help || $ACTION == -h ]] && exit 0
  exit 2
fi
