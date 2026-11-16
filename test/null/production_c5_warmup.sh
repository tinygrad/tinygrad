#!/usr/bin/env bash
# C5/D34 NULL-test harness. Text-only VISION_MMPROJ-unset loading is allowed only for this NULL/CPU test case.
# Production C5 always requires the verified projector file; the service launcher and CLI have no text-only startup mode.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
if [[ -n ${VISION_MMPROJ:-} ]]; then
  export VISION_MMPROJ VISION_BUCKETS=576
fi
# Match the launcher: the benchmark/default leaves these unset; the explicit G3 profile supplies them.
for _extra in VISION_SLICE_MS VISION_MAX_TOKENS VISION_MAX_IMAGES VISION_CACHE_MB VISION_URL_CACHE_MB SERVE_RENDER_CACHE; do
  [[ -z ${!_extra+x} ]] || export "$_extra"
done
MODEL=${MODEL:-/weights/GLM-5.3-Flash-UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf} \
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$PWD/.llm/null-c5-cache} \
DEV=${NULLDEV:-NULL:HIP:gfx1100} PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME="$HOME" \
AM_POWER_LIMIT=225 \
HOST_STASH=0 HOST_MIN=4096 SLOT_HOST=1 SLOT_CKPT=1 BATCH_SLOTS=8 \
SEQ_IDX_SHARD=1 SEQ_KV4=1 SEQ_COPY_SCRATCH=1 \
SEQ_ROWS=3916800 SEQ_SLOTS=18 SEQ_MAX_ACTIVE=18 SEQ_SIZES=1,2,4,8,12,13,18 \
SEQ_STASH_GB=0 SEQ_STASH_ENTRIES=128 LOAD_DEFER=1 \
PREFILL_PARTS=4 PREFILL_CHUNK=768 PREFILL_SIZES=192,768 PREFILL_MB1_ROWS=1200 \
PREFILL_SLICE=1536 SEQ_STARVE_ALT=1 \
AMD_SCRATCH_MIN=576 AMD_SCRATCH_MAX=64 WARMUP_JITS=all \
python3 test/null/production_c5_warmup.py
