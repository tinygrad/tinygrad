#!/usr/bin/env bash
# Production C2 start line on NULL, including the picture tower. Power is inert without hardware.
# NULL stash defaults to 2 GB/device: the warm graph shapes are unchanged, without allocating 46 GB of real host RAM.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
MODEL=${MODEL:-/weights/Qwen3.8-Flash-Next-UD-IQ4_XS-V31/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf} \
VISION_MMPROJ=${VISION_MMPROJ:-/weights/Qwen3.8-Flash-Next-UD-IQ4_XS-V31/mmproj/mmproj-F16.gguf} \
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$PWD/.llm/null-c2-cache} \
DEV=${NULLDEV:-NULL:HIP:gfx1100} PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME="$HOME" \
AM_MEC_PIPES=2 AM_IDLE_MS=1000 AM_POWER_LIMIT=250 \
SPEC_DRAFTS=2 QSA_KV_BITS4_LAYERS=39,43,47,48 \
SEQ_ROWS=4310016 SEQ_SLOTS=64 SEQ_SIZES=1,8,16,24,30 SEQ_MAX_ACTIVE=30 \
SEQ_STASH_GB="${SEQ_STASH_GB:-2}" SEQ_CKPT2=1 SEQ_BASE_ALIGN=24 \
PREFILL_CHUNK=2304 PREFILL_SIZES=1152 PREFILL_PARTS=2 PREFILL_MB1_ROWS=1200 PREFILL_PK_CTX=131072 \
PARALLEL=40 TOP_K=20 \
CLOCK_FAST=2000 CLOCK_MANY=1900 CLOCK_PREFILL=2000 \
VISION_SLICE_MS=200 VISION_MAX_IMAGES=1024 VISION_CACHE_MB=1024 VISION_URL_CACHE_MB=512 \
python3 test/null/production_c2_warmup.py
