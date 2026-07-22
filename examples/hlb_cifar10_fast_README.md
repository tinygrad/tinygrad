# hlb_cifar10_fast — CIFAR-10 speedrun (airbench94 port)

Attempt at the tinygrad bounty *"<10s (wall time) hlb_cifar training on anything"* (≥93.5% eval, single eval at end, warm generic caches allowed).

## What works (validated, RTX 4090 + H100)
- **Correct recipe:** airbench94 port trains to **93.94%** (4090) / **94.28%** (H100) eval at 9.9 epochs, BS=512. Reliable, deterministic.
- Run: `DEFAULT_FLOAT=HALF MATMUL_CONV=1 TC_OPT=2 CONTIG=1 SCHEDULE_CACHE=1 PROGRAM_CACHE=1 TARGET_EVAL_ACC_PCT=93.5 python3 examples/hlb_cifar10_fast.py`
- Knobs: `BS EPOCHS TTA(2/1/0) MATMUL_CONV TC_OPT CONTIG BEAM_VALIDATE SCHEDULE_CACHE PROGRAM_CACHE`.

## Contributions (committed on branch hlb_cifar_10s)
1. **airbench94 port** (`examples/hlb_cifar10_fast.py`): patch-whitening frozen conv, dirac init, fp32 BN islands, triangular LR, decoupled bias/wd groups, alternating flip, lookahead-EMA pullback, TTA.
2. **Persistent schedule/program disk caches** + **fork-safe sqlite connection** (`helpers.py`, `schedule/__init__.py`, `codegen/__init__.py`): cut JIT scheduling/lowering ~65s→~5.7s warm, bit-identical. Same legality as the existing binary/beam caches.
3. **BEAM output validation** (`BEAM_VALIDATE=1`, `codegen/opt/search.py`): rejects numerically-wrong beam candidates against the un-optimized kernel. Fixes a real tinygrad TC/WMMA **miscompile** on backward-conv shapes (beam otherwise collapses training 72%→13%).
4. **MatmulConv2d** (`MATMUL_CONV=1`): im2col conv-as-GEMM so the default heuristic applies *correct* tensor cores with no beam (matmul TC is correct where conv-backward TC miscompiles). Eager 12 TFLOPS / 48ms-step, bit-identical to nn.Conv2d.

## Status on <10s: NOT achieved — blocked by beam-search limits
- Fastest **reliable + correct** config is eager matmul-conv (~70 ms/step, ~72s total for 9.9 ep) on both 4090 and H100. H100 eager is *not* faster (default TC doesn't exploit it).
- <10s needs beam-quality tensor cores (~40–200 TFLOPS). But beam is impractical for this model:
  - **direct-conv** kernels are beamable but the backward TC **miscompiles** (worked around, not fixed, by BEAM_VALIDATE — which then strips backward TC → slow).
  - **matmul-conv** kernels are correct under default TC but too **large to beam**: parallel pool workers OOM (`BrokenPipe`), `BEAM_ESTIMATE` hits `inf→int OverflowError`, serial beam hangs, and searches exceed the per-kernel timeout.
- Root fix for <10s: repair the tinygrad conv-backward TC codegen so direct-conv (FLOP-efficient, beamable) beams correctly. That's a core codegen change beyond this attempt.

## Reproduce accuracy
`time DEFAULT_FLOAT=HALF MATMUL_CONV=1 TC_OPT=2 CONTIG=1 SCHEDULE_CACHE=1 PROGRAM_CACHE=1 CACHEDB=/dev/shm/tg.db EPOCHS=9.9 BS=512 TARGET_EVAL_ACC_PCT=93.5 python3 examples/hlb_cifar10_fast.py`
(run twice; second run warm-cache. CACHEDB on tmpfs — overlayfs breaks sqlite WAL locking.)
