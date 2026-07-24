# hlb_cifar10_fast — CIFAR-10 speedrun (airbench94 port)

tinygrad bounty *"<10s (wall time) hlb_cifar training on anything"* (≥93.5% eval, single eval at end, warm generic caches allowed): **ACHIEVED on a single RTX 4090.**

## Record (2026-07-24, RTX 4090, warm caches)
**9.79s / 9.79s / 9.94s / 9.98s @ 93.88 / 93.60 / 93.59 / 93.57%** across 4 seeds (1337/7/2024/42) — every run <10s and ≥93.5%.

```sh
export PYTHONPATH=. DEV=CUDA DEFAULT_FLOAT=HALF CACHEDB=/dev/shm/tg.db
export BS=1024 EPOCHS=8.0 TTA=2 MATMUL_CONV=1 CONTIG=1 SCHEDULE_CACHE=1 PROGRAM_CACHE=1
export JITBEAM=4 BEAM_ESTIMATE=0 BEAM_TC_SELECT=4 BEAM_VALIDATE=1 IGNORE_JIT_FIRST_BEAM=1 LOG_INTERVAL=0
time python3 examples/hlb_cifar10_fast.py    # 1st run: beam search (~2h, one-time, cached)
time python3 examples/hlb_cifar10_fast.py    # 2nd run: the record
```
All caches are generic and content-keyed (beam/compile/schedule/program) — user-code changes invalidate them naturally, nothing ever needs manual deletion. CACHEDB on tmpfs because overlayfs breaks sqlite WAL locking. More margin: `EPOCHS=8.25` = ~10.2s @ ≥93.75 worst-seed. Full recipe: `EPOCHS=9.9` = ~11.5s @ 94.05.

H100 note: same config ties the 4090 (11.5s at 9.9ep) — tinygrad emits sm_89-style mma.sync for sm_90; wgmma/TMA would be needed to exploit it.

## Contributions (branch hlb_cifar_10s; 74.7s -> 9.8s)
1. **airbench94 port** (`examples/hlb_cifar10_fast.py`): patch-whitening frozen conv, dirac init, fp32 BN islands, triangular LR, decoupled bias/wd groups, alternating flip, lookahead-EMA, TTA. Plus: `contiguous_backward()` barriers (the scheduler otherwise mega-fuses bn-bias grads with a full transposed-conv recompute, 6ms @ 2.9TFLOPS), im2col `MatmulConv2d` (convs as single-reduce GEMMs = the shape tensor cores like), jit-Variable batch offsets, eval capture overlapped with the GPU queue drain, pinned-buffer cifar load.
2. **gpudims fix**: WARP folded into threadIdx.x high bits scrambled WMMA lane ownership when TC kernels stacked ≥3 LOCALs — deterministic miscompile, cause of beam training collapse (72%→13%). Warp now owns threadIdx.x whole.
3. **BEAM_VALIDATE=1**: beam candidates validated against the unoptimized kernel on sparse-integer inputs (reduce reorders exact — only real miscompiles fail). Caught the gpudims bug; zero rejects since the fix.
4. **BEAM_TC_SELECT=N**: beam's tc_select=-1 only ever tries the first dtype-matching tensor core; seeding explicit tc choices (m8n16k8 vs m16n8k16) is worth 14-34% on the hot GEMMs. With BEAM_ESTIMATE=0 (exact timing; scaled timing mispicks followups).
5. **Pre-compile uops gate** (`COMPILE_UOPS_MAX`): BEAM_UOPS_MAX was checked after nvrtc already ran; pathological candidates wedged workers for hours (C call holds the GIL, SIGALRM useless).
6. **Persistent schedule/program caches** + fork-safe sqlite (~65s of python scheduling -> warm).
7. **jit capture fixes** (`engine/`): identity-keyed runtime cache, direct CALL build, local_size early-out — capture stopped re-hashing 130k-uop graphs (1.5->1.1s).

## Reproduce accuracy only (no beam, eager tensor cores)
`DEFAULT_FLOAT=HALF MATMUL_CONV=1 TC_OPT=2 CONTIG=1 EPOCHS=9.9 BS=512 python3 examples/hlb_cifar10_fast.py` -> 93.94% (4090) / 94.28% (H100), ~75s.
