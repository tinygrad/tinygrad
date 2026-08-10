# Kimi K3 on 8× MI350X

This branch targets text generation directly from the official `moonshotai/Kimi-K3` checkpoint at `/raid/weights/kimi-k3`. It intentionally ignores the vision tower and multimodal projector. The checkpoint remains in its official 96-shard format; the loader never converts, rewrites, or creates a second 1.56 TB copy.

The checked TP8 layout consumes 196.78 GB (183.27 GiB) of text weights per GPU. The compressed MLA cache adds 28.99 GB (27 GiB) per GPU at the full 1,048,576-token context, leaving approximately 62.23 GB of each nominal 288 GB MI350X for execution buffers and allocator overhead. Start much smaller.

## Before renting the machine

- Keep the existing 96 shards in `/raid/weights/kimi-k3`; no additional model-sized free space is required. Leave ordinary headroom for logs and temporary files.
- The host should have roughly 3 TB RAM, in line with AMD's MI350X platform guidance. The loader itself is streaming and must not need checkpoint-sized RAM.
- Use a recent kernel/ROCm stack supported by the host vendor, although tinygrad uses its own AMD userspace driver when `DEV=AMD`.
- Clone this exact commit/branch and keep the official checkpoint directory separate from the repository.

Validate the existing directory without modifying it:

```sh
python examples/kimi_k3_prepare.py /raid/weights/kimi-k3 --context 4096
```

For a metadata-only preflight, place the official `config.json` and `model.safetensors.index.json` in a directory and run:

```sh
python examples/kimi_k3_prepare.py /raid/weights/kimi-k3 --metadata-only
```

## Hardware admission checks

Do these before loading weights. Stop if any device is missing or reports a different architecture.

```sh
lspci -d 1002:75a0
amd-smi list
DEV=AMD DEBUG=2 python - <<'PY'
from tinygrad import Device
for i in range(8):
  dev = Device[f"AMD:{i}"]
  print(i, dev.arch)
PY
```

Expected architecture: `gfx950` on all eight devices. Then run the small TP8 graph tests:

```sh
python -m pytest test/unit/test_llm_k3.py test/null/test_kimi_k3.py -q -n12
DEV=NULL:HIP:gfx950 NULL_ALLOW_COPYOUT=1 python -m pytest \
  test/unit/test_llm_k3.py::TestKimiK3::test_chunked_recurrent_generate -q -n1
DEV=AMD python examples/kimi_k3_smoke.py --devices 8
```

The last two commands are deliberately small. They compile CDNA4 kernels and then exercise the complete TP8 topology without loading the checkpoint.

For performance iteration, use the exact-width fake-weight harness before another official load:

```sh
DEV=AMD python extra/benchmark_kimi_k3_fake.py --mode attention --iterations 20
DEV=AMD python extra/benchmark_kimi_k3_fake.py --mode block --iterations 20
```

It retains K3's 7,168-wide residual stream, 12,288-wide KDA state, 96 heads, 128×128 recurrent matrices, TP8 layouts, top-k 16 routing, packed MXFP4 expert shapes, collectives, and decode JIT, but uses one layer and 16 fake experts. Fake attention weights initialize in about 0.9 seconds and the full block in about 3 seconds. The retained path measured 0.630 ms per fake attention layer and 1.367 ms per complete fake block, projecting about 7.87 tok/s across 93 identical blocks versus 6.25 tok/s for the official heterogeneous model. Treat this as a candidate admission benchmark, not a correctness substitute for official weights.

## First official load

Start at a short context so cache allocation and compilation are bounded. The loader reads disk-backed safetensors, TP-shards every destination before realizing it, and drops each source shard/projection immediately afterward.

```sh
/usr/bin/time -v env DEV=AMD DEBUG=1 python -m tinygrad.llm.cli \
  --model /raid/weights/kimi-k3 --devices 8 --max_context 128 </dev/null 2>&1 | tee kimi-k3-load.log
```

Watch host RAM, swap, HBM, temperatures, and XGMI traffic from a second terminal. Do not start with a one-million-token cache. If loading fails, preserve the first exception and the last loader progress line; do not retry with a larger host-side cache.

## Correctness and performance sequence

1. Load with context 128 and generate one token.
2. Repeat a fixed prompt twice and confirm token-for-token deterministic greedy output.
3. Compare the first several greedy tokens against the official Transformers implementation at temperature zero.
4. Benchmark decode only after two warm-up tokens.
5. Benchmark prefill at 128, 512, 2K, and 8K tokens. Increase context only while HBM and compile time remain healthy.
6. Use `VIZ=1` plus `python -m tinygrad.viz.cli` to inspect kernels; use `VIZ=2` only for short SQTT captures because it adds overhead.

Example decode benchmark:

```sh
DEV=AMD DEBUG=1 python -m tinygrad.llm.cli --model /raid/weights/kimi-k3 \
  --devices 8 --max_context 4096 --warmup --benchmark 20
```

## MI350X validation results (2026-08-10)

The official directory was audited in place: 96 shards, 497,220 indexed tensors, 497,052 language tensors, and 1,560,860,324,864 total bytes. All eight devices reported `gfx950`. No checkpoint file was converted, copied, or modified, and every model run used a single process. The actual text tower is 1,559,965,606,912 bytes; its checked TP8 layout is 196,784,397,312 bytes per GPU.

The preserved first full-checkpoint error was an `A_log` shape mismatch, `(128,) -> (96, 1)`. K3 stores one decay value per 128-wide KDA channel, not one per head. The loader now keeps this field replicated and applies the official channel-wise broadcast. A numerical unit test covers the distinction from the older head-wise Kimi Linear behavior.

Load speed was fixed before generation. The original loader opened thousands of individual expert tensors and independently realized eight strided TP slices. The MI350 path now does the following without changing the checkpoint:

- parses safetensor headers selectively, constructing disk-backed tensors only for the 2,460 non-expert entries consumed by that pass instead of materializing metadata objects for every expert entry twice;
- copies contiguous axis-zero shards and replicas directly into their final device buffers;
- reads a replicated tensor once and fans it out over XGMI instead of issuing eight identical direct reads (14.31 GB less RAID traffic);
- stages an inner-axis tensor once and schedules all eight TP slices together;
- reads each layer's contiguous 15.72 GB expert region once, reorders its lexicographically stored expert records on GPU 0, and realizes all six packed/scale destinations together;
- retains only final MultiBuffer identities, drops the reorder graph, and flushes the 15.72 GB staging allocation before the next layer.

One real expert layer leaves exactly 1,965,293,568 bytes resident on each GPU and zero bytes in the GPU-0 allocator cache. Complete context-128 loads measured 527.20 seconds before the final staging cleanup and 490.05/489.59 seconds afterward. Peak host RSS for the unprofiled correctness run was 2.11 GiB with zero swap. RAID variability produced later loads from 489.06 to 532.85 seconds.

The selective-metadata and bounded-GC pass reduced non-expert loading from 125.77 to 57.77 seconds. A subsequent full official context-128 load completed in 411.49 seconds, 78.10 seconds (16.0%) faster than the 489.59-second baseline. It read the 96 shards in place with 1,049,688 KiB peak host RSS and zero swap; no weight payload was converted, copied, or modified. Direct-I/O probes measured approximately 6.9 GB/s aggregate for both one and eight concurrent 1 GiB reads. At that rate the 1.56 TB checkpoint has a roughly 227-second cold-read lower bound, so this RAID cannot meet a true cold sub-three-minute startup regardless of loader overhead.

Expert staging graphs are acyclic and are released by reference counting after each layer, so the loader now suppresses unnecessary cyclic-collector scans only around that loop and restores its prior state on every exit. A quiet context-128 load then completed in 391.54 seconds, 30.14 seconds (7.1%) faster than the immediately preceding 421.68-second run, with 1.04 GiB peak RSS and zero swap, although storage variability contributes to run-to-run timing. The host used for these measurements actually mounts `/raid` from one 3.5 TB XFS NVMe, not a multi-drive RAID; shard 28 has 218 extents and live reads fell to roughly 160 MB/s there. This storage layout, plus the physical checkpoint size, remains the limiting cold-start constraint. The weights were not defragmented, copied, or modified.

The fixed XTML prompt `Reply with exactly: OK` encodes to 93 tokens. After excluding the cold JIT capture from replay comparison, two greedy runs produced the identical eight-token sequence:

```text
[9545, 59991, 10580, 14404, 9545, 59991, 9545, 59991]
```

At context 128, steady prefill was 14.32 seconds (6.49 tok/s) and eight-token decode was 2.27 seconds (3.53 tok/s, 283.3 ms/token). The same first tokens remained stable at every admitted context. These rates are much lower than the planning estimates below and should be treated as the current measured baseline.

The retained gfx950 serving pass enables the validated wave64 recurrent prefill kernel with 128-token chunks, uses exact BF16 decode projections, combines the routed/shared final TP partials into one collective, and tiles four adjacent packed-expert outputs during multi-token execution. On the same 93-token prompt, two replay trials produced the identical sequence `[198, 92652, 220, 80225]`. Prefill replay measured 2.418--2.482 seconds (37.47--38.46 tok/s), and eight-token decode measured 1.294 seconds (6.18 tok/s, 161.81 ms/token). Peak RSS was 2.77 GiB with zero swap. The packed prefill tile changes floating-point reduction order: direct official-layer comparison against the original kernel had maximum differences of 0.015625 for gate and 0.0078125 for down, and the end-to-end greedy sequence was stable across replay.

A subsequent gfx950 decode pass split the 7,168-wide replicated BF16 projections across eight waves per 16 output channels and used CDNA4 BF16 MFMA, with one FP32 LDS reduction at the end. It is enabled only for batch-one/token-one replicated projections whose dimensions satisfy the hardware tile; prefill, the FP32 router, and the output-sharded 12,288-wide KDA gate remain unchanged. The official retained path uses it for MLA q-a/kv-a and KDA f-a. Isolated TP8 measurements improved replicated 128/576-output projections by about 16--18%; applying it to the already output-sharded KDA gate was slower and was rejected. Random-shape comparison against the generic graph had maximum/mean absolute BF16 differences of 2.0/0.1114 because the split changes reduction order. Against a serial FP32 accumulation rounded once to BF16, the 7,168-to-1,536 kernel was bit-exact in the tested sample.

The final official context-128 validation loaded in 389.84 seconds with 2.71 GiB peak RSS and zero swap. Two replay trials produced the identical four-token sequence `[198, 59675, 9817, 12519]`; prefill remained 2.406 seconds (38.65 tok/s), while eight-token decode improved to 1.280 seconds (6.25 tok/s, 160.00 ms/token). A one-wave MFMA variant and a full-wave fused decode recurrence were both rejected: the former delivered 6.02 tok/s, and the latter 6.179 tok/s, while both changed the greedy sequence without a useful speed gain.

A final load-first experiment increased the disk-to-HBM io_uring queue depth from one to the 32 existing bounded 2 MiB staging buffers. On a direct 1 GiB read from fragmented shard 28 it measured 6.834 GB/s versus 6.832 GB/s for the original path, so the change was rejected. The subsequent unmodified official 96-shard load completed in 389.48 seconds, confirming both the prior result and the single-NVMe lower bound. Peak RSS was 2.75 GiB with zero swap.

Two direct packed-expert MFMA prototypes were also rejected after that load. A fused gate/up kernel was about 29% faster in isolation at the TP8-local shape, and a routed-down kernel which combined projection, probability weighting, and route reduction measured 1.45 ms versus 2.42 ms in isolation. End-to-end, however, stable replay produced `[198, 2338, 2127, 148297]`, prefill measured 38.87 tok/s, and decode measured 6.263 tok/s. That is indistinguishable from the retained 38.65/6.25 tok/s path while changing floating-point reduction order, so neither kernel was retained.

A whole-core KDA decode experiment fused convolution, Q/K normalization, channel decay, recurrence, RMS normalization, output gating, and four persistent state updates. Its raw kernel replayed in about 109 microseconds per local KDA layer and matched a one-step synthetic reference within `9.77e-4` output and `8.13e-4` state maximum error. The exact-width fake-layer gate caught that it was slower than the retained attention path (0.665 versus 0.633 ms/layer). The already-running official validation was stopped after its first invalid greedy sequence, `[198, 163840, 163840, 163840]`, where 163840 is outside the checkpoint's vocabulary. The kernel was rejected and removed.

| Maximum context | Load | Short-prompt replay | Result |
|---:|---:|---:|---|
| 128 | 489.59s | 14.32s | stable 8-token replay |
| 4,096 | 489.06s | 14.32s | stable replay, zero swap |
| 32,768 | 532.85s | 14.33s | stable replay, zero swap |
| 131,072 | 520.91s | 14.37s | stable first token, zero swap |
| 262,144 | 497.34s | 14.41s | stable first token, zero swap |

These are maximum-context/cache admission tests with the same 93-token prompt, not full-length 32K/131K/262K prefills. The full cache allocation path was exercised, but filling those contexts remains a separate long-running throughput test.

Runtime profiling bracketed four steady decode tokens. It recorded 6,304 kernel events and about 474--478 ms of summed GPU work across the eight devices inside a roughly 1.5-second profiled wall interval. The packed `mxfp4_expert_linear_wave64` kernels accounted for only about 22.5 ms summed; the largest families were small 1,792-wide reductions. This identifies launch/synchronization granularity as the immediate MI350 bottleneck rather than packed-weight bandwidth. `JIT_BATCH_SIZE=64` produced the same original 3.53 tok/s as 32. A gfx950 fused MXFP8 QDQ experiment was bit-exact but slower on the real device (about 95 microseconds versus 57--64 microseconds), so it was rejected. Combining the routed and shared final TP partials removed one collective per routed decode layer and helped raise unprofiled decode to 6.18 tok/s, but the remaining sequential launch boundaries still dominate.

The checkpoint's bundled Transformers code was used as the architectural reference for channel decay and tensor mapping. A full independent Transformers/vLLM token comparison was not run on this host because the required `compressed_tensors`/serving backend is not installed; deterministic tinygrad replay and the numerical KDA, loader-layout, NULL gfx950 compile, and real TP8 smoke tests are the completed correctness gates.

## Known hardware-only gate

The correctness path now consumes packed MXFP4 expert weights directly on gfx950 with a wave64 software-decode kernel, so it does not create selected-expert BF16 weight expansions. MXFP8 activation quantization is still emulated. tinygrad has gfx950/CDNA4 BF16 and FP8 matrix-core support, but this branch does not yet have a hardware-validated native MXFP4×MXFP8 expert GEMM. Expect the first run to be a correctness bring-up, not production throughput. Capture profiles on MI350X before changing the representation: native FP4 work cannot be validated faithfully on the available gfx1100 cards.

Recurrent prefill is fused. The gfx950 wave-parallel kernel was compared directly with the portable graph at the official per-GPU shape through 128 tokens: maximum core/state differences remained below `8e-6`/`1e-6`, outputs were finite, and replay was about 2.7 ms versus about 8 ms for the portable kernel in the isolated test. Full K3 therefore uses 128-token recurrent chunks on gfx950. Chunk size remains part of the numerical configuration because different reduction orders can select different final greedy tokens.

The following serving changes apply to the official K3 path: recurrent-state reset graph capture, direct AMD scalar readback without rebuilding a scheduler graph, materialized gate/up boundaries, separate greedy decode JITs, K3's uncorrected routed probability semantics, gfx950 KDA Q/K/V and exact BF16 partial projections, one combined routed/shared final collective, a gfx950 greedy output-head kernel, the wave64 packed-expert path, and the multi-token four-output packed tile. Software MXFP8 remains in use.

After hardware admission on MI350X, profile before porting those kernels. The likely implementation order is:

1. A native packed MXFP4×MXFP8 grouped expert GEMM using CDNA4 matrix instructions.
2. A wave64/MFMA KDA Q/K/V decode projection.
3. Combined routed/shared down-projection TP partials so each layer performs one XGMI all-reduce.
4. A CDNA4 output-head matvec and router matvec if they remain visible in the profile.

Every port needs a direct numerical comparison with the generic graph and an end-to-end greedy-token comparison before performance measurements. The wave64 packed-expert kernel has compile coverage through `NULL:HIP:gfx950`; numerical and performance validation still require real MI350X hardware. None of the remaining gfx11-only kernels should be enabled on gfx950 by changing only the architecture guard.

## MI350X performance expectation

Treat the first rental as bring-up, not a guaranteed throughput run. The loader reads every official expert tensor once into a transient GPU-0 staging buffer (at most one packed projection), then redistributes TP8 slices over the GPU fabric; it does not generate files or require checkpoint-sized host RAM. A reasonable planning range for the full text model on eight MI350X cards is 3–8 minutes to stream and TP-shard the 1.56 TB checkpoint, 150–400 tok/s for initial short/medium prefill, and 25–60 tok/s decode with the software packed-expert path. After a native CDNA4 MXFP4×MXFP8 grouped expert kernel, wave64/MFMA recurrent projections, and XGMI collective tuning, 500+ tok/s prefill and roughly 80–150 tok/s decode are plausible targets. These ranges are engineering estimates, not measurements.

The nominal HBM bandwidth is not the main uncertainty: eight MI350X devices have enough aggregate bandwidth for K3's active weights. Utilization is limited by 93 sequential layers, small routed projections, and synchronization after TP input-sharded projections. Record actual HBM and XGMI counters before deciding whether the next port should target matrix instructions or collective count.

The official checkpoint also contains MoonViT-V2 and multimodal projector weights. They are skipped by the text loader. Image input remains a separate implementation and validation task.

## Local TP4 performance baseline

The pre-rental benchmark uses the converted `Kimi-Linear-48B-A3B-Instruct-MXFP4-v2` checkpoint on four gfx1100 GPUs. It is a useful regression test for the KDA/MLA/MoE text path, not a projection of K3 throughput on MI350X.

```sh
DEV=AMD JIT_BATCH_SIZE=64 python extra/benchmark_kimi.py \
  /raid/models/Kimi-Linear-48B-A3B-Instruct-MXFP4-v2 \
  --devices 4 --max-context 128 --prompt-tokens 32 --decode-tokens 32 --chunk-size 32
```

Results from 2026-08-10:

- load from RAID: 44.28s for the 29.27 GB checkpoint
- first 32-token prefill includes roughly 10s of compilation/capture
- steady fresh-prompt prefill replay: 0.118s, 270.20 tok/s
- steady context-32 decode replay: 101.82 tok/s, 9.82 ms/token
- peak host RSS: 729.9 MiB; swap was not used

The load, prefill, and decode targets are all met in the bounded prompt-32 run. Decode improved from 23.03 tok/s to 101.82 tok/s. The retained greedy output was checked across 32 decode steps; rejected half-wave and unrounded recurrent reductions were faster but diverged and eventually collapsed to a repeated token.

Four 7900 XTX cards provide 96 GB aggregate VRAM and about 3.84 TB/s aggregate physical memory bandwidth. Their nominal aggregate vector FP16 rate is about 245.6 TFLOP/s, or about 492 TFLOP/s through matrix instructions. Kimi Linear activates roughly 3.107B parameters per token; a simple active-weight accounting gives approximately 4.05 GB/token and an optimistic bandwidth-only ceiling near 948 tok/s. The measured decode rate is much lower because this MoE decode workload is a collection of small matrix-vector operations plus PCIe collectives, not one ideal streaming kernel.

The generic loader currently rereads logical TP shards and accounts for roughly 227 GB of disk traffic for a TP4 load. RAID bandwidth hides that inefficiency locally, but a direct one-pass shard loader remains worthwhile before slow remote storage is used. It was not retained here because the attempted direct-shard graph exposed an unresolved scheduler/renderer edge; correctness and bounded memory take priority over avoiding the redundant reads.

Different chunk sizes can choose a different final token because their matrix kernels use different floating-point reduction orders. Each measured shape was repeatable between cold and captured execution. For official K3 validation, compare logits/tokens against the reference at one fixed chunk size and greedy settings rather than requiring bitwise agreement between performance shapes.
