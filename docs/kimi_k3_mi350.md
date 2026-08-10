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

## Known hardware-only gate

The correctness path now consumes packed MXFP4 expert weights directly on gfx950 with a wave64 software-decode kernel, so it does not create selected-expert BF16 weight expansions. MXFP8 activation quantization is still emulated. tinygrad has gfx950/CDNA4 BF16 and FP8 matrix-core support, but this branch does not yet have a hardware-validated native MXFP4×MXFP8 expert GEMM. Expect the first run to be a correctness bring-up, not production throughput. Capture profiles on MI350X before changing the representation: native FP4 work cannot be validated faithfully on the available gfx1100 cards.

Recurrent prefill is fused. Kimi Linear defaults to 32-token chunks; full K3 remains capped at eight tokens because its larger recurrent state makes that the safer pre-rental compile shape. The portable kernel compiles for gfx950, but the wave-parallel recurrent version is deliberately restricted to gfx11 because it uses RDNA3 wave32 swizzles. A gfx950-tuned wave64/MFMA recurrent kernel remains a performance task for the rented machine; do not enable the gfx11 recurrent kernel on CDNA without rewriting its lane reduction and validating every recurrent-state transition.

The following serving changes apply to the official K3 path: recurrent-state reset graph capture, direct AMD scalar readback without rebuilding a scheduler graph, materialized gate/up boundaries, separate greedy decode JITs, K3's uncorrected routed probability semantics, exact fused BF16 gate/up pairs, and the gfx950 wave64 packed-expert path. Software MXFP8, fused KDA Q/K/V, combined TP down-partial, and the greedy output-head kernel remain gfx11-only and fall back to the generic implementation on gfx950.

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
