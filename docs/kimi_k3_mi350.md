# Kimi K3 on 8× MI350X

This branch targets text generation from the official `moonshotai/Kimi-K3` checkpoint. It intentionally ignores the vision tower and multimodal projector. The checkpoint remains in its official 96-shard format; no conversion or second 1.56 TB copy is required.

The checked TP8 layout consumes 196.78 GB (183.27 GiB) of text weights per GPU. The compressed MLA cache adds 28.99 GB (27 GiB) per GPU at the full 1,048,576-token context, leaving approximately 62.23 GB of each nominal 288 GB MI350X for execution buffers and allocator overhead. Start much smaller.

## Before renting the machine

- Reserve at least 1.7 TB of local model storage. More headroom is preferable for download caches and logs.
- The host should have roughly 3 TB RAM, in line with AMD's MI350X platform guidance. The loader itself is streaming and must not need checkpoint-sized RAM.
- Use a recent kernel/ROCm stack supported by the host vendor, although tinygrad uses its own AMD userspace driver when `DEV=AMD`.
- Clone this exact commit/branch and keep the official checkpoint directory separate from the repository.

Download on a machine with the storage bandwidth and network allocation intended for the run:

```sh
hf download moonshotai/Kimi-K3 --local-dir /models/Kimi-K3
python examples/kimi_k3_prepare.py /models/Kimi-K3 --context 4096
```

For a metadata-only preflight, place the official `config.json` and `model.safetensors.index.json` in a directory and run:

```sh
python examples/kimi_k3_prepare.py /models/Kimi-K3-metadata --metadata-only
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
  --model /models/Kimi-K3 --devices 8 --max_context 128 </dev/null 2>&1 | tee kimi-k3-load.log
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
DEV=AMD DEBUG=1 python -m tinygrad.llm.cli --model /models/Kimi-K3 \
  --devices 8 --max_context 4096 --warmup --benchmark 20
```

## Known hardware-only gate

The correctness path expands only selected MXFP4 expert weights and emulates MXFP8 activation quantization. tinygrad has gfx950/CDNA4 BF16 and FP8 matrix-core support, but this branch does not yet have a hardware-validated fused native MXFP4×MXFP8 expert GEMM. Expect the first run to be a correctness bring-up, not production throughput. Capture profiles on MI350X before changing the representation: native FP4 work cannot be validated faithfully on the available gfx1100 cards.

Recurrent prefill is fused and defaults to 32-token chunks. The portable custom kernel compiles for gfx950, but the wave-parallel version is deliberately restricted to gfx11 because it uses RDNA3 wave32 swizzles. A gfx950-tuned wave64/MFMA recurrent kernel remains a performance task for the rented machine; do not enable the gfx11 kernel on CDNA without rewriting its lane reduction and validating every recurrent-state transition.

The following serving changes are portable and therefore apply to the official K3 path: recurrent-state reset graph capture, correct fresh-prompt benchmark resets, 32-token recurrent chunks, materialized gate/up boundaries that prevent pathological fused reduction kernels, separate greedy decode JITs, and K3's uncorrected routed probability semantics. The new packed expert, software MXFP8, fused KDA Q/K/V, combined TP down-partial, and exact greedy output-head kernels are intentionally gated to gfx11. They improve the local 7900 XTX bring-up but will fall back to the generic implementation on gfx950.

After hardware admission on MI350X, profile before porting those kernels. The likely implementation order is:

1. A native packed MXFP4×MXFP8 grouped expert GEMM using CDNA4 matrix instructions.
2. A wave64/MFMA KDA Q/K/V decode projection.
3. Combined routed/shared down-projection TP partials so each layer performs one XGMI all-reduce.
4. A CDNA4 output-head matvec and router matvec if they remain visible in the profile.

Every port needs a direct numerical comparison with the generic graph and an end-to-end greedy-token comparison before performance measurements. None of the gfx11 custom kernels should be enabled on gfx950 by changing only the architecture guard.

The official checkpoint also contains MoonViT-V2 and multimodal projector weights. They are skipped by the text loader. Image input remains a separate implementation and validation task.

## Local TP4 performance baseline

The pre-rental benchmark uses the converted `Kimi-Linear-48B-A3B-Instruct-MXFP4-v2` checkpoint on four gfx1100 GPUs. It is a useful regression test for the KDA/MLA/MoE text path, not a projection of K3 throughput on MI350X.

```sh
DEV=AMD JIT_BATCH_SIZE=64 python extra/benchmark_kimi.py \
  /raid/models/Kimi-Linear-48B-A3B-Instruct-MXFP4-v2 \
  --devices 4 --max-context 128 --prompt-tokens 32 --decode-tokens 8 --chunk-size 32
```

Results from 2026-08-10:

- load from RAID: 43.9–44.4s for the 29.27 GB checkpoint
- first 32-token prefill includes roughly 10s of compilation/capture
- steady fresh-prompt prefill replay: 0.120s, 267.18 tok/s
- steady decode replay: 66.19 tok/s, 15.11 ms/token
- peak host RSS: 799.9 MiB; swap was not used

The load and prefill targets of less than 60 seconds and more than 200 tok/s are met on this host. Decode improved from 23.03 tok/s to 66.19 tok/s but remains below the 100 tok/s target. The remaining local profile is dominated by many small router/shared projections, collective and graph-launch overhead, and the unavoidable active expert traffic; this result must not be reported as reaching the decode goal.

Four 7900 XTX cards provide 96 GB aggregate VRAM and about 3.84 TB/s aggregate physical memory bandwidth. Their nominal aggregate vector FP16 rate is about 245.6 TFLOP/s, or about 492 TFLOP/s through matrix instructions. Kimi Linear activates roughly 3.107B parameters per token; a simple active-weight accounting gives approximately 4.05 GB/token and an optimistic bandwidth-only ceiling near 948 tok/s. The measured decode rate is much lower because this MoE decode workload is a collection of small matrix-vector operations plus PCIe collectives, not one ideal streaming kernel.

The generic loader currently rereads logical TP shards and accounts for roughly 227 GB of disk traffic for a TP4 load. RAID bandwidth hides that inefficiency locally, but a direct one-pass shard loader remains worthwhile before slow remote storage is used. It was not retained here because the attempted direct-shard graph exposed an unresolved scheduler/renderer edge; correctness and bounded memory take priority over avoiding the redundant reads.

Different chunk sizes can choose a different final token because their matrix kernels use different floating-point reduction orders. Each measured shape was repeatable between cold and captured execution. For official K3 validation, compare logits/tokens against the reference at one fixed chunk size and greedy settings rather than requiring bitwise agreement between performance shapes.
