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

Recurrent prefill is fused and defaults to eight-token chunks. The portable custom kernel compiles for gfx950, but the wave-parallel version is deliberately restricted to gfx11 because it uses RDNA3 wave32 swizzles. A gfx950-tuned wave64/MFMA recurrent kernel remains a performance task for the rented machine; do not enable the gfx11 kernel on CDNA without rewriting its lane reduction and validating every recurrent-state transition.

The official checkpoint also contains MoonViT-V2 and multimodal projector weights. They are skipped by the text loader. Image input remains a separate implementation and validation task.

## Local TP4 performance baseline

The pre-rental benchmark uses the converted `Kimi-Linear-48B-A3B-Instruct-MXFP4-v2` checkpoint on four gfx1100 GPUs. It is a useful regression test for the KDA/MLA/MoE text path, not a projection of K3 throughput on MI350X.

```sh
DEV=AMD python extra/benchmark_kimi.py \
  /home/tiny/models/Kimi-Linear-48B-A3B-Instruct-MXFP4-v2 \
  --devices 4 --max-context 128 --prompt-tokens 32 --decode-tokens 8 --chunk-size 8
```

Results from 2026-08-10:

- load: 310.280s for 29.27 GB
- cold prefill including compilation: 47.622s
- captured prefill: 1.020s, 31.38 tok/s
- steady prefill replay: 0.898s, 35.64 tok/s
- steady decode replay: 23.03 tok/s, 43.43 ms/token
- peak host RSS: 901,192 KiB; swap: 0

The same resident model measured 19.60 tok/s with tokenwise prefill, so the selected eight-token chunk is 1.82× faster. A 32-token chunk fell to 3.03 tok/s because the current selected-expert path expands packed weights per token; larger batches multiply that temporary dequantization work. Chunk 8 is therefore a conservative default until native grouped MXFP4×MXFP8 expert GEMM exists.

Different chunk sizes can choose a different final token because their matrix kernels use different floating-point reduction orders. Each measured shape was repeatable between cold and captured execution. For official K3 validation, compare logits/tokens against the reference at one fixed chunk size and greedy settings rather than requiring bitwise agreement between performance shapes.
