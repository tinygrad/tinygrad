# Qwen Flash on tinyr9

This branch targets the full 48-layer Qwen3.8-Flash-Next UD-IQ4_XS with shared Q8_0 MTP on six RX 7900 XTX GPUs.
The current goals are **8,000 tok/s prefill and 400 tok/s single-user decode**. Neither has been verified yet.

## Default policy

No optimization environment flags are required. The branch uses GPU P2P collectives with explicit L2 writeback,
Q8 reduce-scatter for aligned chunks, packed-Q8 allgather, staged RDNA3 Q8 GEMM, and 128-wide expert slices for six-GPU Qwen.
The experimental small-weight disk cache and unproven SDMA readback were removed.

## Earlier full-model measurements

These measurements precede the default-policy cleanup; they are not a claim that the targets have been met.

| Workload / placement | Result |
| --- | ---: |
| Earlier fp16 reduce-scatter, slice 128 | 6,158–6,168 prefill tok/s |
| Q8 reduce-scatter, slice 128 | 6,698 / 6,743 prefill tok/s |
| Q8 reduce-scatter, slice 640 (whole experts) | 6,699 / 6,693 prefill tok/s |
| Lighthouse story, slice 640, two warm continuations | 211 / 253 decode tok/s |
| Sky/Rayleigh explanation, slice 128, warm continuation | 303 decode tok/s |
| Slice 640, model load plus deferred preparation | 27.26 seconds |

Decode figures are separate workloads/captures, not an average or guaranteed range. The sky measurement reused an earlier non-profiled
speculative capture; it is not a paired measurement of the whole-expert configuration. Whole-expert placement did not improve prefill.
Generated text was coherent, including a correct Paris answer. This is a smoke check, not a full quality evaluation or token-identical guarantee.
File-cache state was not controlled: the load figure is not an uncached-storage benchmark.

## Timing protocol

Context 4096, one user, four speculative drafts, 2048-token prefill (internal padded shape 2112).
A resident process separates loading and first-time JIT capture from warm measurements; each request has a 75-second deadline.
Loading and compilation are excluded from throughput. First captures took roughly 30–60 seconds, so these are not fast-TTFT figures.

Before every prefill, reset `model._cached_tokens` and `model._ckpt_tokens` to empty lists. Generate input IDs
`[(1000 + 7*i) % 200000 for i in range(2048)]` and time through the first yielded output token. Repeat after capture.
This includes last-token sampling and does not reuse a cached prompt. Synthetic input is for timing, not quality evaluation.

Decode times subsequent 128-token generator continuations after warmup using chat-formatted prompts:
- `Write a vivid story about a lighthouse keeper who discovers a mysterious message during a storm.`
- `Explain why the sky looks blue during the day and red at sunset.`

## Component validation

The collective benchmark uses GPU steady clocks and checks every result on every rank:

```sh
DEV=AMD TEST_GPUS=6 OP=rs ROWS=192,768,2112 K=4 timeout 80 python test/external/external_benchmark_llm_allreduce.py
```

Earlier Q8 reduce-scatter measurements at width 2560: 70.30 / 177.29 / 515.77 microseconds, with zero reference error for those inputs.
Remote Q8 partials are independently quantized; the local contribution remains unquantized.

Run GPU suites separately, with no model server owning the devices:

```sh
DEV=AMD TEST_GPUS=6 timeout 80 python -m pytest test/device/amd/test_flag_allreduce.py -n12 -x -q
DEV=AMD timeout 80 python -m pytest test/device/amd/test_q8_gemm.py -n12 -x -q
DEV=AMD TEST_GPUS=6 timeout 80 python -m pytest test/device/amd/test_expert_slices.py -n12 -x -q
DEV=AMD TEST_GPUS=6 timeout 80 python -m pytest test/device/amd/test_gguf_load.py -n12 -x -q
```

Tests cover fresh-input replay, all ranks, mixed sizes, epoch wrap, transport bit patterns, Q8 quantization, expert partitions,
GEMM tails, and byte-exact loading with short-read/EOF handling. RDNA4 hardware regression testing was not available.
The collective poll no longer silently consumes stale data on timeout; missing peers instead leave the host queue wait to report failure.
Failure recovery and the gfx11-specific peer-mapping/writeback interface still need hardening.
