# Qwen Flash on tinyr9

This branch targets the full 48-layer Qwen3.8-Flash-Next UD-IQ4_XS with shared Q8_0 MTP on six RX 7900 XTX GPUs.
The goals are **8,000 tok/s prefill and 400 tok/s single-user decode on normal coding-agent workloads**, with testing at a
**262,144-token context**. **These goals are not yet met.** The earlier short-context measurements below do not establish agent-workload
or long-context performance.

## Long-context coding-agent baseline (work in progress)

`external_benchmark_llm_agent.py` constructs exact-sized prompts from actual repository source, relevant implementation excerpts,
and coding/debugging tasks, without random-ID or repeated-text padding. Its bounded-step driver separates compilation from warm work
and stops decoding at EOS or the model's context limit. Summed chunk times exclude RPC gaps; only `prefill_all()` measures an uninterrupted
end-to-end prefill.

The full model has processed a **262,000-token sampling-code-review prompt** with native max context **262,144**:
- Late 2112-token prefill chunks: **265 / 263 / 261 / 259 tok/s**. Compilation of the final 112-token chunk is excluded from these figures.
- Initial 128-token decode window: **37.8 tok/s** (3.3883 seconds of generator consumption).
- A checkpoint-replayed 128-token decode window: **155.7 tok/s** (four consecutive 32-token segments, 0.82225 seconds total).
- A separate 2048-token coding-task prefill: **8047.7 tok/s**, illustrating why short prefill is not sufficient evidence.

The timing difference between initial decode and replay has not been isolated. Checkpoint replay also produced a text variation and
needs investigation; this baseline is not a token-identical quality certification. The generated review correctly
identified that removing greedy/stochastic capture separation is unsafe, but the response reached the context boundary before a full patch.
The remaining speculative reserve meant the baseline decode stopped at 262,143 total tokens, not EOS.

### Sparse-prefill checkpoint

The new path scores sequence-local query rows, gathers compact 32-query pooled masks, compacts active key tiles, and runs masked tensor-core
attention. It preserves inclusive threshold ties and causal masking. Ballots are materialized before predicated compaction stores to avoid
incorrect ballot simplification and unwritten index-list entries. Qwen retains 2K prefill chunks at long contexts.

A full-model resident run with the new code, using the **same frozen 262,000 input IDs**, completed:
- Late prefill chunks: **3106–3624 tok/s**, versus approximately 259 tok/s before. The target is still missed.
- Entire staged prefill: **92.95 seconds active time**, including **26.82 seconds** for the final chunk's first capture. This is not an
  uninterrupted warm-prefill measurement and excludes RPC gaps.
- 128-token coding-review decode: **143.4 tok/s** (0.8924 seconds); decode has not been optimized in this checkpoint.
- **262,144-token boundary exercised:** restore the actual 262,000-token prompt checkpoint and prefill a 144-token source continuation.
  All 48 layers completed through position 262,144. This is a capacity/boundary check, with no remaining output headroom.

These model measurements used hot-loaded methods, not a fresh-process performance certification. Fresh standalone six-GPU sparse-attention,
compaction, long-pool-list and replay regressions passed (3 tests, 17.42 seconds); existing dense-prefill regressions passed (4 tests,
22.78 seconds). CPU regressions passed: 18 tests, 40 subtests, one skip. Ruff/diff checks pass; the same three pre-existing mypy errors remain.
Unmeasured histogram tuning was discarded. The resident benchmark process has been stopped to release the GPUs.

Next: profile long-context sparse attention/indexing, investigate checkpoint-replay text variation, improve decode, expand the coding-task
matrix, and rerun fresh-process end-to-end benchmarks. Do not treat this checkpoint as meeting 8000/400.

Reproduction artifacts retained on tinyr9: `/tmp/agent262prompt.i32` (frozen native-int32 prompt IDs),
`/tmp/agent262baseline.jsonl`, `/tmp/agent262sparse.jsonl`, and `/tmp/qp262k.log`.
The frozen prompt SHA-256 is `5d68f6e0b7853ae9c038fe24ee5ad077b47ee409f5620c506f7fad2b916c7906`.
Use `AgentBenchmark.prepare(..., prompt=frozen_ids)` for paired runs so source edits cannot silently change the workload.
The temporary RPC server overwrote top-level `seconds` with operation time; derive generator time from `tokens / tok_s` in those logs.

## Default policy

No optimization environment flags are required. The branch uses GPU P2P collectives with explicit L2 writeback,
Q8 reduce-scatter for aligned chunks, packed-Q8 allgather, staged RDNA3 Q8 GEMM, and 128-wide expert slices for six-GPU Qwen.
The experimental small-weight disk cache and unproven SDMA readback were removed.

Additional defaults include RDNA3 IQ3_S K256 prefill, fused gated normalization/mixing, split-K narrow Q8 GEMM, sequence-local routing,
small-batch bitonic routing, chunked large allgather, cache-only MTP prompt priming, and last-token final mixing. Decode also avoids unused
MTP catch-up queries/FFNs, uses narrow-slice/five-slot GEMV geometry, and broadcasts reduced chunks directly from the reduce kernel.
Greedy generation has separate captures and skips noise/nucleus sampling; nonzero-temperature sampling retains its own captures.
Four speculative drafts remain the default; five- and six-draft experiments did not improve the compared sky continuations overall.

## Verified warm default measurements

Same fresh process, full model, six GPUs, no hot-loaded methods or optimization overrides:

| Workload | Three runs (tok/s) |
| --- | ---: |
| 2048-token synthetic prefill through first output token | **8,176.46 / 8,220.65 / 8,210.90** |
| Binary-search explanation and Python code, 768 decoded tokens | **425.77 / 430.57 / 425.58** |

The decode runs took 1.804 / 1.784 / 1.805 seconds and included the first, slower decode segment; no end-of-response token was reached
within these 768-token windows. Separate longer runs reached end-of-response at 1,497 and 1,489 tokens at 399.75 and 403.20 tok/s;
a 1,536-token capped run measured 407.85 tok/s. These longer runs preceded the final fresh-process check.

Other warm natural prompts remain slower. In one four-prompt check, consecutive 128-token windows measured 327 / 369 / 349 tok/s for the
sky explanation, 292 / 321 / 297 for TCP/UDP, and 268 / 273 / 257 for the lighthouse story. Do not present the coding result as a general
minimum or compare it directly to the earlier 128-token sky/story windows as if acceptance and timing windows were identical.

Fresh-process loading plus deferred preparation took **27.91 seconds**. Initial large-prefill capture took 55.03 seconds; the short-prompt
and speculative captures took 29.85 and 23.36 seconds. These are excluded from the warm figures. The roughly 10-second load aspiration
is still unmet, and storage page-cache state was not controlled.

## Earlier full-model measurements

These measurements precede the default-policy cleanup and subsequent optimizations.

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

For the 768-token decode result, warm the short-prompt and speculative captures first, then reset both prompt-cache lists and
`model.spec_stats` before each independent run. Use:

`Explain how binary search works, give a Python implementation, and discuss its time complexity.`

The chat template is `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.
Obtain the first yielded token before starting the decode timer. Consume up to 768 subsequent tokens, stopping at `tokenizer.is_end(token)`;
never count tokens after end-of-response. Time generator consumption, not text detokenization. Repeat three times without profiling.
The model uses its ordinary greedy generation path and all 48 layers; there is no truncated-model extrapolation.

The earlier and comparison decode measurements time 128-token generator continuations using chat-formatted prompts:
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
DEV=AMD timeout 80 python -m pytest test/device/amd/test_ghc_prefill.py -n12 -x -q
DEV=AMD timeout 80 python -m pytest test/device/amd/test_ghc_decode.py -n12 -x -q
DEV=AMD timeout 80 python -m pytest test/device/amd/test_softmax_router.py -n12 -x -q
DEV=AMD timeout 80 python -m pytest test/device/amd/test_moe_combine.py -n12 -x -q
DEV=AMD timeout 80 python -m pytest test/device/amd/test_moe_decode.py -n12 -x -q
DEV=AMD timeout 80 python -m pytest test/device/amd/test_large_allgather.py -n12 -x -q
DEV=AMD timeout 80 python -m pytest test/device/amd/test_mtp_cache.py -n12 -x -q
```

Run the grouped IQ3_S, fused gate/up, and GDN LDS-scan cases in `test/device/amd/test_llm.py` separately as well.
CPU regressions include `test/runtime/test_llm_mtp_cache.py` and `test/runtime/test_llm_sampling.py`.

Tests cover fresh-input replay, all ranks, mixed sizes, epoch wrap, transport bit patterns, Q8 quantization, expert partitions,
GEMM tails, and byte-exact loading with short-read/EOF handling. New checks include NumPy MoE/GEMV references, exact fused-output Q8
quantization, signed-zero router ties, cache-write bounds, symbolic selected MTP rows, and switching greedy/stochastic capture modes.
Synthetic float32 MTP paths allow FP16 rounding differences in updated cache rows but require untouched rows to remain bit-exact.
An actual quantized-model cache comparison was exact on all six ranks for the checked inputs. Generated iterative and recursive binary
search implementations also passed 1,170 edge-case checks. These checks are not a full quality evaluation or token-identical certification.
RDNA4 hardware regression testing was not available. Whole-repository ruff and diff checks passed; targeted CPU regressions passed
(13 tests, 40 subtests, one skip), and the affected GPU suites passed in separate processes. Mypy still reports the pre-existing mock-HIP
hashability error and two `pool_k` attribute-inference errors in `tinygrad/llm/model.py`.
The collective poll no longer silently consumes stale data on timeout; missing peers instead leave the host queue wait to report failure.
Failure recovery and the gfx11-specific peer-mapping/writeback interface still need hardening.
