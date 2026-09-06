# MXFP4 short-K optimization

The selected kernel in `mxfp4_gemm_shortk.py` targets M=16384, N=4096,
K=4096. It is a standalone assembly generator. The dispatcher selects it
for this exact shape and retains this branch's existing implementation for other shapes.

## Selected implementation

The workgroup tile is 128x512, with four waves and a 128x128 output per
wave. The launch is 8x128 workgroups. A uses two padded K256 LDS buffers
(35,840 bytes total); B and packed scales are prefetched into registers.
The allocation uses 248 regular VGPR entries and 256 accumulator entries
per thread, allowing one resident wave per SIMD.

The first K128 contribution initializes accumulators directly through the
MFMA source operand, eliminating the separate accumulator-clear sequence.
At the last K256 iteration, the lower output rows finish first. Their
accumulator reads, BF16 conversions, and stores are interleaved with the
remaining MFMAs for the upper rows. The remaining output batches have
separate packed-value and address registers. Stores use sc1=1 and nt=1.

The source explicitly emits both wave schedules and the first/final K
paths. It does not import, inspect, or transform another kernel's generated
instruction stream. There is no runtime tile selection in this module.

## Measured performance

Measurements below use the full correctness test on this MI350P, including
the output stores in the timed GEMM. Input quantization is a separate kernel
and is excluded consistently from both GEMM times.

| Implementation | GEMM time | Result |
| --- | ---: | --- |
| Initial two-group prototype | about 880 us | Correct, substantial regression |
| Original 256x256 kernel, seven-run median | 247.44 us | Correct |
| Selected 128x512 pipeline, seven-run median | 234.92 us | Correct |
| Final source with exact LDS allocation, five-run median | 236.52 us | Correct |

The improvement over the original is approximately 4–5% in elapsed time,
not a large MFU breakthrough. At 236.52 us, 2*M*N*K is about 2.324 PFLOPS,
or 50.5% MFU using the 4.6 PFLOPS denominator from our previous comparisons.
**The high-MFU objective has not been achieved.**

Additional tested designs included staggered wave groups, independent
smaller workgroups, direct global-to-LDS payloads and scales, deeper
register prefetch, A-only shared memory, and independent waves without
LDS. The correct variants were slower than the selected implementation.
The 32x32 MFMA experiment was also discarded; its initial output-layout
check failed, so its timing is not a valid performance result.

## Profiling evidence

The earlier 128x512 SQTT capture showed approximately 35–37k cycles of
MFMA work per tile versus an ideal 32,768 issue cycles. Initialization,
drain, and the interval before the next workgroup consumed another
approximately 12–14k cycles. This motivated first-K initialization and
final-K output overlap. It does not establish that every remaining delay
is caused by one mechanism.

The final capture verifies the intended overlap: on each traced SIMD,
128 of 256 output-store instructions issue during the final MFMA span
across eight tiles. Thus half the output stores overlap remaining matrix
instructions. This is instruction overlap, not proof that all store latency
is hidden. Counts are saved in `/tmp/mxfp4_opt_final_overlap.json`.

Captures and logs are in `/tmp`:

- `mxfp4_opt_128_sqtt.jsonl`: earlier 128x512 trace used for diagnosis.
- `mxfp4_opt_final_sqtt.jsonl`: selected kernel's final trace.
- `mxfp4_opt_baseline_final.log`: original 256x256 comparison.
- `mxfp4_opt_candidate_final.log` and `mxfp4_opt_final.log`: candidate timings.
- `mxfp4_opt_signed_final.log` and `mxfp4_opt_signed_exact_final.log`: signed random input checks.
- `mxfp4_opt_mutation_final.log`: deliberate early-return failure.

## Validation

```sh
K=4096 CNT=5 DEV=AMD DEBUG=2 PYTHONPATH=. python test/backend/test_asm_gemm.py TestMXFP4.test_correctness2
K=4096 VIZ=-2 DEV=AMD DEBUG=2 PYTHONPATH=. python test/backend/test_asm_gemm.py TestMXFP4.test_correctness2
python -m tinygrad.viz.cli -s 'mxfp4_gemm_sk_16384_4096_4096 SQTT SE:0 PKTS' --json > /tmp/mxfp4_opt_final_sqtt.jsonl
python -m ruff check .
python -m ruff check extra/gemm/mxfp4_gemm_shortk.py
python -m mypy tinygrad/
```

Full-output comparisons pass with zero-initialized output and the existing
rtol=0.005, atol=0.001 tolerance. Signed random inputs passed separately.
Inserting `s_endpgm` at kernel entry produced ten printed zeros and the
expected `MXFP4 GEMM forward mismatch` assertion. The mutation was removed
before the final positive capture. Ruff and mypy passed.

## Port to fast_fp4_best

The kernel source is identical to the validated source on `profile_gemm`.
Only the exact supported shape uses the new launch; other shapes retain
this branch's existing tile selection and persistent launch configuration.
Initialized output support and `test_correctness2` are included for validation.

Offline assembly checks passed for the new kernel, another K4096 production
shape, and a small generic shape. Repository lint, explicit kernel lint,
and mypy passed. The GPU correctness rerun on this branch stopped during
input allocation (`PTE already mapped: 0xffffffffffffffff`), before GEMM
execution; the runtime subsequently reported a fatal hardware error.
Consequently, the performance and GPU correctness measurements above are
from the source branch, not a successful rerun after this port.
