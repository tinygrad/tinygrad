# MI350P short-K MXFP4 experiment

Shape: M=16384, N=4096, K=4096.

The default `gemm_mxfp4_shortk.py` kernel reuses the existing 256x256 matrix
pipeline and changes output-tile scheduling for this exact shape and tile size.
Other shapes keep the generic kernel. `MXFP4_SHORT_K=0` restores the generic
kernel for comparisons.

```sh
# Existing kernel, correctness and timings (15 measurements)
MXFP4_SHORT_K=0 K=4096 CNT=15 DEV=AMD DEBUG=2 PYTHONPATH=. python test/backend/test_asm_gemm.py TestMXFP4.test_correctness2

# New kernel, same correctness check and timings
K=4096 CNT=15 DEV=AMD DEBUG=2 PYTHONPATH=. python test/backend/test_asm_gemm.py TestMXFP4.test_correctness2
```

The test initializes each output to zero, prints ten output/reference elements,
and compares every element with a Tensor reference computed from decoded FP4
operands. Reference computation, output initialization, and comparison are outside
the displayed GEMM timing and run with DEBUG=0.

## Results

Final comparison on this machine, 2026-09-06, medians of 15 successful runs:

| Kernel | GEMM time | Throughput | MFU against 4.6 PFLOPS |
|---|---:|---:|---:|
| Generic | 248.52 us | 2.212 PFLOPS | 48.09% |
| Persistent + next-tile prefetch | 241.04 us | 2.281 PFLOPS | 49.58% |

This is a 1.031x speedup (3.0% less time). It does not reach the 10-15% improvement
suggested in the supplied planning conversation. MFU is `2*M*N*K / time / 4.6e15`;
it excludes quantization and is a kernel-throughput ratio, not end-to-end model MFU.

Exploration measurements below were separate 7-run sessions unless noted. Small
differences between nearby variants can be timing noise; only the final comparison
above was repeated with 15 measurements per kernel.

| Experiment | Median us | Correct | Decision |
|---|---:|---|---|
| Initial generic baseline | 248.36 | yes | reference |
| Constant tile-coordinate setup | 246.52 | yes | retained in short-K path |
| Persistence, 2 tiles/workgroup | 245.52 | yes | available for comparison |
| Persistence, 4 tiles/workgroup | 245.64 | yes | available for comparison |
| Persistence, 8 tiles/workgroup | 246.56 | yes | little benefit by itself |
| Persistence + next-tile prefetch, 8 tiles | 241.96 | yes | retained |
| Persistence + next-tile prefetch, 2 tiles (9 runs) | 243.64 | yes | available for comparison |
| Persistence + next-tile prefetch, 4 tiles (9 runs) | 241.88 | yes | indistinguishable from 8 within noise |
| Swizzled LDS output staging / contiguous stores | 252.16 | yes | reverted |
| Separate last-two-iteration drain, with prefetch | 244.12 | yes | reverted |
| First-MFMA zero initialization, with prefetch | 241.88 | yes | reverted; no clear additional win |

`MXFP4_TILES_PER_WG={1,2,4,8}` controls static persistence. Default is 8, giving
128 workgroups. `MXFP4_PREFETCH=0` disables cross-tile input prefetch; it defaults
to enabled when tiles/workgroup > 1. These controls apply to the supported short-K shape.

## What changed

The generic kernel maps one workgroup to one output tile. The new kernel assigns
`(group_x, group_y + iteration * (64 / tiles_per_wg))` to a persistent workgroup.
The N dimension has exactly 16 tiles, so the generic coordinate division/swizzle
reduces to the original workgroup coordinates and is omitted.

After finishing a matrix tile, the new kernel:

1. Waits for the old input accesses and synchronizes all four waves.
2. Saves the current C descriptor in s84:87.
3. Builds the next tile's input descriptors and issues the initial A/B and scale loads.
4. Converts/stores the current output through the saved C descriptor, clearing each
   accumulator only after reading its result.
5. Waits for the required input transfers, synchronizes, and enters the next tile's
   LDS-read/matrix pipeline. The last tile takes the ordinary final epilogue.

The register lifetimes allow input prefetch to overlap output processing: the
input prologue uses v0:7 and v136:234, while output conversion uses v8:19 and
output addresses remain in v235:250. The next tile's LDS reads into the output
scratch registers are deliberately deferred until output conversion finishes.

This overlaps next-tile **input transfers** with current-tile output processing.
It does not overlap the next tile's MFMAs with the current tile's epilogue, as the
more elaborate NVIDIA warp-specialized architecture does. Most of the store tail
therefore remains exposed.

## Assembly / trace map

Offsets below are from the original generic kernel assembled for this shape,
not the SQTT export's partially mapped PC annotations. Ranges are half-open.

| Byte offsets | Region |
|---|---|
| 0x0000-0x0424 | Arguments, coordinate mapping, descriptors, input addresses |
| 0x0424-0x0df8 | Initial global loads, 256 accumulator clears, wait/barrier |
| 0x0df8-0x0fb4 | Initial LDS reads and output address setup |
| 0x0fb4-0x3a68 | Two wave-group matrix pipelines and loop control |
| 0x3a68-0x3a70 | Matrix-to-epilogue wait/barrier |
| 0x3a70-0x4af0 | Accumulator reads, BF16 conversion, lane rearrangement, stores |
| 0x4af0-0x4af8 | Final wait and termination |

Separate VIZ=-2 captures corroborate the performance direction:

| SE:0 trace measurement | Generic | New |
|---|---:|---:|
| Completed hardware wave lifetimes | 32 | 4 |
| Output-tile wave portions | 32 | 32 |
| Mean last-MFMA to next-tile first-MFMA gap | 13,481 cycles | 11,700 cycles |
| Median gap | 13,454 cycles | 11,156 cycles |
| Mean first-to-last MFMA region | 37,279 cycles | 36,968 cycles |
| Full exported span | 408,756 cycles | 392,816 cycles |

Gap means are over 28 tile transitions in four SIMD streams. The mean gap fell
13.2%. The body span includes intervening non-MFMA instructions and waits; it is
not a measurement of pure matrix-unit busy cycles. These are sampled trace data,
not a whole-GPU stall attribution.

VALU_MAI includes accumulator reads/writes as well as MFMA. The analysis separates
them using the verified instruction sequence: each tile has 2,048 MFMAs per wave,
256 initial accumulator clears, and 256 epilogue reads. Persistent transitions
interleave 256 reads with 256 clears. Raw PC annotations are not used to classify
those phases.

## Captures and validation

```sh
MXFP4_SHORT_K=0 K=4096 VIZ=-2 DEV=AMD DEBUG=2 PYTHONPATH=. python test/backend/test_asm_gemm.py TestMXFP4.test_empty
python -m tinygrad.viz.cli -s 'mxfp4_gemm_16384_4096_4096 SQTT SE:0 PKTS' --json > /tmp/baseline.jsonl

K=4096 VIZ=-2 DEV=AMD DEBUG=2 PYTHONPATH=. python test/backend/test_asm_gemm.py TestMXFP4.test_empty
python -m tinygrad.viz.cli -s 'mxfp4_gemm_16384_4096_4096 SQTT SE:0 PKTS' --json > /tmp/shortk.jsonl
```

The current experiment's data and analysis script are packaged in
`/tmp/mxfp4_shortk_results.zip`; raw VIZ profiles are preserved separately at
`/tmp/mxfp4_{baseline,prefetch}_{profile,rewrites}.pkl`.

An early `s_endpgm` temporarily inserted at the start of the **new** kernel made
`TestMXFP4.test_correctness2` fail with zero output and `MXFP4 GEMM forward mismatch`.
The mutation was removed. All measurements in the table used the full elementwise
correctness comparison, including repeated output initialization. Long-K fallback
and the restored new kernel were checked separately. Ruff and mypy checks passed.

Next architectural step: overlap actual next-tile matrix work with the old output
stores. That requires a different accumulator/output handoff or a different operand
register schedule; merely extending the persistent loop did not produce it here.
