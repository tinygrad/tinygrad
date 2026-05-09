# Agentic GEMM/Flash-Attention Kernel Optimization

This document defines the narrower first-cut interface for an agentic optimizer
that improves selected tinygrad kernels by replacing kernel source code.

The initial scope is intentionally small:

- Target AMD first.
- Optimize only selected GEMM and fused flash-attention kernels.
- Match kernels by custom-kernel program name or kernel class.
- Preserve tinygrad's surrounding graph, buffer plan, argument ABI, and visible
  read/write behavior.
- Let candidates use AMD-specific source features such as HIP C++, AMD builtins,
  inline assembly, and target-specific launch geometry.

This version does not pass the full UOp spec or raw UOp graph as primary agent
inputs. The agent receives kernel source plus structured semantic metadata.

## Goal

The optimizer should take a known expensive GEMM or fused flash-attention kernel,
generate a replacement source implementation for the same callable kernel,
compile and run it, verify correctness, profile it, and iterate against the
best known candidate.

The first version does not support adding new global/workspace buffers. Candidate
kernels may change implementation strategy and launch geometry, but must preserve
the same externally visible behavior.

## Kernel Selection

The optimizer should not target every tinygrad kernel. It should target only
known slow kernel classes that dominate the step time in `flat_llama.py`:

- GEMM.
- FP8 GEMM.
- GEMM backward.
- Flash-attention forward.
- Flash-attention backward variants.

In tinygrad, `Tensor.custom_kernel` does not normally become an
`Ops.CUSTOM_FUNCTION`. It becomes a `CALL` around either an `Ops.SINK` before
compilation or an `Ops.PROGRAM` after compilation. `Ops.CUSTOM_FUNCTION` is used
for runtime pseudo-functions such as graph execution, validation, and enc/dec.

Practical matching should use:

- `CALL(SINK[KernelInfo.name])` before compilation.
- `CALL(PROGRAM[ProgramInfo.name])` after compilation.
- `ProgramInfo.name` / `KernelInfo.name` inside codegen hooks.

Initial whitelist examples:

- `gemm_*`
- `hk_fp8_gemm_*`
- `uop_gemm_*`
- `fa_custom_forward`
- `fa_custom_backward_q`
- `fa_custom_backward_kv`
- `custom_fa_forward`
- `custom_fa_backward_pre`
- `custom_fa_backward`
- `custom_fa_backward_post`

The whitelist should exclude unrelated custom kernels such as quantization,
RMSNorm/quantize fusion, FP8 transpose, fused CE, and gradient accumulation until
they are deliberately added as new kernel classes.

## Baseline Kernel Path

For the optimizer-friendly path, `flat_llama.py` should be able to route GEMM and
flash attention through UOp/custom-kernel implementations instead of directly
using handwritten ASM/HIPCC kernels.

The purpose of this baseline path is not to limit candidate kernels to tinygrad's
normal renderer. It is to provide:

- A stable ABI.
- A known kernel class.
- A rendered reference source.
- Dense contiguous buffer contracts.
- A correctness reference for candidate evaluation.

Candidate kernels may then replace the rendered source with port-specific source
for the selected compiler and architecture.

## Kernel Source Constraints

Candidate source must preserve:

- Kernel name.
- Kernel signature.
- Argument order, count, and meaning, including pointer and scalar arguments.
- Externally visible read/write semantics.
- The meaning of all scalar `vals` passed by tinygrad.

Candidate source may change:

- Loop structure.
- Tiling strategy.
- Register usage.
- Shared/LDS/threadgroup memory usage.
- Vectorization and instruction strategy.
- MFMA/WMMA/tensor-core usage.
- AMD builtins and inline assembly.
- Launch dimensions, if returned explicitly with the candidate.

Candidate source may not require:

- New global/workspace buffers.
- Persistent state across launches.
- Reordered, added, or removed kernel arguments.
- Different buffer layouts than the ABI describes.

## Buffer Layout Contract

For `Tensor.custom_kernel`, tinygrad forces inputs through `.contiguous()` before
creating placeholders. Normal custom-kernel outputs are also empty/invalid dense
buffers in many of the intended GEMM/FA paths. However, the optimizer should not
rely on an undocumented blanket layout assumption.

The MCP should report layout fields explicitly for every tensor argument:

- dtype.
- shape.
- strides.
- contiguous flag.
- layout order, if known.
- offset, if known.
- read/write role.
- semantic meaning, for example `A`, `B`, `C`, `Q`, `K`, `V`, `O`, `l_vec`.

If the evaluator can prove a tensor is dense, zero-offset, row-major contiguous,
it should say so explicitly. Otherwise the layout should be reported from the
actual tinygrad buffer/view metadata or marked unknown rather than inferred.

For GEMM, buffer layout alone is not enough. The MCP should also report semantic
math meaning:

- `A[M, K]`.
- `B[K, N]` or `B[N, K]` if the ABI stores a transposed operand.
- `C[M, N]`.
- batch dimensions, if present.
- scale tensors for FP8 GEMM, if present.
- output dtype and accumulation dtype.

For flash attention, the MCP should report:

- forward/backward kernel class.
- `Q`, `K`, `V`, `O` shapes.
- `B`, `N`, `H`, `H_KV`, `D`.
- causal or non-causal.
- mask/dropout support, if any.
- GQA/group size.
- scale convention.
- auxiliary outputs/inputs such as `l_vec` and `delta_vec`.
- precision expectations for softmax and accumulation.

## Agent Inputs

Each optimization task should include:

- Kernel class: `gemm`, `fp8_gemm`, `gemm_backward`, `fa_forward`,
  `fa_backward_q`, `fa_backward_kv`, `fa_backward_pre`, or `fa_backward_post`.
- Port information.
- Kernel ABI.
- Buffer metadata.
- Baseline rendered source.
- Baseline launch dimensions.
- Baseline correctness and profile metrics.
- Source constraints.
- Optimization guidance.
- Historical candidates and their metrics.

### Port Information

Required fields:

- Hardware family: string, for example `AMD`.
- Hardware architecture: string, for example `gfx942` or `gfx950`.
- Source language/dialect: string, for example `HIP C++ with AMDGCN builtins and
  inline assembly allowed`.
- Compiler: string, for example `hipcc`, `COMGR`, or a clang HIP path.
- Compiler version: string.

The source dialect and compiler should both be provided. They are not always
redundant, and candidate validity depends on what the actual compiler accepts.

### Baseline Source

The baseline source is the current rendered source for the selected kernel.

It is a compatibility and semantics reference, not a structure candidates must
preserve. Candidate kernels are allowed to replace it with direct target-specific
source as long as the ABI and visible behavior remain unchanged.

### Optimization Guidance

The agent may receive brief domain guidance such as:

- Prefer MFMA/tensor-core paths for GEMM and flash-attention matmul blocks.
- Use LDS/register tiling where it improves reuse.
- Consider vectorized global loads/stores.
- Tune workgroup size, waves per block, and tile sizes against resource metrics.
- Use inline assembly or AMD builtins when useful and supported by the compiler.
- Watch VGPR/SGPR/AGPR pressure, LDS use, scratch spills, and occupancy.

Guidance is not a required checklist. The agent should respond to measured
compile/runtime feedback.

## Candidate Response

A candidate response should include:

- Replacement source code.
- Launch dimensions, if changed or if required by the interface.
- Short explanation of the intended optimization.
- Any assumptions about dtype, shape, or architecture.

The evaluator may reject candidates that do not compile, violate ABI constraints,
fail correctness, exceed resource limits, or regress runtime beyond the accepted
exploration policy.

## Candidate Evaluation

The evaluator should:

- Compile candidate source using the declared compiler/toolchain.
- Execute with representative inputs.
- Verify correctness against the baseline or another trusted reference.
- Profile runtime over warmup and repeated measurements.
- Collect compiler/resource metrics.
- Derive normalized performance metrics.
- Store candidate history.

Correctness should use the same buffer/argument contract as the baseline kernel.
For numerical kernels, tolerances should be explicit per kernel class and dtype.

## Profiling And Metric Data

The loop needs both stable device descriptors and per-candidate metrics. Some
fields are available directly from tinygrad, some from AMD tooling, and some must
be derived by the MCP.

### Stable Device Inputs

| Item | tinygrad availability / trust | AMD availability / trust | MCP handling |
| --- | --- | --- | --- |
| `cu_cnt` | Available, derived from `iface.props`. Trustworthy for tinygrad's runtime model, but normalize whether it means per-XCC or whole device. | Available from ROCm/KFD/topology/device properties. Trust high. | Provide normalized value and scope. |
| `se_cnt` | Available, derived from `iface.props`. Trust high, with the same per-XCC normalization caveat. | Available from device topology/properties. Trust high. | Provide normalized value and scope. |
| `xccs` | Available from `iface.props.get("num_xcc", 1)`. Trust high. | Available from ROCm/device topology. Trust high. | Provide directly. |
| `waves_per_cu` | Available, derived from `max_waves_per_simd * simd_per_cu`. Trust reasonable. | Available from architecture/device docs/properties. Trust high. | Provide directly or derived with provenance. |
| `wave_cnt` | Available, but tinygrad-runtime-specific and derived with gfx9 special caps. Trust medium unless normalized. | Derivable from CU count, wave slots, and XCC layout. Not usually a single profiler field. Trust high if derived carefully. | Derive and label scope. |
| `lds_size_per_cu` | Partially available. tinygrad uses `iface.props["lds_size_in_kb"]` for gfx950 and `0x10000` otherwise. Trust medium as a clean exported metric. | Available from architecture/device properties/docs. Trust high. | Prefer AMD/device property value when available. |
| VGPR/SGPR capacity and allocation granularity | Partially hardcoded/implicit in tinygrad, not exposed cleanly. Trust low as-is for MCP use. | Available from AMD architecture docs or profiler occupancy model. Trust high, but usually needs normalization. | Provide normalized MCP fields. |
| peak FLOPs | Not available. | From SKU specs or calibration, not compiler. Trust medium-high if SKU, clock, dtype, and MFMA path are explicit. | Provide externally or calibrate. |
| peak memory bandwidth | Not available. | From SKU specs or calibration. Trust medium; practical bandwidth depends on access pattern. | Optional external/calibrated input. |

### Per-Candidate Metrics

| Item | tinygrad availability / trust | AMD availability / trust | MCP handling |
| --- | --- | --- | --- |
| runtime | Available through tinygrad GPU timing when `wait=True`. Trust high if aggregated over warmup/repeats. | `rocprofv3` kernel trace gives start/end timestamps. Trust high. | Report min/median/p95 or equivalent. |
| correctness | tinygrad can execute, but MCP must compare against baseline/reference. Not a profiling metric. | AMD tools do not provide this. | Implement in evaluator. |
| VGPR/SGPR/AGPR | Not normalized today. tinygrad keeps raw resource registers/code object data, but no clean metric. Trust low as current API, high if proper extraction is added. | Directly available via AMD tooling such as `rocprofv3` kernel trace fields `VGPR_Count`, `SGPR_Count`, and `Accum_VGPR_Count`. Trust high. | Collect from AMD tooling or code-object metadata. |
| LDS bytes | Directly available as `AMDProgram.group_segment_size` from the kernel descriptor. Trust high. | Directly available as `LDS_Block_Size`. Trust high. | Report directly. |
| scratch bytes | Directly available as `AMDProgram.private_segment_size`. Trust high. | Directly available as `Scratch_Size`. Trust high. | Report directly. |
| wave mode | Directly available as `AMDProgram.wave32`. Trust high. | Available from code object/profiler metadata. Trust high. | Report wave32/wave64. |
| occupancy estimate | Not exposed. | ROCm Compute Profiler can report occupancy views, but theoretical occupancy should be derived from resources plus device descriptor. | Derive. |
| occupancy limiter | Not available. | ROCm Compute Profiler can report limiter-style metrics, but a simple limiter should still be derived from VGPR/SGPR/LDS/scratch/workgroup/waveslot constraints. | Derive. |
| MFU | Not directly available. | Not directly a compiler/profiler primitive. | Derive from FLOPs, runtime, and peak FLOPs. |
| speedup vs baseline/best | Not directly available. | Not an AMD metric. | Derive from candidate history. |

Short version:

- Trust tinygrad for runtime, LDS bytes, scratch bytes, wave mode, and basic
  device topology.
- Trust AMD tooling for VGPR, SGPR, AGPR, LDS, scratch, timestamps,
  grid/workgroup trace fields, and hardware counters.
- Derive occupancy, occupancy limiter, MFU, speedup, and bandwidth utilization.
- Provide externally or calibrate peak FLOPs and peak memory bandwidth.

Even if launch geometry is an agent decision, record the actual `global_size`
and `local_size` used for each candidate. Occupancy and saturation estimates are
not meaningful without the final launch geometry.

## Historical Candidates

Historical records should include:

- Kernel class and semantic shape.
- Source code.
- Launch dimensions.
- Compiler/toolchain identity.
- Compile result and diagnostics.
- Correctness result.
- Runtime metrics.
- Resource metrics.
- Derived occupancy/MFU/speedup metrics.
- Failure reason, if rejected.

History should be available for the exact same kernel and, when useful, for
nearby kernels with similar shape, dtype, and architecture.

## Non-Goals For First Cut

- Optimizing arbitrary low-MFU tinygrad kernels.
- Inferring flash attention from a naive SDPA graph.
- Passing the full UOp graph as the primary semantic representation.
- Passing `spec/tinyspec.tex` as required context.
- Adding new global/workspace buffers.
- Changing kernel argument ABI.
- Cross-port generality beyond AMD.

The full UOp graph may still be useful as optional debug context, but it is not
part of the primary first-cut agent input.

## Open Questions

- What exact environment variable names should enable the optimizer-friendly
  GEMM and flash-attention baseline paths in `flat_llama.py`?
- Should candidate launch dimensions be required in every response, or only when
  changed from the baseline?
- Should history retrieval be exact-match only at first, or should it include
  nearest-neighbor kernels by shape/dtype/kernel class?
- Which AMD profiling path should be the default for VGPR/SGPR/AGPR collection:
  `rocprofv3`, code-object metadata extraction, or both?
- How many correctness inputs and shape bindings are required before accepting a
  candidate as best?
- Should disassembly be collected by default, or only after performance/resource
  metrics suggest an instruction-selection problem?
