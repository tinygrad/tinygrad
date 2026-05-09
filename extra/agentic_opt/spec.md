# Agentic Kernel Optimization MCP

This document sketches the first-cut MCP interface for optimizing tinygrad
kernels by replacing generated kernel source while preserving the surrounding
tinygrad graph and runtime contract.

## Goal

The optimizer receives enough context to understand a single kernel, generate
candidate source for the same callable kernel, compile and validate candidates,
profile them, and compare against prior attempts.

The first version intentionally does not support adding new global/workspace
buffers. Candidate kernels may change their implementation strategy and launch
geometry, but must preserve the same externally visible behavior.

## Kernel Source Constraints

Candidate source must preserve:

- Kernel name.
- Kernel signature.
- Argument order, count, and meaning, including pointer and scalar arguments.
- Externally visible read/write semantics.

Candidate source may change:

- Loop structure.
- Tiling strategy.
- Register/shared/LDS/threadgroup memory usage.
- Vectorization and instruction strategy.
- Launch dimensions, if returned with the candidate source.

Candidate source may not require:

- New global/workspace buffers.
- Persistent state across launches.
- Reordered, added, or removed kernel arguments.

## MCP Inputs

### Port Information

Identifies the hardware port being optimized for.

Required first-cut fields:

- Hardware family: string, for example `AMD`.
- Hardware architecture: string, for example `gfx942`.
- Source language/dialect: string, for example `HIP C++`, `CUDA C++`, `PTX`,
  `AMDGCN assembly`, `LLVM IR`, `OpenCL C`, or `Metal Shading Language`.
- Compiler: string, for example `hipcc`, `COMGR`, `nvrtc`, `nvcc`, or a
  tinygrad assembler path.
- Compiler version: string.

### AST Semantic Representation

The pre-lowered UOp graph for the kernel.

This is the semantic source of truth. It should describe what the kernel
computes before codegen rewrites, scheduling optimizations, and renderer-specific
lowering obscure the original operation.

### UOp Spec

The UOp specification document, currently `spec/tinyspec.tex`.

This is background context for interpreting the AST. The optimizer should treat
the live kernel AST and runtime ABI as authoritative if the spec and current
code disagree.

### Naive Render

The unoptimized baseline kernel source generated from the kernel.

This is primarily used to pin the UOp graph to a concrete kernel representation
and target source dialect. It is a correctness and compatibility reference, not
a loop structure that candidates must preserve.

### Candidate Evaluation

An endpoint that accepts candidate source and launch dimensions, then returns:

- Compile result.
- Execute result.
- Correctness verification result.
- Profile statistics.
- Compiler diagnostics and outputs.
- Runtime or validation failure details.

The verification path should compare the candidate against the baseline kernel
or another trusted reference under the same input and scalar argument contract.

### Historical Kernels

Prior candidate kernels and their statistics for this same kernel, or for
similar kernels on the same port.

Historical records should let the optimizer determine whether the latest pass
improved over previous attempts. They should include source, launch dimensions,
compile status, verification status, profile stats, and relevant compiler
diagnostics when available.

## Open Questions

- What is the minimum structured ABI data to expose beyond the kernel signature
  and source constraints?
- Should launch dimensions be part of every candidate response, or only when
  they differ from the baseline?
- Should historical kernels be limited to exact AST/cache-key matches, or should
  the MCP expose nearest-neighbor examples across similar kernels?
- Which compiler outputs should be normalized in the first version: register
  count, shared/LDS usage, spills, occupancy, disassembly, or raw logs only?
- Should BEAM/heuristic tinygrad results be included as historical baselines, or
  kept separate to avoid anchoring candidate generation?
- How many correctness inputs and shape/var bindings should each candidate be
  tested against before being accepted?
