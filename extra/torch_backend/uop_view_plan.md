# UOp-Oriented Torch Backend Plan

## Context
- Torch backend currently mirrors PyTorch view semantics by recording `_view_ops` and replaying them through the `View`/`ShapeTracker` classes in `view_tracking.py`.
- `wrap_tensor.cpp` extracts shape/strides/offset from `tensor._view_st`, so every view mutation must keep the Python-side tracker in sync.
- The recorded ops are reconverted into movement ops for `as_strided`, then replayed back into actual `Tensor` movement ops when a base buffer gets realized or when PyTorch queries stride/contiguity.
- This duplicates tinygrad's native movement semantics (already encoded in `Tensor._mop` and `UOp`), adds a lot of custom stride math, and is brittle when new view ops are added.

## What PyTorch Needs From Us
- **Accurate striding metadata** for every tensor we hand back to PyTorch: final shape, per-dimension stride, and a single storage offset relative to the base buffer.
- **Alias tracking** so that in-place mutations on a view write into the base buffer seen by all materialized aliases.
- **`as_strided` support** that can produce arbitrary stride/offset combinations while still sharing storage with the canonical base.
- **Fallible validation** for cases reshape/expand cannot be represented (must raise instead of silently copying).
- **Queries on demand**: PyTorch will call things like `.stride()`, `.is_contiguous()`, etc., after `wrap`, and expects consistent answers with the data it later reads or mutates.

## Design Goals
1. Remove the bespoke `View`/`ShapeTracker` stack in favour of information we can derive directly from the `Tensor.uop` tree.
2. Keep all transformations expressed as native movement `Ops` (`RESHAPE`, `SHRINK`, `EXPAND`, `PERMUTE`, `FLIP`) so we do not drift from tinygrad semantics.
3. Compute (shape, strides, offset) lazily from the current `UOp` chain; cache aggressively so repeated PyTorch queries stay cheap.
4. Rebuild Python views for in-place operations using the same derived movement chain instead of replaying a recorded log.
5. Implement `_as_strided` and `_reshape_alias` by constructing UOp-based views directly, without detouring through the current `to_movement_ops` back-conversion layer.

## Proposed Architecture

### 1. `UOp` Movement Inspection
- Implement a helper `collect_movement_chain(uop: UOp) -> tuple[list[tuple[Ops, Any]], UOp]` that walks from `uop` down to its base buffer while `uop.op in GroupOp.Movement` (also transparently skipping `DETACH`, `BITCAST`, `CONTIGUOUS`, `CONTIGUOUS_BACKWARD`, `MULTI` wrappers).
- The function should return the ordered list of movement ops (closest to base first) and the base `UOp` whose shape/stride acts as the storage origin.
- Rely on existing `uop.marg` property to extract reshapes/expands/shrinks arguments with symbolic support.

### 2. `ViewSpec`: derive stride metadata from movement ops
- Define a lightweight Python dataclass (not callable from movement ops) that stores `shape`, `strides`, `offset`, plus simple helpers like `is_contiguous`.
- Construction algorithm:
  1. Start from the base `UOp`: `shape = base.shape`, `strides = strides_for_shape(shape)`, `offset = 0`.
  2. Fold each movement op from the collected chain:
     - **RESHAPE**: use `tinygrad.helpers.get_contraction` (or reuse the logic in `merge_dims`) to map old axes to new axes. Recompute strides by distributing products inside each contraction block; validate that the reshape is compatible with current strides (should be if the `UOp` exists, but keep guards for symbolic mismatch).
     - **EXPAND**: broadcast dimensions of size 1. New axes inserted to the left simply get stride 0; existing axes keep stride unless the expansion breaks broadcast rules (then raise).
     - **SHRINK**: adjust `offset += start * stride` per axis and shrink the shape to `(end-start)`. Fully shrunk axes (start==end) collapse to size 0 without special casing.
     - **PERMUTE**: reorder `shape` and `strides` with the provided axis order.
     - **FLIP**: negate the relevant stride(s) and add `(size-1)*original_stride` to the offset for each flipped axis.
     - **PAD**: mark as unsupported for aliasing (PyTorch view ops never call pad), so encountering PAD should trigger a fallback copy.
  3. Canonicalize zero-stride broadcast dims with `canonicalize_strides` so 1-dim expands present as stride 0, matching PyTorch's query semantics.
- Cache the resulting `ViewSpec` on the tensor (e.g., `_cached_view_spec` + `_spec_version`) and invalidate it whenever the tensor's `uop` mutates.

### 3. View Rebuild for In-Place Ops
- Replace `_view_ops` with an on-demand `rebuild_view(base: Tensor, view: Tensor) -> Tensor` helper:
  1. Use `collect_movement_chain(view.uop)` to recover the ordered operations relative to `base`.
  2. Apply them to `base` using normal tensor methods (`reshape`, `shrink`, `expand`, `permute`, `flip`).
  3. Ensure we share storage by calling `Tensor._mop`/movement mixin rather than materializing copies.
- Store only weak references in `_views` as today, but drop the tuple of recorded ops.
- `maybe_realize_storage` now calls `rebuild_view` when it needs to hand back a live view, eliminating divergence between recorded metadata and the actual `Tensor` graph.

### 4. UOp-native `as_strided`
- Introduce `Tensor.strided_view(size, stride, offset)` that constructs a new `Tensor` sharing the base buffer via a dedicated UOp (either extend `Ops.BUFFER_VIEW` to accept a stride vector, or add a new `Ops.AS_STRIDED`).
  - The UOp should simply carry `(size, stride, offset)` metadata; scheduling checks can treat it as a pure movement op with arbitrary strides.
  - Implement backward compat by lowering arbitrary strides into a small `Ops.INDEX` expression when kernels are generated (mirrors how current `to_movement_ops` forces explicit operations, but now centralized in the UOp lowering step).
- `_as_strided_impl` becomes: find canonical base, call `base.strided_view(...)`, attach `_view_base`, register weak ref, and reuse the resulting tensor directly (no more `_strided_view_ops`).

### 5. Pybind integration
- Adjust `wrap_tensor.cpp` to query a Python attribute (e.g., `_view_spec` method) that returns `(sizes, strides, storage_offset)` computed via the cached `ViewSpec`. This keeps C++ minimal and makes the transition transparent to PyTorch.
- Provide a fast path so repeated `.stride()` calls reuse the cached tuple instead of recomputing the whole chain.

### 6. Contiguity / metadata helpers
- `Tensor.is_contiguous()` can simply check `ViewSpec.strides == strides_for_shape(shape)` and storage offset 0.
- For debug logging, optionally expose a helper that prints the movement chain derived from UOps; this replaces `to_movement_ops` entirely.

## Migration Steps
1. **Scaffolding**: add `collect_movement_chain` and `ViewSpec` with unit tests using plain tinygrad tensors (no torch dependency yet).
2. **Remove `_view_ops` writes**: swap recorders in `view_ops` to rely on `_view_base` only; ensure existing tests still pass using the legacy rebuild path temporarily (e.g., call `rebuild_view` inside recorders to validate equivalence).
3. **Swap stride source**: change `wrap` to use `ViewSpec` instead of `_view_st`; update C++ binding accordingly.
4. **Delete `view_tracking.py`** by replacing all imports with the new helpers.
5. **Implement `Tensor.strided_view`** and rewire `_as_strided_impl` / `_reshape_alias`.
6. **Clean up**: remove now-dead helpers (`MovementOps`, `_movement_to_view_op`, etc.).
7. **Run full suite**: `extra/torch_backend/test.py`, `test_inplace.py`, `test_multigpu.py`, `run_all_tests.sh` smoke subset, plus core `tinygrad` movement tests.

## Validation Checklist
- Compare `ViewSpec` output against the current ShapeTracker for a corpus of existing movement chains (write a temporary assertion while both paths exist).
- Explicit tests for tricky cases: nested shrinks and permutes, zero-stride expands, symbolic shapes (use `Tensor.arange(0, sym)`), negative strides via `flip`, and chained reshapes that merge/split dims.
- Ensure in-place view mutations still recycle the base buffer correctly (existing tests in `test_inplace.py` plus `aten::copy_` scenarios).
- Verify PyTorch’s `test_unary`/`test_binary` suites still pass to confirm alias metadata stays coherent.

## Open Points / Follow-ups
- Decide whether `Ops.BUFFER_VIEW` should grow stride metadata or whether a brand-new movement op is cleaner; whichever choice we make, update the scheduler/type verifier accordingly.
- Audit other torch bridge features (e.g., `_local_scalar_dense`, `nonzero`) to ensure they don’t rely on `_view_ops` side effects.
- Once stable, document the UOp-based view derivation inside `docs/developer` so future backends can reuse it.
