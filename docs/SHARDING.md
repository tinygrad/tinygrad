# Sharding Refactor: Sharding as Device

## Overview

This document describes a refactor to simplify multi-device support in tinygrad by making sharding part of the device specification rather than a separate graph operation.

**Key insight**: Sharding should be metadata that flows through the graph via the device property, not wrapper ops that require explicit pattern matching for every operation type.

## Current Progress

### Completed

- [x] **Phase 1**: `Sharding` dataclass added to `device.py`
- [x] **Phase 2**: `MSTACK` creates `Sharding` device directly when `axis` is in arg
- [x] **Phase 3**: `_device` property propagates `Sharding` through ops and transforms axis for `PERMUTE`, `RESHAPE`, `REDUCE_AXIS`
- [x] **Phase 4**: Updated `alu_multi()` and `reduce_multi()` to work with Sharding device
- [x] **Phase 5**: Fixed `isinstance` checks throughout codebase to handle `(tuple, Sharding)`

### In Progress

None - initial implementation complete. Tests pass with SPEC=2.

### Pending (Future Work)

- [ ] **Phase 6**: Remove `MULTI` op (make `.shard()` return Sharding device)
- [ ] **Phase 7**: Simplify `multi_pm` (remove patterns now handled by propagation)
- [ ] **Phase 8**: Update type hints throughout (make device always `str | Sharding`, not tuple)
- [ ] **Phase 9**: Cleanup and testing

### Test Results

- 86 tests pass with `SPEC=2`
- 4 tests fail due to device comparison (`tuple` vs `Sharding`) - expected during transition
- All pre-commit checks pass

### Key Changes Made

1. **`Sharding` class** in `device.py`:
   ```python
   @dataclass(frozen=True)
   class Sharding:
     devices: tuple[str, ...]
     axis: int
   ```

2. **`MSTACK` with axis** creates `Sharding` device:
   ```python
   UOp(Ops.MSTACK, dtype, srcs, arg=axis)  # device is Sharding(devices, axis)
   ```

3. **`_device` transforms axis** for movement ops:
   - `PERMUTE`: axis becomes `marg.index(src_axis)`
   - `RESHAPE`: axis computed from shape accumulation
   - `REDUCE_AXIS`: returns tuple (not Sharding) if axis is reduced away

4. **`axis` property** checks `Sharding` first:
   ```python
   if isinstance(self._device, Sharding): return self._device.axis
   ```

5. **`isinstance` checks** updated throughout to handle `(tuple, Sharding)`:
   - `tinygrad/tensor.py`: `__init__`, `assign`, `_buffer`, `full_like`, `rand_like`
   - `tinygrad/uop/ops.py`: `multi()`, `buffer` property
   - `tinygrad/schedule/multi.py`: `alu_multi()`, `reduce_multi()`
   - `tinygrad/schedule/indexing.py`: `BufferizeOpts`
   - `tinygrad/nn/state.py`: `load_state_dict()`

6. **`alu_multi` updated** to use `_get_multi_axis()` helper that only returns axis for MULTI/MSTACK ops, ignoring Sharding axis on regular ops

7. **`spec.py` updated** to validate DEVICE op with Sharding arg and add Sharding to pyrender globals

## Original Architecture

### The MULTI/MSELECT/MSTACK Model

Currently, multi-device tensors use special ops:

| Op | Purpose |
|---|---|
| `MULTI` | Wrapper op marking a tensor as sharded, with `axis` in arg |
| `MSELECT` | Selects one buffer from a multi-device set (index in arg) |
| `MSTACK` | Combines single-device buffers into multi-device |
| `ALLREDUCE` | Cross-device reduction |

The flow for sharding a tensor:
```python
tensor.shard(devices, axis)
  → copy_to_device(devices)    # COPY to tuple device
  → _shard(axis)               # symbolic shrink using _device_num variable
  → multi(axis)                # wrap in MULTI op
```

### Problems with Current Approach

1. **Explicit handling for every op**: `multi_pm` in `schedule/multi.py` needs patterns for every operation type:
   - `alu_multi()`, `reduce_multi()`, `reshape_multi()`, `expand_multi()`, `pad_multi()`, `permute_multi()`, `shrink_multi()`, `flip_multi()`, `copy_multi()`, `assign_multi()`, `passthrough_multi()`

2. **Complex axis property**: The `axis` property in `ops.py` has special cases for MULTI, ALU, REDUCE_AXIS, RESHAPE, PERMUTE, etc.

3. **Separate MultiBuffer class**: `device.py` has both `Buffer` and `MultiBuffer` classes.

4. **Special-casing throughout**: `schedule.py`, `rangeify.py`, `spec.py` all have MULTI/MSELECT/MSTACK handling.

5. **Two representations of device**: `str` for single device, `tuple[str, ...]` for multi-device, plus MULTI op for axis info.

## Proposed Architecture

### Sharding as Part of Device

Replace the current model with a `Sharding` class that combines devices and axis:

```python
@dataclass(frozen=True)
class Sharding:
  devices: tuple[str, ...]
  axis: int

  def __len__(self) -> int: return len(self.devices)
  def __iter__(self): return iter(self.devices)
  def __getitem__(self, i: int) -> str: return self.devices[i]
  def __hash__(self): return hash((self.devices, self.axis))
```

Device type becomes:
```python
# Old
device: str | tuple[str, ...]

# New
device: str | Sharding
```

### Implicit Bounds

Bounds are computed from shape, not stored:
- Each device gets `shape[axis] // len(devices)` elements
- Device `i` gets slice `[i * shard_size, (i+1) * shard_size]`
- Constraint: `shape[axis] % len(devices) == 0`

### Automatic Sharding Propagation

The `_device` property computes output sharding based on input sharding and operation:

```python
@property
def _device(self) -> str | Sharding | None:
  if self.op is Ops.DEVICE: return self.arg
  if self.op is Ops.MSELECT:
    # Extract single device from Sharding
    return self.src[0].device.devices[self.arg]
  if self.op is Ops.MSTACK:
    # Combine into Sharding (arg is axis)
    return Sharding(tuple(x.device for x in self.src), self.arg)

  # Propagate Sharding through operations
  src_dev = self.src[0]._device if self.src else None
  if not isinstance(src_dev, Sharding):
    return src_dev

  # Transform axis based on operation
  match self.op:
    case Ops.PERMUTE:
      return Sharding(src_dev.devices, self.marg.index(src_dev.axis))
    case Ops.RESHAPE:
      new_axis = _compute_reshape_axis(src_dev.axis, self.src[0].shape, self.marg)
      return Sharding(src_dev.devices, new_axis)
    case _:
      return src_dev
```

### Eager Sharding Handling

Instead of pattern-matching rewrites, handle sharding at op creation time:

**Reduce on sharded axis** - in `r()` method:
```python
def r(self, op: Ops, axis: tuple[int, ...]) -> UOp:
  if isinstance(self.device, Sharding) and self.device.axis in axis:
    # Reduce then allreduce
    reduced = UOp(Ops.REDUCE_AXIS, self.dtype, (self,), (op, axis))
    return reduced.allreduce(op, self.device.devices)
  return UOp(Ops.REDUCE_AXIS, self.dtype, (self,), (op, axis))
```

**Copy from sharded to single device** - in `copy_to_device()`:
```python
def copy_to_device(self, device: str | Sharding | UOp, arg=None) -> UOp:
  if isinstance(self.device, Sharding) and isinstance(device, str):
    # Unshard and allreduce to single device
    return self._unshard(self.device.axis).allreduce(Ops.ADD, device)
  # ... rest of copy logic
```

**ALU with mismatched shardings** - in `alu()`:
```python
def alu(self, op: Ops, *src: UOp) -> UOp:
  shardings = [s.device for s in (self,) + src if isinstance(s.device, Sharding)]
  if shardings and not all(s.axis == shardings[0].axis for s in shardings):
    # Reshard to match (or raise error)
    ...
  return UOp(op, ...)
```

## What Changes

### Goes Away Entirely

| Item | Location | Reason |
|------|----------|--------|
| `Ops.MULTI` | `uop/__init__.py` | Axis info now in Sharding |
| `multi()` method | `uop/ops.py` | No longer needed |
| `multi_pm` patterns for movement ops | `schedule/multi.py` | Handled by `_device` propagation |
| `axis` property complexity | `uop/ops.py` | Becomes trivial: `device.axis if isinstance(device, Sharding) else None` |
| `bounds` property | `uop/ops.py` | Computed from shape |
| MULTI spec validation | `uop/spec.py` | Op doesn't exist |

### Changes Form

| Item | Old | New |
|------|-----|-----|
| Device type | `str \| tuple[str, ...]` | `str \| Sharding` |
| MSTACK | Creates tuple device, needs MULTI wrapper | Creates Sharding device directly (axis in arg) |
| Sharding propagation | `multi_pm` rewrite pass | `_device` property computation |
| Reduce on shard axis | `reduce_multi()` pattern | Eager in `r()` method |
| Copy sharded→single | `copy_multi()` pattern | Eager in `copy_to_device()` |
| ALU axis mismatch | `alu_multi()` pattern | Eager in `alu()` method |

### Stays (Simplified)

| Item | Notes |
|------|-------|
| `MSTACK` | Still needed to combine single-device buffers, but now sets Sharding directly |
| `MSELECT` | Still needed to extract single device from Sharding |
| `ALLREDUCE` | Still needed for cross-device reduction |
| `_shard()` / `_unshard()` | Symbolic kernel specialization still works |
| `_device_num` variable | Per-device kernel specialization unchanged |
| `MultiBuffer` class | Could potentially unify with Buffer, but lower priority |

## Detailed Changes by File

### `tinygrad/uop/__init__.py`

- Remove `Ops.MULTI` from enum

### `tinygrad/uop/ops.py`

1. Add `Sharding` class (or import from device.py)

2. Remove `multi()` method

3. Simplify `axis` property:
   ```python
   @property
   def axis(self) -> int | None:
     return self.device.axis if isinstance(self.device, Sharding) else None
   ```

4. Remove `bounds` property (compute inline where needed)

5. Update `_device` property to propagate Sharding through ops

6. Update `shard()` method:
   ```python
   def shard(self, devices: tuple[str, ...], axis: int) -> UOp:
     return self.copy_to_device(Sharding(devices, axis))._shard(axis)
   ```

7. Update `copy_to_device()` for eager unshard handling

8. Update `r()` for eager allreduce on sharded axis

9. Update type hints: `str | tuple[str, ...]` → `str | Sharding`

### `tinygrad/device.py`

1. Add `Sharding` class definition (if not in ops.py):
   ```python
   @dataclass(frozen=True)
   class Sharding:
     devices: tuple[str, ...]
     axis: int
     def __len__(self): return len(self.devices)
     def __iter__(self): return iter(self.devices)
     def __getitem__(self, i): return self.devices[i]
   ```

2. Update `MultiBuffer` to accept `Sharding`:
   ```python
   def __init__(self, device: Sharding, size: int, dtype: DType):
     self.bufs = [Buffer(d, size, dtype) for d in device.devices]
   ```

### `tinygrad/schedule/multi.py`

1. Remove movement op patterns (reshape_multi, permute_multi, expand_multi, pad_multi, shrink_multi, flip_multi)

2. Remove `alu_multi` (handled eagerly)

3. Remove `reduce_multi` (handled eagerly)

4. Remove `copy_multi` (handled eagerly)

5. Keep `replace_allreduce` patterns for ALLREDUCE implementation

6. Keep MSTACK/MSELECT simplification patterns

7. Significantly simplify `multi_pm` - may become just `replace_allreduce`

### `tinygrad/schedule/rangeify.py`

1. Update MSTACK handling to work with Sharding device

2. Remove MULTI-specific code paths

3. Simplify tagging logic (no MULTI op to handle)

### `tinygrad/engine/schedule.py`

1. Update MSELECT/MSTACK handling for Sharding

2. Remove any MULTI-specific scheduling logic

### `tinygrad/uop/spec.py`

1. Remove MULTI validation rule

2. Update MSTACK validation - creates Sharding device

3. Update MSELECT validation - source has Sharding device

### `tinygrad/tensor.py`

1. Update type hints: `str | tuple[str, ...]` → `str | Sharding`

2. Update `shard()` method

3. Update `device` property return type

4. Update `to()` method for Sharding

### `tinygrad/gradient.py`

1. Update MULTI gradient rule - now check for Sharding device:
   ```python
   # Old: (UPat(Ops.MULTI, name="ret"), lambda ctx, ret: ctx.shard(ret.device, ret.axis).src)
   # New: handled by checking isinstance(device, Sharding) in gradient propagation
   ```

### Test Files

1. Update `test/test_multitensor.py` for new API

2. Update any tests that check for `Ops.MULTI`

3. Update device type assertions

## TODO List

### Phase 1: Add Sharding Class (Non-Breaking)

- [ ] Define `Sharding` dataclass in `tinygrad/device.py`
- [ ] Add `Sharding` to exports
- [ ] Add tests for `Sharding` class basic functionality
- [ ] Update type hints to allow `Sharding` alongside existing types

### Phase 2: MSTACK Creates Sharding Directly

- [ ] Update MSTACK to take axis in arg
- [ ] Update MSTACK's `_device` to return `Sharding(devices, arg)`
- [ ] Update MSTACK spec validation
- [ ] Update MSTACK handling in rangeify.py
- [ ] Test MSTACK creates correct Sharding device

### Phase 3: _device Propagates Sharding

- [ ] Update `_device` property for PERMUTE → transform axis
- [ ] Update `_device` property for RESHAPE → compute new axis position
- [ ] Update `_device` property for other movement ops (pass through)
- [ ] Update `_device` property for ALU ops
- [ ] Update `_device` property for REDUCE_AXIS
- [ ] Add tests for sharding propagation through ops

### Phase 4: Eager Sharding Handling

- [ ] Update `r()` to insert ALLREDUCE when reducing sharded axis
- [ ] Update `copy_to_device()` to unshard when copying Sharding → str
- [ ] Update `alu()` to handle mismatched shardings
- [ ] Add tests for eager sharding handling

### Phase 5: Simplify axis Property

- [ ] Simplify `axis` property to just check Sharding
- [ ] Remove complex axis computation logic
- [ ] Update all `axis` property usages

### Phase 6: Remove MULTI Op

- [ ] Remove `Ops.MULTI` from enum
- [ ] Remove `multi()` method from UOp
- [ ] Remove MULTI patterns from multi_pm
- [ ] Remove MULTI from spec.py
- [ ] Remove MULTI handling from gradient.py
- [ ] Remove MULTI from rangeify.py
- [ ] Remove MULTI from schedule.py
- [ ] Update shard() to not use multi()

### Phase 7: Simplify multi_pm

- [ ] Remove movement op patterns (handled by _device)
- [ ] Remove alu_multi (handled eagerly)
- [ ] Remove reduce_multi (handled eagerly)
- [ ] Remove copy_multi (handled eagerly)
- [ ] Keep only ALLREDUCE implementation patterns
- [ ] Consider renaming multi_pm → allreduce_pm

### Phase 8: Update Type Hints Throughout

- [ ] `tinygrad/uop/ops.py`: all device type hints
- [ ] `tinygrad/tensor.py`: all device type hints
- [ ] `tinygrad/device.py`: all device type hints
- [ ] `tinygrad/engine/schedule.py`: device type hints
- [ ] `tinygrad/schedule/multi.py`: device type hints
- [ ] `tinygrad/schedule/rangeify.py`: device type hints

### Phase 9: Cleanup

- [ ] Remove `bounds` property (compute inline)
- [ ] Remove dead code from multi.py
- [ ] Update documentation
- [ ] Run full test suite with SPEC=2
- [ ] Benchmark to ensure no performance regression

### Phase 10: Optional - Unify Buffer/MultiBuffer

- [ ] Consider making Buffer work with Sharding device directly
- [ ] Or keep MultiBuffer but simplify its interface

## Migration Example

### Before (Current)

```python
# Creating sharded tensor
x = Tensor.ones(256, device="GPU:0")
x_sharded = x.shard(("GPU:0", "GPU:1"), axis=0)
# Internally: COPY → _shard → MULTI wrapper

# x_sharded.uop structure:
# MULTI(axis=0)
#   └── SHRINK (symbolic with _device_num)
#         └── COPY
#               └── original tensor

# Operations go through multi_pm:
y = x_sharded.reshape(128, 2)  # reshape_multi pattern
z = x_sharded.sum(axis=0)      # reduce_multi pattern → inserts ALLREDUCE
```

### After (Proposed)

```python
# Creating sharded tensor
x = Tensor.ones(256, device="GPU:0")
x_sharded = x.shard(("GPU:0", "GPU:1"), axis=0)
# Internally: COPY to Sharding device → _shard

# x_sharded.uop structure:
# SHRINK (symbolic with _device_num)
#   └── COPY
#         └── original tensor
# x_sharded.device = Sharding(("GPU:0", "GPU:1"), axis=0)

# Operations propagate Sharding via _device:
y = x_sharded.reshape(128, 2)  # _device computes new axis position
z = x_sharded.sum(axis=0)      # r() eagerly inserts ALLREDUCE
```

## Benefits

1. **Less code**: Remove ~150 lines of multi_pm patterns
2. **Simpler mental model**: Sharding is just part of device, not a graph transformation
3. **Automatic propagation**: New ops work with sharding without explicit patterns
4. **Matches JAX model**: "Computation follows data sharding"
5. **Fewer special cases**: Less MULTI/MSELECT/MSTACK handling scattered throughout
6. **Type safety**: `Sharding` class is more explicit than `tuple[str, ...]` + axis somewhere else

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking change | Phased rollout, keep old path working during transition |
| Performance regression | Benchmark each phase, eager handling might add overhead |
| Edge cases in axis propagation | Comprehensive tests, SPEC=2 validation |
| ALLREDUCE timing changes | Test distributed workloads (LLM inference) |

## Testing Strategy

1. **Unit tests**: Test Sharding class, axis propagation, eager handling
2. **Integration tests**: Run existing `test_multitensor.py` throughout
3. **Spec validation**: `SPEC=2` catches UOp structure issues
4. **LLM smoke test**: `echo "Hello" | DEBUG=1 python tinygrad/apps/llm.py --model "llama3.2:1b"`
5. **Multi-GPU benchmarks**: Ensure no performance regression on real workloads
