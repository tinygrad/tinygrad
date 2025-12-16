# Sharding Refactor Progress

## Current State (WIP)

This document tracks the Sharding refactor progress. The goal is to make sharding part of the device specification rather than using wrapper ops.

## What's Working
- Simple sharding tests (test_shard, test_four_add, test_allreduce_*)
- Convolutions (test_conv_data_shard, test_backprop_conv)
- Basic operations on sharded tensors

## Known Issues
- Complex tests (test_backprop_conv_wino, test_data_parallel_resnet) cause infinite loops in pattern matching
- The issue is that broader pattern matching now matches more cases, and some transforms create cycles

# Sharding Shape Architecture

This document explains how shapes work with multi-device sharding in tinygrad.

## Core Concepts

### Two Representations

Sharded tensors have TWO shape representations:

1. **External shape** (what users see): Full logical shape, e.g., `(256,)`
2. **Internal shape** (what gets executed): Per-shard shape, e.g., `(128,)` for 2 devices

### Components

1. **Sharding device**: `Sharding(devices=('CPU:1', 'CPU:2'), axis=0)`
   - Carries device list and sharding axis
   - Propagates through ops via `_device` property
   - Source of truth for which axis is sharded

2. **MULTI op**: A shape-scaling wrapper
   - `arg` = sharding axis (redundant with Sharding.axis, kept for shape calculation)
   - Scales shape from per-shard to full: `shape[axis] * num_devices`
   - Does NOT affect execution - purely for shape presentation

## Shape Flow Example

```
Input: Tensor with shape (256,) on "CPU"

Step 1: copy_to_device(Sharding(('CPU:1','CPU:2'), axis=0))
  Op: COPY
  Device: Sharding(('CPU:1','CPU:2'), axis=0)
  Shape: (256,)  # Unchanged - COPY is elementwise

Step 2: ._shard(0)
  Applies: shrink with symbolic device index
  Op: SHRINK (with Sharding device propagated)
  Shape: (128,)  # Per-shard shape

Step 3: .multi(0)
  Op: MULTI (arg=0)
  Source: SHRINK with shape (128,)
  Output shape: (256,) = 128 * 2 devices  # Scaled back up

Final result:
  External shape: (256,)
  Internal shape: (128,) per device
```

## Operations on Sharded Tensors

When operating on sharded tensors, `multi_pm` transforms handle the unwrap/rewrap:

```
Example: A + B (both sharded on axis 0)

A: MULTI(src=inner_A, axis=0)
   External shape: (256,)
   inner_A shape: (128,)

B: MULTI(src=inner_B, axis=0)
   External shape: (256,)
   inner_B shape: (128,)

alu_multi transformation:
  1. Unwrap: inner_A, inner_B
  2. Apply: inner_A.alu(ADD, inner_B) -> result with shape (128,)
  3. Rewrap: result.multi(0) -> shape (256,)

Result: MULTI(src=add_result, axis=0)
  External shape: (256,)
  Internal shape: (128,)
```

## Why MULTI is Still Needed

MULTI serves one critical purpose: **shape scaling**

Without MULTI:
- After `._shard(axis)`, the shape is per-shard (128,)
- Users expect to see the full shape (256,)
- The shape calculation in `_shape` property needs something to scale

With Sharding device alone:
- We'd need to add shape scaling logic that detects when to scale
- Complex because not all ops with Sharding device should scale shape
- Only the "boundary" between internal/external should scale

## Current Architecture

```
          User sees (256,)
               │
         ┌─────┴─────┐
         │   MULTI   │  <- Shape scaling: 128 * 2 = 256
         │  axis=0   │
         └─────┬─────┘
               │
        Internal (128,)
               │
    ┌──────────┴──────────┐
    │  Ops with Sharding  │  <- Per-shard operations
    │       device        │
    └──────────┬──────────┘
               │
    Executed on each device
```

## Future Direction

Phase 6 goal: Make Sharding device the primary source of truth for axis.

Current state:
- MULTI.arg carries axis for shape scaling
- Sharding.axis carries axis for device operations
- Both should always match (enforced by shard() creating both)

Potential future:
- MULTI only used for shape scaling at graph boundaries
- All internal logic uses Sharding device for axis info
- Eventually: add shape scaling to _shape based on Sharding device, eliminate MULTI

## Key Insight

The `.axis` property already handles Sharding device:

```python
@functools.cached_property
def axis(self) -> int|None:
  if isinstance(self._device, Sharding): return self._device.axis
  if self.op is Ops.MULTI: return self.arg
  ...
```

So for internal logic, ops with Sharding device already report their axis correctly.
The only remaining use of MULTI is shape scaling in `_shape` property.
