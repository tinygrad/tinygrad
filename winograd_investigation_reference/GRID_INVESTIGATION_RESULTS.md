# Grid Organization Investigation Results

## Executive Summary

The grid difference between OLD and NEW is **fundamental** - it comes from how the entire UOp graph is structured, not just axis ordering in the final bufferize() call.

**Key Finding:** NEW systematically puts **channels on X axis**, OLD puts **spatial dims on X/Y axes**.

## Test Results: Grid Comparison

```
OLD vs NEW Grid Comparison (first 3 dims = global grid)
========================================================================
Input 24×24:
  OLD: r_2_2_2...      → 2×2×2
  NEW: r_2_2_2...      → 2×2×2
  ✅ MATCH!

Input 32×32:
  OLD: r_6_6_8...      → 6×6×8   (spatial 6×6, channels 8)
  NEW: r_8_6_6...      → 8×6×6   (channels 8, spatial 6×6)
  ❌ DIFFERENT - channels/spatial swapped!

Input 40×40:
  OLD: r_5_5_16...     → 5×5×16  (spatial 5×5, channels 16)
  NEW: r_5_5_2...      → 5×5×2
  ❌ DIFFERENT

Input 48×48:
  OLD: r_2_3_3...      → 2×3×3
  NEW: r_6_3_2...      → 6×3×2
  ❌ DIFFERENT

Input 64×64:
  OLD: r_6_6_16...     → 6×6×16  (spatial 6×6, channels 16)
  NEW: r_16_6_6...     → 16×6×6  (channels 16, spatial 6×6)
  ❌ DIFFERENT - channels/spatial swapped!
```

## The Pattern

**OLD:** Consistently puts spatial dimensions first (on gid.x, gid.y)
**NEW:** Consistently puts channel dimension first (on gid.x)

For our 64×64 test case:
```
OLD: r_6_6_16
  gid.x = 0-5   (spatial tile X)
  gid.y = 0-5   (spatial tile Y)
  gid.z = 0-15  (channels)

NEW: r_16_6_6
  gid.x = 0-15  (channels)
  gid.y = 0-5   (spatial tile X)
  gid.z = 0-5   (spatial tile Y)
```

## Why This Happens

The grid is determined by the **scheduler**, not by the axis order in bufferize(). The scheduler analyzes the entire UOp computation graph and decides:

1. Which axes are most parallelizable
2. Which axes should be global vs local work
3. How to factor/split dimensions for optimal performance

### OLD Implementation (Tensor-level)

```python
# tensor.py:2584-2587
dfactors = _apply_winograd_matrix(winograd_Bt, d, len(HW)).reshape(*HWI, bs, groups, 1, cin, *tyx)
gfactors = _apply_winograd_matrix(winograd_G, g, len(HW)).reshape(*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
ret = _apply_winograd_matrix(winograd_At, (gfactors * dfactors).sum(axis=-1-len(HW), dtype=dtype), len(HW))
```

Creates shape: `(4, 4, bs, groups, cout, ty, tx)`
- Winograd dims (4×4) become inner loops or local work
- Spatial tiles (ty, tx) are prominent in the graph
- Scheduler sees spatial as primary parallelization opportunity
- **Result: spatial first → r_6_6_16**

### NEW Implementation (UOp-level)

```python
# rangeify.py:292-299
MHAT = mhat_redu.bufferize(*(tile_ranges1+other_loop_ranges_ghat+other_loop_ranges_xhat+inner6_1), ...)
output = winograd_kron(MHAT, winograd_At, tile_ranges1+other_loop_ranges_ghat+other_loop_ranges_xhat, inner6_1, ...)
  .index(*other_loops_x, *other_loops_w, *[ox//4 for ox in o_axes], *[ox%4 for ox in o_axes])
```

Creates axes: `[16, 16, 16, ...]` (tile_ranges1 + other_loop_ranges_ghat)
- All appear equally parallelizable to scheduler
- Channel dimension (16) is in the mix with tile dimensions (16, 16)
- Scheduler chooses to parallelize channels on X axis
- **Result: channels first → r_16_6_6**

## Why Axis Reordering Didn't Work

We tried reordering axes in bufferize():
```python
# Tried: tile_ranges1 + other_loop_ranges_ghat
# Still got: r_16_6_6
```

**Why it failed:** The grid isn't determined by the order of axes in one bufferize() call. It's determined by analyzing the **entire computation graph** including:
- All intermediate buffers (XHAT, GHAT, MHAT)
- How they're indexed and combined
- The structure of the UOp tree
- Scheduler heuristics for performance

## The Root Cause

**OLD and NEW create fundamentally different UOp graph structures:**

**OLD:**
- Uses Tensor operations (reshape, expand, sum)
- Creates one graph structure where spatial dims are "outer"
- Scheduler sees spatial as primary parallelization axis

**NEW:**
- Uses UOp-level winograd_kron with index() operations
- Creates different graph structure where all dims appear similar
- Scheduler chooses channels as primary parallelization axis

## Current State: HYBRID

Our current code has:
- **Buffer order:** OLD-style (GHAT, XHAT)
- **Grid:** NEW-style (r_16_6_6)
- **Correctness:** ✅ Works perfectly!
- **Performance:** ❓ Unknown (scattered writes, not coalesced)

## Options to Match OLD Grid

### Option 1: Mimic OLD's Tensor Operations

Rewrite NEW winograd to use the same Tensor-level operations as OLD:
- Use reshape() and expand() like `_apply_winograd_matrix()`
- Create the same graph structure
- Should result in same scheduler decisions

**Pros:** Would get r_6_6_16 grid
**Cons:** Defeats the purpose of UOp-level optimization

### Option 2: Modify Scheduler

Change the scheduler to prefer spatial dimensions over channels for winograd:
- Add heuristics to recognize winograd pattern
- Force spatial-first grid assignment

**Pros:** Could work with current UOp structure
**Cons:** Requires understanding and modifying complex scheduler code

### Option 3: Accept NEW's Grid (Transpose Approach)

Fully embrace the transpose optimization:
- Revert buffer order to NEW-style (XHAT, GHAT)
- Keep r_16_6_6 grid
- Scattered writes perform the transpose

**Pros:** Clean, might be performant for different reasons
**Cons:** Different from OLD, scattered writes

### Option 4: Keep HYBRID

Ship what we have:
- OLD-style buffer order
- NEW-style grid
- Correct results

**Pros:** Works, minimal risk
**Cons:** Not fully optimized, unclear performance

## Recommendation

**I recommend Option 4 (Keep HYBRID) with performance testing:**

1. ✅ Current HYBRID is numerically correct
2. Benchmark it vs OLD
3. If performance is within 20% of OLD, ship it
4. If performance is poor, investigate Option 3 (full transpose approach)

Trying to match OLD's grid (Options 1-2) would require major refactoring and might not be worth it if the HYBRID already performs reasonably.

## Next Steps

1. **Benchmark HYBRID performance** - critical decision point
2. Remove debug logging from rangeify.py
3. If performance is acceptable, document and close
4. If not, investigate full transpose approach (Option 3)

## Files Generated

- `hybrid_wino_kernel.metal` - Current HYBRID kernel
- `test_grid_sizes.py` - Grid testing for OLD
- `test_grid_sizes_new.py` - Grid comparison OLD vs NEW
- `test_correctness.py` - Numerical correctness verification
- This document

---

**Bottom line:** The grid difference is fundamental to how OLD vs NEW structure the computation. We can't easily change it without restructuring the entire winograd implementation. The HYBRID works correctly; we should test performance before deciding next steps.
