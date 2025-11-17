# Attempts to Match OLD's Grid (r_6_6_16)

## Goal
Match OLD's grid organization `r_6_6_16` instead of current `r_16_6_6` to enable coalesced float2 writes.

## Root Cause Identified

The grid dimensions come from **sorted axis IDs** assigned during range creation (gpudims.py:61):
```python
global_dims = sorted(dedup([x.arg[0:-1] for x in all_ranges.values() if x.arg[-1] in (AxisType.GLOBAL, AxisType.THREAD)]))
```

**Current state:**
- **OLD:** Axis IDs `[(1,), (2,), (4,)]` → sizes `[6, 6, 16]` → grid `r_6_6_16`
- **NEW:** Axis IDs `[(1,), (4,), (5,)]` → sizes `[16, 6, 6]` → grid `r_16_6_6`

The problem: **ID 1 has size 16 in NEW but size 6 in OLD.**

## Attempts Made

### Attempt 1: Reorder axes in final bufferize()
**Code:** Changed `MHAT.bufferize()` to use `tile_ranges1+other_loop_ranges_ghat+...`
**Result:** ❌ No change - still r_16_6_6
**Why it failed:** Bufferize axis order doesn't determine grid; only range IDs matter

### Attempt 2: Reorder axes passed to final winograd_kron()
**Code:** Changed `winograd_kron(MHAT, At, tile_ranges1+other_loop_ranges_xhat+other_loop_ranges_ghat, ...)`
**Result:** ❌ No change - still r_16_6_6
**Why it failed:** `other_loop_ranges_xhat` is empty, and axis order in function arguments doesn't affect range IDs

### Attempt 3: Create result_axes before new_outer_axes
**Code:** In `winograd_kron()`, moved `result_axes` creation before `new_outer_axes`
**Result:** ❌ No change - still r_16_6_6
**Why it failed:** This only affects order within ONE winograd_kron call; final IDs depend on order across ALL calls

### Attempt 4: Create channel ranges (ghat) last
**Code:** In `winowrite()`, moved `other_loop_ranges_ghat` creation to the end
**Result:** ❌ No change - still r_16_6_6
**Why it failed:** These ranges (AxisType.LOOP) don't directly become the grid; they're inputs to winograd_kron which creates new ranges

## Why This Is Hard

The ranges that end up in the final grid are NOT the ones we create initially. They go through:

1. **Initial creation** (winowrite): tile_ranges, inner6, kranges, other_loop_ranges_*
2. **winograd_kron calls** (3×): Each creates `new_outer_axes` and `result_axes`
3. **Indexing operations**: `.index()` call eliminates some dimensions
4. **Scheduling/optimization**: Many ranges get eliminated or merged
5. **Final SINK**: Only 3 ranges survive with IDs [1, 4, 5]

**The issue:** The ranges that survive to the final kernel are determined by complex graph transformations, not just creation order.

## The Fundamental Difference

**OLD (Tensor-level):**
```python
# tensor.py:2584-2587
dfactors = _apply_winograd_matrix(Bt, d, ...)  # Creates Tensor with certain shape
gfactors = _apply_winograd_matrix(G, g, ...)   # Creates Tensor with certain shape
ret = _apply_winograd_matrix(At, (gfactors * dfactors).sum(...), ...)

# These Tensor operations lower to UOps with certain structure
# The structure determines which dimensions get low IDs
```

**NEW (UOp-level):**
```python
# rangeify.py:280-306
XHAT = winograd_kron(X_tiled, Bt, ...)  # Creates UOps directly
GHAT = winograd_kron(w_sub, G, ...)      # Creates UOps directly
output = winograd_kron(MHAT, At, ...).index(...)  # Creates UOps directly

# Direct UOp creation has different graph structure
# Different structure → different dimension ordering → different IDs
```

The Tensor operations in OLD create intermediate shapes that guide the scheduler differently than our direct UOp operations.

## What Would Actually Work

### Option A: Mimic OLD's Tensor Structure (Major Refactor)
Rewrite winowrite() to use Tensor-level operations like `_apply_winograd_matrix()`:
- Create tensors with reshape/expand like OLD
- Let those lower to UOps naturally
- Should produce same graph structure as OLD

**Pros:** Would match OLD's grid
**Cons:**
- Defeats purpose of UOp-level optimization
- Large refactor (100+ lines)
- Loses winograd_kron() abstraction

### Option B: Modify Scheduler Grid Assignment
Change gpudims.py to detect winograd patterns and prefer spatial-first:
```python
# Pseudo-code
if is_winograd_pattern(s):
    # Force spatial dims to come before channel dim in grid
    global_dims = reorder_for_spatial_first(global_dims)
```

**Pros:** Could work with current UOp structure
**Cons:**
- Requires deep scheduler knowledge
- Pattern detection might be fragile
- Affects core scheduling logic

### Option C: Accept HYBRID and Optimize Elsewhere
Keep r_16_6_6 grid but try to make scattered writes faster:
- Investigate if Metal compiler can optimize the pattern
- Benchmark real performance impact
- If acceptable (within 20% of OLD), ship it

**Pros:** Minimal risk, works today
**Cons:** Not perfectly matching OLD

## My Recommendation

After trying 4 different approaches to fix range ordering, I believe:

1. **The grid difference is fundamental to Tensor vs UOp structure**
   - It's not a simple ordering bug we can fix
   - Would require major refactoring (Option A) or scheduler changes (Option B)

2. **The HYBRID is numerically perfect**
   - ✅ Produces identical results to OLD (0.00e+00 error)
   - ✅ Correct buffer order (GHAT, XHAT)
   - ❓ Performance unknown but likely acceptable

3. **Next step should be performance testing, not more fixing**
   - Benchmark HYBRID vs OLD
   - If performance is acceptable → ship HYBRID
   - If performance is poor → consider major refactor (Option A)

We've spent significant effort trying to match the grid without success. The pragmatic path forward is to test whether the grid difference actually matters for performance.

## Test Command

```bash
# Benchmark HYBRID vs OLD
python3 -c "
import time
from tinygrad import Tensor, dtypes
import os

for name, wino_old in [('OLD', True), ('HYBRID', False)]:
    os.environ['RANGEIFY'] = '1'
    os.environ['WINO_OLD'] = '1' if wino_old else '0'
    os.environ['WINO'] = '0' if wino_old else '1'

    x = Tensor.randn(1, 16, 64, 64, dtype=dtypes.float32).realize()
    w = Tensor.randn(16, 16, 3, 3, dtype=dtypes.float32).realize()

    # Warmup
    for _ in range(5):
        out = x.conv2d(w, padding=1).realize()

    # Benchmark
    times = []
    for _ in range(20):
        start = time.perf_counter()
        out = x.conv2d(w, padding=1).realize()
        times.append((time.perf_counter() - start) * 1000)

    print(f'{name}: {min(times):.2f}ms (min of 20 runs)')
"
```

If HYBRID is within ~20% of OLD, I recommend accepting it.
