# Winograd Performance Investigation Plan

## Objective
Identify why NEW (unified kron) has slower compile time and runtime than OLD (tensor.py) implementation.

## Current Performance (64×64×64 shape)
- **BASELINE**: Compile ~2ms, Run ~1.2ms
- **NEW**: Compile ~137ms (64×), Run ~7.1ms (6× slower)
- **OLD**: Compile ~113ms (53×), Run ~2.9ms (2.4× slower)

**NEW is 1.2× slower to compile and 2.4× slower at runtime vs OLD**

## Hypothesis
NEW creates excessive ranges and bufferizes, preventing fusion that OLD gets naturally from tensor-level operations.

## Range Analysis - NEW Implementation

### Ranges Created Upfront (lines 173-176):
```python
tile_ranges     = [new_range((o+3)//4) for o in o_axes]  # Tile count
tile_ranges1    = [new_range((o+3)//4) for o in o_axes]  # DUPLICATE for multiply
inner6          = [new_range(6) for _ in o_axes]         # Winograd tiles
inner6_1        = [new_range(6) for _ in o_axes]         # DUPLICATE for multiply
other_loop_ranges_xhat = [new_range(r.vmax+1) for r in other_loops_x]  # DUPLICATE of batch/cout
other_loop_ranges_ghat = [new_range(r.vmax+1) for r in other_loops_w]  # DUPLICATE of batch/cin
kranges         = [new_range(3) for _ in o_axes]         # Kernel spatial (3×3)
```

### Ranges Created Inside kron() (×3 calls):
Each `kron()` call creates:
```python
new_outer_axes = [new_range(r.vmax+1) for r in outer_axes]  # DUPLICATE outer ranges
result_axes    = [new_range(s) for s in transformed_shape]  # Result shape ranges
```

### Total Range Creations:
For 2D conv (len(o_axes)=2), Cin=64, Cout=64:
- Upfront: 2+2+2+2+Cout+Cin+2 = 138 ranges
- kron() for XHAT: ~68 ranges (Cin + batch + tiles)
- kron() for GHAT: ~68 ranges (Cin + batch + kernel)
- kron() for output: ~70 ranges

**Total: ~344 ranges created!**

OLD creates ranges implicitly during scheduling - likely far fewer.

## Investigation Phases

### Phase 1: XHAT Only
**Test if basic kron transformation is slow**

Modify NEW winowrite to:
```python
X_tiled = act_like.substitute({add: tr*4 + u for add, tr, u in zip(o_adds, tile_ranges, inner6)})
XHAT = kron(X_tiled, winograd_Bt, other_reduces+other_loops_x+tile_ranges, inner6, device, ctx)
return XHAT  # STOP HERE - skip GHAT, MHAT, output transform
```

Modify OLD (tensor.py ~line 2584) to:
```python
dfactors = _apply_winograd_matrix(winograd_Bt, d, len(HW))
return dfactors  # STOP HERE
```

**Measure**: Compile time, runtime, kernel count

**Expected**: If basic kron is the issue, we'll see the difference here.

### Phase 2: XHAT with Indexing
**Test if the indexing pattern adds overhead**

Modify NEW:
```python
XHAT = kron(...)
return XHAT.index(*other_loops_x, *tile_ranges[:len(o_axes)], *inner6[:len(o_axes)])
```

Modify OLD:
```python
dfactors = _apply_winograd_matrix(...)
return dfactors.permute(...)  # Some permutation to match
```

**Expected**: Check if indexing operation is expensive.

### Phase 3: XHAT + GHAT (No Multiply)
**Test if creating both transforms in parallel is slow**

Modify NEW:
```python
XHAT = kron(X_tiled, winograd_Bt, ...)
GHAT = kron(w_sub, winograd_G, ...)
return GHAT  # Or return XHAT + GHAT somehow
```

Modify OLD:
```python
dfactors = _apply_winograd_matrix(winograd_Bt, d, len(HW))
gfactors = _apply_winograd_matrix(winograd_G, g, len(HW))
return gfactors
```

**Expected**: If compile time jumps here, it's about managing two transforms.

### Phase 4: XHAT * GHAT (No Reduce)
**Test the multiplication/indexing pattern**

Modify NEW:
```python
XHAT = kron(...)
GHAT = kron(...)
result = XHAT.index(...) * GHAT.index(...)
return result.bufferize(*appropriate_ranges)  # No reduce yet
```

Modify OLD:
```python
result = gfactors * dfactors
return result
```

**Expected**: Check if the indexing pattern for multiply is inefficient.

### Phase 5: Full MHAT
**Test the reduction**

Modify NEW:
```python
mhat_redu = (XHAT.index(...) * GHAT.index(...)).reduce(*other_reduces, arg=Ops.ADD)
MHAT = mhat_redu.bufferize(...)
return MHAT  # Skip output transform
```

Modify OLD:
```python
result = (gfactors * dfactors).sum(axis=-1-len(HW), dtype=dtype)
return result
```

**Expected**: Check if reduction pattern differs.

### Phase 6: Full Algorithm
Run complete implementations and verify we're back to the original difference.

## Key Constraints to Maintain

1. **Range Usage**: Every LOOP range created must be used in exactly ONE bufferize
2. **No Dangling Ranges**: All ranges must be consumed
3. **Valid Returns**: Must return a properly formed UOp

## Measurement Strategy

For each phase, measure:
```python
import time

# Compile time
t0 = time.time()
with Context(WINO=1):  # or WINO_OLD=1
    out = x.conv2d(w, padding=1)
    out.realize()
compile_time = time.time() - t0

# Runtime
t1 = time.time()
result = out.numpy()
run_time = time.time() - t1
```

## What to Look For

1. **Sudden compile time jump**: Indicates that phase introduces optimization blocker
2. **Gradual runtime degradation**: Indicates accumulating overhead
3. **Kernel count differences**: More kernels = less fusion = slower
4. **Memory patterns**: Excessive bufferizes = more memory traffic

## Expected Outcome

We should find the specific phase where:
- Compile time diverges significantly
- Runtime performance degrades

This will point us to the root cause (likely range duplication or excessive bufferization).
