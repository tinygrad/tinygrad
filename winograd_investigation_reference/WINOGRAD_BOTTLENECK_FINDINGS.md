# Winograd Performance Investigation - Findings

## Test Configuration
- Shape: B=1, Cin=16, Cout=16, H=32×32
- Baseline (no winograd): ~76ms compile, ~0.6ms runtime

## Phase-by-Phase Results

### Phase 1: XHAT Only
```python
# Only create XHAT transform, skip GHAT/MHAT/output
```
- **NEW compile: 197ms (2.6× overhead)**
- **OLD compile: 547ms (7.2× overhead)**
- ✅ **NEW is 2.8× FASTER than OLD**

### Phase 2: XHAT + GHAT
```python
# Create both XHAT and GHAT transforms, no multiply
```
- **NEW compile: 154ms (2.0× overhead)**
- **OLD compile: 538ms (7.1× overhead)**
- ✅ **NEW is 3.5× FASTER than OLD**

### Phase 3: XHAT + GHAT + MHAT (Multiply + Reduce)
```python
# Add the problematic code:
mhat_redu = (XHAT.index(*fresh_ranges_x...) *
             GHAT.index(*fresh_ranges_g...)).reduce(...)
MHAT = mhat_redu.bufferize(*all_fresh_ranges...)
```
- **NEW compile: 764ms (9.0× overhead)**
- **OLD compile: 569ms (6.7× overhead)**
- ❌ **NEW is 1.3× SLOWER than OLD** (34% slower compile)
- ❌ **NEW runtime: 16.3ms vs OLD 0.4ms** (40× slower!)

### Phase 4: Full Algorithm (+ Output Transform)
```python
# Add final kron transform and indexing
return kron(MHAT, winograd_At, ...).index(...)
```
- **NEW compile: 737ms (9.8× overhead)**
- **OLD compile: 561ms (7.5× overhead)**
- ❌ **NEW is 1.3× SLOWER than OLD** (31% slower compile)
- **NEW runtime: 0.6ms vs OLD 0.5ms** (13% slower, much better than Phase 3!)

## 🎯 ROOT CAUSE IDENTIFIED

The bottleneck is in **Phase 3: The multiply + reduce + MHAT bufferize step**.

### The Problematic Code (lines 187-190)

```python
# Create fresh ranges for indexing
tile_ranges1 = [ctx.new_range((int(b.vmax+1)+3)//4, AxisType.LOOP) for b in o_axes]
inner6_1 = [ctx.new_range(6, AxisType.LOOP) for _ in o_axes]
other_loop_ranges_xhat = [ctx.new_range(r.vmax+1, AxisType.LOOP) for r in other_loops_x]
other_loop_ranges_ghat = [ctx.new_range(r.vmax+1, AxisType.LOOP) for r in other_loops_w]

# Index both buffers with fresh ranges, then multiply/reduce
mhat_redu = (XHAT.index(*other_reduces, *other_loop_ranges_xhat, *tile_ranges1, *inner6_1) *
             GHAT.index(*other_reduces, *other_loop_ranges_ghat, *inner6_1)).reduce(*other_reduces, arg=Ops.ADD)

# Bufferize with ALL those fresh ranges
MHAT = mhat_redu.bufferize(*other_loop_ranges_xhat, *other_loop_ranges_ghat, *tile_ranges1, *inner6_1,
                           arg=BufferizeOpts(device=device, addrspace=AddrSpace.GLOBAL))
```

### Why This is Slow

1. **Range Duplication**: We create duplicates of ranges (tile_ranges vs tile_ranges1, inner6 vs inner6_1, etc.) because each bufferize consumes ranges
2. **Prevents Fusion**: Indexing XHAT and GHAT with fresh ranges prevents the scheduler from fusing these operations
3. **Extra Bufferize**: We bufferize MHAT with 4+ dimensions worth of ranges, creating a large intermediate buffer
4. **OLD Avoids This**: OLD does multiplication at tensor level where fusion is easier:
   ```python
   # OLD (tensor.py line 2587)
   ret = (gfactors * dfactors).sum(axis=-1-len(HW), dtype=dtype)
   # Tensor-level ops can fuse the multiply+reduce better
   ```

## Performance Impact Summary

| Phase | Operation | NEW Compile | OLD Compile | Winner | Speedup |
|-------|-----------|-------------|-------------|--------|---------|
| 1 | XHAT only | 197ms | 547ms | NEW | 2.8× faster |
| 2 | + GHAT | 154ms | 538ms | NEW | 3.5× faster |
| 3 | + Multiply/Reduce/MHAT | **764ms** | **569ms** | **OLD** | **1.3× slower** ⚠️ |
| 4 | + Output transform | 737ms | 561ms | OLD | 1.3× slower |

**Jump from Phase 2 → Phase 3**: 154ms → 764ms = **5× compile time increase**

This 5× jump identifies the exact bottleneck!

## Recommendations

### Option 1: Eliminate Duplicate Ranges (Preferred)
Try to reuse the original ranges instead of creating fresh ones for indexing:
```python
# Instead of:
tile_ranges1 = [ctx.new_range(...) for ...]  # Fresh ranges
XHAT.index(..., *tile_ranges1, ...)

# Try:
# Can we reuse tile_ranges somehow?
# Maybe return unbufferized expressions from kron?
```

### Option 2: Defer Bufferization
Modify `kron()` to optionally return an unbufferized expression:
```python
def kron(source, matrix, outer_axes, inner_axes, device, ctx, bufferize=True):
    ...
    if bufferize:
        return result.bufferize(...)
    else:
        return result, new_outer_axes, result_axes  # Return expression + ranges
```

Then in winowrite:
```python
XHAT_expr, xhat_outer, xhat_inner = kron(..., bufferize=False)
GHAT_expr, ghat_outer, ghat_inner = kron(..., bufferize=False)

# Multiply/reduce WITHOUT intermediate bufferize
mhat_redu = (XHAT_expr * GHAT_expr).reduce(...)
MHAT = mhat_redu.bufferize(...)  # Single bufferize
```

### Option 3: Fuse XHAT×GHAT into Single Operation
Create a specialized "winograd_multiply" that does the transforms + multiply as one fused op.

## Key Insight

The unified `kron()` implementation is NOT the problem!

- Basic kron transformations (XHAT, GHAT) are 2-3× FASTER than OLD
- The problem is the **PATTERN of how we use kron** - specifically:
  1. Bufferizing each transform separately
  2. Creating duplicate ranges for multiply
  3. Indexing buffers with fresh ranges
  4. Bufferizing again

This prevents fusion that OLD gets "for free" by operating at tensor level.

## Next Steps

1. Try Option 2 (defer bufferization) - add `bufferize=False` flag to kron
2. Test if reusing ranges or returning unbufferized expressions helps
3. Measure compile time after changes
4. Target: Get Phase 3 compile time down to ~200-300ms (closer to Phase 2)

The ~80 lines of code we saved with unified kron is still valuable - we just need to optimize the usage pattern!
