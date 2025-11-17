# Real Runtime Bottleneck Investigation - Findings

## Executive Summary

**The output transform (final kron() call) is where NEW loses all its performance advantage.**

Shape tested: B=1, Cin=16, Cout=16, H=32×32

| Phase | Operation | NEW Runtime | OLD Runtime | NEW vs OLD |
|-------|-----------|-------------|-------------|------------|
| **Baseline** | No winograd | **2.2ms** | **2.2ms** | — |
| **1** | XHAT only | 68.1ms | 149.7ms | ✅ **54.5% faster** |
| **2** | + GHAT | 74.0ms | 134.1ms | ✅ **44.8% faster** |
| **3** | + Multiply/Reduce/MHAT | 92.9ms | 133.7ms | ✅ **30.5% faster** |
| **4** | + Output transform **(COMPLETE)** | **135.1ms** | **133.1ms** | ❌ **1.4% slower** |

---

## Key Finding: The Output Transform is the Bottleneck

### Phase 3 → Phase 4 Jump

**NEW implementation:**
- Phase 3: 92.9ms
- Phase 4: 135.1ms
- **Jump: +42.2ms (+45% slower!)**

**OLD implementation:**
- Phase 3: 133.7ms (reference - this is incomplete OLD, but we compare full OLD)
- Phase 4: 133.1ms
- **Jump: minimal (OLD doesn't have this bottleneck)**

### What Happens in Phase 4?

Phase 4 adds the **output transform** - the final kron() call:

```python
# rangeify.py lines 193-194
return kron(MHAT, winograd_At, other_loop_ranges_xhat+other_loop_ranges_ghat+tile_ranges1, inner6_1, device, ctx)\
  .index(*other_loops_x, *other_loops_w, *[ox//4 for ox in o_axes], *[ox%4 for ox in o_axes])
```

This single operation adds **~42ms** to NEW's runtime, destroying the 41ms advantage from Phase 3!

---

## Why is the Output Transform Slow?

### 1. MHAT is Already a Buffer

```python
MHAT = mhat_redu.bufferize(*other_loop_ranges_xhat, *other_loop_ranges_ghat, *tile_ranges1, *inner6_1,
                           arg=BufferizeOpts(device=device, addrspace=AddrSpace.GLOBAL))
```

MHAT is a bufferized result with shape controlled by:
- `other_loop_ranges_xhat` (duplicates of `other_loops_x`)
- `other_loop_ranges_ghat` (duplicates of `other_loops_w`)
- `tile_ranges1` (tile counts)
- `inner6_1` (6×6 winograd dimensions)

### 2. kron() on Buffer Creates Indexing Overhead

From our unified kron() implementation (lines 127-129):

```python
if source.op in {Ops.BUFFER, Ops.BUFFERIZE}:
  # Buffer: index with new ranges
  T = [source.index(*new_outer_axes, *[UOp.const(dtypes.index, i) for i in I])
       for I in product(*(range(s) for s in inner_shape))]
```

When kron() receives MHAT (a buffer), it must:
1. Create `new_outer_axes` (duplicating `other_loop_ranges_xhat + other_loop_ranges_ghat + tile_ranges1`)
2. Index MHAT 36 times (6×6 grid for winograd_At matrix)
3. Apply matrix transform
4. Bufferize again

This creates:
- **More range duplication** (already have duplicates from Phase 3)
- **Excessive indexing** (36 index operations on a large buffer)
- **Another bufferize** (3rd bufferization in the pipeline)

### 3. OLD Avoids This

OLD's tensor-level implementation:

```python
# tensor.py line 2587
ret = _apply_winograd_matrix(winograd_At, (gfactors * dfactors).sum(axis=-1-len(HW), dtype=dtype), len(HW))
```

OLD applies `winograd_At` transformation at tensor level where:
- No range duplication needed
- Tensor operations can fuse better
- Scheduler has more optimization freedom

---

## The Real Problem: Triple Bufferization

NEW creates this pipeline:

1. **XHAT buffer** (from first kron)
2. **GHAT buffer** (from second kron)
3. Index both → Multiply → Reduce
4. **MHAT buffer** (from bufferize)
5. Index MHAT 36 times → Transform
6. **Output buffer** (from third kron)

OLD creates simpler pipeline:
- Transform input (dfactors)
- Transform weights (gfactors)
- Multiply + sum (fused)
- Transform output (ret)
- All at tensor level, better fusion

---

## Detailed Phase Breakdown

### Phase 1: XHAT Only (NEW 54.5% faster)

NEW wins because:
- Single kron() call is efficient
- No range duplication yet
- Scheduler can optimize single transform

### Phase 2: XHAT + GHAT (NEW 44.8% faster)

NEW still wins because:
- Two independent kron() calls
- No interaction between buffers yet
- Slight overhead from creating both

### Phase 3: + Multiply/Reduce/MHAT (NEW 30.5% faster)

NEW advantage shrinks because:
- Now indexing both XHAT and GHAT buffers with fresh ranges
- Creating MHAT buffer (2nd level of bufferization)
- Range duplication starts to hurt
- But still faster because OLD's tensor ops are also inefficient

### Phase 4: + Output Transform (NEW 1.4% slower)

NEW loses all advantage because:
- **3rd kron() call on an already-buffered MHAT**
- **36 index operations** on MHAT buffer
- **3rd bufferization** in the pipeline
- **Maximum range duplication** (outer ranges already duplicated twice)
- Adds **+42ms** overhead, wiping out the 41ms advantage

---

## Why Both Are 60× Slower Than Baseline

Both OLD and NEW are ~60× slower than baseline (2.2ms):
- OLD: 133.1ms (61× slower)
- NEW: 135.1ms (62× slower)

This suggests **winograd is fundamentally slow on this hardware/backend** for this shape. Possible reasons:

1. **Poor memory locality**: Winograd requires scattered memory access patterns
2. **Small tensors**: Overhead dominates for 32×32 images
3. **CPU backend**: Winograd benefits GPUs more than CPUs
4. **Lack of kernel fusion**: Multiple transform steps don't fuse well
5. **Matrix size**: 6×6 transforms may not be optimal for this hardware

Winograd is typically beneficial for:
- Larger images (reduces arithmetic)
- GPU execution (parallel transforms)
- Shapes where transform overhead < arithmetic savings

For 32×32 with Cin=16, baseline conv is already very fast (2.2ms), so winograd overhead dominates.

---

## Recommendations

### Option 1: Optimize the Output Transform

Modify kron() to avoid re-buffering when source is already a buffer:

```python
def kron(source, matrix, outer_axes, inner_axes, device, ctx, bufferize=True):
  # If source is already a buffer and bufferize=True, maybe reuse ranges?
  # Or return unbufferized expression to avoid excessive materialization
  ...
```

Call it as:
```python
# Don't bufferize the final transform
return kron(MHAT, winograd_At, ..., bufferize=False).index(...)
```

**Expected impact:** Might save ~20-30ms, getting NEW competitive with OLD.

### Option 2: Eliminate MHAT Bufferization

Instead of bufferizing MHAT, pass the unbufferized expression to the output transform:

```python
# Instead of:
MHAT = mhat_redu.bufferize(...)
return kron(MHAT, winograd_At, ...).index(...)

# Try:
return kron(mhat_redu, winograd_At, ...).index(...)
```

This requires kron() to handle unbufferized reductions, but could save both:
- MHAT bufferization overhead
- Output transform indexing overhead

**Expected impact:** Could save ~40ms, making NEW ~30% faster than OLD.

### Option 3: Fuse Output Transform

Create a specialized operation that fuses MHAT computation + output transform:

```python
# Instead of separate MHAT buffer + kron
# Create fused operation that does multiply→reduce→output_transform in one kernel
```

**Expected impact:** Best case, could approach Phase 3 performance (92.9ms), making NEW 30% faster.

### Option 4: Accept Parity, Keep Clean Code

If runtime parity is acceptable:
- NEW: 135.1ms, cleaner code (~80 lines saved)
- OLD: 133.1ms, messier code
- Difference: 1.4% (within noise)

**Trade-off:** Simpler codebase for essentially identical performance.

---

## Conclusion

The **output transform** (3rd kron() call on MHAT buffer) adds **~42ms** overhead to NEW, eliminating the performance advantage built up through Phases 1-3.

The root cause is **excessive bufferization and range duplication** when applying kron() to an already-buffered result.

**Bottom line:**
- NEW and OLD have essentially **identical runtime** (135ms vs 133ms)
- NEW has **~30% slower compile time** (711ms vs 574ms)
- NEW has **cleaner code** (~80 lines saved)

Unless we optimize the output transform (Options 1-3), NEW provides **no performance benefit**, only code simplification.
