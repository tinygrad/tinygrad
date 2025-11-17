# Winograd Runtime Investigation - Phase Modifications

## IMPORTANT: Focus on RUNTIME, not compile time!

Previous findings showed Phase 3 had **40× slower runtime** but Phase 4 recovered to only 13% slower.
This investigation will identify exactly where and why runtime degrades.

## Setup

Shape: B=1, Cin=16, Cout=16, H=32×32
Baseline: ~0.6ms runtime

## Backup Original

```bash
cp tinygrad/schedule/rangeify.py tinygrad/schedule/rangeify.py.backup
```

---

## PHASE 1: XHAT Only

### Goal
Test if basic kron transformation affects runtime.

### Modify winowrite (lines 165-194)

Replace the entire function with:

```python
def winowrite(ctx: IndexingContext, lhs: UOp, rhs: UOp, redu: UOp):
  if not (g := winoguard(lhs, rhs, redu)): return None
  act_like, w_like, k_axes, o_axes, o_adds = g
  reduce_ranges = list(redu.src[1:]); device = redu.device
  other_reduces = [ax for ax in act_like.ranges if ax not in k_axes and ax in reduce_ranges]
  other_loops_x = [ax for ax in act_like.ranges if ax not in reduce_ranges+o_axes]
  other_loops_w = [ax for ax in w_like.ranges if ax not in reduce_ranges]

  # Only create ranges we actually need
  tile_ranges = [ctx.new_range((int(b.vmax+1)+3)//4, AxisType.LOOP) for b in o_axes]
  inner6 = [ctx.new_range(6, AxisType.LOOP) for _ in o_axes]

  # PHASE 1: Just XHAT transform
  X_tiled = act_like.substitute({add: tr*4 + u for add, tr, u in zip(o_adds, tile_ranges, inner6)})
  XHAT = kron(X_tiled, winograd_Bt, other_reduces+other_loops_x+tile_ranges, inner6, device, ctx)

  # Return with proper indexing to match expected output shape
  return XHAT.index(*other_loops_x, *other_loops_w, *[ox//4 for ox in o_axes], *[ox%4 for ox in o_axes])
```

### OLD Modification (tensor.py ~line 2584)

```python
# In Tensor.conv2d, around line 2584, comment out gfactors and multiplication:
dfactors = Tensor._apply_winograd_matrix(winograd_Bt, d, len(HW))
return dfactors  # STOP HERE - skip gfactors and multiply
```

### Run
```bash
python3 wino_runtime_investigation.py
```

---

## PHASE 2: XHAT + GHAT

### Goal
Test if creating both transforms affects runtime.

### Modify winowrite

```python
def winowrite(ctx: IndexingContext, lhs: UOp, rhs: UOp, redu: UOp):
  if not (g := winoguard(lhs, rhs, redu)): return None
  act_like, w_like, k_axes, o_axes, o_adds = g
  reduce_ranges = list(redu.src[1:]); device = redu.device
  other_reduces = [ax for ax in act_like.ranges if ax not in k_axes and ax in reduce_ranges]
  other_loops_x = [ax for ax in act_like.ranges if ax not in reduce_ranges+o_axes]
  other_loops_w = [ax for ax in w_like.ranges if ax not in reduce_ranges]

  tile_ranges = [ctx.new_range((int(b.vmax+1)+3)//4, AxisType.LOOP) for b in o_axes]
  inner6 = [ctx.new_range(6, AxisType.LOOP) for _ in o_axes]
  kranges = [ctx.new_range(3, AxisType.LOOP) for _ in o_axes]

  # PHASE 2: Create both XHAT and GHAT
  X_tiled = act_like.substitute({add: tr*4 + u for add, tr, u in zip(o_adds, tile_ranges, inner6)})
  XHAT = kron(X_tiled, winograd_Bt, other_reduces+other_loops_x+tile_ranges, inner6, device, ctx)

  w_sub = w_like.substitute({k: r for k, r in zip(k_axes, kranges)})
  GHAT = kron(w_sub, winograd_G, other_reduces+other_loops_w, kranges, device, ctx)

  # Still return XHAT (GHAT is created but not used)
  return XHAT.index(*other_loops_x, *other_loops_w, *[ox//4 for ox in o_axes], *[ox%4 for ox in o_axes])
```

### OLD Modification (tensor.py)

```python
dfactors = Tensor._apply_winograd_matrix(winograd_Bt, d, len(HW))
gfactors = Tensor._apply_winograd_matrix(winograd_G, g, len(HW))
return dfactors  # Create both but return dfactors
```

---

## PHASE 3: XHAT * GHAT + REDUCE (MHAT)

### Goal
**CRITICAL PHASE** - This is where 40× runtime slowdown was observed!

### Modify winowrite

```python
def winowrite(ctx: IndexingContext, lhs: UOp, rhs: UOp, redu: UOp):
  if not (g := winoguard(lhs, rhs, redu)): return None
  act_like, w_like, k_axes, o_axes, o_adds = g
  reduce_ranges = list(redu.src[1:]); device = redu.device
  other_reduces = [ax for ax in act_like.ranges if ax not in k_axes and ax in reduce_ranges]
  other_loops_x = [ax for ax in act_like.ranges if ax not in reduce_ranges+o_axes]
  other_loops_w = [ax for ax in w_like.ranges if ax not in reduce_ranges]

  # PHASE 3: Need duplicate ranges for the multiply step
  tile_ranges, tile_ranges1 = [[ctx.new_range((int(b.vmax+1)+3)//4, AxisType.LOOP) for b in o_axes] for _ in range(2)]
  inner6, inner6_1 = [[ctx.new_range(6, AxisType.LOOP) for _ in o_axes] for _ in range(2)]
  other_loop_ranges_xhat = [ctx.new_range(r.vmax+1, AxisType.LOOP) for r in other_loops_x]
  other_loop_ranges_ghat = [ctx.new_range(r.vmax+1, AxisType.LOOP) for r in other_loops_w]
  kranges = [ctx.new_range(3, AxisType.LOOP) for _ in o_axes]

  X_tiled = act_like.substitute({add: tr*4 + u for add, tr, u in zip(o_adds, tile_ranges, inner6)})
  XHAT = kron(X_tiled, winograd_Bt, other_reduces+other_loops_x+tile_ranges, inner6, device, ctx)

  w_sub = w_like.substitute({k: r for k, r in zip(k_axes, kranges)})
  GHAT = kron(w_sub, winograd_G, other_reduces+other_loops_w, kranges, device, ctx)

  # PHASE 3: The problematic multiply + reduce + bufferize
  mhat_redu = (XHAT.index(*other_reduces, *other_loop_ranges_xhat, *tile_ranges1, *inner6_1) *
               GHAT.index(*other_reduces, *other_loop_ranges_ghat, *inner6_1)).reduce(*other_reduces, arg=Ops.ADD)
  MHAT = mhat_redu.bufferize(*other_loop_ranges_xhat, *other_loop_ranges_ghat, *tile_ranges1, *inner6_1,
                             arg=BufferizeOpts(device=device, addrspace=AddrSpace.GLOBAL))

  # Return MHAT with indexing (no output transform yet)
  return MHAT.index(*other_loop_ranges_xhat, *other_loop_ranges_ghat, *tile_ranges1, *inner6_1)
```

### OLD Modification (tensor.py)

```python
dfactors = Tensor._apply_winograd_matrix(winograd_Bt, d, len(HW))
gfactors = Tensor._apply_winograd_matrix(winograd_G, g, len(HW))
ret = (gfactors * dfactors).sum(axis=-1-len(HW), dtype=dtype)
return ret  # STOP before output transform
```

---

## PHASE 4: Full Algorithm

### Modify winowrite

Restore the complete original implementation (or use backup):

```bash
cp tinygrad/schedule/rangeify.py.backup tinygrad/schedule/rangeify.py
```

### OLD Modification

Restore complete OLD implementation in tensor.py.

---

## Analysis

After running all phases, create a table:

| Phase | Operation | NEW Runtime | OLD Runtime | Slowdown | Notes |
|-------|-----------|-------------|-------------|----------|-------|
| Baseline | No wino | 0.6ms | 0.6ms | 1.0× | Reference |
| 1 | XHAT only | ? | ? | ? | Basic transform |
| 2 | + GHAT | ? | ? | ? | Both transforms |
| 3 | + Multiply/Reduce | ? | ? | **??×** | Expected bottleneck |
| 4 | + Output transform | ? | ? | ? | Should recover |

**Key Questions:**
1. Does Phase 3 show massive runtime regression? (Previous: 40×)
2. Does Phase 4 recover? (Previous: only 1.13×)
3. If so, WHY does adding output transform fix the runtime issue?

**Hypothesis:**
Phase 3 creates MHAT buffer that is never fully consumed, causing inefficient materialization.
Phase 4 adds the output transform which properly consumes MHAT and allows better fusion.

## Restore Original

```bash
cp tinygrad/schedule/rangeify.py.backup tinygrad/schedule/rangeify.py
rm tinygrad/schedule/rangeify.py.backup
```
