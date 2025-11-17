# Understanding Grid Organization

## Question: Why do we get r_16_6_6 instead of r_6_6_16?

The grid comes from the **axes we pass to the final bufferize()** call.

### Current Code (rangeify.py:286-292)

```python
# Line 286-287: MHAT bufferize
MHAT = mhat_redu.bufferize(*(tile_ranges1+other_loop_ranges_ghat+other_loop_ranges_xhat+inner6_1),
                            arg=BufferizeOpts(device=device, addrspace=AddrSpace.GLOBAL))

# Line 292-293: Final output transform
return winograd_kron(MHAT, winograd_At, tile_ranges1+other_loop_ranges_ghat+other_loop_ranges_xhat, inner6_1, device, ctx, bufferize=True)\
  .index(*other_loops_x, *other_loops_w, *[ox//4 for ox in o_axes], *[ox%4 for ox in o_axes])
```

### Axis Sizes (from DEBUG_AXES output)

```
tile_ranges1:           [16, 16]  (spatial tiles - but wait, these are 16 not 6!)
other_loop_ranges_ghat: [16]      (channels)
other_loop_ranges_xhat: []        (empty)
inner6_1:               [6, 6]    (winograd transform dims)
```

**Wait! tile_ranges1 is [16, 16] not [6, 6]!**

This is because with 64×64 input:
- Tiles of 4×4 pixels
- 64÷4 = 16 tiles per dimension
- So tile_ranges1 = [16, 16]

### What winograd_kron Does

Inside `winograd_kron(MHAT, At, [16,16,16], [6,6])`:
1. Creates `new_outer_axes` = [16, 16, 16] (copies of input outer axes)
2. Creates `result_axes` = [4, 4] (At transforms 6→4)
3. Bufferizes with: [16, 16, 16, 4, 4]
4. **Returns to caller**

Then the `.index()` call:
- Selects specific elements from the [16,16,16,4,4] buffer
- Final shape: [1, 16, 64, 64]

### Where Does Grid Come From?

The scheduler looks at the **top 3 dimensions** for the global grid:
```
Axes:        [16,    16,    16,    4,    4]
             └────┬─────┘  └──┬──┘
                  Grid         Local
```

So we get: `r_16_16_16...` → scheduler optimizes to `r_16_6_6...`?

**Mystery:** How does [16,16,16] become r_16_6_6?

## Hypothesis: Scheduler Optimizations

The scheduler may be:
1. Merging/splitting dimensions
2. Reordering for performance
3. Choosing which axes go to global vs local work

We need to understand the scheduler's grid assignment logic.

## OLD Winograd Approach

OLD uses Tensor operations (`_apply_winograd_matrix`) which:
- Creates different axis ordering at the Tensor level
- Results in different UOp graph structure
- Scheduler sees different pattern → assigns different grid

The grid comes from **how the entire computation graph is structured**, not just the final bufferize() call!

## Key Insight

We can't just reorder axes in the final bufferize() to change the grid. The grid is determined by:
1. The entire UOp graph structure
2. Which axes are "global" (parallelizable) vs "local" (sequential)
3. Scheduler heuristics for performance

To get r_6_6_16, we might need to restructure the **whole computation**, not just reorder axes.
