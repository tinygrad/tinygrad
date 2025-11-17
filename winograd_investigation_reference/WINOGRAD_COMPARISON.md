# Winograd NEW vs OLD Comparison Scripts

## Quick Performance Table

Run this to see the actual performance comparison (correcting the misleading test averages):

```bash
python3 wino_performance_table.py
```

**Key Finding:** For most practical shapes (Cin > 1), the OLD implementation is faster in BOTH compile time and runtime.

## Visual Comparison with VIZ

Use `wino_viz_compare.py` to visually inspect the compiled kernels for a large representative shape (64×64×64).

### Run BASELINE (no Winograd):
```bash
WINO=0 WINO_OLD=0 VIZ=1 python3 wino_viz_compare.py
```

### Run NEW (unified kron in schedule/rangeify.py):
```bash
WINO=1 WINO_OLD=0 VIZ=1 python3 wino_viz_compare.py
```

### Run OLD (tensor ops in tensor.py):
```bash
WINO=0 WINO_OLD=1 VIZ=1 python3 wino_viz_compare.py
```

## Expected Performance (64×64×64 shape)

From benchmarks:
- **BASELINE**: Compile ~2ms, Run ~1.2ms
- **NEW (unified kron)**: Compile ~137ms (64×), Run ~7.1ms (0.16× speedup = **6× slower**)
- **OLD (tensor.py)**: Compile ~113ms (53×), Run ~2.9ms (0.40× speedup = **2.4× slower**)

**Winner: OLD is ~1.2× faster to compile and ~2.4× faster at runtime**

## Why OLD is Faster

1. **Compile Time**: OLD operates at tensor level before scheduling → less graph rewrite overhead
2. **Runtime**: OLD gets better fusion opportunities in the tensor graph
3. **NEW**: Operates at schedule/UOp level → more explicit range management → less optimization

## What NEW Gives Us

1. **~80 lines less code** (single unified implementation)
2. **More maintainable** (one place to optimize, not three)
3. **Cleaner abstraction** (explicit range handling)
4. **Better foundation** for future schedule-level optimizations

## Trade-off Decision

- **Use OLD** if runtime performance is critical (current recommendation)
- **Use NEW** if code maintainability/future extensibility matters more
- **Both are correct** (numerical errors < 1e-3)

## Example VIZ Session

```bash
# Compare kernels side-by-side:

# Terminal 1 - Run OLD
WINO=0 WINO_OLD=1 VIZ=1 python3 wino_viz_compare.py

# Terminal 2 - Run NEW
WINO=1 WINO_OLD=0 VIZ=1 python3 wino_viz_compare.py

# Look for:
# - Number of kernels generated
# - Buffer sizes and memory usage
# - Loop structure and fusion opportunities
# - AST complexity
```

## Debugging Performance

If you want to understand WHY one is faster, try:

```bash
# See detailed kernel scheduling
DEBUG=2 WINO=1 WINO_OLD=0 python3 wino_viz_compare.py

# Track compilation stages
DEBUG=3 WINO=0 WINO_OLD=1 python3 wino_viz_compare.py
```
