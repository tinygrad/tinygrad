# Claude Code Guide for tinygrad

## Architecture Overview

tinygrad compiles tensor operations into optimized kernels. The pipeline:

1. **Tensor** (`tensor.py`) - User-facing API, creates UOp graph
2. **UOp** (`uop/ops.py`) - Unified IR for all operations (both tensor and kernel level)
3. **Schedule** (`engine/schedule.py`, `schedule/`) - Converts tensor UOps to kernel UOps
4. **Codegen** (`codegen/`) - Converts kernel UOps to device code
5. **Runtime** (`runtime/`) - Device-specific execution

## Key Concepts

### UOp (Universal Operation)
Everything is a UOp - tensors, operations, buffers, kernels. Key properties:
- `op`: The operation type (Ops enum)
- `dtype`: Data type
- `src`: Tuple of source UOps
- `arg`: Operation-specific argument
- `tag`: Optional tag for graph transformations

UOps are **immutable and cached** - creating the same UOp twice returns the same object (ucache).

### PatternMatcher
Used extensively for graph transformations:
```python
pm = PatternMatcher([
  (UPat(Ops.ADD, src=(UPat.cvar("x"), UPat.cvar("x"))), lambda x: x * 2),
])
result = graph_rewrite(uop, pm)
```

### Schedule Cache
Schedules are cached by graph structure. BIND nodes (variables with bound values) are unbound before cache key computation so different values hit the same cache.

## Directory Structure

```
tinygrad/
├── tensor.py          # Tensor class, user API
├── device.py          # Buffer, device management
├── dtype.py           # Data types
├── helpers.py         # Utilities, environment vars
├── uop/
│   ├── ops.py         # UOp class, Ops enum, PatternMatcher
│   ├── spec.py        # UOp type verification
│   └── symbolic.py    # Symbolic math simplification
├── engine/
│   ├── schedule.py    # Schedule creation, caching
│   ├── realize.py     # Tensor realization
│   ├── jit.py         # JIT compilation
│   └── memory.py      # Memory planning
├── schedule/
│   ├── rangeify.py    # Convert movements to ranges
│   └── indexing.py    # Index calculations
├── codegen/
│   ├── kernel.py      # Kernel optimization
│   └── uopgraph.py    # UOp graph transformations
├── renderer/          # Code generation (CUDA, Metal, etc.)
└── runtime/           # Device backends
```

## Testing

```bash
# Run specific test
python -m pytest test/unit/test_schedule_cache.py -xvs

# Run with timeout
python -m pytest test/test_symbolic_ops.py -x --timeout=60

# Debug with print
DEBUG=2 python -m pytest test/test_schedule.py::test_name -xvs

# Visualize UOp graphs
VIZ=1 python -c "from tinygrad import Tensor; Tensor.ones(10).sum().realize()"
```

## Common Environment Variables

- `DEBUG=1-4` - Increasing verbosity
- `VIZ=1` - Enable graph visualization
- `SPEC=1` - Enable UOp spec verification
- `NOOPT=1` - Disable optimizations
- `DEVICE=CPU/CUDA/AMD/METAL` - Set default device

## Debugging Tips

1. **Print UOp graphs**: `print(tensor.uop)` or `print(tensor.uop.sink())`
2. **Check schedule**: `tensor.schedule()` returns list of ScheduleItems
3. **Trace graph rewrites**: Use `VIZ=1` or add print in PatternMatcher callbacks
4. **Find UOps by type**: `[u for u in uop.toposort() if u.op is Ops.SOMETHING]`

## Style Notes

- 2-space indentation, 150 char line limit
- PatternMatchers should be defined at module level (slow to construct)
- Prefer `graph_rewrite` over manual graph traversal
- UOp methods like `.replace()` preserve tags unless explicitly changed
- Use `.rtag(value)` to add tags to UOps

## Common Patterns

### Graph Transformation
```python
def my_transform(ctx, x):
  # Return new UOp or None to skip
  return x.replace(arg=new_arg)

pm = PatternMatcher([
  (UPat(Ops.SOMETHING, name="x"), my_transform),
])
result = graph_rewrite(input_uop, pm, ctx={})
```

### Finding Variables
```python
# Get all variables in a UOp graph
variables = uop.variables()

# Get bound variable values
var, val = bind_uop.unbind()
```

### Shape Handling
```python
# Shapes can be symbolic (contain UOps)
shape = tensor.shape  # tuple[sint, ...] where sint = int | UOp
```

## Performance Optimization

When optimizing tinygrad internals:

1. **Measure wall time, not just call counts** - Reducing `graph_rewrite` calls doesn't always improve wall time. The overhead of conditional checks can exceed the cost of the operation being skipped.

2. **Profile each optimization individually** - Run benchmarks with and without each change to measure actual impact. Use `test/external/external_benchmark_schedule.py` for schedule/rewrite timing.

3. **Early exits in hot paths are effective** - Simple checks like `if self.op is Ops.CONST: return self` in `simplify()` can eliminate many unnecessary `graph_rewrite` calls.

4. **`graph_rewrite` is expensive** - Each call has overhead even for small graphs. Avoid calling it when the result is trivially known (e.g., simplifying a CONST returns itself).

5. **Beware iterator overhead** - Checks like `all(x.op is Ops.CONST for x in self.src)` can be slower than just running the operation, especially for small sequences.

6. **Verify cache hit rates before adding/keeping caches** - Measure actual hit rates with real workloads. A cache with 0% hit rate is pure overhead (e.g., `pm_cache` was removed because the algorithm guarantees each UOp is only passed to `pm_rewrite` once).
