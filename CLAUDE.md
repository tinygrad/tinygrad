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

### TinyJit Behavior
TinyJit captures a schedule on the second call (cnt=1) and replays it on subsequent calls. **Critical**: The Python code inside a jitted function only runs during warmup (cnt=0,1). After that, only the captured schedule executes.

**Side effects and assigns**: If a tensor is modified via `.assign()` inside a jitted function but not included in the `realize()` call, those assigns won't be captured in the schedule. This is especially important for:
- **BatchNorm running stats** (`running_mean`, `running_var`) - These are updated via `.assign()` during forward pass but are NOT dependencies of the loss
- Any stateful tensor updated as a side effect

```python
# ❌ BROKEN with JIT - BatchNorm stats only update during warmup (2 iterations)
@TinyJit
def train_step():
    loss = model(x).mean().backward()
    Tensor.realize(loss, grads)  # running_mean.assign() not captured!

# ✅ CORRECT - explicitly realize buffers so assigns are in the schedule
@TinyJit
def train_step():
    loss = model(x).mean().backward()
    Tensor.realize(*params, *buffers, loss, grads)  # buffers includes running stats
```

**Debugging JIT issues**: If training works with `JIT=0` but fails with JIT, check if stateful tensors are being realized. You can verify ASSIGN chains:
```python
def count_assign_chain(uop, depth=0):
    if uop.op.name != 'ASSIGN': return depth
    return count_assign_chain(uop.src[0], depth+1)
print(count_assign_chain(bn.running_mean.uop))  # Should increase each step, not plateau at 2
```

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
- `JIT=0` - Disable JIT (useful for debugging JIT-related issues)

## Debugging Tips

1. **Print UOp graphs**: `print(tensor.uop)` or `print(tensor.uop.sink())`
2. **Check schedule**: `tensor.schedule()` returns list of ScheduleItems
3. **Trace graph rewrites**: Use `VIZ=1` or add print in PatternMatcher callbacks
4. **Find UOps by type**: `[u for u in uop.toposort() if u.op is Ops.SOMETHING]`
5. **JIT vs non-JIT**: If something works with `JIT=0` but not with JIT, the issue is likely unrealized side-effect tensors (see TinyJit Behavior above)
6. **Check tensor state**: `tensor.uop.op` shows current state - `Ops.BUFFER` means realized, `Ops.ASSIGN` means pending write

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
