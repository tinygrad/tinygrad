# Diagonal Operations Fix - Key Insights

## Problem
5 diagonal tests were failing in torch backend, returning all zeros instead of diagonal matrices.

## Root Cause
PyTorch's `torch.diag()` decomposes into `aten.diag_embed` which relies on:
1. `prims.mask_tensor` - a primitive operation to mask values
2. The decomposition wasn't registered for our backend
3. Backward pass needed `aten.diagonal_backward` decomposition
4. `diagonal_backward` uses `aten.diagonal_scatter` which wasn't implemented

## Solution
Three changes needed:

### 1. Register `prims::mask_tensor` implementation
```python
@torch.library.impl("prims::mask_tensor", "privateuseone")
def mask_tensor(mask: torch.Tensor, t: torch.Tensor):
  mask, t = unwrap(mask), unwrap(t)
  return wrap(t.logical_and(mask) if t.dtype == dtypes.bool else Tensor.where(mask, t, 0))
```

### 2. Add decompositions to decomposition list
```python
decomps = [
  # ... other decomps ...
  aten.diag_embed,        # Forward pass: 1D vector -> 2D diagonal matrix
  aten.diagonal_backward, # Backward pass: gradients for diagonal extraction
]
```

### 3. Implement `aten.diagonal_scatter`
```python
"aten.diagonal_scatter": lambda input, src, offset=0, dim1=0, dim2=1: (
  input + src.diag() if offset == 0 and dim1 == 0 and dim2 == 1 and input.ndim == 2 
  else NotImplemented
),
```

## Key Insights

1. **No custom diagonal implementation needed** - At commit c2e63aef9 there was NO custom diagonal code and tests passed. PyTorch's decompositions are sufficient.

2. **Decompositions aren't automatic** - Even though PyTorch has decompositions defined, they must be explicitly registered via `get_decompositions()` for custom backends.

3. **Prims vs Aten** - Primitive operations (`prims::`) need different registration than aten operations. Use `@torch.library.impl("prims::...", "privateuseone")` directly, not in the `tiny_backend` dict.

4. **mask_tensor is critical** - The `diag_embed` decomposition creates a zero matrix then uses `mask_tensor(cond, values)` which is essentially `torch.where(cond, values, 0)` to fill in the diagonal.

5. **Backward pass needs separate handling** - Forward decompositions don't automatically provide backward support. Need to register `diagonal_backward` decomposition separately.

6. **diagonal_scatter is the inverse** - `diagonal_scatter` fills diagonal of a matrix with values. It's used in the backward pass to scatter gradients back to the diagonal positions.

7. **Tinygrad's native diag works** - `Tensor.diag()` in tinygrad works perfectly. The issue was purely in the PyTorch decomposition path.

## Results
- **Before**: 5 tests failing (diagonal forward and backward tests)
- **After**: All 15 diagonal tests passing ✓

### Fixed Tests
1. ✓ `test_diag_1d_input` - Forward: 1D → 2D diagonal matrix
2. ✓ `test_diag_1d_still_works` - Forward: 1D → 2D diagonal matrix  
3. ✓ `test_diag_vector_to_matrix` - Forward: vector to diagonal matrix
4. ✓ `test_diag_2d_to_1d_backward` - Backward: gradient for 2D → 1D extraction
5. ✓ `test_diagonal_backward_gradient_values` - Backward: gradient values for diagonal

Plus 10 other diagonal-related tests that were already passing.
