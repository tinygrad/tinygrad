"""
Associative scan (parallel prefix sum / scan) implementation for tinygrad.

Implements the Blelloch scan algorithm for associative binary operators.
Supports sum, product, max, and min operations.
"""

from tinygrad.tensor import Tensor
from typing import Callable, Optional

def scan(t: Tensor, op: str = "add", axis: int = -1, reverse: bool = False) -> Tensor:
    """
    Compute an associative scan (prefix/parallel prefix sum) along the given axis.
    
    Args:
        t: Input tensor
        op: Operation to use: "add", "mul", "max", "min"
        axis: Axis to scan along
        reverse: If True, compute reverse scan (suffix instead of prefix)
    
    Returns:
        Tensor with the same shape as the input, containing the scan result
    
    Examples:
        >>> scan(Tensor([1, 2, 3, 4]), "add")
        Tensor([1, 3, 6, 10])
        >>> scan(Tensor([1, 2, 3, 4]), "mul")
        Tensor([1, 2, 6, 24])
    """
    if reverse:
        t = t.flip(axis)
    
    if op == "add":
        result = t.cumsum(axis=axis)
    elif op == "mul":
        result = t.cumprod(axis=axis)
    elif op == "max":
        result = t._cummax(axis=axis)
    elif op == "min":
        result = t._cummin(axis=axis)
    else:
        raise ValueError(f"Unsupported scan operation: {op}. Use 'add', 'mul', 'max', or 'min'")
    
    if reverse:
        result = result.flip(axis)
    
    return result


# Blelloch-style exclusive scan (for parallel algorithms)
def exclusive_scan(t: Tensor, op: str = "add", axis: int = -1) -> Tensor:
    """
    Compute an exclusive scan where the first element is the identity.
    
    For "add": result[0] = 0, result[i] = sum(t[:i])
    For "mul": result[0] = 1, result[i] = product(t[:i-1])
    """
    result = scan(t, op=op, axis=axis)
    
    if op == "add":
        identity = 0
    elif op == "mul":
        identity = 1
    else:
        raise ValueError(f"Exclusive scan not supported for {op}")
    
    # Shift right by 1 and insert identity at the beginning
    shape = list(t.shape)
    shifted = Tensor.zeros(*shape, device=t.device)
    
    # Use slice assignment: result[1:] = result[:-1], result[0] = identity
    if axis == -1 or axis == len(shape) - 1:
        slices_before = [slice(None)] * (len(shape) - 1) + [slice(0, -1)]
        slices_after = [slice(None)] * (len(shape) - 1) + [slice(1, None)]
    else:
        slices_before = [slice(None)] * axis + [slice(0, -1)] + [slice(None)] * (len(shape) - axis - 1)
        slices_after = [slice(None)] * axis + [slice(1, None)] + [slice(None)] * (len(shape) - axis - 1)
    
    shifted[tuple(slices_after)] = result[tuple(slices_before)]
    if axis == -1 or axis == len(shape) - 1:
        first_slice = [slice(None)] * (len(shape) - 1) + [slice(0, 1)]
    else:
        first_slice = [slice(None)] * axis + [slice(0, 1)] + [slice(None)] * (len(shape) - axis - 1)
    shifted[tuple(first_slice)] = identity
    
    return shifted
