# tinygrad/shape_utils.py
from typing import Tuple

def normalize_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Remove 1s from the shape tuple."""
    return tuple(d for d in shape if d != 1)

def broadcast_shape(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    """Broadcast two shapes like NumPy does."""
    la, lb = len(a), len(b)
    max_len = max(la, lb)
    a = (1,) * (max_len - la) + a
    b = (1,) * (max_len - lb) + b
    return tuple(max(x, y) for x, y in zip(a, b))

def reduce_chomp(shape: Tuple[int, ...], n: int) -> Tuple[int, ...]:
    """Reduce (drop) the n rightmost dimensions."""
    assert 0 <= n <= len(shape), f"cannot chomp {n} from {shape}"
    return shape[:-n]

def add_gpu_dims(shape: Tuple[int, ...], gpu_dims: Tuple[int, ...]) -> Tuple[int, ...]:
    """Append GPU-specific layout dimensions at the end."""
    return shape + gpu_dims

