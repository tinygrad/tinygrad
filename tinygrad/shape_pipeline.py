# tinygrad/shape_pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from tinygrad.shape_utils import normalize_shape, reduce_chomp, add_gpu_dims

@dataclass
class ShapeMeta:
    """Represents logical (yellow), reduced (red), and physical (green) shape states."""
    logical: Tuple[int, ...]
    reduced: Tuple[int, ...]
    physical: Optional[Tuple[int, ...]] = None

def normalize_shape_pipeline(
    shape: Tuple[int, ...],
    reduce_n: int = 0,
    gpu_dims: Optional[Tuple[int, ...]] = None
) -> ShapeMeta:
    """
    Full shape processing pipeline:
      1️⃣ Remove 1s (logical)
      2️⃣ Reduce rightmost dims (red)
      3️⃣ Append GPU-special dims (green)
    """
    gpu_dims = gpu_dims or ()

    # Step 1: normalize shape (remove trivial 1s)
    logical = normalize_shape(shape)

    # Step 2: assert reduction only applies to rightmost dims
    if reduce_n > 0:
        assert reduce_n <= len(logical), f"Cannot reduce {reduce_n} from {logical}"
    reduced = reduce_chomp(logical, reduce_n)

    # Step 3: append GPU-special dims last
    physical = add_gpu_dims(reduced, gpu_dims)

    return ShapeMeta(logical=logical, reduced=reduced, physical=physical)

