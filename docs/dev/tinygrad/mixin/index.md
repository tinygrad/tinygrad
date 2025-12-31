# Mixin

The `tinygrad/mixin/` directory contains mixin classes that add functionality to other classes (like `Tensor` or `UOp`).

## `math.py`

Defines `MathMixin` (or similar, sometimes integrated into `OpMixin`).
It adds standard math operations (`__add__`, `__sub__`, `__mul__`, `sin`, `log`, etc.) that dispatch to underlying methods (e.g., `_binop`, `_apply_uop`).

## `movement.py`

Defines logic related to movement operations (reshape, permute, etc.).
- **`_align_left`**: Helper for broadcasting shapes.
