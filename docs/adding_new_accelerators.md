# Adding a new accelerator to tinygrad

It's pretty easy to add a new accelerator to tinygrad. All you need to do is implement a total of 27 (optionally 28) low level ops. Then tinygrad takes care of the rest, handling derivatives and syntactic sugar.

## llops

These are the ops that you must implement for your accelerator of choice. Compiled Accelerators do not need to implement movement_ops, as they are handled by the ShapeTracker.
```
Buffer                                                       # class of memory on this device
unary_op  (NOOP, EXP2, LOG2, CAST, SIN, SQRT)                # A -> A
reduce_op (SUM, MAX)                                         # A -> B (smaller size, B has 1 in shape)
binary_op (ADD, SUB, MUL, DIV, CMPEQ, MAX)                   # A + A -> A (all the same size)
movement_op (EXPAND, RESHAPE, PERMUTE, PAD, SHRINK, STRIDE)  # A -> B (different size)
load_op   (EMPTY, RAND, CONST, FROM, CONTIGUOUS, CUSTOM)     # -> A   (initialize data on device)
ternary_op (WHERE)                                           # A, A, A -> A
ternary_op [[optional]] (MULACC)                             # A * A -> B
```

## mlops

These are the mid level ops that handle the derivatives.
```
Relu, Log, Exp, Sin                            # unary ops
Sum, Max                                       # reduce ops (with axis argument)
Maximum, Add, Sub, Mul, Pow, Div, Equal        # binary ops (no broadcasting, use expand)
Expand, Reshape, Permute, Pad, Shrink, Flip    # movement ops
Where                                          # ternary ops
```
These are implemented in [mlops.py](/tinygrad/mlops.py).

## hlops

These are the syntax sugar. They are built on top of the mlops and support most of the things that you could expect from a tensor library.

These are implemented in [tensor.py](/tinygrad/tensor.py).
