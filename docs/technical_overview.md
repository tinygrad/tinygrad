# Tinygrad overview
Tinygrad is a deep learning framework, that provides.
* Tensor manipulation and fast computation, using various backends.
* Auto grading.
* Lazy evaluation.

## What it is all about
At the user api level tinygrad looks like any modern tensor library, where one works with the Tensor class to perform computation.
The computation of the Tensors are deferd until the result is needed. This deference allows tinygrad to merge Tensor operations, factor out operations that do nothing, and optimize memory access paterns in the underlying hardware for speed. This allows for fast computation while without making a sacrifice to the developer experience.

## Structure under the hood

* Tensor: what you work with when creating Nuralnets, the operations performed on Tensor's are called hlops (High level operations)(tensor.py)
* mlops: Mid level operations, IR(intermediary representation) of the ops(tensor.py) used by tinygrad to track gradients and optimize for speed.
* llops Low level operations, what the computational accelerator(GPU,or other accelerator) deals with.
