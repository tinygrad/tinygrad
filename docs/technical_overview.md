# Tinygrad overview
Tinygrad is a deep learning framework, that provides.
* Tensor manipulation and fast computation, using various backends.
* Auto grading.
* Lazy evaluation.

## What it is all about
At the user api level tinygrad looks like any modern tensor library, where one works with the Tensor class to perform computation.
The computation of the Tensors are deferd until the result is needed. This deference allows tinygrad to merge Tensor operations, factor out operations that do nothing, and optimize memory access paterns in the underlying hardware for speed. This allows for fast computation without making a sacrifice to developer experience.

## Structure under the hood

* Tensor: what you work with when creating Nuralnets, the operations performed on Tensor's are called hlops (High level operations)(tensor.py)
* mlops: Mid level operations, IR(intermediary representation) of the ops(tensor.py) used by tinygrad to track gradients.
* llops: Low level operations, what the computational accelerator(GPU,or other accelerator) deals with.

## Accelerators
An accelerator extends the ExplicitExecAST class, If you have a accelerator named Nuralacel it should be named something like NuralacelBuffer as it representing the backing buffer of a Tensor object. The NuralacelBuffer should have 3 methods
* fromCPU (takes data and "moves" the data from python)
* toCPU (moves data back to python)
* exec_ast (Executes a tree structure of Tensor Operations)

### exec_ast
exec_ast takes a tree structure of Tensor Operations, compiles the operations to a program for the hardware accelerator.
Then executes the program.


## Current status - jan 2023
Work is being done to verify core ideas regarding optimizations like operation fusing and memory access speedups. As that is verified, it will become more clear where the boundaries between abstraction layers should be, which will make later refactoring and cleanup easier.
