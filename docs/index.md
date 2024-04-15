Welcome to the docs for tinygrad. This page is for users of the tinygrad library. We also have [developer docs](developer.md)

tinygrad is not 1.0 yet, but it will be soon. The API has been pretty stable for a while.

## tinygrad Usage

The main class you will interact with is [Tensor](tensor.md). It functions very similarly to PyTorch, but has a bit more of a functional style. tinygrad supports [many datatypes](dtypes.md).  All operations in tinygrad are lazy, meaning they won't do anything until you realize.

* tinygrad has a built in [neural network library](nn.md) with some classes, optimizers, and load/save state management.
* tinygrad has a JIT to make things fast. Decorate your pure function with `TinyJit`
* tinygrad has amazing support for multiple GPUs, allowing you to shard your Tensors with `Tensor.shard`

To understand what training looks like in tinygrad, you should read `beautiful_mnist.py`

We have a [quickstart guide](quickstart.md) and a [showcase](showcase.md)

## Differences from PyTorch

If you are migrating from PyTorch, welcome. We hope you will find tinygrad both familiar and somehow more "correct feeling"

### tinygrad doesn't have nn.Module

There's nothing special about a "Module" class in tinygrad, it's just a normal class. `get_parameter`

### tinygrad is functional

<!-- link these methods -->

In tinygrad, you can do `x.conv2d(w, b)` or `x.sparse_categorical_cross_entropy(y)`

### tinygrad is lazy

When you do `a+b` in tinygrad, nothing happens.

### tinygrad requires @TinyJIT to be fast

PyTorch spends a lot of development effort to make dispatch very fast. tinygrad doesn't. We have a simple decorator that will replay the kernels used in the decorated function.