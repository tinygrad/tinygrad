Welcome to the docs for tinygrad. This page is for users of the tinygrad library. We also have [Developer Docs](developer.md)

As a user of tinygrad, the main class you will interact with is [Tensor](tensor.md). It functions very similarly to PyTorch, but has a bit more of a functional style. tinygrad supports [many datatypes](dtypes.md).

All operations in tinygrad are lazy, meaning they won't do anything until you realize.

tinygrad has a built in [neural network library](nn.md) with some classes, optimizers, and load/save state management.

tinygrad has a JIT to make things fast. Decorate your pure function with `TinyJit`

We have a [quickstart guide](quickstart.md) and a [showcase](showcase.md)
