<div align="center">

[![logo](https://raw.githubusercontent.com/geohot/tinygrad/master/docs/logo.png)](https://tinygrad.org)

tinygrad: For something between [PyTorch](https://github.com/pytorch/pytorch) and [karpathy/micrograd](https://github.com/karpathy/micrograd). Maintained by [tiny corp](https://tinygrad.org).

<h3>

[Homepage](https://github.com/geohot/tinygrad) | [Documentation](/docs) | [Examples](/examples) | [Showcase](/docs/showcase.md) | [Discord](https://discord.gg/ZjZadyC7PK)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/geohot/tinygrad)](https://github.com/geohot/tinygrad/stargazers)
[![Unit Tests](https://github.com/geohot/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/geohot/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)
[![Lines of code](https://img.shields.io/tokei/lines/github/geohot/tinygrad)](https://github.com/geohot/tinygrad)

</div>

---

This may not be the best deep learning framework, but it is a deep learning framework.

Due to its extreme simplicity, it aims to be the easiest framework to add new accelerators to, with support for both inference and training.

Eventually, we will have a [tinygrad accelerator](https://geohot.github.io/blog/jekyll/update/2021/06/13/a-breakdown-of-ai-chip-companies.html), then tinygrad will be ***fast***. But, for now, it is slow.

## Features

### LLaMA and Stable Diffusion

tinygrad can run [LLaMA](/docs/showcase.md#llama) and [Stable Diffusion](/docs/showcase.md#stable-diffusion)!

### Laziness

Try a matmul. See how, despite the style, it is fused into one kernel with the power of laziness.

```sh
DEBUG=3 OPTLOCAL=1 python3 -c "from tinygrad.tensor import Tensor;
N = 1024; a, b = Tensor.rand(N, N), Tensor.rand(N, N);
c = (a.reshape(N, 1, N) * b.permute(1,0).reshape(1, N, N)).sum(axis=2);
print((c.numpy() - (a.numpy() @ b.numpy())).mean())"
```

And we can change `DEBUG` to `4` to see the generated code.

### Neural networks

As it turns out, 90% of what you need for neural networks are a decent autograd/tensor library.
Throw in an optimizer, a data loader, and some compute, and you have all you need.

#### Neural network example (from test/models/test_mnist.py)

```py
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim

class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor.uniform(784, 128)
    self.l2 = Tensor.uniform(128, 10)

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).log_softmax()

model = TinyBobNet()
optim = optim.SGD([model.l1, model.l2], lr=0.001)

# ... complete data loader here

out = model.forward(x)
loss = out.mul(y).mean()
optim.zero_grad()
loss.backward()
optim.step()
```

## Accelerators

tinygrad already supports numerous accelerators, including:

- [x] CPU
- [x] GPU (OpenCL)
- [x] C Code (Clang)
- [x] LLVM
- [x] METAL
- [x] CUDA
- [x] Triton
- [x] PyTorch

And it is easy to add more! Your accelerator of choice only needs to support a total of 20 (optionally 21) low level ops.
More information can be found in the [documentation for adding new accelerators](/docs/adding_new_accelerators.md).

## Installation

The current recommended way to install tinygrad is from source.

### From source

```sh
git clone https://github.com/geohot/tinygrad.git
cd tinygrad
python3 -m pip install -e . # or `py3 -m pip install -e .` if you are on windows
```
Don't forget the `.` at the end!

## Documentation

Documentation along with a quick start guide can be found in the [docs/](/docs) directory.

### Quick example comparing to PyTorch

```py
from tinygrad.tensor import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.numpy())  # dz/dx
print(y.grad.numpy())  # dz/dy
```

The same thing but in PyTorch:
```py
import torch

x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.numpy())  # dz/dx
print(y.grad.numpy())  # dz/dy
```

## Contributing

There has been a lot of interest in tinygrad lately. Here are some basic guidelines for contributing:

- Bug fixes are the best and always welcome! Like [this one](https://github.com/geohot/tinygrad/pull/421/files).
- If you don't understand the code you are changing, don't change it!
- All code golf PRs will be closed, but [conceptual cleanups](https://github.com/geohot/tinygrad/pull/372/files) are great.
- Features are welcome. Though if you are adding a feature, you need to include tests.
- Improving test coverage is great, with reliable non-brittle tests.

Additional guidelines can be found in [CONTRIBUTING.md](/CONTRIBUTING.md).

### Running tests

For more examples on how to run the full test suite please refer to the [CI workflow](.github/workflows/test.yml).

Some examples:
```sh
python3 -m pip install -e '.[testing]'
python3 -m pytest
python3 -m pytest -v -k TestTrain
python3 ./test/models/test_train.py TestTrain.test_efficientnet
```
