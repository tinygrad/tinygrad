<p align="center">
  <img src="https://raw.githubusercontent.com/geohot/tinygrad/master/docs/logo.png">
</p>

--------------------------------------------------------------------

![Unit Tests](https://github.com/geohot/tinygrad/workflows/Unit%20Tests/badge.svg)

[![tinygrad discord](https://discordapp.com/api/guilds/1068976834382925865/widget.png?style=banner2)](https://discord.gg/ZjZadyC7PK)

For something in between a [pytorch](https://github.com/pytorch/pytorch) and a [karpathy/micrograd](https://github.com/karpathy/micrograd)

This may not be the best deep learning framework, but it is a deep learning framework.

The sub 1000 line core of it is in `tinygrad/`

Due to its extreme simplicity, it aims to be the easiest framework to add new accelerators to, with support for both inference and training. Support the simple basic ops, and you get SOTA [vision](https://arxiv.org/abs/1905.11946) `models/efficientnet.py` and [language](https://arxiv.org/abs/1706.03762) `models/transformer.py` models.

We are working on support for the Apple Neural Engine and the Google TPU in the `accel/` folder. Eventually, [we will build custom hardware](https://geohot.github.io/blog/jekyll/update/2021/06/13/a-breakdown-of-ai-chip-companies.html) for tinygrad, and it will be blindingly fast. Now, it is slow.

This project is maintained by [tiny corp](https://tinygrad.org/).

### Installation

```bash
git clone https://github.com/geohot/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

### Contributing

There's a lot of interest in tinygrad lately. Here's some guidelines for contributing:

* Bugfixes are the best and always welcome! Like [this one](https://github.com/geohot/tinygrad/pull/421/files).
* If you don't understand the code you are changing, don't change it!
* All code golf PRs will be closed, but [conceptual cleanups](https://github.com/geohot/tinygrad/pull/372/files) are great.
* Features are welcome. Though if you are adding a feature, you need to include tests.
* Improving test coverage is great, with reliable non brittle tests.

### Example

```python
from tinygrad.tensor import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.numpy())  # dz/dx
print(y.grad.numpy())  # dz/dy
```

### Same example in torch

```python
import torch

x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
```

## Is tinygrad fast?

Try a matmul. See how, despite the style, it is fused into one kernel with the power of laziness.

```python
DEBUG=3 OPTLOCAL=1 GPU=1 python3 -c "from tinygrad.tensor import Tensor;
N = 1024; a, b = Tensor.randn(N, N), Tensor.randn(N, N);
c = (a.reshape(N, 1, N) * b.permute(1,0).reshape(1, N, N)).sum(axis=2);
print((c.numpy() - (a.numpy() @ b.numpy())).mean())"
```

Change to `DEBUG=4` to see the generated code.

## Neural networks?

It turns out, a decent autograd tensor library is 90% of what you need for neural networks. Add an optimizer (SGD, RMSprop, and Adam implemented) from tinygrad.nn.optim, write some boilerplate minibatching code, and you have all you need.

### Neural network example (from test/test_mnist.py)

```python
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

# ... and complete like pytorch, with (x,y) data

out = model.forward(x)
loss = out.mul(y).mean()
optim.zero_grad()
loss.backward()
optim.step()
```

## GPU and Accelerator Support

tinygrad supports GPUs through PyOpenCL.

```python
from tinygrad.tensor import Tensor
(Tensor.ones(4,4).gpu() + Tensor.ones(4,4).gpu()).cpu()
```

### ANE Support?! (broken)

If all you want to do is ReLU, you are in luck! You can do very fast ReLU (at least 30 MEGAReLUs/sec confirmed)

Requires your Python to be signed with `ane/lib/sign_python.sh` to add the `com.apple.ane.iokit-user-access` entitlement, which also requires `sudo nvram boot-args="amfi_get_out_of_my_way=1 ipc_control_port_options=0"`. Build the library with `ane/lib/build.sh`

In order to set boot-args and for the AMFI kext to respect that arg, run `csrutil enable --without-kext --without-nvram` in recovery mode.

```python
from tinygrad.tensor import Tensor

a = Tensor([-2,-1,0,1,2]).ane()
b = a.relu()
print(b.cpu())
```

Warning: do not rely on the ANE port. It segfaults sometimes. So if you were doing something important with tinygrad and wanted to use the ANE, you might have a bad time.

### hlops (in tensor.py)

hlops are syntactic sugar around mlops. They support most things torch does.

### mlops

mlops are mid level ops. They understand derivatives. They are very simple.

```
Log, Exp                                       # unary ops
Sum, Max                                       # reduce ops (with axis argument)
Maximum, Add, Sub, Mul, Pow, Div, Equal        # binary ops (no broadcasting, use expand)
Expand, Reshape, Permute, Pad, Shrink, Flip    # movement ops
```

You no longer need to write mlops for a new accelerator

### Adding an accelerator (llops)

The autodiff stuff is all in mlops now so you can focus on the raw operations

```
Buffer                                                     # class of memory on this device
unary_op  (NOOP, NEG, NOT, EXP, LOG)                       # A -> A
reduce_op (SUM, MAX)                                       # A -> B (smaller size, B has 1 in shape)
binary_op (ADD, SUB, MUL, DIV, POW, CMPEQ, MAX)            # A + A -> A (all the same size)
movement_op (EXPAND, RESHAPE, PERMUTE, PAD, SHRINK, FLIP)  # A -> B (different size)
fused_op [[optional]] (MULACC)                             # A * A -> B
```

## ImageNet inference

Despite being tiny, tinygrad supports the full EfficientNet. Pass in a picture to discover what it is.

```bash
ipython3 examples/efficientnet.py https://media.istockphoto.com/photos/hen-picture-id831791190
```

Or, if you have a webcam and cv2 installed

```bash
ipython3 examples/efficientnet.py webcam
```

PROTIP: Set "GPU=1" environment variable if you want this to go faster.

PROPROTIP: Set "DEBUG=1" environment variable if you want to see why it's slow.

### tinygrad supports Stable Diffusion!

You might need to download the [weight](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt) of Stable Diffusion and put it into weights/

Run `GPU=1 python3 examples/stable_diffusion.py`

<p align="center">
  <img src="https://raw.githubusercontent.com/geohot/tinygrad/master/docs/stable_diffusion_by_tinygrad.jpg">
</p>

<p align="center">
"a horse sized cat eating a bagel"
</p>

### tinygrad supports GANs

See `examples/mnist_gan.py`

<p align="center">
  <img src="https://raw.githubusercontent.com/geohot/tinygrad/master/docs/mnist_by_tinygrad.jpg">
</p>

### tinygrad supports yolo

See `examples/yolov3.py`

<p align="center">
  <img src="https://raw.githubusercontent.com/geohot/tinygrad/master/docs/yolo_by_tinygrad.jpg">
</p>

## The promise of small

tinygrad will always be below 1000 lines. If it isn't, we will revert commits until tinygrad becomes smaller.

### Drawing Execution Graph

* Nodes are Tensors
* Black edge is a forward pass
* Blue edge is a backward pass
* Red edge is data the backward pass depends on
* Purple edge is intermediates created in the forward

```bash
GRAPH=1 python3 test/test_mnist.py TestMNIST.test_sgd_onestep
# requires dot, outputs /tmp/net.svg
```

### Running tests

For more examples on how to run the full test suite please refer to the [CI workflow](.github/workflows/test.yml).

```bash
python3 -m pip install -e '.[testing]'
python3 -m pytest
python3 -m pytest -v -k TestTrain
python3 ./test/models/test_train.py TestTrain.test_efficientnet
```

