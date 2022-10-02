<p align="center">
  <img src="https://raw.githubusercontent.com/geohot/tinygrad/master/docs/logo.png">
</p>

--------------------------------------------------------------------

![Unit Tests](https://github.com/geohot/tinygrad/workflows/Unit%20Tests/badge.svg)

For something in between a [pytorch](https://github.com/pytorch/pytorch) and a [karpathy/micrograd](https://github.com/karpathy/micrograd)

This may not be the best deep learning framework, but it is a deep learning framework.

The sub 1000 line core of it is in `tinygrad/`

Due to its extreme simplicity, it aims to be the easiest framework to add new accelerators to, with support for both inference and training. Support the simple basic ops, and you get SOTA [vision](https://arxiv.org/abs/1905.11946) `models/efficientnet.py` and [language](https://arxiv.org/abs/1706.03762) `models/transformer.py` models.

We are working on support for the Apple Neural Engine and the Google TPU in the `accel/` folder. Eventually, [we will build custom hardware](https://geohot.github.io/blog/jekyll/update/2021/06/13/a-breakdown-of-ai-chip-companies.html) for tinygrad, and it will be blindingly fast. Now, it is slow.

### Installation

```bash
git clone https://github.com/geohot/tinygrad.git
cd tinygrad
python3 setup.py develop
```

### Example

```python
from tinygrad.tensor import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
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
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

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

mlops are mid level ops, there's 15 of them. They understand memory allocation and derivatives

```
Relu, Log, Exp                          # unary ops
Sum, Max                                # reduce ops (with axis argument)
Add, Sub, Mul, Pow                      # binary ops (no broadcasting, use expand)
Reshape, Permute, Slice, Expand, Flip   # movement ops
Conv2D(NCHW)                            # processing op (Matmul is also Conv2D)
```

You no longer need to write mlops for a new accelerator

### Adding an accelerator (llops)

The autodiff stuff is all in mlops now so you can focus on the raw operations

```
Buffer                                                     # class of memory on this device
unary_op  (RELU, EXP, LOG, NEG, SIGN)                      # A -> A
reduce_op (SUM, MAX)                                       # A -> B (smaller size, B has 1 in shape)
binary_op (ADD, SUB, MUL, DIV, POW, CMPEQ)                 # A + B -> C (all the same size)
movement_op (RESHAPE, PERMUTE, PAD, SHRINK, EXPAND, FLIP)  # A -> B (different size)
processing_op (CONV)                                       # A + B -> C
```

When tinygrad moves to lazy evaluation, optimizations will happen here.

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

Run `TORCH=1 python3 examples/stable_diffusion.py`

(or without torch: `OPT=2 OPENCL=1 python3 examples/stable_diffusion.py`)

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

```bash
python3 -m pytest
```

