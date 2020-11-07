<p align="center">
  <img src="https://raw.githubusercontent.com/geohot/tinygrad/master/docs/logo.png">
</p>

--------------------------------------------------------------------

![Unit Tests](https://github.com/geohot/tinygrad/workflows/Unit%20Tests/badge.svg)

For something in between a [pytorch](https://github.com/pytorch/pytorch) and a [karpathy/micrograd](https://github.com/karpathy/micrograd)

This may not be the best deep learning framework, but it is a deep learning framework.

The Tensor class is a wrapper around a numpy array, except it does Tensor things.

Also a city in Russia.

### Installation

```bash
pip3 install tinygrad --upgrade
```

### Example

```python
from tinygrad.tensor import Tensor

x = Tensor.eye(3)
y = Tensor([[2.0,0,-2.0]])
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

### Neural networks?

It turns out, a decent autograd tensor library is 90% of what you need for neural networks. Add an optimizer (SGD, RMSprop, and Adam implemented) from tinygrad.optim, write some boilerplate minibatching code, and you have all you need.

### Neural network example (from test/test_mnist.py)

```python
from tinygrad.tensor import Tensor
import tinygrad.optim as optim
from tinygrad.utils import layer_init_uniform

class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor(layer_init_uniform(784, 128))
    self.l2 = Tensor(layer_init_uniform(128, 10))

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

model = TinyBobNet()
optim = optim.SGD([model.l1, model.l2], lr=0.001)

# ... and complete like pytorch, with (x,y) data

out = model.forward(x)
loss = out.mul(y).mean()
loss.backward()
optim.step()
```

### GPU Support?!

tinygrad supports GPUs through PyOpenCL. Not all ops are supported yet on the backward pass.

```python
from tinygrad.tensor import Tensor
(Tensor.ones(4,4).cuda() + Tensor.ones(4,4).cuda()).cpu()
```

### ImageNet inference

Despite being tiny, tinygrad supports the full EfficientNet. Pass in a picture to discover what it is.

```bash
python3 examples/efficientnet.py https://upload.wikimedia.org/wikipedia/commons/4/41/Chicken.jpg
```

Or, if you have a webcam and pygame+cv2 installed

```bash
python3 examples/efficientnet.py webcam
```

PROTIP: Set "GPU=1" environment variable if you want this to go faster.

### The promise of small

tinygrad will always be below 1000 lines. If it isn't, we will revert commits until tinygrad becomes smaller.

### Running tests

```bash
python -m pytest
```

### TODO

* Train an EfficientNet on ImageNet
  * Make broadcasting work on the backward pass (simple please)
  * EfficientNet backward pass
  * Tensors on GPU (GPU support, must support Mac)
* Make tinygrad work on comma two and run driving model
* Reduce code
* Increase speed
* Add features

