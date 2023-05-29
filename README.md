<p align="center">
  <img src="https://raw.githubusercontent.com/geohot/tinygrad/master/docs/logo.png">
</p>

--------------------------------------------------------------------

![Unit Tests](https://github.com/geohot/tinygrad/workflows/Unit%20Tests/badge.svg)

[![tinygrad discord](https://discordapp.com/api/guilds/1068976834382925865/widget.png?style=banner2)](https://discord.gg/ZjZadyC7PK)


This project is maintained by [tiny corp](https://tinygrad.org/).

### Description :

*You like pytorch? You like micrograd? You love tinygrad! <3*

Tinygrad is something in between a [pytorch](https://github.com/pytorch/pytorch) and a [karpathy/micrograd](https://github.com/karpathy/micrograd)

*This may not be the best deep learning framework, but it is a deep learning framework.*

The sub 1000 line core of it is in `tinygrad/`

Due to its extreme simplicity, it aims to be the easiest framework to add new accelerators to, with support for both inference and training. Support the simple basic ops, and you get SOTA [vision](https://arxiv.org/abs/1905.11946) ([efficientnet](https://github.com/geohot/tinygrad/blob/master/models/efficientnet.py)) & [language](https://arxiv.org/abs/1706.03762) ([transformer](https://github.com/geohot/tinygrad/blob/master/models/transformer.py)) models.



**We are working on support for the Apple Neural Engine and the Google TPU in the `accel/` folder**. 

Eventually, [we will build custom hardware](https://geohot.github.io/blog/jekyll/update/2021/06/13/a-breakdown-of-ai-chip-companies.html) for tinygrad, and it will be blindingly fast. For now, it is slow.


### Installation

```bash
git clone https://github.com/geohot/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```



### Example 

#### Tinygrad : 
```python
from tinygrad.tensor import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.numpy())  # dz/dx
print(y.grad.numpy())  # dz/dy
```

####  Torch :

```python
import torch

x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
```

### Is tinygrad fast?

Try a matmul. See how, despite the style, it is fused into one kernel with the power of laziness.


```python
DEBUG=3 OPTLOCAL=1 python3 -c "from tinygrad.tensor import Tensor;
N = 1024; a, b = Tensor.randn(N, N), Tensor.randn(N, N);
c = (a.reshape(N, 1, N) * b.permute(1,0).reshape(1, N, N)).sum(axis=2);
print((c.numpy() - (a.numpy() @ b.numpy())).mean())"
```
Change `DEBUG=4` in above snippet to see the generated code. 


### Build Neural networks

It turns out, a decent autograd tensor library is 90% of what you need for neural networks. Add an optimizer (SGD, Adam, AdamW implemented) from tinygrad.nn.optim, write some boilerplate minibatching code, and you have all you need.

#### Example 

[Mnist](https://github.com/geohot/tinygrad/blob/master/test/models/test_mnist.py) example provided in titnygrad

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

### GPU and Accelerator Support

tinygrad supports GPUs through `PyOpenCL`.

```python
from tinygrad.tensor import Tensor
(Tensor.ones(4,4).gpu() + Tensor.ones(4,4).gpu()).cpu()
```

### Drawing Execution Graph

```bash
GRAPH=1 python3 test/models/test_mnist.py TestMNIST.test_sgd_onestep
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
--------------------------------------------------------------------

### OPS

#### hlops

hlops are syntactic sugar around mlops. They support most things torch does. Like in [tensor.py](https://github.com/geohot/tinygrad/blob/master/tinygrad/tensor.py). 

#### mlops

mlops are mid level ops. They understand derivatives. They are very simple.

```
Relu, Log, Exp, Sin                            # unary ops
Sum, Max                                       # reduce ops (with axis argument)
Maximum, Add, Sub, Mul, Pow, Div, Equal        # binary ops (no broadcasting, use expand)
Expand, Reshape, Permute, Pad, Shrink, Flip    # movement ops
```

You no longer need to write mlops for a new accelerator

#### Adding an accelerator (llops)

The autodiff stuff is all in mlops now so you can focus on the raw operations

```
Buffer                                                       # class of memory on this device
unary_op  (NOOP, EXP, LOG, CAST, SIN)                        # A -> A
reduce_op (SUM, MAX)                                         # A -> B (smaller size, B has 1 in shape)
binary_op (ADD, SUB, MUL, DIV, POW, CMPEQ, MAX)              # A + A -> A (all the same size)
movement_op (EXPAND, RESHAPE, PERMUTE, PAD, SHRINK, STRIDE)  # A -> B (different size)
fused_op [[optional]] (MULACC)                               # A * A -> B
```

### ImageNet inference

Despite being tiny, tinygrad supports the full EfficientNet. Pass in a picture to discover what it is.

```bash
python3 examples/efficientnet.py https://media.istockphoto.com/photos/hen-picture-id831791190
```

Or, if you have a webcam and cv2 installed

```bash
python3 examples/efficientnet.py webcam
```

**PROTIP**: Set `DEBUG=2` environment variable if you want to see why it's slow.

### Tinygrad supports :-
#### 1. Stable Diffusion

You might need to download the [weight](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt) of Stable Diffusion and put it into weights/

Run `python3 examples/stable_diffusion.py`

<p align="center">
  <img src="https://raw.githubusercontent.com/geohot/tinygrad/master/docs/stable_diffusion_by_tinygrad.jpg">
</p>

<p align="center">
"a horse sized cat eating a bagel"
</p>

#### 2. LLaMA

After putting the weights in weights/LLaMA, you can have a chat with Stacy. She lives inside tinygrad.

```bash
python3 examples/llama.py
```

#### 3. GANs

See `examples/mnist_gan.py`

<p align="center">
  <img src="https://raw.githubusercontent.com/geohot/tinygrad/master/docs/mnist_by_tinygrad.jpg">
</p>

### 4. Yolo

See `examples/yolov3.py`

<p align="center">
  <img src="https://raw.githubusercontent.com/geohot/tinygrad/master/docs/yolo_by_tinygrad.jpg">
</p>

**and many more!!**

### Contributing

There's a lot of interest in tinygrad lately. Here's some guidelines for contributing:

* Bugfixes are the best and always welcome! Like [this one](https://github.com/geohot/tinygrad/pull/421/files).
* If you don't understand the code you are changing, don't change it!
* All code golf PRs will be closed, but [conceptual cleanups](https://github.com/geohot/tinygrad/pull/372/files) are great.
* Features are welcome. Though if you are adding a feature, you need to include tests.
* Improving test coverage is great, with reliable non brittle tests.
