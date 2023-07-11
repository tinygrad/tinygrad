import time
import numpy as np
from tinygrad import nn
from tinygrad.nn import optim
from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.state import get_parameters

class Net:
    def __init__(self):
        self.conv = nn.Conv2d(3, 64, kernel_size=1)
        self.linear= nn.Linear(64*32*32, 10, bias=False)        

    def __call__(self, x):
        x = self.conv(x)
        x = x.reshape((256, 64*32*32))
        x = self.linear(x)
        return x.log_softmax()
    
if __name__=="__main__":
    x = Tensor.ones((256,3,32,32))
    y = Tensor.ones((256,10))
    net = Net()
    # did get_parameters add grad????
    optim = optim.SGD(get_parameters(net), lr=0.001)

    out = net(x)
    loss = out.mul(y).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()

