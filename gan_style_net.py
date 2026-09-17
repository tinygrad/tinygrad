import tinygrad.nn as nn
from tinygrad import Tensor

net1, net2 = nn.Linear(10, 10), nn.Linear(10, 10)
optim1, optim2 = (
    nn.optim.Adam(nn.state.get_parameters(net1)),
    nn.optim.Adam(nn.state.get_parameters(net2)),
)

inp1 = Tensor.randn(5, 10)
with Tensor.train():
    out1 = net1(inp1)

    out2 = net2(out1.detach())
    optim2.zero_grad()
    out2.sum().backward()
    optim2.step()

    optim1.zero_grad()
    out1.sum().backward()
    optim1.step()