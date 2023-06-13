from extra.ddp import DDP
if __name__ == "__main__":
  DDP.preinit() # setup everything for DDP

# it's safe to import other things now
from tinygrad.nn import optim, Linear
from extra.training import sparse_categorical_crossentropy
from datasets import fetch_mnist
import numpy as np

# setup our model
HS = 8192
class TinyNet:
  def __init__(self):
    self.l1 = Linear(784, HS)
    self.hl1 = Linear(HS, HS)
    self.l2 = Linear(HS, 10)

  def __call__(self, x):
    x = self.l1(x)
    x = x.leakyrelu()
    x = self.hl1(x)
    x = x.leakyrelu()
    x = self.l2(x)
    return x.log_softmax()

# setup our ddp world
def model_fn():
  return TinyNet()

def optim_fn(params):
  return optim.LAMB(params, lr=0.0001)

if __name__ == "__main__":
  ddp = DDP(
    devices=["gpu:0", "gpu:1", "gpu:0", "gpu:1"],
    model_fn=model_fn,
    optim_fn=optim_fn,
    loss_fn=sparse_categorical_crossentropy,
  )
  print(ddp)

  X_train, Y_train, X_test, Y_test = fetch_mnist()

  for step in range(1000):
    # random sample a batch for each device
    samp = [np.random.randint(0, X_train.shape[0], size=(8192)) for _ in ddp.devices]
    batches = [X_train[s] for s in samp]
    # get the corresponding labels
    labels = [Y_train[s] for s in samp]

    # zero gradients
    ddp.zero_grad()

    # forward pass
    loss = ddp.forward(batches, labels)
    print("loss", loss)

    # backward pass
    ddp.backward()

    # update parameters
    ddp.step()
