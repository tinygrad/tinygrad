from extra.ddp import DDP
if __name__ == "__main__":
  DDP.preinit() # setup everything for DDP

# it's safe to import other things now
from extra.ddp import ZeroGradCommand, ForwardBackwardCommand, AllReduceCommand, StepCommand
from tinygrad.nn import optim, Linear
from extra.training import sparse_categorical_crossentropy
from datasets import fetch_mnist
import numpy as np
import time

# setup our model
HS = 8192
class TinyNet:
  def __init__(self):
    self.l1 = Linear(784, HS, bias=False)
    self.l2 = Linear(HS, 10, bias=False)

  def __call__(self, x):
    x = self.l1(x)
    x = x.leakyrelu()
    x = self.l2(x)
    return x.log_softmax()

def model_fn():
  return TinyNet()

def optim_fn(params):
  return optim.LAMB(params, lr=0.0001)

if __name__ == "__main__":
  ddp = DDP(
    devices=["gpu:0", "gpu:1"],
    model_fn=model_fn,
    optim_fn=optim_fn,
    loss_fn=sparse_categorical_crossentropy,
  )

  X_train, Y_train, X_test, Y_test = fetch_mnist()

  for step in range(1000):
    # random sample a batch for each device
    samp = [np.random.randint(0, X_train.shape[0], size=(64)) for _ in ddp.devices]
    batches = [X_train[s] for s in samp]
    # get the corresponding labels
    labels = [Y_train[s] for s in samp]

    # zero gradients
    for cmd_queue in ddp.cmd_queues:
      cmd_queue.put(ZeroGradCommand())
    ddp.wait_barrier.wait()

    # forward & backward pass
    for cmd_queue, batch, labels in zip(ddp.cmd_queues, batches, labels):
      cmd_queue.put(ForwardBackwardCommand(batch, labels))
    ddp.wait_barrier.wait()

    losses = [ddp.out_queue.get() for _ in ddp.devices]
    print("losses", losses)

    # all reduce
    print("all reduce")
    st = time.perf_counter()
    for cmd_queue in ddp.cmd_queues:
      cmd_queue.put(AllReduceCommand())
    ddp.wait_barrier.wait()
    print(f"all reduce time: {time.perf_counter() - st:.3f}s")

    # update parameters
    for cmd_queue in ddp.cmd_queues:
      cmd_queue.put(StepCommand())
    ddp.wait_barrier.wait()
