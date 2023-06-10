from tinygrad.lazy import Device, LazyBuffer
from tinygrad.ops import LoadOps
from tinygrad.tensor import Tensor
from datasets import fetch_mnist
from tinygrad.nn import optim
from tinygrad.nn import Linear
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import time

Tensor.training = True

class TinyNet:
  def __init__(self):
    self.l1 = Linear(784, 1024, bias=False)
    self.hidden = [Linear(1024, 1024) for _ in range(3)]
    self.l2 = Linear(1024, 10, bias=False)

  def __call__(self, x):
    x = self.l1(x)
    x = x.leakyrelu()
    for h in self.hidden:
      x = h(x)
      x = x.leakyrelu()
    x = self.l2(x)
    return x.log_softmax()

def net_to(net, device):
  for p in optim.get_parameters(net):
    device = Device.canonicalize(device)
    p.lazydata = p.lazydata if p.device == device else LazyBuffer.loadop(LoadOps.FROM, p.shape, p.dtype, device, src=p.lazydata)
    p.realize() # TODO: do we need to realize here?
  return net

X_train, Y_train, X_test, Y_test = fetch_mnist()

def sparse_categorical_crossentropy(out, Y):
  num_classes = out.shape[-1]
  YY = Y.flatten().astype(np.int32)
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  # correct loss for NLL, torch NLL loss returns one per row
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y, device=out.device)
  return out.mul(y).mean()

devices = ["cpu", "llvm", "cpu"]

# create a copy of the network on each device
nets = [net_to(TinyNet(), device) for device in devices]

# create a copy of the optimizer on each device
opts = [net_to(optim.Adam(optim.get_parameters(net), lr=3e-4), device) for net, device in zip(nets, devices)]

# this is the command class that will be sent to the process
class Command:
  def __init__(self): ...

class ZeroGradCommand(Command):
  def __init__(self):
    super(ZeroGradCommand, self).__init__()

class ForwardBackwardCommand(Command):
  def __init__(self, x, y):
    super(ForwardBackwardCommand, self).__init__()
    self.x, self.y = x, y

class AllReduceCommand(Command):
  def __init__(self, num_devices):
    super(AllReduceCommand, self).__init__()
    self.num_devices = num_devices

class StepCommand(Command):
  def __init__(self):
    super(StepCommand, self).__init__()

# this is a process to just manage one instance of the network on one device
class NetDeviceProcess(mp.Process):
  def __init__(self, net, lossfn, opt, devices, rank, wait_barrier, ring_queues, shared_cache, out_queue):
    super(NetDeviceProcess, self).__init__()
    self.net = net
    self.lossfn = lossfn
    self.opt = opt
    self.devices = devices
    self.rank = rank
    self.device = Device.canonicalize(devices[rank])

    self.wait_barrier = wait_barrier
    self.cmd_queue = mp.Queue()
    self.our_queue = ring_queues[rank]
    self.next_queue = ring_queues[(rank + 1) % len(devices)]
    self.shared_cache = shared_cache
    self.out_queue = out_queue

  def run(self):
    while True:
      cmd = self.cmd_queue.get()
      if cmd.__class__ == ZeroGradCommand:
        self.opt.zero_grad()
      elif cmd.__class__ == ForwardBackwardCommand:
        # rebuild the tensor from shared memory
        x = Tensor(cmd.x, device=self.device, requires_grad=False)
        print(x.realize())
        out = self.net(x).realize()
        loss = self.lossfn(out, cmd.y)
        loss.backward()
        # realize grads
        for p in optim.get_parameters(self.net):
          if p.grad is not None:
            p.grad.realize()
        self.out_queue.put(loss.numpy().item())
      elif cmd.__class__ == AllReduceCommand:
        num_devices = cmd.num_devices
        state_dict = optim.get_state_dict(self.net)

        # calculate our bucket
        st = time.perf_counter()
        bucket = {}
        for i, (k, v) in enumerate(state_dict.items()):
          if i % num_devices == self.rank:
            if v.grad is not None:
              # we move the grad here to shared memory
              if k not in self.shared_cache:
                s = shared_memory.SharedMemory(create=True, size=v.grad.nbytes())
                self.shared_cache[k] = s.name
              else:
                s = shared_memory.SharedMemory(name=self.shared_cache[k])
              t = np.ndarray(v.grad.shape, dtype=v.grad.dtype.np, buffer=s.buf)
              t[:] = v.grad.numpy()

              bucket[k] = (v.grad.shape, v.grad.dtype, s.name)

              s.close()
        if len(self.devices) > 1:
          # send to next device
          self.next_queue.put(bucket)
        print(f"device {self.device}:{self.rank} all reduce send time {time.perf_counter() - st}")

        # reduce
        for i in range(num_devices - 1):
          print(f"device {self.device}:{self.rank} waiting for {i}")
          recv_bucket = self.our_queue.get()
          for k, (shape, dtype, name) in recv_bucket.items():
            # rebuild tensor from shared memory
            s = shared_memory.SharedMemory(name)
            sn = np.ndarray(shape, dtype=dtype.np, buffer=s.buf)
            n = np.ndarray(shape, dtype=dtype.np)
            n[:] = sn[:]
            t = Tensor(n, device=self.device, requires_grad=False)

            # reduce with our grad
            t = t + state_dict[k].grad

            # move back to shared memory
            sn[:] = t.numpy()

            # update bucket
            recv_bucket[k] = (shape, dtype, name)

            # if we are the last device we can set our grad
            if i == num_devices - 2:
              state_dict[k].grad.assign(t)

            s.close()
          self.next_queue.put(recv_bucket)

        # gather
        for i in range(num_devices - 1):
          recv_bucket = self.our_queue.get()
          for k, (shape, dtype, name) in recv_bucket.items():
            # rebuild tensor from shared memory
            s = shared_memory.SharedMemory(name)
            sn = np.ndarray(shape, dtype=dtype.np, buffer=s.buf)
            n = np.ndarray(shape, dtype=dtype.np)
            n[:] = sn[:]
            t = Tensor(n, device=self.device, requires_grad=False)

            # set our grad
            state_dict[k].grad.assign(t)

            s.close()
          # send back to next device
          if i != num_devices - 2:
            self.next_queue.put(recv_bucket)
      elif cmd.__class__ == StepCommand:
        self.opt.step()

      self.wait_barrier.wait()

wait_barrier = mp.Barrier(len(devices) + 1)
ring_queues = [mp.Queue(1) for _ in devices]
shared_cache = {}
out_queue = mp.Queue()
processes = [NetDeviceProcess(net, sparse_categorical_crossentropy, opt, devices, i, wait_barrier, ring_queues, shared_cache, out_queue) for i, (net, opt) in enumerate(zip(nets, opts))]
for process in processes:
  process.start()

print("training on {} devices".format(len(devices)))
print(devices)

for step in range(1000):
  # random sample a batch for each device
  samp = [np.random.randint(0, X_train.shape[0], size=(64)) for _ in devices]
  batches = [X_train[s] for s in samp]
  # get the corresponding labels
  labels = [Y_train[s] for s in samp]

  # zero gradients
  print("zero grad")
  for process in processes:
    process.cmd_queue.put(ZeroGradCommand())
  wait_barrier.wait()

  # forward & backward pass
  print("forward backward")
  for process, batch, labels in zip(processes, batches, labels):
    process.cmd_queue.put(ForwardBackwardCommand(batch, labels))
  wait_barrier.wait()

  losses = [out_queue.get() for _ in devices]
  print("losses", losses, end="\r")

  # all reduce
  print("all reduce")
  st = time.perf_counter()
  for process in processes:
    process.cmd_queue.put(AllReduceCommand(len(devices)))
  wait_barrier.wait()
  print(f"all reduce time: {time.perf_counter() - st:.3f}s")

  # update parameters
  for process in processes:
    process.cmd_queue.put(StepCommand())
  wait_barrier.wait()
