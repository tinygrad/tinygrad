# the imports have to be careful here, since we need to delay the runtime initialization
# we can't import anything that might setup the runtimes
import numpy as np
from multiprocessing import shared_memory as shm
import multiprocessing as mp
import os
import signal
import time

# process control commands
class Command: ...
class ZeroGradCommand(Command): ...
class ForwardBackwardCommand(Command):
  def __init__(self, x, y):
    self.x, self.y = x, y
class AllReduceCommand(Command): ...
class StepCommand(Command): ...

class DDP:
  @staticmethod
  def preinit():
    # delay runtime initialization till we get into the child processes
    os.environ["DELAYED_RUNTIME_INIT"] = "1" # TODO: this is kinda cursed, find a way to do this without env vars
    mp.set_start_method("spawn")

  def __init__(self, devices, model_fn, optim_fn, loss_fn):
    from tinygrad.lazy import Device
    self.devices = list(map(Device.canonicalize, devices))

    self.wait_barrier = mp.Barrier(len(devices) + 1)
    self.cmd_queues = [mp.Queue() for _ in devices]
    self.ring_queues = [mp.Queue() for _ in devices]
    self.out_queue = mp.Queue()
    self.shm_cache = {}

    self.processes = [DDPProcess(i, device, len(devices), model_fn, optim_fn, loss_fn, cmd_queue, self.out_queue, self.wait_barrier, self.ring_queues[i], self.ring_queues[(i+1)%len(devices)], self.shm_cache) for i, (device, cmd_queue) in enumerate(zip(self.devices, self.cmd_queues))]
    _ = [p.start() for p in self.processes]

    def shutdown(*_):
      for p in self.processes: p.terminate()
      for v in self.shm_cache.values(): shm.SharedMemory(name=v).unlink()
      exit(1)
    signal.signal(signal.SIGINT, shutdown)

class DDPProcess(mp.Process):
  def __init__(self, rank, device, device_count, model_fn, optim_fn, loss_fn, cmd_queue, out_queue, wait_barrier, our_ring_queue, next_ring_queue, shm_cache):
    super(DDPProcess, self).__init__()
    self.rank, self.device, self.device_num, self.device_count = rank, device, 0 if ":" not in device else int(device.split(":")[-1]), device_count
    print(f"DDPProcess {rank} initialized for device {self.device}")
    self.model_fn, self.optim_fn, self.loss_fn = model_fn, optim_fn, loss_fn
    self.cmd_queue, self.out_queue, self.wait_barrier, self.our_ring_queue, self.next_ring_queue, self.shm_cache = cmd_queue, out_queue, wait_barrier, our_ring_queue, next_ring_queue, shm_cache

  def run(self):
    # this is now running in the process
    if "GPU" in self.device:
      from tinygrad.runtime.ops_gpu import init
      init(self.device_num)
    print(f"DDPProcess {self.rank} initialized runtime for device {self.device}")

    from tinygrad.nn.optim import get_parameters, get_state_dict
    self.model = self.model_fn()
    self.optim = self.optim_fn(get_parameters(self.model))

    from tinygrad.tensor import Tensor
    # we can now start the main loop
    while True:
      cmd = self.cmd_queue.get()
      if cmd.__class__ is ZeroGradCommand:
        self.optim.zero_grad()
      elif cmd.__class__ is ForwardBackwardCommand:
        x = Tensor(cmd.x, requires_grad=False)
        out = self.model(x)
        loss = self.loss_fn(out, cmd.y)
        loss.backward()
        # realize grads
        for p in get_parameters(self.model):
          if p.grad is not None:
            p.grad.realize()
        self.out_queue.put(loss.numpy().item())
      elif cmd.__class__ is AllReduceCommand:
        state_dict = get_state_dict(self.model)

        # this is our first step so we just send our grads to the next device
        st = time.perf_counter()
        bucket = {}
        for i, (k, v) in enumerate(state_dict.items()):
          if i % self.device_count == self.rank:
            if v.grad is not None:
              # we move the grad here to shared memory
              if k not in self.shm_cache:
                s = shm.SharedMemory(create=True, size=v.grad.nbytes())
                self.shm_cache[k] = s.name
              else:
                s = shm.SharedMemory(name=self.shm_cache[k])
              t = np.ndarray(v.grad.shape, dtype=v.grad.dtype.np, buffer=s.buf)
              t[:] = v.grad.numpy()
              bucket[k] = (v.grad.shape, v.grad.dtype, s.name)
              s.close()
        if self.device_count > 1:
          # send to next device
          self.next_ring_queue.put(bucket)

        for i in range(self.device_count - 1):
          recv_bucket = self.our_ring_queue.get()
          for k, (shape, dtype, name) in recv_bucket.items():
            # rebuild tensor from shared memory
            s = shm.SharedMemory(name=name)
            sn = np.ndarray(shape, dtype=dtype.np, buffer=s.buf)
            n = np.ndarray(shape, dtype=dtype.np)
            n[:] = sn[:]
            t = Tensor(n, requires_grad=False)

            # reduce with our grad
            t = t + state_dict[k].grad

            # move back to shared memory
            sn[:] = t.numpy()

            # update bucket
            recv_bucket[k] = (shape, dtype, name)

            # if we are the last device we can set our grad
            if i == self.device_count - 2:
              state_dict[k].grad.assign(t)

            s.close()
          self.next_ring_queue.put(recv_bucket)

        # gather
        for i in range(self.device_count - 1):
          recv_bucket = self.our_ring_queue.get()
          for k, (shape, dtype, name) in recv_bucket.items():
            # rebuild tensor from shared memory
            s = shm.SharedMemory(name=name)
            sn = np.ndarray(shape, dtype=dtype.np, buffer=s.buf)
            n = np.ndarray(shape, dtype=dtype.np)
            n[:] = sn[:]
            t = Tensor(n, requires_grad=False)

            # set our grad
            state_dict[k].grad.assign(t)

            s.close()
          # send back to next device
          if i != self.device_count - 2:
            self.next_ring_queue.put(recv_bucket)
      elif cmd.__class__ == StepCommand:
        self.optim.step()
      self.wait_barrier.wait()
