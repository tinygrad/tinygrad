# this file needs to be very careful with its imports as to not accidentally initialize the runtimes
import os
import multiprocessing as mp

from tinygrad.helpers import DEBUG

# this needs to be called before everything else if you are using multidevice
def preinit():
  os.environ["DELAYED_RUNTIME_INIT"] = "1" # TODO: this is kinda cursed, find a way to do this without env vars
  mp.set_start_method("spawn")

class _OOB:
  def __init__(self, queues):
    self.queues = queues
  def send(self, data, target_rank):
    self.queues[target_rank].put(data)
  def recv(self):
    return self.queues[RANK].get()
OOB = None

def init_oob(world_size):
  return [mp.Queue() for _ in range(world_size)]

RANK = -1
def _process_wrap(rank, world_size, device, oob, fn, args=()):
  # setup the rank
  global RANK
  RANK = rank

  # setup out of band communication
  global OOB
  OOB = _OOB(oob)

  # initialize the runtime
  from tinygrad.lazy import Device
  device, device_num = Device.canonicalize(device), 0 if ":" not in device else int(device.split(":")[-1])
  if "GPU" in device:
    from tinygrad.runtime.ops_gpu import CL
    CL.post_init(device_num)
  if DEBUG >= 1: print(f"DDPProcess {rank} initialized runtime for device {device}")

  fn(rank, world_size, device, *args)

# wrapper around mp.Process that initializes the runtime
def spawn(rank, world_size, device, oob, fn, args=()):
  (p := mp.Process(target=_process_wrap, args=(rank, world_size, device, oob, fn, args))).start()
  return p
