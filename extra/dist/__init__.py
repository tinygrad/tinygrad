# this file needs to be very careful with its imports as to not accidentally initialize the runtimes
import os
import multiprocessing as mp

from tinygrad.helpers import DEBUG

# this needs to be called before everything else if you are using multidevice
def preinit():
  os.environ["DELAYED_RUNTIME_INIT"] = "1" # TODO: this is kinda cursed, find a way to do this without env vars
  mp.set_start_method("spawn")

class _OOB:
  def __init__(self, pipes):
    self.pipes = pipes
  def send(self, data, target_rank):
    self.pipes[target_rank * (WORLD_SIZE - 1) + RANK][1].send(data)
  def recv(self, target_rank):
    return self.pipes[RANK * (WORLD_SIZE - 1) + target_rank][0].recv()
OOB = None

def init_oob(world_size):
  return [mp.Pipe(False) for _ in range(world_size * (world_size - 1))]

RANK = -1
WORLD_SIZE = -1
def _process_wrap(rank, world_size, device, oob, fn, args=()):
  # setup the rank
  global RANK, WORLD_SIZE
  RANK, WORLD_SIZE = rank, world_size

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
