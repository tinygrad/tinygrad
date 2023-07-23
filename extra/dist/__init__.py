# this file needs to be very careful with its imports as to not accidentally initialize the runtimes
from multiprocessing.connection import Connection
from typing import Any, Callable, List, Tuple
import multiprocessing as mp
import os

from tinygrad.helpers import DEBUG, getenv

# this needs to be called before everything else if you are using distributed
def preinit():
  os.environ["DELAYED_RUNTIME_INIT"] = "1"
  mp.set_start_method("spawn")

class _OOB:
  def __init__(self, pipes:List[Tuple[Connection, Connection]]):
    self.pipes = pipes
  def send(self, data:Any, target_rank:int):
    self.pipes[getenv("RANK") * getenv("WORLD_SIZE") + target_rank][1].send(data)
  def recv(self, target_rank:int) -> Any:
    return self.pipes[target_rank * getenv("WORLD_SIZE") + getenv("RANK")][0].recv()
OOB = None

def init_oob(world_size:int):
  os.environ["WORLD_SIZE"] = str(world_size)

  global OOB
  OOB = _OOB([mp.Pipe(False) for _ in range(world_size * world_size)])

def _process_wrap(rank:int, device:str, oob:_OOB, fn:Callable, args=()):
  # setup the rank
  os.environ["RANK"] = str(rank)

  # setup out of band communication
  global OOB
  OOB = oob

  # initialize the runtime
  from tinygrad.lazy import Device
  device, device_num = Device.canonicalize(device), 0 if ":" not in device else int(device.split(":")[-1])
  if "GPU" in device:
    from tinygrad.runtime.ops_gpu import CL
    CL.post_init(device_num)
  elif "HIP" in device:
    import extra.hip_wrapper as hip
    hip.hipSetDevice(device_num)
  if DEBUG >= 1: print(f"DDPProcess {rank} initialized runtime for device {device}")

  # convert device to be process specific
  Device.DEFAULT = device.split(":")[0]

  fn(*args)

# wrapper around mp.Process that initializes the runtime
def spawn(rank:int, device:str, fn:Callable, args=()) -> mp.Process:
  (p := mp.Process(target=_process_wrap, args=(rank, device, OOB, fn, args))).start()
  return p
