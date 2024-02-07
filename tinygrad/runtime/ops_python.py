# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Tuple, List, Optional, Any
import pickle, base64, itertools, time
from tinygrad.dtype import DType
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.codegen.uops import UOp, UOps
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps

class PythonProgram:
  def __init__(self, name:str, lib:bytes):
    self.uops: List[Tuple[UOp, Optional[DType], List[int], Any]] = pickle.loads(lib)
  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    st = time.perf_counter()
    # TODO: abstract this out so it can be used for constant folding
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]+local_size[::-1]]):
      ul = {}
      pbufs: List[memoryview] = list(bufs)
      for i, (uop, dtype, idp, arg) in enumerate(self.uops):
        inp = [ul[v] for v in idp]
        if uop is UOps.DEFINE_GLOBAL:
          ul[i] = pbufs.pop(0).cast(dtype.fmt)
        elif uop is UOps.SPECIAL:
          if arg[1][0] == 'g': ul[i] = idxs[2-arg[0]]
          elif arg[1][0] == 'l': ul[i] = idxs[5-arg[0]]
        elif uop is UOps.CONST: ul[i] = arg
        elif uop is UOps.CAST:
          if dtype.sz > 1:
            ul[i] = inp
          else:
            # TODO: add real cast
            ul[i] = inp[0]
        elif uop is UOps.STORE:
          if isinstance(inp[2], list):
            for j,val in enumerate(inp[2]):
              inp[0][inp[1] + j] = val
          else:
            inp[0][inp[1]] = inp[2]
        elif uop is UOps.LOAD:
          if dtype.sz > 1:
            ul[i] = [inp[0][inp[1]+j] for j in range(dtype.sz)]
          else:
            ul[i] = inp[0][inp[1]]
        elif uop is UOps.GEP:
          ul[i] = inp[0][arg]
        elif uop is UOps.ALU:
          if arg == BinaryOps.MUL:
            ul[i] = inp[0] * inp[1]
          elif arg == BinaryOps.ADD:
            ul[i] = inp[0] + inp[1]
          elif arg == BinaryOps.MAX:
            ul[i] = max(inp[0], inp[1])
          elif arg == BinaryOps.CMPLT:
            ul[i] = inp[0] < inp[1]
          elif arg == UnaryOps.NEG:
            ul[i] = -inp[0]
        assert uop in {UOps.STORE} or i in ul, (uop, dtype, arg)
    return time.perf_counter() - st

class PythonCompiler(Compiler):
  linearizer_opts = LinearizerOptions()
  def render(self, name:str, uops:List[UOp]) -> str:
    lops = [(u.uop, u.dtype, [uops.index(v) for v in u.vin], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops))
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class PythonAllocator(Allocator):
  def _alloc(self, size): return memoryview(bytearray(size))
  def copyin(self, dest, src:memoryview): dest[:] = src
  def copyout(self, dest:memoryview, src): dest[:] = src

class PythonDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, PythonAllocator(), PythonCompiler(), PythonProgram)