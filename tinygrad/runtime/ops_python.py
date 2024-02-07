# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Tuple, List, Optional, Any
import pickle, base64, itertools, time
from tinygrad.dtype import DType
from tinygrad.helpers import all_same
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.codegen.uops import UOp, UOps
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps

class PythonProgram:
  def __init__(self, name:str, lib:bytes):
    self.uops: List[Tuple[UOp, Optional[DType], List[int], Any]] = pickle.loads(lib)
  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      # TODO: abstract this out so it can be used for constant folding
      ul = {}
      pbufs: List[memoryview] = list(bufs)
      i = 0
      loop_ends = {}
      while i < len(self.uops):
        uop, dtype, idp, arg = self.uops[i]
        inp = [ul[v] for v in idp]
        if uop is UOps.DEFINE_GLOBAL:
          ul[i] = [pbufs.pop(0).cast(dtype.fmt)] * warp_size
        elif uop is UOps.SPECIAL:
          if arg[1][0] == 'g':
            ul[i] = [idxs[2-arg[0]]] * warp_size
          elif arg[1][0] == 'l':
            ul[i] = [x[2-arg[0]] for x in warp]
        elif uop is UOps.CONST: ul[i] = [arg] * warp_size
        elif uop is UOps.DEFINE_ACC:
          if dtype.sz > 1:
            ul[i] = [[arg] * warp_size for _ in range(dtype.sz)]
          else:
            ul[i] = [arg] * warp_size
        elif uop is UOps.LOOP:
          if i not in ul:
            ul[i] = [0] * warp_size
          else:
            for j in range(len(ul[i])):
              ul[i][j] += 1
            if ul[i][0] == inp[1][0]:
              i = loop_ends[i] + 1
              continue
        elif uop is UOps.END:
          loop_ends[idp[0]] = i
          i = idp[0] - 1
        elif uop is UOps.CAST:
          if dtype.sz > 1:
            ul[i] = inp
          else:
            # TODO: add real cast
            ul[i] = inp[0]
        elif uop is UOps.STORE:
          if isinstance(inp[2][0], list):
            for j,val in enumerate(inp[2]):
              for m,o,v in zip(inp[0], inp[1], val):
                m[o+j] = v
          else:
            for m,o,v in zip(*inp):
              m[o] = v
        elif uop is UOps.LOAD:
          if dtype.sz > 1:
            ul[i] = [[m[x+j] for m,x in zip(inp[0], inp[1])] for j in range(dtype.sz)]
          else:
            ul[i] = [m[x] for m,x in zip(inp[0], inp[1])]
        elif uop is UOps.PHI:
          for j in range(len(inp[0])):
            inp[0][j] = inp[1][j]
          ul[i] = inp[0]
        elif uop is UOps.GEP:
          ul[i] = inp[0][arg]
        elif uop is UOps.ALU:
          assert all_same([len(x) for x in inp]), f"{[len(x) for x in inp]} doesn't match on {arg}"
          if arg == BinaryOps.MUL:
            ul[i] = [x*y for x,y in zip(inp[0], inp[1])]
          elif arg == BinaryOps.ADD:
            ul[i] = [x+y for x,y in zip(inp[0], inp[1])]
          elif arg == TernaryOps.MULACC:
            ul[i] = [x*y+z for x,y,z in zip(inp[0], inp[1], inp[2])]
          #elif arg == BinaryOps.MAX:
          #  ul[i] = max(inp[0], inp[1])
          #elif arg == BinaryOps.CMPLT:
          #  ul[i] = inp[0] < inp[1]
          #elif arg == UnaryOps.NEG:
          #  ul[i] = -inp[0]
        assert uop in {UOps.STORE, UOps.END} or i in ul, (uop, dtype, idp, arg)
        #print(i, uop, dtype, arg, ul[i] if i in ul else None)
        i += 1
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