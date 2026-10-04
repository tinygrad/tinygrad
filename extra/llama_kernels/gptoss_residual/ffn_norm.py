"""Exact post-compilation GPTOSS FFN forward fusion; never changes allocation or autograd."""
import functools, hashlib, pathlib
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, Target
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import UOp, Ops, ProgramInfo, KernelInfo

# Full generated sources, normalizing only the function identifier. Each pins
# indices, dtypes, arithmetic association, and all BF16 rounding boundaries.
_SPECS = (
  ("49e89f4524f19508ddbd832955b4eaf3173ccbcd1a7e95b67feb8af7524f637d", (4096, 1, 1), (16, 4, 4), 4),
  ("0709d6d75442b14db7244d5145d1842a9b02c660ec514f7308902f2f8610ea19", (45, 1024, 1), (16, 16, 1), 4),
  ("7185cca723c04a1e0977b9e82c6d94e6d7877863115198712ff7550dd859c679", (45, 512, 1), (16, 8, 1), 6),
  ("49b4e6155a0dd7207c006252792f44460bbe147790c2ba862e8ecbd72ed39be3", (256, 1, 1), (8, 8, 1), 2),
)
_SIZES = (47185920, 16384, 47185920, 16384, 47185920, 50331648, 2880, 2880)
_DTYPES = (dtypes.bfloat16, dtypes.float32, dtypes.bfloat16, dtypes.float32) + (dtypes.bfloat16,)*4

@functools.cache
def _matches(prg:UOp, stage:int) -> bool:
  sha, gs, ls, nargs = _SPECS[stage]
  if prg.op is not Ops.PROGRAM or not isinstance(p:=prg.arg, ProgramInfo) or len(prg.src) != 4: return False
  if p.target.device != "AMD" or p.target.arch != "gfx950" or p.vars: return False
  if p.global_size != gs or p.local_size != ls or p.globals != tuple(range(nargs)) or p.outs != (0,) or p.ins != tuple(range(1, nargs)):
    return False
  if prg.src[2].op is not Ops.SOURCE or prg.src[3].op is not Ops.BINARY: return False
  src = prg.src[2].arg.replace(p.function_name + "(", "FFN_REFERENCE(", 1)
  return hashlib.sha256(src.encode()).hexdigest() == sha

def _byte_range(u:UOp) -> tuple[UOp, int, int]|None:
  # BUFFER denotes owned storage. PARAM slots can bind overlapping runtime
  # views, so even different slots are deliberately NOT an alias proof.
  if u.op is Ops.BUFFER: root, lo, hi = u, 0, u.max_numel()*u.dtype.itemsize
  elif u.op is Ops.SLICE and u.src[0].op is Ops.BUFFER and u.src[1].op is Ops.CONST and isinstance(u.src[1].arg, int):
    root, lo, hi = u.src[0], u.src[1].arg, u.src[1].arg + u.arg*u.dtype.itemsize
  elif u.op is Ops.SHRINK and u.src[0].op is Ops.BUFFER and len(u.src) == 3 and \
       all(s.op is Ops.CONST and isinstance(s.arg, int) for s in u.src[1:]):
    # Lowered SHRINK uses (offset, extent), not (start, end).
    root, lo, hi = u.src[0], u.src[1].arg*u.dtype.itemsize, (u.src[1].arg+u.src[2].arg)*u.dtype.itemsize
  else: return None
  if not all(isinstance(v, int) for v in (lo, hi)) or lo < 0 or hi > root.max_numel()*root.dtype.itemsize or hi <= lo: return None
  return root, lo, hi

@functools.cache
def _program(target:Target) -> UOp:
  from tinygrad.codegen import to_program
  from tinygrad.renderer.cstyle import HIPRenderer
  from tinygrad.runtime.support.compiler_amd import HIPCompiler
  args = [UOp.placeholder((n,), dt, i) for i, (n, dt) in enumerate(zip(_SIZES, _DTYPES))]
  zero = UOp.const(0, dtypes.int32)
  accesses = [u.index(zero).store(UOp.const(0, u.dtype)) if i < 4 else u.index(zero).load() for i,u in enumerate(args)]
  sink = UOp.sink(*accesses, UOp.special(128, "lidx0"), UOp.special(4096, "gidx0"),
                  arg=KernelInfo("ffn_residual_norm_packed_raw",
                                 estimates=Estimates(ops=6*47185920+68*16384, mem=8*47185920+8*16384+4*2880)))
  source = (pathlib.Path(__file__).parent/"ffn_norm.cpp").read_text()
  prg = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=source),
                             UOp(Ops.BINARY, arg=HIPCompiler("gfx950").compile_cached(source))))
  # Register a normal final View Program trace, despite this pass running after
  # the enclosing LINEAR was compiled. Source extraction sees the loaded binary.
  return to_program(prg, HIPRenderer(target))

def fuse_gptoss_ffn_norm(linear:UOp) -> UOp:
  if not getenv("GPTOSS_FFN_NORM_FUSION", 0) or getenv("AMD_COMGR_FAST_MATH", 0) or linear.op is not Ops.LINEAR: return linear
  result, i = [], 0
  while i < len(linear.src):
    chain = linear.src[i:i+4]
    if len(chain) == 4 and all(c.op is Ops.CALL and len(c.src) == _SPECS[j][3]+1 and _matches(c.src[0], j) for j,c in enumerate(chain)):
      den, save, norm, reciprocal = (c.src[1:] for c in chain)
      args = (save[0], den[0], norm[0], reciprocal[0], *den[1:], norm[5])
      device = args[0].device
      if den[1:] == save[1:] == norm[1:4] and den[0] is norm[4] is reciprocal[1] and \
         all(c.src[0].arg.target == chain[0].src[0].arg.target for c in chain) and \
         all(u.dtype == dt and u.shape == (n,) and u.device == device for u,dt,n in zip(args, _DTYPES, _SIZES)) and \
         all(d.split(":")[0] == "AMD" for d in (device if isinstance(device, tuple) else (device,))):
        ranges = [_byte_range(u) for u in args]
        if all(r is not None for r in ranges):
          concrete = [r for r in ranges if r is not None]
          if all(lo % 4 == 0 for _,lo,_ in concrete) and all(root is not other or hi <= start or stop <= lo
                 for j,(root,lo,hi) in enumerate(concrete) for other,start,stop in concrete[:j]):
            result.append(_program(chain[0].src[0].arg.target).call(*args))
            i += 4
            continue
    result.append(linear.src[i])
    i += 1
  return linear.replace(src=tuple(result)) if len(result) != len(linear.src) else linear
