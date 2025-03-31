from __future__ import annotations
import ctypes, os, mmap, tempfile, pathlib, array, functools, threading, contextlib, sys, subprocess, struct
assert sys.platform != 'win32'
from tinygrad.device import BufferSpec, Compiled, Allocator, Compiler, MallocAllocator
from tinygrad.dtype import dtypes, DType, PtrDType
from tinygrad.ops import Ops, UOp
from tinygrad.codegen.symbolic import gep_pushing
from tinygrad.helpers import from_mv, getenv, round_up, mv_address, to_mv, cpu_objdump, DEBUG, dedup, all_same
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.autogen import libc, qcom_dsp
if getenv("IOCTL"): import extra.dsp.run # noqa: F401 # pylint: disable=unused-import

from tinygrad.ops import PatternMatcher, UPat

def multi_mul(a0, a1, b0, b1, c0, c1, d0=None, d1=None, acc=None):
  swizzle = []
  for i in range(32):
    swizzle.append(i)
    swizzle.append(32+i)
    swizzle.append(64+i)
    swizzle.append(96+i)
  swizzle = tuple(swizzle)
  if a0.op is not Ops.CAST:
    #print("rejected on a0")
    return None
  if a1.op is not Ops.CAST:
    #print("rejected on a1")
    return None
  if d0 is None:
    if c0.src[0].op is Ops.GEP and c0.src[0].arg == (2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62,
                                                     66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126):
      d0 = c0.src[0].src[0].gep(tuple(i+1 for i in c0.src[0].arg)).cast(dtypes.int.vec(32))
    else:
      d0 = UOp.const(dtypes.uchar.vec(32), 0).cast(dtypes.int.vec(32))
  if d1 is None:
    #if c1.src[0].op is Ops.GEP: print("here1", c1.src[0].arg)
    d1 = UOp.const(dtypes.uchar.vec(32), 0).cast(dtypes.int.vec(32))
  assert a0.op is Ops.CAST
  assert b0.op is Ops.CAST
  assert c0.op is Ops.CAST
  assert d0.op is Ops.CAST
  assert a1.op is Ops.CAST
  assert b1.op is Ops.CAST
  assert c1.op is Ops.CAST
  assert d1.op is Ops.CAST
  dt1 = a0.src[0].dtype.scalar().vec(128)
  dt2 = a1.src[0].dtype.scalar().vec(128)
  m0 = UOp(Ops.CAT, dt1, src=(a0.src[0],b0.src[0],c0.src[0],d0.src[0])).gep(swizzle)
  m1 = UOp(Ops.CAT, dt2, src=(a1.src[0],b1.src[0],c1.src[0],d1.src[0])).gep(swizzle)
  simp_m1 = m1.simplify()
  if simp_m1.op is Ops.GEP and simp_m1.arg == simp_m1.arg[0:4]*32:
    # Vx32.w+=vrmpy(Vu32.ub,Rt32.b) -> __builtin_HEXAGON_V6_vrmpybus_acc
    # Vx32.uw+=vrmpy(Vu32.ub,Rt32.ub) -> __builtin_HEXAGON_V6_vrmpyub_acc
    scalar_m1 = simp_m1.src[0].gep(simp_m1.arg[0:4])
    if acc is not None:
      return UOp(Ops.CUSTOMI, dtypes.int.vec(32), (acc, m0, scalar_m1.bitcast(dtypes.uint)), "__builtin_HEXAGON_V6_vrmpybus_acc_128B({0}, {1}, {2})")
    else:
      return UOp(Ops.CUSTOMI, dtypes.int.vec(32), (m0, scalar_m1.bitcast(dtypes.uint)), "__builtin_HEXAGON_V6_vrmpybus_128B({0}, {1})")
  if acc is not None:
    return UOp(Ops.CUSTOMI, dtypes.int.vec(32), (acc, m0, m1), "__builtin_HEXAGON_V6_vrmpybusv_acc_128B({0}, {1}, {2})")
  else:
    return UOp(Ops.CUSTOMI, dtypes.int.vec(32), (m0, m1), "__builtin_HEXAGON_V6_vrmpybusv_128B({0}, {1})")

# __builtin_HEXAGON_A2_vraddub_acc

def gep_on_reduce(gep, alu):
  if gep.dtype.vcount == 1: return None
  return UOp(alu.op, alu.dtype.scalar().vec(gep.dtype.count),
     tuple(x.gep(gep.arg) if x.op is not Ops.RANGE else x for x in alu.src), alu.arg) if not isinstance(gep.dtype, PtrDType) and \
      alu.dtype.count >= gep.dtype.count else None

def multi_add_int32(**aa):
  if 'acc' in aa:
    acc = aa['acc']
    del aa['acc']
  else:
    acc = None
  if 'd0' not in aa:
    c0 = aa['c0']
    if c0.src[0].op is Ops.GEP and c0.src[0].arg == (2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62,
                                                     66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126):
      d0 = c0.src[0].src[0].gep(tuple(i+1 for i in c0.src[0].arg)).cast(dtypes.int.vec(32))
    else:
      d0 = UOp.const(dtypes.uchar.vec(32), 0).cast(dtypes.int.vec(32))
    aa['d0'] = d0
  swizzle = []
  for i in range(32):
    swizzle.append(i)
    swizzle.append(32+i)
    swizzle.append(64+i)
    swizzle.append(96+i)
  for x in aa.values():
    assert x.src[0].dtype.scalar() is dtypes.uchar
    assert x.op is Ops.CAST
  swizzle = tuple(swizzle)
  m0 = UOp(Ops.CAT, dtypes.uchar.vec(128), src=tuple(aa[k].src[0] for k in sorted(aa.keys()))).gep(swizzle)
  if acc is not None:
    return UOp(Ops.CUSTOMI, dtypes.int.vec(32), (acc, m0, UOp.const(dtypes.uint, 0x01010101)),
               "__builtin_HEXAGON_V6_vrmpybus_acc_128B({0}, {1}, {2})")
  else:
    return UOp(Ops.CUSTOMI, dtypes.int.vec(32), (m0, UOp.const(dtypes.uint, 0x01010101)), "__builtin_HEXAGON_V6_vrmpybus_128B({0}, {1})")

def multi_add_int2(**aa):
  if 'acc' in aa:
    acc = aa['acc']
    del aa['acc']
  else:
    acc = None
  r0 = UOp(Ops.VECTORIZE, dtypes.uchar.vec(8), tuple(x.src[0].gep(0) for x in aa.values()))
  r1 = UOp(Ops.VECTORIZE, dtypes.uchar.vec(8), tuple(x.src[0].gep(1) for x in aa.values()))
  if acc is not None:
    return UOp(Ops.CUSTOMI, dtypes.int.vec(2), (acc, r0.bitcast(dtypes.int64), r1.bitcast(dtypes.int64)),
               arg="__builtin_HEXAGON_A2_vraddub_acc({0}, {1}, {2})")
  else:
    return UOp(Ops.CUSTOMI, dtypes.int.vec(2), (r0.bitcast(dtypes.int64), r1.bitcast(dtypes.int64)), arg="__builtin_HEXAGON_A2_vraddub({0}, {1})")

conv_pm = PatternMatcher([
  # __builtin_HEXAGON_V6_vrmpybus x3
  (UPat(dtype=dtypes.int.vec(32), name="a0")*UPat(name="a1") + UPat(name="b0")*UPat(name="b1") + \
   UPat(name="c0")*UPat(name="c1"), multi_mul),
  (UPat(name="acc") + UPat(dtype=dtypes.int.vec(32), name="a0")*UPat(name="a1") + UPat(name="b0")*UPat(name="b1") + \
   UPat(name="c0")*UPat(name="c1"), multi_mul),

  # __builtin_HEXAGON_V6_vrmpybus x3
  (UPat(Ops.CAST, dtype=dtypes.int.vec(32), name="a0") + UPat(Ops.CAST, name="b0") + UPat(Ops.CAST, name="c0"), multi_add_int32),
  (UPat(name="acc") + UPat(Ops.CAST, dtype=dtypes.int.vec(32), name="a0") + UPat(Ops.CAST, name="b0") + UPat(Ops.CAST, name="c0"), multi_add_int32),
])

dsp_pm = PatternMatcher([
  # convert load char32 to load char128
  (UPat(Ops.LOAD, (dtypes.uchar.vec(96), dtypes.uchar.vec(64), dtypes.uchar.vec(32)), src=(UPat.var("buf").cast(),), name="load"),
   lambda load, buf: load.replace(dtype=dtypes.uchar.vec(128),
     src=(buf.cast(buf.dtype.base.vec(128).ptr(size=buf.dtype.size, local=buf.dtype.local)),)+load.src[1:]).gep(tuple(range(0, load.dtype.count)))),
  # GEP on REDUCE
  (UPat(Ops.GEP, src=(UPat(Ops.REDUCE, name='alu'),), name='gep'), gep_on_reduce),
  # no swizzle down convert
  (((UPat.var('x').maximum(0) ^ -1).maximum(-256) ^ -1).cast(dtypes.uchar.vec(128)),
   lambda x: UOp(Ops.CUSTOM, dtypes.uchar.vec(128), src=tuple(x.gep(tuple(range(i, i+32))) for i in range(0, 128, 32)),
     arg="__builtin_HEXAGON_V6_vpackhub_sat_128B(__builtin_HEXAGON_V6_vpackwh_sat_128B({3}, {2}), __builtin_HEXAGON_V6_vpackwh_sat_128B({1}, {0}))")),

  # REDUCE int4 -> 2xint2, int8 -> 4xint2
  (UPat(Ops.REDUCE, dtype=dtypes.int.vec(4), name="r"),
   lambda r: UOp(Ops.CAT, r.dtype, (gep_on_reduce(r.gep((0,1)), r), gep_on_reduce(r.gep((2,3)), r)))),
  (UPat(Ops.REDUCE, dtype=dtypes.int.vec(8), name="r"),
   lambda r: UOp(Ops.CAT, r.dtype, (gep_on_reduce(r.gep((0,1)), r), gep_on_reduce(r.gep((2,3)), r),
                                    gep_on_reduce(r.gep((4,5)), r), gep_on_reduce(r.gep((6,7)), r)))),

  # __builtin_HEXAGON_V6_vrmpybus
  (UPat(dtype=dtypes.int.vec(32), name="a0")*UPat(name="a1") + UPat(name="b0")*UPat(name="b1") + \
   UPat(name="c0")*UPat(name="c1") + UPat(name="d0")*UPat(name="d1"), multi_mul),
  (UPat(name="acc") + UPat(dtype=dtypes.int.vec(32), name="a0")*UPat(name="a1") + UPat(name="b0")*UPat(name="b1") + \
   UPat(name="c0")*UPat(name="c1") + UPat(name="d0")*UPat(name="d1"), multi_mul),

  # build __builtin_HEXAGON_V6_vrmpybus_128B
  (UPat(Ops.CAST,dtype=dtypes.int.vec(32),name="a0")+UPat(Ops.CAST,name="b0")+UPat(Ops.CAST,name="c0")+UPat(Ops.CAST,name="d0"), multi_add_int32),
  (UPat(name="acc")+UPat(Ops.CAST,dtype=dtypes.int.vec(32),name="a0")+UPat(Ops.CAST,name="b0")+
   UPat(Ops.CAST,name="c0")+UPat(Ops.CAST,name="d0"), multi_add_int32),

  # build __builtin_HEXAGON_A2_vraddub
  (UPat(Ops.CAST,dtype=dtypes.int.vec(2),name="a0")+UPat(Ops.CAST,name="a1")+UPat(Ops.CAST,name="a2")+UPat(Ops.CAST,name="a3")+ \
   UPat(Ops.CAST,dtype=dtypes.int.vec(2),name="a4")+UPat(Ops.CAST,name="a5")+UPat(Ops.CAST,name="a6")+UPat(Ops.CAST,name="a7"), multi_add_int2),
  (UPat(name="acc")+UPat(Ops.CAST,dtype=dtypes.int.vec(2),name="a0")+UPat(Ops.CAST,name="a1")+UPat(Ops.CAST,name="a2")+UPat(Ops.CAST,name="a3")+ \
   UPat(Ops.CAST,dtype=dtypes.int.vec(2),name="a4")+UPat(Ops.CAST,name="a5")+UPat(Ops.CAST,name="a6")+UPat(Ops.CAST,name="a7"), multi_add_int2),

  # we upcast 3 as 4
  (UPat(Ops.REDUCE, name="r", src=(UPat(dtype=dtypes.int.vec(32), name="a0")*UPat(name="a1") +
                                   UPat(name="b0")*UPat(name="b1") + UPat(name="c0")*UPat(name="c1"),), allow_any_len=True),
   lambda r, **kwargs: r.replace(src=(multi_mul(**kwargs),)+r.src[1:])),
  (UPat(Ops.REDUCE, name="r", src=(UPat(Ops.CAST,dtype=dtypes.int.vec(32),name="a0")+UPat(Ops.CAST,name="b0")+UPat(Ops.CAST,name="c0"),
                                   ), allow_any_len=True),
   lambda r, **kwargs: r.replace(src=(multi_add_int32(**kwargs),)+r.src[1:])),

  # mul by const on GEP
  (UPat(Ops.GEP, src=(UPat.var('x'),), name="gep")*UPat.cvar("c", vec=False),
   lambda x, gep, c: (x.gep(gep.arg[0])*c.arg).broadcast(c.dtype.count) if all_same(gep.arg) and c.dtype.count > 1 else None),
])+gep_pushing

def add_to_mul(c:UOp, x:UOp):
  if c.arg.startswith("__builtin_HEXAGON_V6_vrmpybus_128B"):
    return UOp(Ops.CUSTOMI, dtypes.int.vec(32), (x, c.src[0], c.src[1]), "__builtin_HEXAGON_V6_vrmpybus_acc_128B({0}, {1}, {2})")
  elif c.arg.startswith("__builtin_HEXAGON_V6_vrmpybusv_128B"):
    return UOp(Ops.CUSTOMI, dtypes.int.vec(32), (x, c.src[0], c.src[1]), "__builtin_HEXAGON_V6_vrmpybusv_acc_128B({0}, {1}, {2})")
  elif 'acc' in c.arg and x.op is not Ops.CUSTOM:
    return c.replace(src=(x+c.src[0], c.src[1], c.src[2]))
  else:
    return None

def prefetch_l1(ld:UOp, idx:UOp):
  if ld.src[-1].op is Ops.CUSTOM: return None
  ranges = sorted([x for x in ld.src[0].src[0].toposort if x.op is Ops.RANGE], key=lambda x: x.arg)
  x1 = UOp(Ops.CUSTOM, dtypes.void, src=(idx.src[0], idx.src[1]+UOp.const(dtypes.int, ld.dtype.count*2)),
           arg="__builtin_HEXAGON_Y2_dcfetch({0}+{1});")
  x2 = UOp(Ops.CUSTOM, dtypes.void, src=(idx.src[0], idx.src[1].substitute({ranges[-1]: ranges[-1].src[0]})),
           arg="__builtin_HEXAGON_Y2_dcfetch({0}+{1});")
  return ld.replace(src=ld.src+(x1, x2))

def prefetch_l2(ld:UOp, idx:UOp):
  if not getenv("PREFETCHL2", 0): return None
  if ld.src[-1].op is Ops.CUSTOM: return None
  ranges = sorted([x for x in ld.src[0].src[0].toposort if x.op is Ops.RANGE], key=lambda x: x.arg)
  if len(ranges):
    nidx = idx.src[1]
    if nidx.op is Ops.ADD and nidx.src[1].op is Ops.CONST: nidx = nidx.src[0]
    zero_ranges = {r:r.const_like(0) for r in ranges[:-1]}
    nlen_uop = (nidx.substitute({ranges[-1]: ranges[-1].src[1], **zero_ranges}) -
                nidx.substitute({ranges[-1]: ranges[-1].src[0], **zero_ranges})).simplify()
    if nlen_uop.arg > 8192: return None  # too much to prefetch. "L2FETCH is best performed in sizes less than 8 KB"
    nidx = nidx.substitute({ranges[-1]: ranges[-1].src[0]})
    x1 = UOp(Ops.CUSTOM, dtypes.void, src=(idx.src[0], nidx, UOp.const(dtypes.int, nlen_uop.arg)), arg="__builtin_HEXAGON_Y4_l2fetch({0}+{1}, {2});")
    return ld.replace(src=ld.src+(x1,))

def vectorize_shuffle(vec:UOp):
  if not all(s.op in {Ops.GEP, Ops.CONST} for s in vec.src): return None
  gepped = dedup([s.src[0] for s in vec.src if s.op is Ops.GEP])
  if len(gepped) == 0: return None
  if len(gepped) == 1:
    arg = []
    for s in vec.src:
      if s.op is Ops.GEP:
        arg.append(s.arg[0])
      else:
        arg.append(-1)
    str_arg = ','.join([f'{y:4d}' for y in arg])
    full_arg = "__builtin_shufflevector({0}, {0}, "+str_arg+")"
    return UOp(Ops.CUSTOM, vec.dtype, tuple(gepped), full_arg)
  if not all(x.dtype.scalar() is dtypes.uchar for x in gepped): return None
  if not all_same([x.dtype.count for x in gepped]) or gepped[0].dtype.count != vec.dtype.count: return None
  if len(gepped) == 2:
    arg = []
    for s in vec.src:
      if s.op is Ops.GEP:
        if s.src[0] is gepped[0]:
          arg.append(s.arg[0])
          continue
        if s.src[0] is gepped[1]:
          arg.append(gepped[0].dtype.count + s.arg[0])
          continue
      arg.append(-1)
    str_arg = ','.join([f'{y:4d}' for y in arg])
    full_arg = "__builtin_shufflevector({0}, {1}, "+str_arg+")"
    return UOp(Ops.CUSTOM, vec.dtype, tuple(gepped), full_arg)
  if len(gepped) != 3: return None
  arg = []
  for s in vec.src:
    if s.op is Ops.GEP:
      if s.src[0] is gepped[0]:
        arg.append(s.arg[0])
        continue
      if s.src[0] is gepped[1]:
        arg.append(gepped[0].dtype.count + s.arg[0])
        continue
    arg.append(-1)
  arg2 = []
  for i,s in enumerate(vec.src):
    if s.op is Ops.GEP:
      if s.src[0] is gepped[2]:
        arg2.append(vec.dtype.count + s.arg[0])
        continue
    if s.op is Ops.CONST:
      arg2.append(-1)
      continue
    arg2.append(i)
  str_arg = ','.join([f'{y:4d}' for y in arg])
  str_arg2 = ','.join([f'{y:4d}' for y in arg2])
  full_arg = "__builtin_shufflevector(__builtin_shufflevector({0}, {1}, "+str_arg+"), {2}, "+str_arg2+")"
  return UOp(Ops.CUSTOM, vec.dtype, tuple(gepped), full_arg)

  #vv = []
  #for i in range(64):
  #src = "__builtin_shufflevector({0}, {1})"
  return None

dsp_pm_late = PatternMatcher([
  # prefetch L1
  (UPat(Ops.LOAD, dtype=(dtypes.uchar.vec(4), dtypes.uchar.vec(8)), src=(UPat(Ops.INDEX, name="idx").cast(),), name="ld"), prefetch_l1),

  # prefetch L2
  (UPat(Ops.LOAD, dtype=dtypes.uchar.vec(128), src=(UPat(Ops.INDEX, name="idx").cast(),), name="ld"), prefetch_l2),

  # 64 -> 128
  #(UPat(Ops.LOAD, dtype=dtypes.uchar.vec(64), src=(UPat(Ops.CAST, src=(UPat(Ops.INDEX, name="idx"),)),)),
  # lambda idx: idx.cast(dtypes.uchar.vec(128).ptr(idx.dtype.size)).load(dtype=dtypes.uchar.vec(128)).gep(tuple(range(0,64)))),

  # unaligned load
  #(UPat(Ops.LOAD, src=(UPat(Ops.CAST, src=(UPat(Ops.INDEX, src=(UPat(), UPat()+UPat.cvar("c"))),), name="ptr"),), dtype=dtypes.uchar.vec(128)),
  # lambda c,ptr: UOp(Ops.CUSTOM, dtype=dtypes.uchar.vec(128), src=(ptr,), arg='vmemu({0})')),

  (UPat(Ops.VECTORIZE, dtypes.uchar.vec(128), name="vec"), vectorize_shuffle),

  # __builtin_HEXAGON_V6_vrmpybus_acc_128B
  (UPat(Ops.CUSTOMI, dtype=dtypes.int.vec(32), name="c")+UPat.var("x"), add_to_mul),

  # add acc to __builtin_HEXAGON_A2_vraddub (must be after the reduce expansion)
  (UPat(Ops.CUSTOMI, name="cu", arg="__builtin_HEXAGON_A2_vraddub({0}, {1})") + UPat.var("x"),
   lambda x,cu: cu.replace(dtype=dtypes.int64, src=(x.bitcast(dtypes.int64), cu.src[0], cu.src[1]),
                           arg="__builtin_HEXAGON_A2_vraddub_acc({0}, {1}, {2})").bitcast(dtypes.int.vec(2))),

  #(UPat(Ops.BITCAST, src=(UPat(Ops.LOAD, name="ld"),), name="bc"),
  # lambda ld, bc: ld.src[0].src[0].cast(bc.dtype.ptr(ld.src[0].dtype.size)).load(dtype=bc.dtype)),
  (UPat(Ops.GEP, name="x"), lambda x: UOp(Ops.CUSTOM, x.dtype, x.src+x.src,
    "__builtin_shufflevector({0}, {1}, "+','.join([f'{y:4d}' for y in x.arg])+")") if len(x.arg) > 1 and x.src[0].dtype.count > 1 else None),
  (UPat.var("x")+UPat(Ops.VECTORIZE,src=UPat.var("y")),
   lambda x,y: x+UOp(Ops.CUSTOMI,x.dtype,(y,),arg="{0}") if x.op is not Ops.CUSTOMI or x.arg != "{0}" else None),
  (UPat.var("x")*UPat(Ops.VECTORIZE,src=UPat.var("y")),
   lambda x,y: x*UOp(Ops.CUSTOMI,x.dtype,(y,),arg="{0}") if x.op is not Ops.CUSTOMI or x.arg != "{0}" else None),
  (UPat.var("x")//UPat(Ops.VECTORIZE,src=UPat.var("y")),
   lambda x,y: x//UOp(Ops.CUSTOMI,x.dtype,(y,),arg="{0}") if x.op is not Ops.CUSTOMI or x.arg != "{0}" else None),
  (UPat(Ops.DEFINE_ACC, src=(UPat(Ops.VECTORIZE, src=UPat(Ops.CONST, arg=0)),), dtype=dtypes.uchar.vec(128), name="d", allow_any_len=True),
   lambda d: d.replace(src=(UOp(Ops.CUSTOMI, d.dtype, arg="__builtin_HEXAGON_V6_vd0_128B()"),)+d.src[1:])),
])

pretty_render = PatternMatcher([
  # makes rendering nicer
  (UPat(Ops.VECTORIZE, src=UPat(Ops.CONST, dtype=(dtypes.uint8, dtypes.int8)), name="v"),
   lambda v: UOp(Ops.VECTORIZE, v.dtype, src=tuple(UOp(Ops.CUSTOMI, x.dtype, src=(UOp.const(dtypes.int, x.arg),), arg="{0}") for x in v.src))),
])

class DSPRenderer(ClangRenderer):
  device = "DSP"
  supports_float4 = True
  buffer_suffix = " restrict __attribute__((align_value(128)))"
  kernel_prefix = "__attribute__((noinline)) "
  pre_matcher = dsp_pm
  extra_matcher = dsp_pm_late+ClangRenderer.extra_matcher+pretty_render
  type_map = { **ClangRenderer.type_map, dtypes.uint64: "unsigned long long", dtypes.int64: "long long" }
  code_for_op = {k:v for k,v in ClangRenderer.code_for_op.items() if k != Ops.SQRT}

  def _render_defines(self, uops) -> list[str]:
    return ['''/* DSP boilerplate */ struct dcvs_v2_req { int type; int _pad; _Bool dcvs_enable; char dcvs_option; _Bool set_latency; int latency;
      _Bool set_dcvs_params; short _pad2; char target_corner; char min_corner; char max_corner; int _pad3[3];};''','int HAP_power_set(void*, void*);',
      'typedef union { struct { void *pv; unsigned int len; } buf; struct { int fd; unsigned int offset; } dma; } remote_arg;',
      'void* HAP_mmap(void *addr, int len, int prot, int flags, int fd, long offset);', 'int HAP_munmap(void *addr, int len);',
      'unsigned long long HAP_perf_get_time_us(void);'] + super()._render_defines(uops)

  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[DType,bool]]]) -> str:
    msrc = ['int entry(unsigned long long handle, unsigned int sc, remote_arg* pra) {',
            'struct dcvs_v2_req req = {.type=7, .dcvs_enable=0, .set_latency=1, .latency=100, .set_dcvs_params=1, .target_corner = 6 /* TURBO */};',
            'HAP_power_set((void*)handle, (void*)&req);']
    msrc += ['if ((sc>>24) != 2) return 0;']
    msrc += [f'int sz_or_val_{i} = ((int*)pra[0].buf.pv)[{i}];' for i,b in enumerate(bufs)]
    msrc += [f'int off{i} = ((int*)pra[1].buf.pv)[{i}];' for i,b in enumerate(bufs) if isinstance(b[1][0], PtrDType)]
    msrc += [f'void *buf_{i} = HAP_mmap(0,sz_or_val_{i},3,0,pra[{i+3}].dma.fd,0)+off{i};' for i,b in enumerate(bufs) if isinstance(b[1][0], PtrDType)]
    msrc += ["unsigned long long start = HAP_perf_get_time_us();"]
    msrc += [f"{function_name}({', '.join([(f'buf_{i}' if isinstance(b[1][0], PtrDType) else f'sz_or_val_{i}') for i,b in enumerate(bufs)])});"]
    msrc += ["*(unsigned long long *)(pra[2].buf.pv) = HAP_perf_get_time_us() - start;"]
    msrc += [f'HAP_munmap(buf_{i}, sz_or_val_{i});' for i,b in enumerate(bufs) if isinstance(b[1][0], PtrDType)]
    msrc += ["return 0; }"]
    return '\n'.join(msrc)

def rpc_sc(method=0, ins=0, outs=0, fds=0): return (method << 24) | (ins << 16) | (outs << 8) | fds
def rpc_prep_args(ins=None, outs=None, in_fds=None):
  ins, outs, in_fds = ins or list(), outs or list(), in_fds or list()

  pra = (qcom_dsp.union_remote_arg * (len(ins) + len(outs) + len(in_fds)))()
  fds = (ctypes.c_int32 * (len(ins) + len(outs) + len(in_fds)))(*([-1] * (len(ins) + len(outs))), *in_fds)
  attrs = (ctypes.c_uint32 * (len(ins) + len(outs) + len(in_fds)))(*([0] * (len(ins) + len(outs))), *([1] * (len(in_fds))))

  for i, mv in enumerate(ins + outs): pra[i].buf.pv, pra[i].buf.len = mv_address(mv) if mv.nbytes > 0 else 0, mv.nbytes
  return pra, fds, attrs, (ins, outs)

class DSPProgram:
  def __init__(self, dev:DSPDevice, name:str, lib:bytes):
    self.dev, self.lib = dev, lib

  def __call__(self, *bufs, vals:tuple[int, ...]=(), wait=False):
    if len(bufs) >= 16: raise RuntimeError(f"Too many buffers to execute: {len(bufs)}")

    pra, fds, attrs, _ = rpc_prep_args(ins=[var_vals_mv:=memoryview(bytearray((len(bufs)+len(vals))*4)), off_mv:=memoryview(bytearray(len(bufs)*4))],
                                       outs=[timer:=memoryview(bytearray(8)).cast('Q')], in_fds=[b.share_info.fd for b in bufs])
    var_vals_mv.cast('i')[:] = array.array('i', tuple(b.size for b in bufs) + vals)
    off_mv.cast('I')[:] = array.array('I', tuple(b.offset for b in bufs))
    self.dev.exec_lib(self.lib, rpc_sc(method=2, ins=2, outs=1, fds=len(bufs)), pra, fds, attrs)
    return timer[0] / 1e6

class DSPBuffer:
  def __init__(self, va_addr:int, size:int, share_info, offset:int=0):
    self.va_addr, self.size, self.share_info, self.offset = va_addr, size, share_info, offset

class DSPAllocator(Allocator):
  def __init__(self, dev:DSPDevice):
    self.dev = dev
    super().__init__()

  def _alloc(self, size:int, options:BufferSpec):
    b = qcom_dsp.ION_IOC_ALLOC(self.dev.ion_fd, len=size, align=0x200, heap_id_mask=1<<qcom_dsp.ION_SYSTEM_HEAP_ID, flags=qcom_dsp.ION_FLAG_CACHED)
    share_info = qcom_dsp.ION_IOC_SHARE(self.dev.ion_fd, handle=b.handle)
    va_addr = libc.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, share_info.fd, 0)
    return DSPBuffer(va_addr, size, share_info, offset=0)

  def _free(self, opaque:DSPBuffer, options:BufferSpec):
    if libc is not None and qcom_dsp is not None:
      libc.munmap(opaque.va_addr, opaque.size)
      os.close(opaque.share_info.fd)
      qcom_dsp.ION_IOC_FREE(self.dev.ion_fd, handle=opaque.share_info.handle)

  def _as_buffer(self, src:DSPBuffer) -> memoryview: return to_mv(src.va_addr, src.size)
  def _copyin(self, dest:DSPBuffer, src:memoryview): ctypes.memmove(dest.va_addr, from_mv(src), src.nbytes)
  def _copyout(self, dest:memoryview, src:DSPBuffer): ctypes.memmove(from_mv(dest), src.va_addr, dest.nbytes)
  def _offset(self, buf, size:int, offset:int): return DSPBuffer(buf.va_addr+offset, size, buf.share_info, buf.offset+offset)

class ClangCompiler(Compiler):
  def __init__(self, cachekey="compile_clang", args:list[str]|None=None, objdump_tool='objdump'):
    self.args = ['-shared', '-march=native'] if args is None else args
    self.objdump_tool = objdump_tool
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output([getenv("CC", 'clang'), *self.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib',
                               '-', '-o', str(output_file.name)], input=src.encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

  def disassemble(self, lib:bytes): return cpu_objdump(lib, self.objdump_tool)

class DSPDevice(Compiled):
  def __init__(self, device:str=""):
    compiler_args = ["--target=hexagon", "-mcpu=hexagonv65", "-fuse-ld=lld", "-nostdlib",  "-mhvx=v65", "-mhvx-length=128b"]
    try:
      self.ion_fd = os.open('/dev/ion', os.O_RDONLY)
      # Generate link script to pass into clang. Aligning all used sections to 4k fixes invoke problem.
      sections = ['hash', 'text', 'rela.plt', 'got', 'got.plt', 'dynamic', 'dynsym', 'dynstr', 'plt', 'data', 'bss']
      sections_link = '\n'.join([f'.{n} : ALIGN(4096) {{ *(.{n}) }}' for n in sections])
      with tempfile.NamedTemporaryFile(delete=False) as self.link_ld:
        self.link_ld.write(f"SECTIONS {{ . = 0x0; {sections_link}\n /DISCARD/ : {{ *(.note .note.* .gnu.hash .comment) }} }}".encode())
        self.link_ld.flush()

      from tinygrad.runtime.graph.cpu import CPUGraph
      super().__init__(device, DSPAllocator(self), DSPRenderer(),
        ClangCompiler("compile_dsp", ["-shared"] + compiler_args + [f"-T{self.link_ld.name}"], 'llvm-objdump'), functools.partial(DSPProgram, self),
        functools.partial(CPUGraph, self))
      fastrpc_shell = memoryview(bytearray(pathlib.Path('/dsp/cdsp/fastrpc_shell_3').read_bytes()))
      self.shell_buf = self.allocator.alloc(round_up(fastrpc_shell.nbytes, 0x1000), BufferSpec(nolru=True))
      ctypes.memmove(self.shell_buf.va_addr, mv_address(fastrpc_shell), fastrpc_shell.nbytes)

      self.init_dsp()
      RPCListener(self).start()
    except FileNotFoundError:
      super().__init__(device, MallocAllocator, MockDSPRenderer(), ClangCompiler(None, ["-static"] + compiler_args, 'llvm-objdump'), MockDSPProgram)

  def open_lib(self, lib):
    self.binded_lib, self.binded_lib_off = lib, 0
    fp = "file:///tinylib?entry&_modver=1.0&_dom=cdsp\0"
    pra, _, _, _ = rpc_prep_args(ins=[memoryview(array.array('I', [len(fp), 0xff])), memoryview(bytearray(fp.encode()))],
                                 outs=[o1:=memoryview(bytearray(0x8)), o2:=memoryview(bytearray(0xff))])
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0, sc=rpc_sc(method=0, ins=2, outs=2), pra=pra)
    if o1.cast('i')[1] < 0: raise RuntimeError(f"Cannot open lib: {o2.tobytes().decode()}")
    return o1.cast('I')[0]

  def close_lib(self, handle):
    pra, _, _, _ = rpc_prep_args(ins=[memoryview(array.array('I', [handle, 0xff]))], outs=[memoryview(bytearray(0x8)), memoryview(bytearray(0xff))])
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=0, sc=rpc_sc(method=1, ins=1, outs=2), pra=pra)

  def exec_lib(self, lib, sc, args, fds, attrs):
    def _exec_lib():
      handle = self.open_lib(lib)
      qcom_dsp.FASTRPC_IOCTL_INVOKE_ATTRS(self.rpc_fd, fds=fds, attrs=attrs, inv=qcom_dsp.struct_fastrpc_ioctl_invoke(handle=handle, sc=sc, pra=args))
      self.close_lib(handle)
    try: _exec_lib()
    except (OSError, PermissionError):
      # DSP might ask for a connection reset or just fail with operation not permitted, try to reset connection.
      self.init_dsp()
      try: _exec_lib()
      except (OSError, PermissionError) as e: raise RuntimeError(e)

  def init_dsp(self):
    if hasattr(self, 'rpc_fd'):
      with contextlib.suppress(OSError):
        qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=4, sc=rpc_sc(method=2, ins=0, outs=0)) # pylint: disable=access-member-before-definition
      os.close(self.rpc_fd) # pylint: disable=access-member-before-definition

    self.rpc_fd: int = os.open('/dev/adsprpc-smd', os.O_RDONLY | os.O_NONBLOCK)
    qcom_dsp.FASTRPC_IOCTL_GETINFO(self.rpc_fd, 3)
    qcom_dsp.FASTRPC_IOCTL_CONTROL(self.rpc_fd, req=0x3)
    qcom_dsp.FASTRPC_IOCTL_INIT(self.rpc_fd, flags=0x1, file=self.shell_buf.va_addr, filelen=self.shell_buf.size, filefd=self.shell_buf.share_info.fd)
    qcom_dsp.FASTRPC_IOCTL_INVOKE(self.rpc_fd, handle=3, sc=rpc_sc(method=3, ins=0, outs=0))

class RPCListener(threading.Thread):
  def __init__(self, device:DSPDevice):
    super().__init__()
    self.device, self.daemon = device, True

  def run(self):
    # Setup initial request arguments.
    context, status, TINYFD = 0, 0xffffffff, 0xffff
    req_args, _, _, _ = rpc_prep_args(ins=[msg_send:=memoryview(bytearray(0x10)).cast('I'), out_buf:=memoryview(bytearray(0x10000)).cast('I')],
                                      outs=[msg_recv:=memoryview(bytearray(0x10)).cast('I'), in_buf:=memoryview(bytearray(0x10000)).cast('I')])
    req_args[1].buf.len = 0

    while True:
      # Update message request and send it.
      msg_send[:] = array.array('I', [context, status, req_args[1].buf.len, in_buf.nbytes])

      try: qcom_dsp.FASTRPC_IOCTL_INVOKE(self.device.rpc_fd, handle=0x3, sc=0x04020200, pra=req_args)
      except OSError: continue # retry

      context, inbufs, outbufs = msg_recv[0], ((sc:=msg_recv[2]) >> 16) & 0xff, (msg_recv[2] >> 8) & 0xff

      in_ptr, out_ptr, objs = mv_address(in_buf), mv_address(out_buf), []
      for i in range(inbufs + outbufs):
        obj_ptr = round_up(in_ptr + 4, 8) if i < inbufs else round_up(out_ptr + 4, 8)
        objs.append(to_mv(obj_ptr, obj_size:=to_mv(in_ptr, 4).cast('I')[0]))
        if i < inbufs: in_ptr = obj_ptr + obj_size
        else:
          to_mv(out_ptr, 4).cast('I')[0] = obj_size
          out_ptr = obj_ptr + obj_size
          in_ptr += 4

      in_args, out_args = objs[:inbufs], objs[inbufs:]
      req_args[1].buf.len = out_ptr - mv_address(out_buf)

      status = 0 # reset status, will set if error
      if sc == 0x20200: pass # greating
      elif sc == 0x13050100: # open
        try: out_args[0].cast('I')[0] = TINYFD if (name:=in_args[3].tobytes()[:-1].decode()) == "tinylib" else os.open(name, os.O_RDONLY)
        except OSError: status = 1
      elif sc == 0x3010000:
        if (fd:=in_args[0].cast('I')[0]) != TINYFD: os.close(fd)
      elif sc == 0x9010000: # seek
        if (fd:=in_args[0].cast('I')[0]) == TINYFD:
          assert in_args[0].cast('I')[2] == qcom_dsp.APPS_STD_SEEK_SET, "Supported only SEEK_SET"
          res, self.device.binded_lib_off = 0, in_args[0].cast('I')[1]
        else: res = os.lseek(fd, in_args[0].cast('I')[1], in_args[0].cast('I')[2])
        status = 0 if res >= 0 else res
      elif sc == 0x4010200: # read
        if (fd:=in_args[0].cast('I')[0]) == TINYFD:
          buf = self.device.binded_lib[self.device.binded_lib_off:self.device.binded_lib_off+in_args[0].cast('I')[1]]
          self.device.binded_lib_off += len(buf)
        else: buf = os.read(fd, in_args[0].cast('I')[1])
        out_args[1][:len(buf)] = buf
        out_args[0].cast('I')[0:2] = array.array('I', [len(buf), int(len(buf) == 0)])
      elif sc == 0x1f020100: # stat
        stat = os.stat(in_args[1].tobytes()[:-1].decode())
        out_stat = qcom_dsp.struct_apps_std_STAT.from_address(mv_address(out_args[0]))
        for f in out_stat._fields_: out_stat.__setattr__(f[0], int(getattr(stat, f"st_{f[0]}", 0)))
      elif sc == 0x2010100: # mmap
        st = qcom_dsp.FASTRPC_IOCTL_MMAP(self.device.rpc_fd, fd=-1, flags=in_args[0].cast('I')[2], vaddrin=0, size=in_args[0].cast('Q')[3])
        out_args[0].cast('Q')[0:2] = array.array('Q', [0, st.vaddrout])
      else: raise RuntimeError(f"Unknown op: {sc=:X}")

# ***** mock DSP *****

mockdsp_boilerplate = '''/* DSP boilerplate */ static long syscall(long r0, long r1, long r2, long r3, long r4, long r5, long r6) {
long retval; __asm__ volatile("r0 = %1; r1 = %2; r2 = %3; r3 = %4; r4 = %5; r5 = %6; r6 = %7; trap0(#1); %0 = r0" : "=r" (retval)
  : "r" (r0), "r" (r1), "r" (r2), "r" (r3), "r" (r4), "r" (r5), "r" (r6) : "r0", "r1", "r2", "r3", "r4", "r5", "r6"); return retval; }
static int read(int fd, void* buf, int len) {{ return syscall(fd, (long)buf, len, 0, 0, 0, 63); }}
static int write(int fd, void* buf, int len) {{ return syscall(fd, (long)buf, len, 0, 0, 0, 64); }}
static int exit(int ret) {{ return syscall(ret, 0, 0, 0, 0, 0, 93); }}
static unsigned int inscount(void) {{ unsigned int ret; __asm__ volatile(".word 0x6a15c000; %0 = R0" : "=r" (ret) : : "r0"); return ret; }}
static void *mmap2(void *addr, unsigned int length, int prot, int flags, int fd, unsigned long offset) {{
return (void*)syscall((long)addr, length, prot, flags, fd, offset, 222); }}'''

class MockDSPRenderer(DSPRenderer):
  def _render_defines(self, uops) -> list[str]: return ClangRenderer._render_defines(self, uops)
  def _render_entry(self, function_name:str, bufs:list[tuple[str,tuple[DType,bool]]]) -> str:
    # https://gpages.juszkiewicz.com.pl/syscalls-table/syscalls.html
    # control register 21 is HEX_REG_QEMU_INSN_CNT, 0x6a15c000 loads it
    msrc = [mockdsp_boilerplate, 'void _start(void) {']
    for i,b in enumerate(bufs):
      if isinstance(b[1][0], PtrDType):
        sz = b[1][0].size*b[1][0].itemsize
        # for loop for big reads
        msrc.append(f"void *buf{i} = mmap2(0, {sz}, 3, 0x21, -1, 0); for(int rd = 0; rd < {sz}; rd += read(0, buf{i}+rd, {sz}-rd));")
      else:
        msrc.append(f"unsigned int val{i}; read(0, &val{i}, 4);")
    msrc.append("unsigned int st = inscount();")
    msrc.append(f"{function_name}({', '.join([(f'(void*)buf{i}' if isinstance(b[1][0], PtrDType) else f'val{i}') for i,b in enumerate(bufs)])});")
    msrc.append("unsigned int et = inscount() - st; write(1, &et, sizeof(et));")
    for i,b in enumerate(bufs):
      if isinstance(b[1][0], PtrDType): msrc.append(f"write(1, buf{i}, {b[1][0].size*b[1][0].itemsize});")
    msrc.append('exit(0); }')
    return '\n'.join(msrc)

class MockDSPProgram:
  def __init__(self, name:str, lib:bytes): self.lib = lib
  def __call__(self, *bufs, vals:tuple[int, ...]=(), wait=False):
    with tempfile.NamedTemporaryFile(suffix=".out") as dsp_lib:
      dsp_lib.write(self.lib)
      dsp_lib.flush()
      os.chmod(dsp_lib.name, 0o0777)
      # NOTE: this timing includes a docker launch
      proc = subprocess.run(["docker", "run", "--rm", "-i", "-v", f"{os.path.abspath(os.path.dirname(dsp_lib.name))}:/work", "-w", "/work",
                            "qemu-hexagon", "-c", f"qemu-hexagon {'-strace' if DEBUG >= 5 else ''} /work/"+os.path.basename(dsp_lib.name)],
                            input=b''.join([bytes(x) for x in bufs] + [struct.pack("I", x) for x in vals]), stdout=subprocess.PIPE, check=True)
    offset = 4
    for x in bufs:
      x[:] = proc.stdout[offset:offset+len(x)]
      offset += len(x)
    assert offset == len(proc.stdout)
    return struct.unpack("I", proc.stdout[0:4])[0] / 1e9  # pretend it's 1 Ghz, but this is an inscount, not a time
