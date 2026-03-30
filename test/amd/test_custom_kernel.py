import unittest
import functools
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.dtype import AddrSpace
from tinygrad.runtime.autogen.amd.rdna3.ins import *
import tinygrad.runtime.autogen.amd.rdna3.ins as r3
import tinygrad.runtime.autogen.amd.rdna4.ins as r4
from tinygrad.renderer.amd.dsl import s, v
from test.amd.helpers import TARGET_TO_ARCH
from extra.gemm.amd_asm_matmul import Kernel

def custom_add_one(A:UOp) -> UOp:
  A = A.flatten()
  assert dtypes.is_float(A.dtype.base), f"buffer dtype must be float32, got {A.dtype}"
  threads = UOp.special(A.size, "lidx0")
  insts = [
    s_load_b64(s[0:1], s[0:1], soffset=NULL),
    s_waitcnt(lgkmcnt=0),
    v_lshlrev_b32_e32(v[0], 2, v[0]), # element offset
    global_load_b32(v[1], v[0], saddr=s[0:1]),
    s_waitcnt(vmcnt=0),
    v_mov_b32_e32(v[2], 1.0),
    v_add_f32_e32(v[1], v[1], v[2]),
    global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]),
    s_endpgm(),
  ]
  sink = UOp.sink(A.base, threads, arg=KernelInfo(f"custom_add_one_{A.size}", estimates=Estimates(ops=A.size, mem=A.size*4*2)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_add_var(A:UOp, B:UOp) -> UOp:
  A,B = A.flatten(), B.flatten()
  assert A.dtype.base == dtypes.uint32, f"buffer dtype must be uint32, got {A.dtype}"
  threads = UOp.special(A.size, "lidx0")
  var = UOp.variable("var", 0, 10)
  insts = [
    s_load_b128(s[4:7], s[0:1]),
    s_load_b32(s[8], s[0:1], offset=0x10), # all threads load the same variable
    s_waitcnt(lgkmcnt=0),
    v_lshlrev_b32_e32(v[0], 2, v[0]), # element offset, different per thread
    global_load_b32(v[1], v[0], saddr=s[6:7]),
    s_waitcnt(vmcnt=0),
    v_add_nc_u32_e32(v[1], s[8], v[1]),
    global_store_b32(addr=v[0], data=v[1], saddr=s[4:5]),
    s_endpgm(),
  ]
  sink = UOp.sink(A.base, B.base, var, threads, arg=KernelInfo(f"custom_add_var_{A.size}"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_wave_sync(A:UOp, arch:str) -> UOp:
  # 4 waves across 1024 WG — enough to saturate a SIMD with many concurrent WGs
  # s_sleep yields the SIMD so waves from different WGs interleave, causing barrier packet reordering
  threads = UOp.special(128, "lidx0")
  wg = UOp.special(1024, "gidx0")
  insts = []
  for _ in range(4):
    insts.append(s_sleep(4))
    insts += [s_barrier()] if arch == "rdna3" else [r4.s_barrier_signal(), r4.s_barrier_wait()]
    insts += [s_nop(0)]*4
  insts.append(s_endpgm())
  sink = UOp.sink(A.base, threads, wg, arg=KernelInfo("custom_wave_sync"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_lds_sync(A:UOp, arch:str) -> UOp:
  A = A.flatten()
  num_threads = A.shape[0]
  threads = UOp.special(num_threads, "lidx0")
  wg = UOp.special(1, "gidx0")
  lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=512, addrspace=AddrSpace.LOCAL), (), 'lds')  # 128 * 4 bytes
  isa = r4 if arch == "rdna4" else r3
  wait_kmcnt = [isa.s_wait_kmcnt(simm16=0)] if arch == "rdna4" else [isa.s_waitcnt(lgkmcnt=0)]
  wait_dscnt = [isa.s_wait_dscnt(simm16=0)] if arch == "rdna4" else [isa.s_waitcnt(lgkmcnt=0)]
  barrier = [isa.s_barrier_signal(ssrc0=-1), isa.s_barrier_wait(simm16=-1)] if arch == "rdna4" else [isa.s_barrier()]
  global_store = [isa.global_store_b32(vaddr=v[6:7], saddr=s[0:1], vsrc=v[5])] if arch == "rdna4" \
      else [isa.global_store_b32(addr=v[6], data=v[5], saddr=s[0:1])]
  insts = [
    isa.s_load_b64(s[0:1], s[0:1], soffset=NULL),
    *wait_kmcnt,
    isa.v_lshlrev_b32_e32(v[1], 2, v[0]),
    # lds[thread_idx] = thread_idx
    isa.ds_store_b32(addr=v[1], data0=v[0]),
    *wait_dscnt,
    *barrier,
    # out[threaed_idx] = thread_idx == num_threads ? -1 : lds[thread_idx + 1]
    isa.v_add_nc_u32_e32(v[2], 4, v[1]),
    isa.v_cmp_gt_u32_e32(num_threads-1, v[0]),
    isa.ds_load_b32(vdst=v[3], addr=v[2]),
    *wait_dscnt,
    isa.v_mov_b32_e32(v[4], -1),
    isa.v_cndmask_b32_e32(v[5], v[4], v[3]),
    isa.v_lshlrev_b32_e32(v[6], 2, v[0]),
    *global_store,
    isa.s_endpgm(),
  ]
  sink = UOp.sink(A.base, lds, threads, wg, arg=KernelInfo("custom_lds_sync"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_handwritten(A:UOp, arch:str) -> UOp:
  A = A.flatten()
  threads = UOp.special(128, "lidx0")
  wg = UOp.special(256, "gidx0")
  lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=512, addrspace=AddrSpace.LOCAL), (), 'lds')  # 128 * 4 bytes
  k = Kernel(arch)
  k.emit(r4.s_nop(0))
  k.emit(r4.v_mov_b32_e32(v[1], 10))
  def emit_alt():
    for i in range(4):
      k.emit(r4.v_mov_b32_e32(v[20+i], 4.0))
      k.emit(r4.v_rcp_f32_e32(v[22+i], v[20+i]))
      k.emit(r4.s_mov_b32(s[20+i], i))
  def emit_wmma():
    for _ in range(4):
      k.emit(r4.v_wmma_f32_16x16x16_f16(v[0:7], v[8:11], v[8:11], 1))
  k.label("start")
  k.emit(s_mov_b32(s[1], 10))
  k.label("loop")
  # wmma should've overlapped here if it was a different unit?
  for _ in range(4):
    emit_wmma()
    emit_alt()
  for _ in range(8): k.emit(s_nop(1))
  k.emit(s_add_u32(s[1], s[1], -1))
  k.emit(s_cmp_eq_i32(s[1], 0))
  k.emit(s_cbranch_scc0(), target="loop")
  k.emit(r4.s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, wg, lds, arg=KernelInfo("custom_handwritten"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_data_deps(A:UOp, arch:str) -> UOp:
  A = A.flatten()
  threads = UOp.special(A.size, "lidx0")
  k = Kernel(arch)
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt(lgkmcnt=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(global_load_b32(v[1], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt(vmcnt=0))
  k.emit(v_mov_b32_e32(v[2], 1.0))
  k.emit(v_add_f32_e32(v[1], v[1], v[2]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, arg=KernelInfo(f"custom_data_deps_{A.size}", estimates=Estimates(ops=A.size, mem=A.size*4*2)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD device")
class TestCustomKernel(unittest.TestCase):
  def setUp(self): self.arch = TARGET_TO_ARCH[Device["AMD"].arch]

  def test_simple(self):
    if self.arch != "rdna3": self.skipTest("only rdna3")
    a = Tensor.full((16, 16), 1.).contiguous().realize()
    a = Tensor.custom_kernel(a, fxn=custom_add_one)[0]
    ei = a.schedule()[-1].lower()
    self.assertEqual(ei.prg.estimates.ops, a.numel())
    self.assertEqual(ei.prg.estimates.mem, a.nbytes()*2)
    ei.run()
    self.assertTrue((a.numpy() == 2.).all())

  def test_variable(self):
    if self.arch != "rdna3": self.skipTest("only rdna3")
    b = Tensor.full((16, 16), 1, dtype=dtypes.uint32).contiguous().realize()
    a = Tensor.zeros_like(b).contiguous().realize()
    a = Tensor.custom_kernel(a, b, fxn=custom_add_var)[0]
    ei = a.schedule()[-1].lower()
    for i in range(4):
      ei.run({"var":i})
      self.assertTrue((a.numpy() == 1+i).all())

  def test_lds_sync(self):
    if self.arch not in ("rdna3", "rdna4"): self.skipTest("only rdna3/rdna4")
    a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
    a = Tensor.custom_kernel(a, fxn=functools.partial(custom_lds_sync, arch=self.arch))[0]
    a.realize()
    ref = Tensor.arange(1, 129, dtype=dtypes.int32)
    ref[127] = -1
    self.assertListEqual(a.tolist(), ref.tolist())

  def test_handwritten(self):
    if self.arch != "rdna4": self.skipTest("only tested on rdna4")
    a = Tensor.empty(1024, dtype=dtypes.int32).contiguous().realize()
    a = Tensor.custom_kernel(a, fxn=functools.partial(custom_handwritten, arch=self.arch))[0]
    a.realize()

  def test_data_deps(self):
    if self.arch != "rdna3": self.skipTest("only tested on rdna3")
    a = Tensor(np.full((16, 16), 5.0, dtype=np.float32)).realize()
    a = Tensor.custom_kernel(a, fxn=functools.partial(custom_data_deps, arch=self.arch))[0]
    a.realize()
    self.assertTrue((a.numpy() == 6.0).all())

if __name__ == "__main__":
  unittest.main()
