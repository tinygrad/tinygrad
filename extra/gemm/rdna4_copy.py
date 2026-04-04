from tinygrad.helpers import getenv
from tinygrad.renderer.amd import Inst
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer.amd.dsl import ttmp
from tinygrad.runtime.autogen.amd.rdna4.ins import *

N = getenv("N", 256)
NUM_THREADS = 32

def kernel(*args:tuple[UOp, ...]) -> UOp:
  I:list[Inst] = []
  def e(i): I.append(i); return i

  # s[0:1] = kernarg ptr, s[2] = gidx0, v[0] = thread_id (0..31)

  # load buffer pointers: a (buf0) at offset 0x00, c (buf2) at offset 0x10
  e(s_load_b64(s[4:5], s[0:1], soffset=NULL))               # s[4:5] = a ptr
  e(s_load_b64(s[6:7], s[0:1], soffset=NULL, ioffset=0x8))  # s[6:7] = b ptr
  e(s_wait_kmcnt(simm16=0))

  # global_thread_id = gidx0 * NUM_THREADS + tid
  e(s_lshl_b32(s[2], ttmp[9], 5))           # s[2] = gidx0 * 32
  e(v_add_nc_u32_e32(v[0], s[2], v[0]))     # v[0] = global thread id

  # byte offset = global_thread_id * 2 (half = 2 bytes), zero-extend to 64-bit
  e(v_lshlrev_b32_e32(v[0], 1, v[0]))       # v[0] = byte offset (32-bit)
  e(v_mov_b32_e32(v[1], 0))                 # v[1] = 0 (high 32 bits)

  # build full 64-bit load address: v[2:3] = s[4:5] + v[0:1]
  e(v_add_co_u32(v[2], VCC_LO, s[4], v[0]))
  e(v_add_co_ci_u32_e32(v[3], s[5], v[1]))

  # build full 64-bit store address: v[4:5] = s[6:7] + v[0:1]
  e(v_add_co_u32(v[4], VCC_LO, s[6], v[0]))
  e(v_add_co_ci_u32_e32(v[5], s[7], v[1]))

  # load from a, store to c (flat 64-bit addressing, saddr=off)
  e(global_load_u16(vdst=v[6], vaddr=v[2:3], saddr=NULL))
  e(s_wait_loadcnt(simm16=0))
  e(global_store_b16(vaddr=v[4:5], vsrc=v[6], saddr=NULL))

  e(s_endpgm())

  workgroups = (N * N // NUM_THREADS, 1, 1)
  gidxs = [UOp.special(n, f"gidx{i}") for i,n in enumerate(workgroups)]
  lidxs = [UOp.special(NUM_THREADS, "lidx0")]
  sink = UOp.sink(*[u.base for u in args], *gidxs, *lidxs, arg=KernelInfo("copy"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in I))))

if __name__ == "__main__":
  import numpy as np
  from tinygrad import Tensor, dtypes, Device
  assert Device[Device.DEFAULT].renderer.target.arch.startswith("gfx12")
  rng = np.random.default_rng(42)
  a = Tensor(rng.random((N, N), dtype=np.float32) - 0.5, dtype=dtypes.half).realize()
  b = Tensor(np.zeros((N, N), dtype=np.half)).realize()
  b = Tensor.custom_kernel(a, b, fxn=kernel)[1].realize()
  np.testing.assert_allclose(b.numpy(), a.numpy())
