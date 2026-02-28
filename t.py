from tinygrad import Tensor, Device, dtypes, TinyJit
from tinygrad.runtime.autogen.amd.rdna4.ins import *
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from extra.gemm.amd_asm_matmul import Kernel
from tinygrad.renderer.amd.sqtt import map_insts, format_packet

device = Device[Device.DEFAULT]
arch = device.arch
N_WAVES = 2

k = Kernel(arch)
"""
k.emit(s_load_b128(s[0:3], s[0:1], NULL))
k.waitcnt(lgkm=0)
k.emit(v_mov_b32_e32(v[0], 0))
k.emit(v_mov_b32_e32(v[1], 0))
k.emit(global_load_b64(v[2:3], v[0:1], s[2:3]))
k.waitcnt(lgkm=0, vm=0)
"""
S_LOOP_CTR = 10

k.emit(s_mov_b32(s[S_LOOP_CTR], 0))

k.label("loop")
"""
for _ in range(10):
  #k.emit(s_nop(1))
  #k.emit(s_mov_b32(s[2], 1.0))
  #k.emit(s_prefetch_inst_pc_rel(s[2]))
"""
for _ in range(100):
  k.emit(v_add_f32_e32(v[5], 1.0, v[5]))
k.emit(s_add_co_i32(s[S_LOOP_CTR], s[S_LOOP_CTR], 1))
k.emit(s_cmp_le_i32(s[S_LOOP_CTR], 2))
k.emit(s_cbranch_scc1(), target='loop')

"""
k.emit(global_store_b64(vaddr=v[0:1], saddr=s[0:1], vsrc=v[2:3]))
"""
k.emit(s_endpgm())

def fxn(out:UOp, A:UOp) -> UOp:
  lidx = UOp.special(1, "lidx0")
  sink = UOp.sink(out, A, lidx, arg=KernelInfo(name="test"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in k.finalize()]))))

@TinyJit
def fn(out, a):
  out = Tensor.custom_kernel(out, a, fxn=fxn)[0]
  return out

for _ in range(6):
  a = Tensor([1, 2], dtype=dtypes.int32).realize()
  out = Tensor.empty_like(a)
  out.uop.buffer.ensure_allocated()
  Tensor.realize(a, out)
  out = fn(out, a)
  out.realize()

programs:dict[int, bytes] = {}
for e in device.profile_events:
  if type(e).__name__ == "ProfileProgramEvent": programs[e.tag] = e.lib
  if type(e).__name__ == "ProfileSQTTEvent":
    for pkt,inst in map_insts(e.blob, programs[e.kern], arch):
      if inst is None: continue
      pass
      #print(format_packet(pkt), inst.inst)
