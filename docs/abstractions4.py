# tinygrad allows you to write kernels at many different abstractions levels.
# This is for RDNA3, but if you don't have one you can run with the emulator
# PYTHONPATH="." MOCKGPU=1 DEV=AMD

from tinygrad import Tensor, Context, GlobalCounters, UOp, Device
from tinygrad.helpers import DEBUG, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.dtype import AddrSpace, dtypes

SZ = 32*1024 if getenv("MOCKGPU") else 1024*1024*1024

if __name__ == "__main__":
  # First define a Tensor and realize it. We will focus on a 1GB sum kernel on Strix Halo with 32 CUs

  a = Tensor.ones(SZ).contiguous().realize()

  def eval_harness(name, fxn, check=None):
    print(f"***** {name}")
    GlobalCounters.reset()
    with Context(DEBUG=max(DEBUG.value, 2)): out = fxn(a).item()
    assert check is None or out == check, f"out was wrong {out}, off by {out/check}x"
    print(f"computed in {GlobalCounters.time_sum_s*1000:.2f} ms, {(a.nbytes()/1e9)/GlobalCounters.time_sum_s:.2f} GB/s")
    return out

  # *****
  # This is the high level tinygrad way.
  # Note that this is split into multiple kernels for speed.

  correct = eval_harness("basic kernel", lambda x: x.sum())

  # *****
  # Now we get to the lower abstraction layers.
  # You can write a kernel in UOps, and it's 2.5x faster.

  # This GPU has 32 CUs, keep them all busy
  CU_COUNT = 32
  def custom_sum(out:UOp, buf:UOp) -> UOp:
    LCLS = 256
    buf = buf.reshape(CU_COUNT, -1, LCLS)

    glbl = UOp.range(CU_COUNT, 0, AxisType.GLOBAL)
    lane = UOp.range(LCLS, 1, AxisType.LOCAL)

    # accumulate the globals into a per lane accumulator
    reduce_loop = UOp.range(buf.shape[1], 2, AxisType.REDUCE)
    acc = UOp.placeholder((1,), dtypes.float, slot=6, addrspace=AddrSpace.REG)
    acc = acc.after(acc.store(0))
    acc = acc.after(acc[0].store(acc.after(reduce_loop)[0] + buf[glbl, reduce_loop, lane]).end(reduce_loop))

    # store all the per lane accumulators to LOCAL
    local_accs = UOp.placeholder((LCLS,), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL)
    local_accs = local_accs.after(local_accs[lane].store(acc[0]).barrier())

    # accumulate LOCALs into a single per CU accumulator
    late_reduce_loop = UOp.range(LCLS, 3, AxisType.REDUCE)
    acc2 = UOp.placeholder((1,), dtypes.float, slot=7, addrspace=AddrSpace.REG)
    acc2 = acc2.after(acc2.store(0))
    acc2 = acc2.after(acc2[0].store(acc2.after(late_reduce_loop)[0] + local_accs[late_reduce_loop]).end(late_reduce_loop))[0]

    # store (NOTE: since the address doesn't depend on the warp, this will be automatically gated)
    return out[glbl].store(acc2).end(lane, glbl).sink(arg=KernelInfo(opts_to_apply=()))

  eval_harness("custom UOp kernel", lambda x: Tensor.empty(CU_COUNT).custom_kernel(x, fxn=custom_sum)[0].sum(), check=correct)

  # *****
  # You can also BEAM search stock tinygrad for a faster kernel.
  # This does even better than the custom kernel in this simple case.

  with Context(BEAM=2): eval_harness("BEAMed kernel", lambda x: x.sum(), check=correct)

  # *****
  # Though if you really want to go crazy with speed, you can code in assembly

  # copied from amd_asm_matmul
  class Kernel:
    def __init__(self, arch='gfx1100'): self.instructions, self.labels, self.pos, self.arch = [], {}, 0, arch
    def label(self, name): self.labels[name] = self.pos
    def emit(self, inst, target=None):
      self.instructions.append(inst)
      inst._target, inst._pos = target, self.pos
      self.pos += inst.size()
      return inst
    def waitcnt(self, lgkm=None, vm=None):
      """Wait for memory operations. lgkm=N waits until N lgkm ops remain, vm=N waits until N vmem ops remain."""
      vmcnt, lgkmcnt, expcnt = vm if vm is not None else 63, lgkm if lgkm is not None else 63, 7
      waitcnt = (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)
      self.emit(s_waitcnt(simm16=waitcnt))
    def finalize(self, sink:UOp) -> UOp:
      for inst in self.instructions:
        if inst._target is None: continue
        offset_dwords = (self.labels[inst._target] - inst._pos - inst.size()) // 4
        if not -32768 <= offset_dwords <= 32767: raise ValueError(f"branch to '{inst._target}' offset {offset_dwords} exceeds simm16 range")
        inst.simm16 = offset_dwords
      return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                                   UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in self.instructions]))))

  from tinygrad.runtime.autogen.amd.rdna3.ins import *
  CU_COUNT = 32
  LANES = 32
  def asm_sum(out:UOp, buf:UOp) -> UOp:
    V_LANE_ID = 0             # lane_id set on startup
    S_WORKGROUP_X = 2         # workgroup_id_x
    S_LOOP_CTR = 3
    k = Kernel()
    # mul lane id by 16 for offsets (4 for float, 4 for b128)
    k.emit(v_mul_lo_u32(v[0], v[0], 16))
    # load both addresses
    k.emit(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0x0, soffset=NULL))
    k.waitcnt(lgkm=0)
    # zero the accumulators
    k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, vdstx=v[8], vdsty=v[9], srcx0=0, srcy0=0))
    k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, vdstx=v[10], vdsty=v[11], srcx0=0, srcy0=0))

    k.emit(s_mov_b32(s[S_LOOP_CTR], 0))

    k.label('LOOP')
    k.emit(global_load_b128(vdst=v[4:7], addr=v[0], saddr=s[6:7]))
    k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, vdstx=v[8], vdsty=v[9], srcx0=v[8], vsrcx1=v[4], srcy0=v[9], vsrcy1=v[5]))
    k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, vdstx=v[10], vdsty=v[11], srcx0=v[10], vsrcx1=v[6], srcy0=v[11], vsrcy1=v[7]))

    k.emit(s_add_i32(s[S_LOOP_CTR], s[S_LOOP_CTR], 1))
    k.emit(s_cmp_ge_i32(s[S_LOOP_CTR], buf.numel()//(CU_COUNT*LANES)))
    k.emit(s_cbranch_scc0(), target='LOOP')


    k.emit(s_sendmsg(simm16=3))  # DEALLOC_VGPRS
    k.emit(s_endpgm())
    return k.finalize(UOp.sink(UOp.special(CU_COUNT, 'gidx0'), UOp.special(LANES, 'lidx0'), out, buf, arg=KernelInfo(opts_to_apply=())))

  out = Tensor.zeros(1,).contiguous().realize()
  eval_harness("custom UOp kernel", lambda x: out.custom_kernel(x, fxn=asm_sum)[0], check=correct)

