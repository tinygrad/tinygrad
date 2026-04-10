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
  correct = SZ

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

  eval_harness("basic kernel", lambda x: x.sum(), check=correct)

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

  # Kernel class copied from amd_asm_matmul
  class Kernel:
    def __init__(self, arch='gfx1100'): self.instructions, self.labels, self.pos, self.arch = [], {}, 0, arch
    def label(self, name): self.labels[name] = self.pos
    def emit(self, inst, target=None):
      self.instructions.append(inst)
      inst._target, inst._pos = target, self.pos
      self.pos += inst.size()
      return inst
    def waitcnt(self, lgkm=None, vm=None):
      # Wait for memory operations. lgkm=N waits until N lgkm ops remain, vm=N waits until N vmem ops remain.
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
  LANES = 64
  def asm_sum(out:UOp, buf:UOp) -> UOp:
    V_LANE_ID = 0             # lane_id set on startup
    S_WORKGROUP_X = 2         # workgroup_id_x
    S_LOOP_CTR = 3
    k = Kernel()
    # mul lane id by 16 for offsets (4 for float, 4 for b128)
    k.emit(v_mul_lo_u32(v[0], v[V_LANE_ID], 16))
    k.emit(v_add_nc_u32_e32(v[1], 4096, v[0]))
    k.emit(v_add_nc_u32_e32(v[2], 4096, v[1]))
    k.emit(v_add_nc_u32_e32(v[3], 4096, v[2]))
    # load both addresses
    k.emit(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0x0, soffset=NULL))
    k.waitcnt(lgkm=0)
    # offset buffer pointer by workgroup_id_x * chunk_size_bytes
    k.emit(s_mul_i32(s[S_LOOP_CTR], s[S_WORKGROUP_X], buf.numel()*4//CU_COUNT))
    k.emit(s_add_u32(s[6], s[6], s[S_LOOP_CTR]))
    k.emit(s_addc_u32(s[7], s[7], 0))
    # zero the accumulators
    k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, vdstx=v[4], vdsty=v[5], srcx0=0, srcy0=0))
    k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, vdstx=v[6], vdsty=v[7], srcx0=0, srcy0=0))

    LOAD_UNROLL = 16
    k.emit(s_mov_b32(s[S_LOOP_CTR], buf.numel()//(CU_COUNT*LANES*4*LOAD_UNROLL) - 1))

    k.label('LOOP')
    # issue all loads (v[8:71]), offset field is signed 13-bit (max 4095), so bump base halfway
    for i in range(LOAD_UNROLL):
      offset = i*LANES*16
      assert offset < 16384
      k.emit(global_load_b128(vdst=v[8+i*4:11+i*4], addr=v[offset//4096], saddr=s[6:7], offset=offset%4096))
    k.emit(s_add_u32(s[6], s[6], LOAD_UNROLL * LANES * 16))
    k.emit(s_addc_u32(s[7], s[7], 0))
    k.waitcnt(vm=0)
    # tree reduce loads into v[8:11], cross float positions so srcx0 (bank 0,1) and vsrcx1 (bank 2,3) avoid bank conflicts
    n = LOAD_UNROLL
    while n > 1:
      half = n // 2
      for j in range(half):
        a, b = 8 + j*4, 8 + (j+half)*4
        # v[a+0](bank0) += v[b+2](bank2), v[a+1](bank1) += v[b+3](bank3)
        k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, vdstx=v[a], vdsty=v[a+1], srcx0=v[a], vsrcx1=v[b+2], srcy0=v[a+1], vsrcy1=v[b+3]))
        # v[a+2](bank2) += v[b+0](bank0), v[a+3](bank3) += v[b+1](bank1)
        k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, vdstx=v[a+2], vdsty=v[a+3], srcx0=v[a+2], vsrcx1=v[b], srcy0=v[a+3], vsrcy1=v[b+1]))
      n = half
    # accumulate into v[4:7]
    k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, vdstx=v[4], vdsty=v[5], srcx0=v[4], vsrcx1=v[8], srcy0=v[5], vsrcy1=v[9]))
    k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, vdstx=v[6], vdsty=v[7], srcx0=v[6], vsrcx1=v[10], srcy0=v[7], vsrcy1=v[11]))

    k.emit(s_sub_u32(s[S_LOOP_CTR], s[S_LOOP_CTR], 1))
    k.emit(s_cbranch_scc0(), target='LOOP')

    # add into v[4]
    k.emit(v_add_f32_e32(v[4], v[4], v[5]))
    k.emit(v_add_f32_e32(v[6], v[6], v[7]))
    k.emit(v_add_f32_e32(v[4], v[4], v[6]))

    # warp shuffle into v[4] on lane 0 using DPP row_shl within each 16-lane row
    for shift in [1, 2, 4, 8]:
      k.emit(v_add_f32_e32(v[4], DPP, v[4], vsrc0=v[4], dpp=0x100 | shift, row_mask=0xf, bank_mask=0xf, bc=1))
    # combine rows: get lane 16's value to lane 0 via permlanex16
    k.emit(v_permlanex16_b32(v[5], v[4], 0, 0))
    k.emit(v_add_f32_e32(v[4], v[4], v[5]))

    # atomic store (only on lane 0)
    k.emit(s_mov_b32(EXEC_LO, 1))
    k.emit(v_mov_b32_e32(v[0], 0))
    k.emit(global_atomic_add_f32(addr=v[0], saddr=s[4:5], data=v[4]))

    k.emit(s_sendmsg(simm16=3))  # DEALLOC_VGPRS
    k.emit(s_endpgm())
    return k.finalize(UOp.sink(UOp.special(CU_COUNT, 'gidx0'), UOp.special(LANES, 'lidx0'), out, buf, arg=KernelInfo(opts_to_apply=())))

  out = Tensor.zeros(1,).contiguous().realize()
  eval_harness("RDNA3 assembly kernel", lambda x: out.custom_kernel(x, fxn=asm_sum)[0], check=correct)

