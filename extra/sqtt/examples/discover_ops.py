#!/usr/bin/env python3
# Run all ALU and memory instructions in the ISA
import functools, inspect
from enum import Enum
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AddrSpace
from tinygrad.renderer.amd.dsl import Inst, Reg, OPERANDS, SrcField, VGPRField, SGPRField, SSrcField, SBaseField, AlignedSGPRField, BitField
from tinygrad.renderer.amd.dsl import FixedBitField, EnumBitField, s, v, NULL, VCC_LO
from extra.gemm.amd_asm_matmul import Kernel

# skip instructions that mutate wave state (PC, EXEC, allocations, signals)
SKIP = {'S_SETPC_B64', 'S_SWAPPC_B64', 'S_RFE_B64', 'S_BARRIER_SIGNAL', 'S_BARRIER_SIGNAL_ISFIRST', 'S_GET_BARRIER_STATE', 'S_ALLOC_VGPR',
        'S_BARRIER_INIT', 'S_BARRIER_JOIN', 'S_SLEEP_VAR', 'S_SENDMSG_RTN_B32', 'S_SENDMSG_RTN_B64', 'S_GETPC_B64',
        'S_FMAAK_F32', 'S_FMAMK_F32'}
SKIP_SUBSTR = ['SAVEEXEC', 'WREXEC', 'MOVREL', 'ATOMIC', 'S_BUFFER_', 'S_ATC_PROBE', 'DS_CMPSTORE_RTN', 'GS_REG', 'BARRIER', 'DS_GWS',
               'DS_WRAP_RTN_B32']

ALU_FORMATS = {'VOP1', 'VOP1_LIT', 'VOP1_SDST', 'VOP2', 'VOP2_LIT', 'VOP3', 'VOP3_SDST', 'VOP3SD', 'VOP3P', 'VOP3P_MFMA', 'VOP3PX2',
               'VOPC', 'SOP1', 'SOP1_LIT', 'SOP2', 'SOP2_LIT', 'SOPC', 'SOPC_LIT', 'SOPK', 'SOPK_LIT', 'VINTERP'}
# intentionally not testing scratch memory ops
# TODO: add mem back
MEM_FORMATS = {'VGLOBAL', 'GLOBAL', 'SMEM', 'DS'}

def should_skip(op: Enum) -> bool: return (name:=op.name) in SKIP or any(sub in name for sub in SKIP_SUBSTR)

# ** named register assignments

# ALU operands
ALU_VGPR_STRIDE = 16          # v[0], v[16], v[32], ... per ALU operand slot
ALU_SGPR_STRIDE = 4           # s[0], s[4], s[8], ... per ALU operand slot

# memory address registers
S_KERNARG_PTR = (0, 1)
S_BUF_PTR = (2, 3)
V_VADDR = (0, 1)
V_DS_ADDR = 0

# memory data registers
MEM_VGPR_BASE = 32            # v[32], v[48], ... for vdst/vdata/vsrc
MEM_VGPR_STRIDE = 16          # spacing between memory data vgpr slots
MEM_SGPR_BASE = 8             # s[8], s[10], ... for SMEM sdata
MEM_SGPR_STRIDE = 2           # spacing between memory data sgpr slots

# ** create an ALU instruction based on the operands

def reg_for_field(field: BitField, nregs: int, slot: int, name: str | None = None, is_sreg: bool = False) -> Reg | None:
  if name == 'sdst' and isinstance(field, SGPRField): return VCC_LO
  if is_sreg and not isinstance(field, VGPRField): return VCC_LO
  base_v, base_s = slot * ALU_VGPR_STRIDE, slot * ALU_SGPR_STRIDE
  if isinstance(field, VGPRField): return v[base_v:base_v+nregs-1] if nregs > 1 else v[base_v]
  if isinstance(field, SSrcField): return VCC_LO if nregs <= 2 else s[base_s:base_s+nregs-1] if nregs > 1 else s[base_s]
  if isinstance(field, SGPRField): return s[base_s:base_s+nregs-1] if nregs > 1 else s[base_s]
  if isinstance(field, SrcField): return v[base_v:base_v+nregs-1] if nregs > 1 else v[base_v]
  return None

def create_alu_inst(op: Enum, builder: functools.partial[Inst]) -> Inst:
  inst_cls, operands, slot = builder.func, OPERANDS.get(op, {}), 0
  kwargs: dict[str, Reg|int] = {}
  for name, field in inst_cls._fields:
    if isinstance(field, (FixedBitField, EnumBitField)): continue
    nregs = max(1, operands[name][1] // 32) if name in operands else 1
    is_sreg = name in operands and 'SREG' in str(operands[name][2])
    reg = reg_for_field(field, nregs, slot, name, is_sreg)
    if reg is not None: kwargs[name] = reg; slot += 1
    elif isinstance(field, BitField): kwargs[name] = field.default
  return builder(**kwargs)

# ** create a memory instruction with pre set address registers

MEM_PRESET_REGS: dict[str, dict[str, Reg]] = {
  'VGLOBAL': {'saddr': s[S_BUF_PTR[0]:S_BUF_PTR[1]], 'vaddr': v[V_VADDR[0]:V_VADDR[1]]},
  'GLOBAL': {'saddr': s[S_BUF_PTR[0]:S_BUF_PTR[1]], 'addr': v[V_DS_ADDR]},  # addr is 32-bit offset when saddr is valid SGPR
  'DS': {'addr': v[V_DS_ADDR]},
  'SMEM': {'sbase': s[S_KERNARG_PTR[0]:S_KERNARG_PTR[1]], 'soffset': NULL},
}

def create_mem_inst(op: Enum, builder: functools.partial[Inst]) -> Inst:
  inst_cls, operands, field_map = builder.func, OPERANDS.get(op, {}), MEM_PRESET_REGS.get(builder.func.__name__, {})
  kwargs: dict[str, Reg|int] = {}
  vslot, sslot = 0, 0
  for name, field in inst_cls._fields:
    if isinstance(field, (FixedBitField, EnumBitField)): continue
    if name in field_map:
      kwargs[name] = field_map[name]
      continue
    nregs = max(1, operands[name][1] // 32) if name in operands else 1
    if isinstance(field, VGPRField):
      vi = MEM_VGPR_BASE + vslot * MEM_VGPR_STRIDE
      kwargs[name] = v[vi:vi+nregs-1] if nregs > 1 else v[vi]
      vslot += 1
    elif isinstance(field, (SGPRField, AlignedSGPRField, SBaseField)):
      si = MEM_SGPR_BASE + sslot * MEM_SGPR_STRIDE
      kwargs[name] = s[si:si+nregs-1] if nregs > 1 else s[si]
      sslot += 1
    elif isinstance(field, BitField): kwargs[name] = field.default
  return builder(**kwargs)

# ** collect all memory and ALU instructions from the ins autogen

def collect_instructions() -> tuple[list[Inst], list[Inst], list[str]]:
  op_map: dict[Enum, functools.partial[Inst]] = {}
  for name, obj in inspect.getmembers(all_insts):
    if isinstance(obj, functools.partial) and len(obj.args) == 1: op_map[obj.args[0]] = obj
  alu_insts: list[Inst] = []
  mem_insts: list[Inst] = []
  skipped: list[str] = []
  for op_enum, builder in op_map.items():
    if should_skip(op_enum) or op_enum not in OPERANDS: skipped.append(op_enum.name); continue
    fmt = builder.func.__name__
    if fmt in ALU_FORMATS: alu_insts.append(create_alu_inst(op_enum, builder))
    elif fmt in MEM_FORMATS: mem_insts.append(create_mem_inst(op_enum, builder))
  return alu_insts, mem_insts, skipped

def exec_insts(insts:list):
  k = Kernel(arch)
  # ** prologue for global memory
  k.emit(s_load_b64(sdata=s[S_BUF_PTR[0]:S_BUF_PTR[1]], sbase=s[S_KERNARG_PTR[0]:S_KERNARG_PTR[1]], soffset=NULL))
  k.waitcnt(lgkm=0)
  k.emit(v_mov_b32_e32(v[V_VADDR[0]], 0))
  k.emit(v_mov_b32_e32(v[V_VADDR[1]], 0))
  # ** emit
  for inst in insts: k.emit(inst)
  k.emit(s_endpgm())
  # ** run
  NUM_THREADS, NUM_GRIDS, BUF_SIZE = 32, 1, 1024*1024
  def fxn(A: UOp, B: UOp, C: UOp) -> UOp:
    lidx, gidx = UOp.special(NUM_THREADS, "lidx0"), UOp.special(NUM_GRIDS, "gidx0")
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=BUF_SIZE, addrspace=AddrSpace.LOCAL), (), 'lds')
    sink = UOp.sink(A.base, B.base, C.base, lds, lidx, gidx, arg=KernelInfo(name="discover_ops"))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in k.finalize()))))
  A = Tensor.empty(BUF_SIZE, dtype=dtypes.uint8)
  B = Tensor.empty(1, dtype=dtypes.uint8)
  C = Tensor.empty(1, dtype=dtypes.uint8)
  Tensor.custom_kernel(A, B, C, fxn=fxn)[0].realize()
  print(f"Ran {len(insts)} instructions.")

def find_first_faulting_index(all_insts):
  def crashes(k):
    try:
      exec_insts(all_insts[:k])
      return False
    except Exception:
      return True

  n = len(all_insts)

  # 1) Find an upper bound where it crashes: (lo, hi] with crash at hi
  lo = 0
  hi = 1
  while hi <= n and not crashes(hi):
    lo = hi
    hi *= 2
  if hi > n:
    hi = n
    if not crashes(hi):
      return None  # never crashes

  # 2) Binary search for first crashing prefix length
  # invariant: crashes(lo) == False, crashes(hi) == True
  while hi - lo > 1:
    mid = (lo + hi) // 2
    if crashes(mid):
      hi = mid
    else:
      lo = mid

  # hi is the first prefix length that crashes, so the bad instruction is hi-1
  return hi - 1

if __name__ == "__main__":
  arch = Device[Device.DEFAULT].renderer.arch
  if arch.startswith("gfx12"):
    from tinygrad.runtime.autogen.amd.rdna4.ins import *
    import tinygrad.runtime.autogen.amd.rdna4.ins as all_insts
  elif arch.startswith("gfx11"):
    from tinygrad.runtime.autogen.amd.rdna3.ins import *
    import tinygrad.runtime.autogen.amd.rdna3.ins as all_insts
  else: raise RuntimeError(f"{arch} not supported yet")
  alu_insts, mem_insts, skipped = collect_instructions()
  print(f"collected {len(alu_insts)} ALU + {len(mem_insts)} memory instructions ({len(skipped)} skipped)")
  all_insts = mem_insts + alu_insts

  bad_i = find_first_faulting_index(all_insts)
  if bad_i is not None:
    print("bad_i =", bad_i)
  if bad_i is not None:
    print("faulting inst =", all_insts[bad_i])
