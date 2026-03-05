# Direct AMD GPU assembly renderer — emits Inst objects, produces GAS text via disasm()
# No LLVM. Uses HIPCompiler (COMGR) to assemble text into ELF.

import functools, math
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import AMDHIPRenderer
from extra.assembly.amd.dsl import Inst, Reg, s, v, NULL, VCC_LO, EXEC_LO, M0
from extra.assembly.amd.autogen.rdna3.ins import (s_load_b64, s_load_b128, s_mov_b32, s_waitcnt, s_endpgm, s_barrier,
  s_branch, s_cbranch_scc0, s_cbranch_scc1, s_cmp_ge_i32, s_add_i32, s_and_b32, s_lshl_b32,
  v_mov_b32_e32, v_add_f32_e32, v_add_nc_u32_e32, v_lshlrev_b32_e32, v_lshrrev_b32_e32,
  v_and_b32_e32, v_mul_lo_u32, v_cmp_lt_i32_e32,
  global_load_b32, global_load_b64, global_load_b128, global_store_b32, global_store_b64, global_store_b128,
  ds_load_b32, ds_store_b32)
from extra.assembly.amd.test.disasm import disasm
from extra.assembly.amd.isel import rdna3_isel, make_inst

# ═══════════════════════════════════════════════════════════════
# Register allocator — simple bump allocator
# ═══════════════════════════════════════════════════════════════

class RegFile:
  """Simple register allocator: bump-allocates VGPRs and SGPRs."""
  def __init__(self):
    self.next_vgpr = 1  # v0 = workitem_id_x (reserved by hardware)
    self.next_sgpr = 0  # s[0:1] = kernarg_ptr (reserved by ABI)
    self.max_vgpr = 1
    self.max_sgpr = 0

  def alloc_vgpr(self, count=1) -> Reg:
    r = v[self.next_vgpr] if count == 1 else v[self.next_vgpr:self.next_vgpr + count - 1]
    self.next_vgpr += count
    self.max_vgpr = max(self.max_vgpr, self.next_vgpr)
    return r

  def alloc_sgpr(self, count=1) -> Reg:
    # align to 2 for 64-bit, 4 for 128-bit
    if count >= 4: self.next_sgpr = (self.next_sgpr + 3) & ~3
    elif count >= 2: self.next_sgpr = (self.next_sgpr + 1) & ~1
    r = s[self.next_sgpr] if count == 1 else s[self.next_sgpr:self.next_sgpr + count - 1]
    self.next_sgpr += count
    self.max_sgpr = max(self.max_sgpr, self.next_sgpr)
    return r

# ═══════════════════════════════════════════════════════════════
# Instruction emitter (like amd_asm_matmul.Kernel)
# ═══════════════════════════════════════════════════════════════

class AsmKernel:
  def __init__(self, arch='gfx1100'):
    self.instructions: list[Inst] = []
    self.labels: dict[str, int] = {}
    self.pos = 0
    self.arch = arch
    self.regs = RegFile()
    self.lds_size = 0

  def emit(self, inst, target=None):
    self.instructions.append(inst)
    inst._target = target
    inst._pos = self.pos
    self.pos += inst.size()
    return inst

  def label(self, name):
    self.labels[name] = self.pos

  def waitcnt(self, lgkm=None, vm=None):
    vmcnt = vm if vm is not None else 63
    lgkmcnt = lgkm if lgkm is not None else 63
    expcnt = 7
    wc = (expcnt & 0x7) | ((lgkmcnt & 0x3f) << 4) | ((vmcnt & 0x3f) << 10)
    self.emit(s_waitcnt(simm16=wc))

  def resolve_branches(self):
    for inst in self.instructions:
      if hasattr(inst, '_target') and inst._target is not None:
        offset_dwords = (self.labels[inst._target] - inst._pos - inst.size()) // 4
        inst.simm16 = offset_dwords

  def to_asm(self, name='kernel', kernarg_size=0, n_params=0) -> str:
    self.resolve_branches()
    body = ['\t' + disasm(inst) for inst in self.instructions]

    hsa = [
      ('group_segment_fixed_size', self.lds_size), ('private_segment_fixed_size', 0), ('kernarg_size', kernarg_size),
      ('user_sgpr_count', 2), ('user_sgpr_kernarg_segment_ptr', 1),
      ('wavefront_size32', 1), ('uses_dynamic_stack', 0), ('enable_private_segment', 0),
      ('system_sgpr_workgroup_id_x', 1), ('system_sgpr_workgroup_id_y', 1), ('system_sgpr_workgroup_id_z', 0),
      ('system_vgpr_workitem_id', 0), ('next_free_vgpr', self.regs.max_vgpr),
      ('next_free_sgpr', max(self.regs.max_sgpr, 4)),  # minimum 4 SGPRs
      ('float_round_mode_32', 0), ('float_round_mode_16_64', 0),
      ('float_denorm_mode_32', 3), ('float_denorm_mode_16_64', 3),
      ('dx10_clamp', 1), ('ieee_mode', 1), ('fp16_overflow', 0),
      ('workgroup_processor_mode', 0), ('memory_ordered', 1), ('forward_progress', 0), ('shared_vgpr_count', 0)]

    args_meta = '\n'.join(
      f'      - .address_space: global\n        .offset: {i*8}\n        .size: 8\n        .value_kind: global_buffer'
      for i in range(n_params))

    return '\n'.join([
      '\t.text', f'\t.amdgcn_target "amdgcn-amd-amdhsa--{self.arch}"',
      f'\t.protected\t{name}', f'\t.globl\t{name}', '\t.p2align\t8', f'\t.type\t{name},@function', f'{name}:',
      *body,
      '\t.section\t.rodata,"a",@progbits', '\t.p2align\t6, 0x0', f'\t.amdhsa_kernel {name}',
      *[f'\t\t.amdhsa_{k} {v}' for k, v in hsa],
      f'\t.end_amdhsa_kernel', '\t.text', f'.Lfunc_end0:', f'\t.size\t{name}, .Lfunc_end0-{name}',
      '\t.amdgpu_metadata', '---', 'amdhsa.kernels:', '  - .args:',
      args_meta,
      f'    .group_segment_fixed_size: {self.lds_size}', '    .kernarg_segment_align: 8',
      f'    .kernarg_segment_size: {kernarg_size}', '    .max_flat_workgroup_size: 1024',
      f'    .name: {name}', '    .private_segment_fixed_size: 0',
      f'    .sgpr_count: {max(self.regs.max_sgpr, 4)}', f'    .symbol: {name}.kd',
      f'    .vgpr_count: {self.regs.max_vgpr}', '    .wavefront_size: 32',
      f'amdhsa.target: amdgcn-amd-amdhsa--{self.arch}',
      'amdhsa.version:', '  - 1', '  - 2', '...', '\t.end_amdgpu_metadata'])

# ═══════════════════════════════════════════════════════════════
# UOp → Inst rendering
# ═══════════════════════════════════════════════════════════════

# dtype → register count for VGPRs
def _dtype_regs(dt: DType) -> int:
  if isinstance(dt, PtrDType): return 2  # 64-bit pointer
  return max(1, dt.itemsize // 4) * (dt.vcount if hasattr(dt, 'vcount') and dt.vcount > 1 else 1)

# dtype → global load instruction
def _global_load(vdst, addr, saddr, offset=0, nregs=1):
  if nregs == 1: return global_load_b32(vdst=vdst, addr=addr, saddr=saddr, offset=offset)
  if nregs == 2: return global_load_b64(vdst=vdst, addr=addr, saddr=saddr, offset=offset)
  if nregs == 4: return global_load_b128(vdst=vdst, addr=addr, saddr=saddr, offset=offset)
  raise RuntimeError(f"unsupported global load size: {nregs} regs")

def _global_store(addr, data, saddr, offset=0, nregs=1):
  if nregs == 1: return global_store_b32(addr=addr, data=data, saddr=saddr, offset=offset)
  if nregs == 2: return global_store_b64(addr=addr, data=data, saddr=saddr, offset=offset)
  if nregs == 4: return global_store_b128(addr=addr, data=data, saddr=saddr, offset=offset)
  raise RuntimeError(f"unsupported global store size: {nregs} regs")

def render_kernel(uops: list[UOp], arch='gfx1100') -> str:
  """Render linearized UOps into GAS assembly text."""
  k = AsmKernel(arch)
  # r maps UOp → register (Reg)
  r: dict[UOp, Reg] = {}
  # s_args: SGPR pairs for kernel argument pointers, loaded from kernarg segment
  # kernarg_ptr is in s[0:1] (set by HSA ABI)
  kernarg_base = k.regs.alloc_sgpr(2)  # s[0:1] = kernarg segment pointer
  # system SGPRs for workgroup IDs come after user SGPRs
  # with user_sgpr_count=2, workgroup_id_x is s[2], workgroup_id_y is s[3]
  wg_id_x_sgpr = 2
  wg_id_y_sgpr = 3

  name = 'test'
  params: list[tuple[int, Reg]] = []  # (param_idx, sgpr_pair)
  specials: dict[str, Reg] = {}
  loop_stack: list[tuple[str, str, Reg]] = []  # (label_start, label_end, range_reg)
  n_params = 0

  # first pass: count params
  for u in uops:
    if u.op is Ops.PARAM: n_params = max(n_params, u.arg + 1)
    if u.op is Ops.SINK and u.arg is not None: name = u.arg.function_name

  kernarg_size = n_params * 8  # each param is 8 bytes (pointer)

  # load all kernel argument pointers
  param_sgprs: dict[int, Reg] = {}
  for i in range(n_params):
    sp = k.regs.alloc_sgpr(2)
    param_sgprs[i] = sp
    k.emit(s_load_b64(sdata=sp, sbase=kernarg_base, offset=i * 8, soffset=NULL))
  k.waitcnt(lgkm=0)

  for u in uops:
    if u.op is Ops.SINK:
      continue

    elif u.op is Ops.PARAM:
      r[u] = param_sgprs[u.arg]

    elif u.op is Ops.CONST:
      if u.dtype == dtypes.float:
        vr = k.regs.alloc_vgpr()
        k.emit(v_mov_b32_e32(vr, u.arg))
        r[u] = vr
      elif u.dtype in (dtypes.int, dtypes.int32, dtypes.uint, dtypes.uint32):
        vr = k.regs.alloc_vgpr()
        k.emit(v_mov_b32_e32(vr, u.arg if isinstance(u.arg, int) and -16 <= u.arg <= 64 else u.arg))
        r[u] = vr
      elif u.dtype == dtypes.bool:
        # booleans: 1=true, 0=false — stored in VGPR as int
        vr = k.regs.alloc_vgpr()
        k.emit(v_mov_b32_e32(vr, 1 if u.arg else 0))
        r[u] = vr
      else:
        raise RuntimeError(f"unsupported CONST dtype {u.dtype}")

    elif u.op is Ops.SPECIAL:
      kind, idx = u.arg[0], int(u.arg[-1])
      if kind == 'l':
        # local thread ID — workitem_id_{x,y,z} already in v0 (only x for 1D)
        if idx == 0:
          r[u] = v[0]  # workitem_id_x is pre-loaded in v0 by hardware
        else:
          raise RuntimeError(f"unsupported local dim {idx}")
      elif kind == 'g':
        # workgroup ID — in system SGPRs (after user SGPRs)
        sgpr_off = wg_id_x_sgpr + idx
        vr = k.regs.alloc_vgpr()
        k.emit(v_mov_b32_e32(vr, s[sgpr_off]))
        r[u] = vr
      else:
        raise RuntimeError(f"unsupported SPECIAL kind {kind}")

    elif u.op is Ops.INDEX:
      # INDEX(ptr, idx) — compute byte address: base_ptr + idx * element_size
      base = r[u.src[0]]
      idx_reg = r[u.src[1]]
      assert isinstance(u.dtype, PtrDType), f"INDEX must produce pointer, got {u.dtype}"
      elem_size = u.dtype.base.itemsize
      # compute byte offset: idx * elem_size
      offset_vr = k.regs.alloc_vgpr()
      if elem_size == 4:
        k.emit(v_lshlrev_b32_e32(offset_vr, 2, idx_reg))
      elif elem_size == 8:
        k.emit(v_lshlrev_b32_e32(offset_vr, 3, idx_reg))
      elif elem_size == 16:
        k.emit(v_lshlrev_b32_e32(offset_vr, 4, idx_reg))
      elif elem_size == 2:
        k.emit(v_lshlrev_b32_e32(offset_vr, 1, idx_reg))
      elif elem_size == 1:
        k.emit(v_mov_b32_e32(offset_vr, idx_reg))
      else:
        k.emit(v_mul_lo_u32(offset_vr, elem_size, idx_reg))
      # base is an SGPR pair (64-bit pointer), offset is VGPR — use scalar+vector addressing
      r[u] = offset_vr  # store offset VGPR; base SGPR pair stored separately
      # stash the base pointer for LOAD/STORE to use
      u._base_sgpr = base

    elif u.op is Ops.LOAD:
      idx_uop = u.src[0]
      assert idx_uop.op is Ops.INDEX or (idx_uop.op is Ops.CAST and idx_uop.src[0].op is Ops.INDEX)
      real_idx = idx_uop.src[0] if idx_uop.op is Ops.CAST else idx_uop
      base_sgpr = real_idx._base_sgpr
      offset_vr = r[real_idx]
      nregs = max(1, u.dtype.itemsize // 4) * (u.dtype.vcount if hasattr(u.dtype, 'vcount') and u.dtype.vcount > 1 else 1)
      dst = k.regs.alloc_vgpr(nregs)
      k.emit(_global_load(dst, offset_vr, base_sgpr, nregs=nregs))
      k.waitcnt(vm=0)
      r[u] = dst

    elif u.op is Ops.STORE:
      idx_uop = u.src[0]
      assert idx_uop.op is Ops.INDEX or (idx_uop.op is Ops.CAST and idx_uop.src[0].op is Ops.INDEX)
      real_idx = idx_uop.src[0] if idx_uop.op is Ops.CAST else idx_uop
      base_sgpr = real_idx._base_sgpr
      offset_vr = r[real_idx]
      val_reg = r[u.src[1]]
      nregs = max(1, u.src[1].dtype.itemsize // 4) * (u.src[1].dtype.vcount if hasattr(u.src[1].dtype, 'vcount') and u.src[1].dtype.vcount > 1 else 1)
      k.emit(_global_store(offset_vr, val_reg, base_sgpr, nregs=nregs))
      r[u] = offset_vr  # stores don't produce values, but map for dependencies

    elif u.op is Ops.ADD:
      a_reg, b_reg = r[u.src[0]], r[u.src[1]]
      dst = k.regs.alloc_vgpr()
      if u.dtype == dtypes.float:
        k.emit(v_add_f32_e32(dst, a_reg, b_reg))
      elif u.dtype in (dtypes.int, dtypes.int32, dtypes.uint, dtypes.uint32):
        k.emit(v_add_nc_u32_e32(dst, a_reg, b_reg))
      else:
        raise RuntimeError(f"unsupported ADD dtype {u.dtype}")
      r[u] = dst

    elif u.op is Ops.MUL:
      a_reg, b_reg = r[u.src[0]], r[u.src[1]]
      dst = k.regs.alloc_vgpr()
      if u.dtype == dtypes.float:
        from extra.assembly.amd.autogen.rdna3.ins import v_mul_f32_e32
        k.emit(v_mul_f32_e32(dst, a_reg, b_reg))
      elif u.dtype in (dtypes.int, dtypes.int32, dtypes.uint, dtypes.uint32):
        k.emit(v_mul_lo_u32(dst, a_reg, b_reg))
      else:
        raise RuntimeError(f"unsupported MUL dtype {u.dtype}")
      r[u] = dst

    elif u.op is Ops.SHL:
      # SHL(val, shift) -> v_lshlrev_b32(shift, val) (reversed operands)
      val_reg, shift_reg = r[u.src[0]], r[u.src[1]]
      dst = k.regs.alloc_vgpr()
      k.emit(v_lshlrev_b32_e32(dst, shift_reg, val_reg))
      r[u] = dst

    elif u.op is Ops.SHR:
      val_reg, shift_reg = r[u.src[0]], r[u.src[1]]
      dst = k.regs.alloc_vgpr()
      k.emit(v_lshrrev_b32_e32(dst, shift_reg, val_reg))
      r[u] = dst

    elif u.op is Ops.AND:
      a_reg, b_reg = r[u.src[0]], r[u.src[1]]
      dst = k.regs.alloc_vgpr()
      k.emit(v_and_b32_e32(dst, a_reg, b_reg))
      r[u] = dst

    elif u.op is Ops.CAST:
      # for now: pointer casts are noops, numeric casts need work
      if isinstance(u.dtype, PtrDType):
        r[u] = r[u.src[0]]
        if hasattr(u.src[0], '_base_sgpr'): u._base_sgpr = u.src[0]._base_sgpr
      else:
        raise RuntimeError(f"unsupported CAST {u.src[0].dtype} -> {u.dtype}")

    elif u.op is Ops.VECTORIZE:
      # VECTORIZE packs scalars into a vector — just allocate contiguous VGPRs
      count = len(u.src)
      dst = k.regs.alloc_vgpr(count)
      for i, src_u in enumerate(u.src):
        src_reg = r[src_u]
        target = v[dst.offset - 256 + i] if count > 1 else dst
        if src_reg.offset != target.offset:
          k.emit(v_mov_b32_e32(target, src_reg))
      r[u] = dst

    elif u.op is Ops.GEP:
      # GEP extracts element from vector — just offset into the VGPR range
      base_reg = r[u.src[0]]
      idx = u.arg[0]
      r[u] = v[base_reg.offset - 256 + idx]

    elif u.op in (Ops.NOOP, Ops.GROUP, Ops.AFTER):
      if u.src: r[u] = r[u.src[0]]

    elif u.op is Ops.RANGE:
      # loop: counter starts at 0, increments by 1, bound is src[0]
      label_start = f'loop_{id(u)}'
      label_end = f'end_{id(u)}'
      ctr = k.regs.alloc_vgpr()
      k.emit(v_mov_b32_e32(ctr, 0))
      k.label(label_start)
      r[u] = ctr
      loop_stack.append((label_start, label_end, ctr))

    elif u.op is Ops.END:
      label_start, label_end, ctr = loop_stack.pop()
      # increment counter
      k.emit(v_add_nc_u32_e32(ctr, 1, ctr))
      # compare and branch: use SGPR compare since loop bound should be uniform
      bound_uop = u.src[1]  # the RANGE uop's src[0] is the bound
      # actually END.src = (range_uop, ...), range_uop.src[0] = bound
      range_uop = u.src[0]
      bound_reg = r[range_uop.src[0]]
      k.emit(v_cmp_lt_i32_e32(ctr, bound_reg))
      k.emit(s_cbranch_scc1(), target=label_start)
      k.label(label_end)

    elif u.op is Ops.BARRIER:
      k.emit(s_barrier())

    elif u.op is Ops.DEFINE_LOCAL:
      # LDS allocation — just track size, address computed at use time
      r[u] = v[0]  # placeholder, LDS addressing handled separately
      k.lds_size = max(k.lds_size, u.dtype.size * u.dtype.base.itemsize if hasattr(u.dtype, 'size') else 0)

    elif u.op is Ops.DEFINE_REG:
      # register "spill" region — allocate VGPRs
      size = u.dtype.size if hasattr(u.dtype, 'size') else 1
      vr = k.regs.alloc_vgpr(size)
      r[u] = vr

    elif u.op is Ops.CMPLT:
      # compare: write result to VCC, then v_cndmask to get bool in VGPR
      a_reg, b_reg = r[u.src[0]], r[u.src[1]]
      dst = k.regs.alloc_vgpr()
      k.emit(v_cmp_lt_i32_e32(a_reg, b_reg))
      from extra.assembly.amd.autogen.rdna3.ins import v_cndmask_b32_e32
      k.emit(v_cndmask_b32_e32(dst, 0, 1))
      r[u] = dst

    elif u.op is Ops.CMPNE:
      from extra.assembly.amd.autogen.rdna3.ins import v_cmp_ne_u32_e32, v_cndmask_b32_e32
      a_reg, b_reg = r[u.src[0]], r[u.src[1]]
      dst = k.regs.alloc_vgpr()
      k.emit(v_cmp_ne_u32_e32(a_reg, b_reg))
      k.emit(v_cndmask_b32_e32(dst, 0, 1))
      r[u] = dst

    elif u.op is Ops.WHERE:
      from extra.assembly.amd.autogen.rdna3.ins import v_cndmask_b32_e32
      cond_reg, true_reg, false_reg = r[u.src[0]], r[u.src[1]], r[u.src[2]]
      dst = k.regs.alloc_vgpr()
      # set VCC from condition (nonzero = true)
      k.emit(v_cmp_lt_i32_e32(0, cond_reg))  # VCC = cond_reg != 0
      k.emit(v_cndmask_b32_e32(dst, false_reg, true_reg))
      r[u] = dst

    else:
      raise RuntimeError(f"unsupported UOp: {u.op} dtype={u.dtype}")

  # epilogue
  k.waitcnt(vm=0, lgkm=0)
  k.emit(s_endpgm())

  return k.to_asm(name=name, kernarg_size=kernarg_size, n_params=n_params)

# ═══════════════════════════════════════════════════════════════
# AMDAssemblyRenderer: Renderer subclass for tinygrad integration
# ═══════════════════════════════════════════════════════════════

class AMDAssemblyRenderer(Renderer):
  device = "AMD"
  suffix = "s"  # GAS assembly
  supports_float4 = True
  has_local = True
  has_shared = True
  global_max = AMDHIPRenderer.global_max
  shared_max = AMDHIPRenderer.shared_max

  def __init__(self, arch: str):
    from tinygrad.runtime.support.compiler_amd import HIPCompiler
    self.arch = arch
    self.compiler = HIPCompiler(arch)

  def render(self, uops: list[UOp]) -> str:
    return render_kernel(uops, arch=self.arch)

  def __reduce__(self): return self.__class__, (self.arch,)
