# RDNA3 Renderer - uses assembly DSL and RDNARegAlloc
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace
from tinygrad.renderer import Renderer
from tinygrad.helpers import get_single_element
from tinygrad.renderer.rdna_regalloc import RDNARegAlloc
from tinygrad.renderer.rdna_uops import rdna_matcher
from tinygrad.renderer.cstyle import create_non_native_float_pats, cast_float_to_bf16
from tinygrad.codegen.late.devectorizer import no_vectorized_alu
from tinygrad.codegen.opt import tc
from extra.assembly.rdna3.lib import Inst
from extra.assembly.rdna3.asm import waitcnt
from extra.assembly.rdna3.autogen import (
  v, s, VGPR, SGPR, VCC_LO, EXEC_LO, NULL, OFF, SrcEnum,
  # VOP1
  v_mov_b32_e32, v_cvt_f32_i32_e32, v_cvt_i32_f32_e32, v_cvt_f32_u32_e32, v_cvt_u32_f32_e32,
  v_cvt_f16_f32_e32, v_cvt_f32_f16_e32, v_rcp_f32_e32, v_rsq_f32_e32, v_sqrt_f32_e32,
  v_exp_f32_e32, v_log_f32_e32, v_trunc_f32_e32, v_sin_f32_e32,
  # VOP2
  v_add_f32_e32, v_sub_f32_e32, v_mul_f32_e32, v_and_b32_e32, v_or_b32_e32, v_xor_b32_e32,
  v_add_nc_u32_e32, v_sub_nc_u32_e32, v_lshlrev_b32_e32, v_lshrrev_b32_e32, v_ashrrev_i32_e32,
  v_max_f32_e32, v_max_i32_e32, v_max_u32_e32,
  # VOP3
  v_add_f32, v_mul_f32, v_fma_f32, v_fma_f64, v_mad_u64_u32, v_lshlrev_b64, v_add3_u32,
  v_mul_lo_u32, v_bfe_u32, v_bfe_i32, v_add_co_u32, v_add_co_ci_u32_e32, v_cndmask_b32,
  v_cmp_lt_f32, v_cmp_eq_f32, v_cmp_neq_f32, v_cmp_gt_f32,
  v_cmp_lt_i32, v_cmp_eq_i32, v_cmp_ne_i32, v_cmp_gt_i32,
  v_cmp_lt_u32, v_cmp_eq_u32, v_cmp_ne_u32, v_cmp_gt_u32,
  # SOPP/SOP
  s_endpgm, s_waitcnt, s_barrier, s_branch, s_cbranch_vccnz, s_cbranch_execz, s_sendmsg,
  s_mov_b32, s_mov_b64, s_and_saveexec_b32, s_or_b32,
  # SMEM
  s_load_b64, s_load_b128,
  # FLAT/GLOBAL
  global_load_b32, global_load_b64, global_load_b128, global_load_u16, global_load_u8, global_load_i8,
  global_store_b32, global_store_b64, global_store_b128, global_store_b16, global_store_b8,
  # DS (local memory)
  ds_load_b32, ds_load_b64, ds_load_b128, ds_store_b32, ds_store_b64, ds_store_b128,
  # WMMA
  v_wmma_f32_16x16x16_f16,
)

class RDNARenderer(Renderer):
  device = "AMD"
  suffix = "RDNA"
  supports_float4 = True
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 1024)
  shared_max = 65536
  max_upcast_size = 16
  rdna_bf16_cast = PatternMatcher([(UPat(Ops.CAST, dtype=dtypes.bfloat16, src=(UPat.var("x", dtype=dtypes.float),)), cast_float_to_bf16)])
  extra_matcher = rdna_matcher + create_non_native_float_pats((dtypes.bfloat16,)) + rdna_bf16_cast
  tensor_cores = tc.amd_rdna3

  def __init__(self, arch: str = "gfx1100"):
    self.arch = arch

  def __reduce__(self): return self.__class__, (self.arch,)

  def render(self, uops: list[UOp]) -> str:
    ra = RDNARegAlloc(uops)  # Register allocator with liveness analysis
    r: dict[UOp, VGPR | SGPR | int] = {}  # UOp -> register mapping
    code: list[Inst] = []  # Generated instructions
    bufs: list[UOp] = []
    kernarg_offset: dict[UOp, int] = {}
    current_offset = 0
    lds_size = 0
    labels: dict[str, int] = {}  # Label -> instruction index
    pending_waits: set[UOp] = set()  # Track loads that need waits before use

    def maybe_wait(srcs):
      """Emit waitcnt if any source (or transitive source) is pending from an async load."""
      def needs_wait(src, visited=None):
        if visited is None: visited = set()
        if src in visited: return False
        visited.add(src)
        if src in pending_waits: return True
        # Check transitive sources (e.g., GEP -> LOAD)
        for s in src.src:
          if needs_wait(s, visited): return True
        return False
      for src in srcs:
        if needs_wait(src):
          code.append(s_waitcnt(waitcnt(vmcnt=0, lgkmcnt=0)))
          pending_waits.clear()
          return

    # Fixed registers
    kernarg_ptr = s[0:2]  # s[0:1] holds kernarg pointer
    group_id = (s[2], s[3], s[4])  # s[2:4] holds group ID xyz
    local_id = (v[0], v[1], v[2])  # v[0:2] holds local ID xyz

    def get_reg(u: UOp) -> VGPR | SGPR | int:
      """Get register for a UOp, handling constants and aliases."""
      import struct, math
      if u in r: return r[u]
      if u.op is Ops.CONST:
        val = u.arg
        if isinstance(val, bool): return 1 if val else 0  # Handle bool before int (bool is subclass of int)
        if isinstance(val, float):
          if val in (0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0): return val
          # Convert inf/nan to hex representation
          if math.isinf(val) or math.isnan(val):
            val = struct.unpack("I", struct.pack("f", val))[0]
        elif isinstance(val, int) and -16 <= val <= 64: return val
        # Load literal constant into register
        reg = ra.alloc_vgpr(u)
        code.append(v_mov_b32_e32(reg, val))
        r[u] = reg
        return reg
      raise ValueError(f"No register for {u}")

    def emit_cmp(op: Ops, dtype: DType, dst: VGPR, a, b):
      """Emit comparison instruction based on dtype."""
      # VOPC encoding: src0 can be constant, vsrc1 must be VGPR
      # For non-symmetric comparisons (LT), we swap and use the opposite (GT)
      cmp_map = {
        (Ops.CMPLT, dtypes.float32): v_cmp_lt_f32, (Ops.CMPLT, dtypes.int32): v_cmp_lt_i32, (Ops.CMPLT, dtypes.uint32): v_cmp_lt_u32,
        (Ops.CMPEQ, dtypes.float32): v_cmp_eq_f32, (Ops.CMPEQ, dtypes.int32): v_cmp_eq_i32, (Ops.CMPEQ, dtypes.uint32): v_cmp_eq_u32,
        (Ops.CMPNE, dtypes.float32): v_cmp_neq_f32, (Ops.CMPNE, dtypes.int32): v_cmp_ne_i32, (Ops.CMPNE, dtypes.uint32): v_cmp_ne_u32,
      }
      # GT versions for swapping CMPLT: a < b ⇔ b > a
      cmp_gt_map = {
        (Ops.CMPLT, dtypes.float32): v_cmp_gt_f32, (Ops.CMPLT, dtypes.int32): v_cmp_gt_i32, (Ops.CMPLT, dtypes.uint32): v_cmp_gt_u32,
      }
      base_dtype = dtypes.float32 if dtypes.is_float(dtype) else dtypes.int32 if dtype in (dtypes.int8, dtypes.int16, dtypes.int32) else dtypes.uint32
      def is_const(x): return isinstance(x, (int, float))
      # For CMPLT with constant in vsrc1 position, swap and use GT
      if op is Ops.CMPLT and is_const(b) and not is_const(a):
        cmp_fn = cmp_gt_map.get((op, base_dtype))
        if cmp_fn:
          code.append(cmp_fn(b, a))  # b > a ⇔ a < b
          code.append(v_cndmask_b32(dst, 0, 1, VCC_LO))
          return
      # For symmetric comparisons, just swap
      cmp_fn = cmp_map.get((op, base_dtype))
      if cmp_fn:
        if op in (Ops.CMPEQ, Ops.CMPNE) and is_const(b) and not is_const(a):
          a, b = b, a
        code.append(cmp_fn(a, b))  # VOPC implicitly writes to VCC
        code.append(v_cndmask_b32(dst, 0, 1, VCC_LO))  # Use VOP3: src2 is the VCC condition

    def emit_alu(u: UOp, dst: VGPR):
      """Emit ALU instruction."""
      op, dtype = u.op, u.dtype
      srcs = [get_reg(s) for s in u.src]
      a, b = (srcs[0], srcs[1]) if len(srcs) >= 2 else (srcs[0], 0)

      # For VOP2: src0 can be constant/literal, but vsrc1 must be VGPR
      # Swap operands for commutative ops when b is constant and a is not
      def is_const(x): return isinstance(x, (int, float))
      if op in (Ops.ADD, Ops.MUL, Ops.AND, Ops.OR, Ops.XOR, Ops.MAX) and is_const(b) and not is_const(a):
        a, b = b, a

      if op is Ops.ADD:
        code.append(v_add_f32_e32(dst, a, b) if dtypes.is_float(dtype) else v_add_nc_u32_e32(dst, a, b))
      elif op is Ops.SUB:
        code.append(v_sub_f32_e32(dst, a, b) if dtypes.is_float(dtype) else v_sub_nc_u32_e32(dst, a, b))
      elif op is Ops.MUL:
        code.append(v_mul_f32_e32(dst, a, b) if dtypes.is_float(dtype) else v_mul_lo_u32(dst, a, b))
      elif op is Ops.AND: code.append(v_and_b32_e32(dst, a, b))
      elif op is Ops.OR: code.append(v_or_b32_e32(dst, a, b))
      elif op is Ops.XOR: code.append(v_xor_b32_e32(dst, a, b))
      elif op is Ops.SHL: code.append(v_lshlrev_b32_e32(dst, b, a))
      elif op is Ops.SHR: code.append(v_lshrrev_b32_e32(dst, b, a) if dtype in (dtypes.uint32, dtypes.uint16, dtypes.uint8) else v_ashrrev_i32_e32(dst, b, a))
      elif op is Ops.MAX:
        if dtypes.is_float(dtype): code.append(v_max_f32_e32(dst, a, b))
        elif dtype in (dtypes.int32, dtypes.int16, dtypes.int8): code.append(v_max_i32_e32(dst, a, b))
        else: code.append(v_max_u32_e32(dst, a, b))
      elif op is Ops.MULACC:
        c = srcs[2] if len(srcs) > 2 else 0
        if dtype == dtypes.float64:
          code.append(v_fma_f64(dst, a, b, c))
        else:
          code.append(v_fma_f32(dst, a, b, c))
      elif op is Ops.RECIPROCAL: code.append(v_rcp_f32_e32(dst, a))
      elif op is Ops.SQRT: code.append(v_sqrt_f32_e32(dst, a))
      elif op is Ops.EXP2: code.append(v_exp_f32_e32(dst, a))
      elif op is Ops.LOG2: code.append(v_log_f32_e32(dst, a))
      elif op is Ops.TRUNC: code.append(v_trunc_f32_e32(dst, a))
      elif op is Ops.SIN: code.append(v_sin_f32_e32(dst, a))
      elif op is Ops.NEG:
        if dtypes.is_float(dtype): code.append(v_mul_f32_e32(dst, -1.0, a))
        else: code.append(v_sub_nc_u32_e32(dst, 0, a))
      elif op in (Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE):
        emit_cmp(op, u.src[0].dtype, dst, a, b)
      elif op is Ops.WHERE:
        cond, true_val, false_val = srcs[0], srcs[1], srcs[2]
        code.append(v_cmp_ne_i32(0, cond))  # VOPC: src0=constant, vsrc1=VGPR; 0 != cond ⇔ cond != 0
        code.append(v_cndmask_b32(dst, false_val, true_val, VCC_LO))  # Use VOP3
      else:
        code.append(v_mov_b32_e32(dst, a))  # Fallback: just move

    # Process UOps
    for i, u in enumerate(uops):
      ra.free_dead_regs(i)  # Free registers that are no longer needed

      if u.op is Ops.SINK: continue

      elif u.op is Ops.DEFINE_GLOBAL:
        bufs.append(u)
        kernarg_offset[u] = current_offset
        current_offset += 8  # 64-bit pointer
        r[u] = ra.alloc_sgpr_pair(u)

      elif u.op is Ops.DEFINE_VAR:
        kernarg_offset[u] = current_offset
        current_offset += 4
        r[u] = ra.alloc_sgpr(u) or ra.alloc_vgpr(u)

      elif u.op is Ops.DEFINE_LOCAL:
        # Size can be from arg (tuple/int) or from dtype.size
        if isinstance(u.arg, tuple):
          size = u.arg[1]
        elif isinstance(u.arg, int) and u.arg > 0:
          size = u.arg
        else:
          size = u.dtype.size if hasattr(u.dtype, 'size') else 0
        u_lds_size = u.dtype.itemsize * size
        lds_size = max(lds_size, u_lds_size)
        r[u] = 0  # LDS base is always 0 (offsets are relative)

      elif u.op is Ops.DEFINE_REG:
        # Register-space buffer for WMMA/tensor core outputs
        # dtype is a pointer to register space with size = element count
        num_regs = u.dtype.size if hasattr(u.dtype, 'size') and u.dtype.size > 0 else 16
        r[u] = ra.alloc_vgpr_range(u, num_regs)

      elif u.op is Ops.SPECIAL:
        # SPECIAL arg is a string like 'lidx0', 'gidx1', etc.
        # Local IDs are packed in v0: bits 0-9=X, 10-19=Y, 20-29=Z
        # Group IDs are in s2, s3, s4
        axis = u.arg
        idx = int(axis[-1])
        dst = ra.alloc_vgpr(u)
        r[u] = dst
        if axis.startswith('l'):  # Local ID - extract from packed v0
          if idx == 0:
            code.append(v_and_b32_e32(dst, 0x3ff, v[0]))
          else:
            code.append(v_bfe_u32(dst, v[0], idx * 10, 10))
        else:  # Group ID - copy from SGPR to VGPR
          code.append(v_mov_b32_e32(dst, s[2 + idx]))

      elif u.op is Ops.CONST:
        pass  # Handled lazily in get_reg

      elif u.op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR,
                    Ops.MAX, Ops.MULACC, Ops.RECIPROCAL, Ops.SQRT, Ops.EXP2, Ops.LOG2,
                    Ops.TRUNC, Ops.NEG, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE, Ops.WHERE, Ops.SIN):
        maybe_wait(u.src)  # Wait for any pending loads used by this operation
        dst = ra.alloc_vgpr_pair(u) if RDNARegAlloc.needs_vgpr_pair(u.dtype) else ra.alloc_vgpr(u)
        r[u] = dst
        emit_alu(u, dst)

      elif u.op is Ops.CAST:
        maybe_wait(u.src)  # Wait for source value
        src_reg = get_reg(u.src[0])
        src_dtype, dst_dtype = u.src[0].dtype, u.dtype
        if src_dtype == dst_dtype:
          r[u] = src_reg
        else:
          dst = ra.alloc_vgpr_pair(u) if RDNARegAlloc.needs_vgpr_pair(dst_dtype) else ra.alloc_vgpr(u)
          r[u] = dst
          # Float conversions
          if dst_dtype == dtypes.float32 and src_dtype in (dtypes.int32, dtypes.int16, dtypes.int8):
            code.append(v_cvt_f32_i32_e32(dst, src_reg))
          elif dst_dtype == dtypes.float32 and src_dtype in (dtypes.uint32, dtypes.uint16, dtypes.uint8):
            code.append(v_cvt_f32_u32_e32(dst, src_reg))
          elif src_dtype == dtypes.float32 and dst_dtype in (dtypes.int32, dtypes.int16, dtypes.int8):
            code.append(v_cvt_i32_f32_e32(dst, src_reg))
          elif src_dtype == dtypes.float32 and dst_dtype in (dtypes.uint32, dtypes.uint16, dtypes.uint8):
            code.append(v_cvt_u32_f32_e32(dst, src_reg))
          elif dst_dtype == dtypes.float16 and src_dtype == dtypes.float32:
            code.append(v_cvt_f16_f32_e32(dst, src_reg))
          elif dst_dtype == dtypes.float32 and src_dtype == dtypes.float16:
            code.append(v_cvt_f32_f16_e32(dst, src_reg))
          elif dst_dtype == dtypes.bfloat16 and src_dtype == dtypes.float32:
            code.append(v_lshrrev_b32_e32(dst, 16, src_reg))
          elif dst_dtype == dtypes.float32 and src_dtype == dtypes.bfloat16:
            code.append(v_lshlrev_b32_e32(dst, 16, src_reg))
          elif dst_dtype == dtypes.bool:
            # Cast to bool: 0 -> 0, non-zero -> 1
            code.append(v_cmp_ne_i32(0, src_reg))  # VCC = (src != 0)
            code.append(v_cndmask_b32(dst, 0, 1, VCC_LO))  # dst = VCC ? 1 : 0
          elif dtypes.is_float(dst_dtype) and src_dtype == dtypes.bool:
            # Cast bool to float: 0 -> 0.0, 1 -> 1.0
            # src_reg is 0 or 1 as an integer, need to convert to VCC condition first
            code.append(v_cmp_ne_i32(0, src_reg))  # VCC = (src != 0)
            code.append(v_cndmask_b32(dst, 0, 0x3f800000, VCC_LO))  # dst = VCC ? 1.0f : 0.0f
          else:
            code.append(v_mov_b32_e32(dst, src_reg))

      elif u.op is Ops.BITCAST:
        r[u] = get_reg(u.src[0])  # Bitcast is just a reinterpretation

      elif u.op is Ops.INDEX:
        buf, idx = u.src[0], u.src[1]
        cond = u.src[2] if len(u.src) > 2 else None  # Optional condition (3rd source)
        if isinstance(buf.dtype, PtrDType) and buf.dtype.addrspace == AddrSpace.LOCAL:
          # Local memory: just the offset
          r[u] = (get_reg(idx), cond)  # Store as tuple with condition
        elif isinstance(buf.dtype, PtrDType) and buf.dtype.addrspace == AddrSpace.REG:
          # Register space: just the offset (will be used to index into VGPR range)
          r[u] = (get_reg(idx), cond)
        else:
          # Global memory: INDEX stores just the byte offset (idx already scaled by rdna_uops)
          # LOAD/STORE will use saddr with the buffer's SGPR pair as base
          idx_reg = get_reg(idx)
          if isinstance(idx_reg, (int, float)):
            # Constant offset - load into VGPR
            dst = ra.alloc_vgpr(u)
            code.append(v_mov_b32_e32(dst, idx_reg))
            r[u] = (dst, cond)
          else:
            r[u] = (idx_reg, cond)

      elif u.op is Ops.LOAD:
        idx_uop = u.src[0]
        default_uop = u.src[1] if len(u.src) > 1 else None  # Optional default value
        # For INDEX, r[idx_uop] is a tuple (addr, cond)
        idx_result = r.get(idx_uop) if idx_uop in r else get_reg(idx_uop)
        if isinstance(idx_result, tuple):
          addr, cond_uop = idx_result
        else:
          addr, cond_uop = idx_result, None
        dtype = u.dtype
        itemsize = dtype.itemsize if hasattr(dtype, 'itemsize') else 4
        buf_uop = idx_uop.src[0] if idx_uop.op is Ops.INDEX else idx_uop

        if isinstance(buf_uop.dtype, PtrDType) and buf_uop.dtype.addrspace == AddrSpace.LOCAL:
          # Local memory load - allocate VGPRs based on size
          if itemsize == 16:
            dst = ra.alloc_vgpr_range(u, 4)
            code.append(ds_load_b128(vdst=dst, addr=addr))
          elif itemsize == 8:
            dst = ra.alloc_vgpr_pair(u)
            code.append(ds_load_b64(vdst=dst, addr=addr))
          else:
            dst = ra.alloc_vgpr(u)
            code.append(ds_load_b32(vdst=dst, addr=addr))
        elif isinstance(buf_uop.dtype, PtrDType) and buf_uop.dtype.addrspace == AddrSpace.REG:
          # Register-space load: return register from the buffer's range
          buf_result = r.get(buf_uop) if buf_uop in r else get_reg(buf_uop)
          buf_reg = buf_result[0] if isinstance(buf_result, tuple) else buf_result
          if isinstance(addr, (int, float)):
            # addr is the element index, not byte offset
            reg_offset = int(addr)
            dst = v[buf_reg.idx + reg_offset]
          else:
            # Variable offset - use first register as fallback (TODO: proper indirect)
            dst = v[buf_reg.idx]
          r[u] = dst
          continue  # Skip the rest, no actual load needed
        else:
          # Global memory load: use buffer SGPR pair as saddr
          buf_result = r.get(buf_uop) if buf_uop in r else get_reg(buf_uop)
          buf_reg = buf_result[0] if isinstance(buf_result, tuple) else buf_result
          if itemsize == 16:
            dst = ra.alloc_vgpr_range(u, 4)  # b128 needs 4 VGPRs
          elif itemsize == 8:
            dst = ra.alloc_vgpr_pair(u)  # b64 needs 2 VGPRs
          else:
            dst = ra.alloc_vgpr(u)  # b32 and smaller

          # Handle conditional load with default value
          if cond_uop is not None and default_uop is not None:
            # First, initialize dst with default value
            default_val = get_reg(default_uop)
            if isinstance(default_val, (int, float)):
              for j in range(itemsize // 4):
                code.append(v_mov_b32_e32(v[dst.idx + j] if hasattr(dst, 'idx') else dst, default_val))
            else:
              # Copy from default vector
              for j in range(itemsize // 4):
                code.append(v_mov_b32_e32(v[dst.idx + j] if hasattr(dst, 'idx') else dst, v[default_val.idx + j] if hasattr(default_val, 'idx') else default_val))

            # Then set up exec mask based on condition and do load
            cond_reg = get_reg(cond_uop)
            code.append(v_cmp_ne_i32(0, cond_reg))  # VCC = (cond != 0)
            code.append(s_and_saveexec_b32(s[12], VCC_LO))  # Save exec, mask with condition
            # Now do the load (only executed by lanes where condition is true)

          if itemsize == 1:
            code.append(global_load_u8(vdst=dst, addr=addr, saddr=buf_reg))
          elif itemsize == 2:
            code.append(global_load_u16(vdst=dst, addr=addr, saddr=buf_reg))
          elif itemsize == 4:
            code.append(global_load_b32(vdst=dst, addr=addr, saddr=buf_reg))
          elif itemsize == 8:
            code.append(global_load_b64(vdst=dst, addr=addr, saddr=buf_reg))
          else:
            code.append(global_load_b128(vdst=dst, addr=addr, saddr=buf_reg))

          # Restore exec mask if we masked it
          if cond_uop is not None and default_uop is not None:
            code.append("s_mov_b32 exec_lo, s12")

        r[u] = dst
        pending_waits.add(u)  # Track that this load result needs wait before use

      elif u.op is Ops.STORE:
        maybe_wait(u.src)  # Wait for value to store
        idx_uop, val_uop = u.src[0], u.src[1]
        # For INDEX, r[idx_uop] is a tuple (addr, cond)
        idx_result = r.get(idx_uop) if idx_uop in r else get_reg(idx_uop)
        if isinstance(idx_result, tuple):
          addr, cond_uop = idx_result
        else:
          addr, cond_uop = idx_result, None
        val = get_reg(val_uop)
        dtype = val_uop.dtype
        itemsize = dtype.itemsize if hasattr(dtype, 'itemsize') else 4
        # STORE data operand must be a VGPR, not an inline constant
        if isinstance(val, (int, float)):
          if itemsize == 8:
            # 64-bit constant needs 2 VGPRs
            tmp = ra.alloc_vgpr_pair(val_uop)
            code.append(v_mov_b32_e32(v[tmp.idx], val & 0xffffffff if isinstance(val, int) else val))
            code.append(v_mov_b32_e32(v[tmp.idx + 1], (val >> 32) & 0xffffffff if isinstance(val, int) else 0))
            val = tmp
          else:
            tmp = ra.alloc_vgpr(val_uop)
            code.append(v_mov_b32_e32(tmp, val))
            val = tmp
        buf_uop = idx_uop.src[0] if idx_uop.op is Ops.INDEX else idx_uop

        if isinstance(buf_uop.dtype, PtrDType) and buf_uop.dtype.addrspace == AddrSpace.LOCAL:
          # Local memory store - use appropriate instruction based on size
          if itemsize == 16:
            code.append(ds_store_b128(addr=addr, data0=val))
          elif itemsize == 8:
            code.append(ds_store_b64(addr=addr, data0=val))
          else:
            code.append(ds_store_b32(addr=addr, data0=val))
        elif isinstance(buf_uop.dtype, PtrDType) and buf_uop.dtype.addrspace == AddrSpace.REG:
          # Register-space store: copy to register range
          buf_result = r.get(buf_uop) if buf_uop in r else get_reg(buf_uop)
          buf_reg = buf_result[0] if isinstance(buf_result, tuple) else buf_result
          if isinstance(addr, (int, float)):
            # addr is the element index, not byte offset
            reg_offset = int(addr)
            code.append(v_mov_b32_e32(v[buf_reg.idx + reg_offset], val))
          else:
            # Variable offset - use first register as fallback (TODO: proper indirect)
            code.append(v_mov_b32_e32(v[buf_reg.idx], val))
        else:
          # Global memory store: use buffer SGPR pair as saddr
          buf_result = r.get(buf_uop) if buf_uop in r else get_reg(buf_uop)
          buf_reg = buf_result[0] if isinstance(buf_result, tuple) else buf_result
          if itemsize == 1:
            code.append(global_store_b8(addr=addr, data=val, saddr=buf_reg))
          elif itemsize == 2:
            code.append(global_store_b16(addr=addr, data=val, saddr=buf_reg))
          elif itemsize == 4:
            code.append(global_store_b32(addr=addr, data=val, saddr=buf_reg))
          elif itemsize == 8:
            code.append(global_store_b64(addr=addr, data=val, saddr=buf_reg))
          else:
            code.append(global_store_b128(addr=addr, data=val, saddr=buf_reg))

      elif u.op is Ops.RANGE:
        loop_var = ra.alloc_vgpr(u)
        r[u] = loop_var
        code.append(v_mov_b32_e32(loop_var, -1))  # Start at -1
        code.append(f"s_branch .L_END_{i}")  # Jump to loop end check
        code.append(f".L_BODY_{i}:")  # Loop body label

      elif u.op is Ops.END:
        if len(u.src) >= 2 and u.src[1].op is Ops.RANGE:
          range_uop = u.src[1]
          range_idx = uops.index(range_uop)
          loop_var = r[range_uop]
          bound = get_reg(range_uop.src[0])
          code.append(f".L_END_{range_idx}:")  # Loop end label
          code.append(v_add_nc_u32_e32(loop_var, 1, loop_var))  # VOP2: src0=constant, vsrc1=VGPR
          # VOPC: src0 can be constant/SGPR, vsrc1 must be VGPR
          # We need loop_var < bound. Use bound > loop_var since loop_var is VGPR (vsrc1)
          code.append(v_cmp_gt_i32(bound, loop_var))  # bound > loop_var ⇔ loop_var < bound
          code.append(f"s_cbranch_vccnz .L_BODY_{range_idx}")  # Branch back to loop body

      elif u.op is Ops.BARRIER:
        code.append(s_barrier())

      elif u.op is Ops.AFTER:
        # AFTER ensures previous operations complete, then returns the buffer
        # src[0] is the buffer, src[1] is the operation that must complete
        buf_uop = u.src[0]
        r[u] = get_reg(buf_uop)

      elif u.op is Ops.VECTORIZE:
        # Allocate contiguous registers for vector
        count = len(u.src)
        dst_range = ra.alloc_vgpr_range(u, count)
        for j, src in enumerate(u.src):
          src_reg = get_reg(src)
          code.append(v_mov_b32_e32(v[dst_range.idx + j], src_reg))
        r[u] = dst_range

      elif u.op is Ops.GEP:
        vec_reg = get_reg(u.src[0])
        idx = u.arg[0] if isinstance(u.arg, tuple) else u.arg
        r[u] = v[vec_reg.idx + idx] if isinstance(vec_reg, VGPR) else vec_reg

      elif u.op is Ops.GROUP:
        # GROUP is handled at a higher level; just skip (NOOP)
        pass

      elif u.op is Ops.IF:
        # Save exec and mask with condition
        cond = get_reg(u.src[0])
        code.append(v_cmp_ne_i32(0, cond))  # condition != 0
        code.append(s_and_saveexec_b32(s[12], VCC_LO))  # Save exec, AND with condition
        code.append(f"s_cbranch_execz .L_ENDIF_{i}")  # Skip if all lanes masked

      elif u.op is Ops.ENDIF:
        # Restore exec mask
        if_uop = u.src[0]
        if_idx = uops.index(if_uop)
        code.append(f".L_ENDIF_{if_idx}:")
        code.append("s_mov_b32 exec_lo, s12")  # exec_lo = saved

    # Emit kernel prologue (load kernargs)
    prologue: list[Inst] = []
    for buf in bufs:
      offset = kernarg_offset[buf]
      dst = r[buf]
      prologue.append(s_load_b64(dst, kernarg_ptr, offset))
    prologue.append(s_waitcnt(waitcnt(lgkmcnt=0)))

    # Emit epilogue - MSG_DEALLOC_VGPRS = 3
    epilogue: list[Inst] = [
      s_waitcnt(waitcnt(vmcnt=0, lgkmcnt=0)),
      s_sendmsg(simm16=3),  # sendmsg(MSG_DEALLOC_VGPRS)
      s_endpgm(),
    ]

    # Combine and convert to text
    all_code = prologue + code + epilogue
    asm_lines = [item if isinstance(item, str) else item.disasm() for item in all_code]

    # Generate kernel header
    v_cnt, s_cnt = ra.max_vgpr, ra.max_sgpr
    header = f""".text
.amdgcn_target "amdgcn-amd-amdhsa--{self.arch}"
.global kernel
.type kernel,@function
.p2align 8
kernel:
"""
    body = "\n".join(asm_lines)
    footer = self._render_metadata("kernel", bufs, v_cnt, s_cnt, lds_size)
    return header + body + "\n" + footer

  def _render_metadata(self, name: str, bufs: list[UOp], v_cnt: int, s_cnt: int, lds_size: int) -> str:
    """Generate AMDHSA kernel descriptor and metadata."""
    import yaml
    kernargs = []
    offset = 0
    for buf in bufs:
      kernargs.append({".name": f"buf{len(kernargs)}", ".offset": offset, ".size": 8, ".value_kind": "global_buffer"})
      offset += 8

    metadata = {
      "amdhsa.kernels": [{
        ".name": name,
        ".symbol": f"{name}.kd",
        ".kernarg_segment_size": offset,
        ".group_segment_fixed_size": lds_size,
        ".private_segment_fixed_size": 0,
        ".kernarg_segment_align": 8,
        ".wavefront_size": 32,
        ".sgpr_count": s_cnt + 6,
        ".vgpr_count": v_cnt,
        ".max_flat_workgroup_size": 1024,
        ".args": kernargs,
      }],
      "amdhsa.version": [1, 2],
    }
    metadata_yaml = yaml.dump(metadata, default_flow_style=False, sort_keys=False)

    return f"""
.rodata
.p2align 6
.amdhsa_kernel {name}
  .amdhsa_group_segment_fixed_size {lds_size}
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size {offset}
  .amdhsa_user_sgpr_count 2
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_next_free_vgpr {v_cnt}
  .amdhsa_next_free_sgpr {s_cnt + 6}
  .amdhsa_wavefront_size32 1
  .amdhsa_float_round_mode_32 0
  .amdhsa_float_round_mode_16_64 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_system_vgpr_workitem_id 2
.end_amdhsa_kernel

.amdgpu_metadata
{metadata_yaml}.end_amdgpu_metadata
"""
