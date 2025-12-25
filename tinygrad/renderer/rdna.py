from typing import cast
from collections import defaultdict
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType, AddrSpace
from tinygrad.renderer import Renderer
from tinygrad.helpers import get_single_element
from tinygrad.codegen.opt import tc
from tinygrad.renderer.cstyle import create_non_native_float_pats, cast_float_to_bf16
from tinygrad.renderer.rdna_uops import rdna_matcher
from tinygrad.renderer.rdna_regalloc import RDNARegAlloc
from tinygrad.renderer.rdna_render import (get_reg_base, render_val, can_inline_const, extract_low_32, asm_for_op, string_rewrite,
                                           RDNARenderMixin, analyze_const_usage, analyze_half16_vectorize, analyze_deferred_stores)

class RDNARenderer(RDNARenderMixin, Renderer):
  device = "AMD"
  suffix = "RDNA"
  supports_float4 = True
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 1024)
  shared_max = 65536
  max_upcast_size = 16
  code_for_op = asm_for_op
  rdna_bf16_cast = PatternMatcher([(UPat(Ops.CAST, dtype=dtypes.bfloat16, src=(UPat.var("x", dtype=dtypes.float),)), cast_float_to_bf16)])
  extra_matcher = rdna_matcher + create_non_native_float_pats((dtypes.bfloat16,)) + rdna_bf16_cast
  tensor_cores = tc.amd_rdna3

  def __init__(self, arch:str="gfx1100"):
    self.arch, self.wave32 = arch, True
  def __reduce__(self): return self.__class__, (self.arch,)

  types: dict[DType, str] = {
    dtypes.int8: "i32", dtypes.int16: "i32", dtypes.int32: "i32", dtypes.int64: "i32",
    dtypes.uint8: "u32", dtypes.uint16: "u32", dtypes.uint32: "u32", dtypes.uint64: "u32",
    dtypes.float16: "f16", dtypes.bfloat16: "bf16", dtypes.float32: "f32", dtypes.float64: "f64", dtypes.bool: "i32"
  }

  def render(self, uops:list[UOp]) -> str:
    kernel:list[str] = []
    bufs = []
    kernarg_offset: dict[UOp, int] = {}
    current_kernarg_offset = 0

    r: dict[UOp, str|list[str]] = {}
    self.r = r
    self.kernarg_offset = kernarg_offset
    self.uops = uops
    self.gated_sgpr = 100  # for gated_load
    self.if_sgpr_base = 101  # for IF/ENDIF
    self.if_save_stack: list[int] = []
    self.max_if_depth = 0
    self.scratch_sgpr_used = False
    self.lds_size = 0

    # Initialize register allocator with liveness analysis
    regalloc = RDNARegAlloc(uops)
    self.get_scratch_vgpr = regalloc.get_scratch_vgpr

    # Run pre-render analysis
    skip_alloc_consts, store_const_uses = analyze_const_usage(uops)
    half16_vectorize_sources, half16_vectorize_ranges, half16_packed, half16_temp_regs, half16_direct_loads = analyze_half16_vectorize(uops, regalloc)
    store_only_indices, recomputable_shls = analyze_deferred_stores(uops, regalloc)

    const_cache: dict[tuple, str] = {}
    name, pending_waits = "test", set()
    deferred_store_addr_vgpr: str | None = None

    def get_half16_range(vec_uop: UOp) -> str:
      if vec_uop not in half16_vectorize_ranges: half16_vectorize_ranges[vec_uop] = regalloc.alloc_vgpr_range(vec_uop, 8)
      return half16_vectorize_ranges[vec_uop]

    for i, u in enumerate(uops):
      regalloc.free_dead_regs(i)
      if u.op in {Ops.NOOP, Ops.GROUP}: continue
      if u.op is Ops.AFTER: r[u] = r[u.src[0]]; continue
      if u.op is Ops.SINK:
        if u.arg is not None: name = u.arg.function_name
        continue
      if u.op is Ops.VECTORIZE:
        if any(src in pending_waits for src in u.src): kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)"); pending_waits.clear()
        if u.dtype.scalar() == dtypes.half and u.dtype.count == 16:
          r[u] = half16_vectorize_ranges.get(u) or regalloc.alloc_vgpr_range(u, 8)
          base = get_reg_base(r[u])
          for j in range(8):
            if j not in half16_packed.get(u, set()): kernel.append(f"v_pack_b32_f16 v{base+j}, {r[u.src[j*2]]}, {r[u.src[j*2+1]]}")
        # For float8 (WMMA accumulator), check if sources are contiguous
        elif u.dtype.scalar() == dtypes.float and u.dtype.count == 8:
          src_regs = [int(rs[1:]) for src in u.src if isinstance((rs := r[src]), str) and rs.startswith('v') and '[' not in rs]
          if len(src_regs) == 8 and src_regs == list(range(src_regs[0], src_regs[0] + 8)):
            r[u] = f"v[{src_regs[0]}:{src_regs[0]+7}]"
            continue
          ru = regalloc.alloc_vgpr_range(u, 8)
          r[u] = ru
          base = get_reg_base(ru)
          for j, src in enumerate(u.src):
            kernel.append(f"v_mov_b32 v{base+j}, {r[src]}")
        # For other vector types, allocate contiguous VGPRs based on size
        elif isinstance(u.dtype, DType) and u.dtype.count > 1:
          vgpr_count = (u.dtype.itemsize + 3) // 4
          ru = regalloc.alloc_vgpr_range(u, vgpr_count)
          r[u] = ru
          base = get_reg_base(ru)
          # Check if this is a packed half type (2 halfs per VGPR)
          if u.dtype.scalar() == dtypes.half and u.dtype.count > 1:
            # Pack pairs of half values into each VGPR using v_pack_b32_f16
            for j in range(vgpr_count):
              lo_idx = j * 2
              hi_idx = j * 2 + 1
              lo_reg = r[u.src[lo_idx]] if lo_idx < len(u.src) else "0"
              hi_reg = r[u.src[hi_idx]] if hi_idx < len(u.src) else "0"
              kernel.append(f"v_pack_b32_f16 v{base+j}, {lo_reg}, {hi_reg}")
          else:
            dst_offset = 0  # Track destination register offset
            for j, src in enumerate(u.src):
              src_reg = r[src]
              if isinstance(src_reg, str) and src_reg.startswith('v['):
                src_base, src_end = get_reg_base(src_reg), int(src_reg[src_reg.index(':')+1:-1])
                for k in range(src_end - src_base + 1):
                  kernel.append(f"v_mov_b32 v{base+dst_offset+k}, v{src_base+k}")
                dst_offset += src_end - src_base + 1  # Advance by the size of the source
              else:
                kernel.append(f"v_mov_b32 v{base+dst_offset}, {src_reg}")
                dst_offset += 1
        else:
          r[u] = [cast(str,r[x]) for x in u.src]
        continue
      if u.op is Ops.GEP:
        if u.src[0] in pending_waits: kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)"); pending_waits.clear()
        src_reg, elem_idx, src_dtype = r[u.src[0]], get_single_element(u.arg), u.src[0].dtype
        if u.src[0] in half16_direct_loads:
          vec_uop, base_vgpr_idx = half16_direct_loads[u.src[0]]
          r[u] = f"v{get_reg_base(half16_vectorize_ranges[vec_uop]) + base_vgpr_idx + elem_idx // 2}"; continue
        if isinstance(src_reg, str) and src_reg.startswith('v['):
          base, end = get_reg_base(src_reg), int(src_reg[src_reg.index(':')+1:-1])
          if isinstance(src_dtype, DType) and src_dtype.count > 1:
            epr = src_dtype.count // (end - base + 1)  # elements_per_vgpr
            if epr > 1:
              vgpr_idx, eiv = elem_idx // epr, elem_idx % epr
              if eiv == 0: r[u] = f"v{base + vgpr_idx}"
              else: dst = regalloc.alloc_vgpr(u); kernel.append(f"v_lshrrev_b32 {dst}, {eiv * (32 // epr)}, v{base + vgpr_idx}"); r[u] = dst
            else:
              scalar = src_dtype.scalar() if hasattr(src_dtype, 'scalar') else src_dtype
              r[u] = f"v[{base + elem_idx * 2}:{base + elem_idx * 2 + 1}]" if scalar.itemsize == 8 else f"v{base + elem_idx}"
          else:
            r[u] = f"v[{base + elem_idx * 2}:{base + elem_idx * 2 + 1}]" if hasattr(src_dtype, 'itemsize') and src_dtype.itemsize == 8 else f"v{base + elem_idx}"
        elif isinstance(src_reg, list): r[u] = src_reg[elem_idx]
        else: r[u] = src_reg
        continue
      if u.op in {Ops.CAST, Ops.BITCAST} and (u.src[0].dtype == u.dtype or isinstance(u.src[0].dtype, PtrDType)):
        r[u] = r[u.src[0]]
        continue

      # Register allocation with liveness-based reuse
      if u.op is Ops.DEFINE_GLOBAL:
        r[u] = regalloc.alloc_sgpr_pair(u)
        kernarg_offset[u] = current_kernarg_offset
        current_kernarg_offset += 8  # Pointers are 8 bytes
        bufs.append((f"data{u.arg}", u.dtype))
      elif u.op is Ops.DEFINE_LOCAL:
        # Local memory - DEFINE_LOCAL.dtype contains the LDS size in the ptr size
        lds_size = u.dtype.size * u.dtype.itemsize if isinstance(u.dtype, PtrDType) else 0
        self.lds_size = max(getattr(self, 'lds_size', 0), lds_size)
        r[u] = u.arg  # Store the offset (arg is the offset in bytes)
        continue
      elif u.op is Ops.DEFINE_VAR:
        sgpr = regalloc.alloc_sgpr(u)
        kernarg_offset[u] = current_kernarg_offset
        current_kernarg_offset += 4  # Variables are 4 bytes (int32)
        r[u] = sgpr or regalloc.alloc_vgpr(u)  # Fall back to VGPR if SGPRs exhausted
        bufs.append((u.arg[0], u.dtype))
      elif u.op is Ops.SPECIAL:
        r[u] = regalloc.alloc_vgpr(u)
      elif u.op is Ops.INDEX:
        # REG addrspace INDEX: index into register array from DEFINE_REG
        if isinstance(u.src[0].dtype, PtrDType) and u.src[0].dtype.addrspace == AddrSpace.REG:
          assert u.src[1].op == Ops.CONST, f"REG INDEX requires CONST index, not {u.src[1].op}"
          r[u] = r[u.src[0]][u.src[1].arg]  # Get specific register from array
          continue  # Skip rendering - handled inline
        # Skip VGPR allocation for store-only indices - address computed inline at store time
        if u in store_only_indices:
          r[u] = "DEFERRED_STORE_ADDR"
        else:
          r[u] = regalloc.alloc_vgpr(u)
      elif u.op is Ops.LOAD:
        # REG addrspace LOAD: handled by REG handling code below, skip allocation
        if u.src[0].op is Ops.INDEX and isinstance(u.src[0].src[0].dtype, PtrDType) and u.src[0].src[0].dtype.addrspace == AddrSpace.REG:
          r[u] = r[u.src[0]]  # Use register from INDEX (which came from DEFINE_REG array)
          continue  # Skip rendering - handled inline
        if u in half16_direct_loads:
          vec_uop, base_vgpr_idx = half16_direct_loads[u]
          range_base = get_reg_base(half16_vectorize_ranges[vec_uop])
          r[u] = f"v[{range_base + base_vgpr_idx}:{range_base + base_vgpr_idx + 1}]"
        elif isinstance(u.dtype, DType) and u.dtype.count > 1:
          # Calculate number of 32-bit VGPRs needed
          vgpr_count = (u.dtype.itemsize + 3) // 4  # Round up to 32-bit chunks
          r[u] = regalloc.alloc_vgpr_range(u, vgpr_count)
        elif u in half16_vectorize_sources:
          # Scalar half LOAD destined for half16 VECTORIZE - use look-ahead packing
          # Allocate a temp VGPR for the load, will pack immediately after
          temp_vgpr = regalloc.alloc_vgpr(u)
          r[u] = temp_vgpr
          half16_temp_regs[u] = temp_vgpr
        else:
          r[u] = regalloc.alloc_vgpr_pair(u) if RDNARegAlloc.needs_vgpr_pair(u.dtype) else regalloc.alloc_vgpr(u)
        pending_waits.add(u)
      elif u.op is Ops.CONST:
        # Skip allocation for constants used as compile-time REG indices
        if u in skip_alloc_consts:
          r[u] = render_val(u.arg, u.dtype)  # Store the value as string, no register needed
          continue  # Skip rendering - it's a compile-time index
        # Check if constant can be inlined (no VGPR needed) - but not if used in STORE
        if can_inline_const(u.arg, u.dtype) and u not in store_const_uses:
          r[u] = render_val(u.arg, u.dtype)  # Store value string, not register
          continue  # Skip rendering - will be inlined
        # Deduplicate non-inlineable constants - reuse register if same value already loaded
        const_key = (u.dtype, u.arg)
        if const_key in const_cache:
          r[u] = const_cache[const_key]
          # Extend the original owner's lifetime to include this use
          reg_str = const_cache[const_key]
          reg_num = int(reg_str[1:]) if reg_str.startswith('v') and '[' not in reg_str else None
          if reg_num is not None and regalloc.is_vgpr_owner(reg_num):
            original_owner = regalloc.get_vgpr_owner(reg_num)
            if original_owner is not None:
              regalloc.extend_lifetime(original_owner, max(regalloc.get_last_use(original_owner), regalloc.get_last_use(u) if regalloc.get_last_use(u) >= 0 else i))
          continue  # Skip rendering - already loaded
        # For 64-bit types used in STORE, need a pair for global_store_b64
        needs_pair = RDNARegAlloc.needs_vgpr_pair(u.dtype) or (u in store_const_uses and u.dtype.itemsize == 8)
        ru = regalloc.alloc_vgpr_pair(u) if needs_pair else regalloc.alloc_vgpr(u)
        r[u] = ru
        const_cache[const_key] = ru
      elif u.op is Ops.RANGE:
        r[u] = regalloc.alloc_vgpr(u)
      elif u.op is Ops.END:
        r[u] = "vcc_lo"  # comparison result
      elif u.op in GroupOp.ALU or u.op in {Ops.CAST, Ops.BITCAST}:
        # Skip allocation for SHL/ADD ops that will be recomputed inline at store time
        if u in recomputable_shls:
          r[u] = "RECOMPUTE_AT_STORE"
          continue
        # Check if any source needs a wait
        for src in u.src:
          if src in pending_waits:
            kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
            pending_waits.clear()
            break
        # CAST optimization: reuse source register when this is its last use and dest fits
        # This avoids allocating 128 extra VGPRs for float32→half conversions
        cast_reused_src = False
        if u.op is Ops.CAST and len(u.src) == 1:
          cast_src = u.src[0]
          cast_src_reg = r.get(cast_src)
          # Can reuse if: single VGPR source, this is last use, dest fits in 32 bits
          if (cast_src_reg and isinstance(cast_src_reg, str) and cast_src_reg.startswith('v') and '[' not in cast_src_reg
              and regalloc.get_last_use(cast_src) == i and isinstance(u.dtype, DType) and u.dtype.itemsize <= 4):
            r[u] = cast_src_reg  # Reuse source register - in-place conversion
            reg_num = int(cast_src_reg[1:])
            # Transfer ownership: old owner is gone, new owner (the CAST) takes over
            regalloc.reschedule_vgpr_death(reg_num, u)
            cast_reused_src = True
        # Only allocate if we didn't already reuse source register for CAST
        if not cast_reused_src:
          # Only direct comparison ops go to SGPR; other bool ops stay in VGPR
          # because their inputs might be in VGPR (from memory loads)
          if u.dtype == dtypes.bool and u.op in {Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ}:
            sgpr = regalloc.alloc_sgpr(u)
            r[u] = sgpr if sgpr is not None else regalloc.alloc_vgpr(u)  # fall back to VGPR if SGPRs exhausted
          elif isinstance(u.dtype, DType) and u.dtype.count > 1:
            # Vector types need contiguous register ranges - size based on total bytes
            vgpr_count = (u.dtype.itemsize + 3) // 4  # Round up to 32-bit chunks
            r[u] = regalloc.alloc_vgpr_range(u, vgpr_count)
          else:
            r[u] = regalloc.alloc_vgpr_pair(u) if RDNARegAlloc.needs_vgpr_pair(u.dtype) else regalloc.alloc_vgpr(u)
      elif u.op is Ops.IF:
        sgpr = regalloc.alloc_sgpr(u)
        assert sgpr is not None, "IF op requires SGPR"
        r[u] = sgpr
      elif u.op is Ops.WMMA:
        # WMMA outputs a vector of floats (8 for RDNA3)
        # For RDNA3 WMMA, we can do in-place accumulation if dst == C source
        # Check if we can reuse the accumulator register range
        acc_src = u.src[2]  # accumulator input
        acc_reg = r.get(acc_src)
        if isinstance(acc_reg, str) and acc_reg.startswith('v['):
          # Accumulator is already a contiguous range - reuse it for output
          r[u] = acc_reg
        else:
          # Allocate contiguous VGPRs for the output using range allocator
          r[u] = regalloc.alloc_vgpr_range(u, 8)
      elif u.op is Ops.DEFINE_REG:
        # For 64-bit types, allocate VGPR pairs
        if RDNARegAlloc.needs_vgpr_pair(u.ptrdtype.base):
          r[u] = [regalloc.alloc_vgpr_pair(u) for _ in range(u.ptrdtype.size)]
        else:
          r[u] = [regalloc.alloc_vgpr(u) for _ in range(u.ptrdtype.size)]
        continue

      # Handle register-based INDEX/LOAD/STORE (accumulator spills)
      if u.op in {Ops.INDEX, Ops.LOAD, Ops.STORE} and isinstance(u.src[0].dtype, PtrDType) and u.src[0].dtype.addrspace == AddrSpace.REG:
        if u.op is Ops.INDEX:
          assert u.src[1].op == Ops.CONST, f"index on REG in rdna only supported on CONST, not {u.src[1].op}"
          reg_list = r[u.src[0]]
          assert isinstance(reg_list, list)
          r[u] = reg_list[u.src[1].arg]
        else:
          r[u] = r[u.src[0]]
          if u.op is Ops.STORE:
            dst_reg, src_reg = r[u.src[0]], r[u.src[1]]
            assert isinstance(dst_reg, str) and isinstance(src_reg, str)
            if '[' in dst_reg or '[' in src_reg:
              # Either source or dest is a pair - handle both 32-bit parts
              dst_num = get_reg_base(dst_reg)
              src_num = get_reg_base(src_reg) if '[' in src_reg else int(src_reg[1:])
              kernel.extend([f"v_mov_b32 v{dst_num}, v{src_num}", f"v_mov_b32 v{dst_num+1}, v{src_num+1}"])
            else:
              kernel.append(f"v_mov_b32 {dst_reg}, {src_reg}")
        continue

      # Skip INDEX as it's handled inline
      if u.op is Ops.INDEX:
        # If this is a store-only index, skip - address computed inline at store time
        if u in store_only_indices:
          continue

        # The byte offset is already computed at UOp level by rdna_matcher
        # We just need to move it to the allocated register (and optionally add base offset for local)
        buf, idx = u.src[0], u.src[1]
        if len(u.src) > 2:  # gated
          pass  # handled in LOAD
        is_local = isinstance(u.dtype, PtrDType) and u.dtype.addrspace == AddrSpace.LOCAL
        base_offset = r[buf] if is_local else 0  # local memory has base offset from DEFINE_LOCAL

        if idx.op is Ops.CONST and idx.arg == 0:
          if is_local and base_offset:
            kernel.append(f"v_mov_b32 {r[u]}, {base_offset}")
          else:
            kernel.append(f"v_mov_b32 {r[u]}, 0")
        else:
          # Wait for idx if needed
          if idx in pending_waits:
            kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
            pending_waits.clear()
          # idx is already the byte offset (computed at UOp level by rdna_matcher)
          if is_local and base_offset:
            kernel.append(f"v_add_nc_u32 {r[u]}, {base_offset}, {r[idx]}")
          else:
            kernel.append(f"v_mov_b32 {r[u]}, {r[idx]}")
        continue

      # Check if STORE needs a wait for any pending loads
      if u.op is Ops.STORE:
        for src in u.src:
          if src in pending_waits or (src.op is Ops.INDEX and any(s in pending_waits for s in src.src)):
            kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
            pending_waits.clear()
            break

        # Handle deferred store address computation - reuse single shared temp VGPR
        if u.src[0].op is Ops.INDEX and r.get(u.src[0]) == "DEFERRED_STORE_ADDR":
          index_op = u.src[0]
          buf, idx = index_op.src[0], index_op.src[1]
          # Use a dedicated scratch VGPR for all deferred store addresses (never freed, won't conflict)
          if deferred_store_addr_vgpr is None:
            deferred_store_addr_vgpr = regalloc.get_deferred_store_vgpr()
          # Compute the byte offset - detect pattern SHL(ADD(base, const), shift) for inline recompute
          # Only recompute if idx is marked as RECOMPUTE_AT_STORE - otherwise it has a valid register
          if idx.op is Ops.CONST and idx.arg == 0:
            kernel.append(f"v_mov_b32 {deferred_store_addr_vgpr}, 0")
          elif r.get(idx) == "RECOMPUTE_AT_STORE" and idx.op is Ops.SHL and idx.src[0].op is Ops.ADD and idx.src[1].op is Ops.CONST:
            # Pattern: SHL(ADD(base, offset), shift) - recompute inline to avoid holding SHL result
            add_op = idx.src[0]
            shift_val = idx.src[1].arg
            add_src0, add_src1 = add_op.src[0], add_op.src[1]
            # Handle both ADD(base, const) and ADD(const, base)
            recompute_const: int|None = None
            recompute_base: UOp|None = None
            if add_src1.op is Ops.CONST:
              recompute_const, recompute_base = add_src1.arg, add_src0
            elif add_src0.op is Ops.CONST:
              recompute_const, recompute_base = add_src0.arg, add_src1
            if recompute_const is not None and recompute_base is not None:
              # ADD(base, const) - use v_lshl_add_u32 to compute (base << shift) + (const << shift)
              base_reg = r.get(recompute_base)
              # v_lshl_add_u32 accepts both VGPR and SGPR sources
              if base_reg and isinstance(base_reg, str) and (base_reg.startswith('v') or base_reg.startswith('s')):
                kernel.append(f"v_lshl_add_u32 {deferred_store_addr_vgpr}, {base_reg}, {shift_val}, {recompute_const << shift_val}")
              else:
                # Fallback: compute the SHL manually if we can't recompute inline
                # This shouldn't normally happen, but handle gracefully
                kernel.append(f"v_mov_b32 {deferred_store_addr_vgpr}, {r[idx] if r.get(idx) != 'RECOMPUTE_AT_STORE' else 0}")
            else:
              # ADD(reg, reg) - compute ADD then SHL
              kernel.append(f"v_add_nc_u32 {deferred_store_addr_vgpr}, {r[add_src0]}, {r[add_src1]}")
              kernel.append(f"v_lshlrev_b32 {deferred_store_addr_vgpr}, {shift_val}, {deferred_store_addr_vgpr}")
          else:
            kernel.append(f"v_mov_b32 {deferred_store_addr_vgpr}, {r[idx]}")
          # Update r[index_op] to point to the temp VGPR so pattern matcher can use it
          r[index_op] = deferred_store_addr_vgpr

      # Render the instruction
      if (l:=cast(str|list[str], string_rewrite.rewrite(u, ctx=self))) is None:
        raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
      if isinstance(l, str):
        kernel.append(l)
      else:
        kernel.extend(l)

      # === LOOK-AHEAD PACKING: pack halfs immediately after load ===
      if u.op is Ops.LOAD and u in half16_vectorize_sources:
        vec_uop, pos = half16_vectorize_sources[u]
        vgpr_idx, is_high_half = pos // 2, pos % 2 == 1
        dst_vgpr = f"v{get_reg_base(get_half16_range(vec_uop)) + vgpr_idx}"

        # Find the pair (the other half that shares this destination VGPR)
        pair_pos = pos ^ 1  # XOR to toggle between even/odd
        pair_src = vec_uop.src[pair_pos]

        # Check if pair is already loaded (its temp reg exists in half16_temp_regs and it's been processed)
        pair_loaded = pair_src in half16_temp_regs and pair_src in r

        if pair_loaded:
          # Both halves ready - wait for loads and pack immediately
          kernel.append("s_waitcnt vmcnt(0)")
          pending_waits.discard(u)
          pending_waits.discard(pair_src)

          # Determine which is low and which is high
          if is_high_half:
            lo_reg = half16_temp_regs[pair_src]
            hi_reg = half16_temp_regs[u]
          else:
            lo_reg = half16_temp_regs[u]
            hi_reg = half16_temp_regs[pair_src]

          # Pack two halfs into destination VGPR
          kernel.append(f"v_pack_b32_f16 {dst_vgpr}, {lo_reg}, {hi_reg}")
          half16_packed[vec_uop].add(vgpr_idx)

          # Immediately free the temp VGPRs since values are now in packed destination
          # This is critical for reducing register pressure
          for temp_src in [u, pair_src]:
            temp_reg = half16_temp_regs[temp_src]
            if temp_reg.startswith('v') and '[' not in temp_reg:
              reg_num = int(temp_reg[1:])
              if regalloc.is_vgpr_owner(reg_num):
                regalloc.free_vgpr(reg_num)

      # Add wait after loads
      if u.op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR}:
        kernel.append("s_waitcnt lgkmcnt(0)")

    # Final waitcnt and end program - always wait to ensure stores complete
    kernel.append("s_waitcnt vmcnt(0) lgkmcnt(0)")
    kernel.extend(['s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)', 's_endpgm', 's_code_end'])

    # Compute actual max VGPR from assembly (not allocation high-water mark)
    # This accounts for VGPRs that were allocated but then freed and reused
    import re
    actual_max_vgpr = 3  # v[0:2] are reserved for local_xyz
    for line in kernel:
      for m in re.finditer(r'v(\d+)', line):
        actual_max_vgpr = max(actual_max_vgpr, int(m.group(1)) + 1)
      # Also check for VGPR ranges v[a:b]
      for m in re.finditer(r'v\[(\d+):(\d+)\]', line):
        actual_max_vgpr = max(actual_max_vgpr, int(m.group(2)) + 1)

    # If scratch SGPRs were used, update max_sgpr to include them
    max_sgpr = regalloc.max_sgpr
    if self.scratch_sgpr_used:
      # gated_sgpr (s100) + s101 for 64-bit comparisons + IF stack SGPRs (s101, s102, ...)
      max_sgpr = max(max_sgpr, self.gated_sgpr + 2)  # +2 for 64-bit comparison scratch
      if self.max_if_depth > 0:
        max_sgpr = max(max_sgpr, self.if_sgpr_base + self.max_if_depth)

    return self.render_kernel(kernel, name, bufs, actual_max_vgpr, max_sgpr, uops)
