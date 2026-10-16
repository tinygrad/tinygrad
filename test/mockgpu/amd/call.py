import struct
from tinygrad.viz.serve import amd_decode, get_cfg, COND_TAKEN, COND_NOT_TAKEN
from tinygrad.uop.ops import UOp, sint, Ops, KernelInfo, PatternMatcher, UPat, graph_rewrite, rewrite_group, uopfunc
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.dtype import Invalid, AddrSpace, dtypes
from tinygrad.helpers import Context, getenv, TracingKey, dedup, to_mv
from tinygrad.runtime.autogen import hsa
from tinygrad.renderer.amd.dsl import EXEC_LO, ttmp
from test.mockgpu.amd.emu import _Ctx, _get_handler, _wave_size, _canonical_info, PC_LO_IDX, PC_HI_IDX, SGPR_COUNT, SCRATCH_STRIDE_IDX, F32_INLINE

# this is meant to replace the old emulator
@uopfunc
def init_wave(wg:UOp, wave:UOp, sgpr:UOp, vgpr:UOp, args_ptr:UOp, gx:int, gy:int, lx:int, ly:int, total_threads:int, wave_size:int,
              scratch_size:int, rsrc2:int, arch:str="rdna3", user_data_count:int=0):
  ci = UOp.range(81, 2, dtype=dtypes.int)
  init_ints = sgpr.index(ci+128).store((ci<65).where(ci, 64-ci).cast(dtypes.uint32)).end(ci)
  fi = UOp.range(len(F32_INLINE), 3, dtype=dtypes.int)
  floats_buf = UOp(Ops.BINARY, arg=struct.pack(f"<{len(F32_INLINE)}I", *F32_INLINE.values())).bitcast(dtypes.uint32)
  init_floats = sgpr.index(fi+240).store(floats_buf[fi]).end(fi)
  # set RANGE registers
  gidx, gidy, gidz = wg%gx, (wg//gx)%gy, wg//(gx*gy)
  n_lanes = (total_threads-wave*wave_size).minimum(wave_size)
  initial:list[tuple[int, sint]] = [(124, 0)] if wave_size == 32 else []
  if user_data_count: initial += [(i, UOp.variable(f"user_data_{i}", 0, dtypes.uint32.max, dtypes.uint32)) for i in range(user_data_count)]
  else: initial += [(0, args_ptr.cast(dtypes.uint32)), (1, (args_ptr>>32).cast(dtypes.uint32))]
  if arch == "rdna4": initial += [(ttmp[7].offset, (gidy&0xFFFF)|((gidz&0xFFFF)<<16)), (ttmp[9].offset, gidx)]
  else:
    sgpr_id = (rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT
    for enabled, gid in [(hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X, gidx),
                         (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y, gidy),
                         (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z, gidz)]:
      if rsrc2 & enabled:
        initial.append((sgpr_id, gid))
        sgpr_id += 1
  initial += [(EXEC_LO.offset, ((UOp.const(1, dtypes.uint64)<<n_lanes.minimum(32).cast(dtypes.uint64))-1).cast(dtypes.uint32)),
              (SCRATCH_STRIDE_IDX, scratch_size), (SGPR_COUNT-16+4, (wave&15)|((wave&3)<<4))]
  if wave_size == 64: initial.append((EXEC_LO.offset+1,
                                     ((UOp.const(1, dtypes.uint64)<<(n_lanes-32).maximum(0).cast(dtypes.uint64))-1).cast(dtypes.uint32)))
  lane = UOp.range(wave_size, 5, dtype=dtypes.int)
  tid = wave*wave_size+lane
  init_vgpr = vgpr.index(lane.valid(tid<total_threads)).store(
      (((tid//(lx*ly))<<20)|(((tid//lx)%ly)<<10)|(tid%lx)).cast(dtypes.uint32)).end(lane)
  return UOp.sink(init_vgpr, *(sgpr.after(init_ints, init_floats).index(i).store(UOp.const(v, dtypes.uint32))
                             for i,v in dict(initial).items()))

def pc_index(idx:int) -> UPat:
  reg, null = UPat.const(idx).cast(), UPat.const(124).cast()
  return UPat.any(reg, reg.ne(null).where(reg, UPat.const(Invalid)))

def move_const_idxs(call:UOp) -> UOp|None:
  idxs = dedup(u.src[1] for u in call.body.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and u.src[0].arg.name == "vmem"
               and u.src[1].op is Ops.CONST)
  if not idxs: return None
  rep = {idx:UOp.param(len(call.src)-1+i, idx.commit_dtype(), name=f"inst_{i}", addrspace=AddrSpace.ALU) for i,idx in enumerate(idxs)}
  return call.replace(src=(call.body.substitute(rep, walk=True), *call.src[1:], *idxs))

pm_asm_call = PatternMatcher([
  # remove PC from CALL body
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat(Ops.PARAM, name="buf").index(UPat.any(pc_index(PC_LO_IDX), pc_index(PC_HI_IDX))),), allow_any_len=True),
   lambda buf: UOp(Ops.NOOP) if buf.arg.name == "sgpr" else None),
  # move CONST outside CALL body
  (UPat(Ops.CALL, src=(UPat(Ops.SINK),), allow_any_len=True, name="call"), move_const_idxs),
])

@rewrite_group(name=lambda *args,ret,**kw: TracingKey(kw.get('name', ''), ("lift",)))
def lift(lib:int, lib_sz:int, gx:int, gy:int, gz:int, lx:int, ly:int, lz:int, rsrc2:int, scratch_size:int, arch:str="rdna3",
         user_data_count:int=0, backend:str|None=None, name="") -> UOp:
  backend = getenv("ASM_CALL_BACKEND", "CPU") if backend is None else backend
  # decode
  view = to_mv(lib, lib_sz)
  insts = amd_decode(view, arch)
  cfg = get_cfg(insts)
  # construct CALL graph
  wave_size, total_threads = _wave_size(arch), lx*ly*lz
  n_waves = (total_threads+wave_size-1)//wave_size
  wg = UOp.range(gx*gy*gz, 0)
  wave = UOp.range(n_waves, 1)
  # alloc register and LDS buffers
  sgpr = UOp.alloc((SGPR_COUNT*n_waves,), dtypes.uint32, 0, AddrSpace.REG)
  vgpr = UOp.alloc((256*wave_size*n_waves,), dtypes.uint32, 1, AddrSpace.REG)
  lds_size = ((rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT)*512
  lds = UOp.alloc((max(lds_size//4, 1),), dtypes.uint32, 3, AddrSpace.REG)
  scratch = UOp.alloc((max(scratch_size*wave_size*n_waves, 1),), dtypes.uint8, 4, AddrSpace.REG)
  accvgpr = UOp.alloc((256*wave_size*n_waves,), dtypes.uint32, 5, AddrSpace.REG) if wave_size == 64 else vgpr
  args_ptr = UOp.variable("args_ptr", 0, dtypes.uint64.max, dtypes.uint64)
  lib_addr = UOp.variable("lib", 0, dtypes.uint64.max, dtypes.uint64)
  inst_addr = UOp.param(-1, dtypes.uint64, name="inst", addrspace=AddrSpace.ALU)
  init = init_wave(wg, wave, sgpr.index(wave*SGPR_COUNT), vgpr.index(wave*256*wave_size), args_ptr, gx, gy, lx, ly, total_threads, wave_size,
                   scratch_size, rsrc2, arch, user_data_count)
  user_vars = [u for u in init.body.toposort() if u.is_variable]
  init = init.replace(src=(init.body.substitute({u:UOp.param(len(init.src)-1+i, u.dtype, name=u.arg.name, addrspace=AddrSpace.ALU)
                                                 for i,u in enumerate(user_vars)}), *init.src[1:], *user_vars))
  ctx = _Ctx(4, wave_size)
  end = init.end(wave)
  mem = {ctx.sgpr:(sgpr,SGPR_COUNT),ctx.vgpr:(vgpr,256*wave_size),ctx.vmem:(ctx.vmem,0),ctx.lds:(lds,0),ctx.scratch:(scratch,scratch_size*wave_size)}
  if wave_size == 64: mem[ctx.accvgpr] = (accvgpr, 256*wave_size)
  for block_idx, (block_pc, block) in enumerate(cfg["blocks"].items()):
    wave = UOp.range(n_waves, lib_sz+block_idx)
    afters = {b:(buf.shrink(((wave*stride, (wave+1)*stride),)) if stride else buf).after(end, wave) for b,(buf,stride) in mem.items()}
    loop_path = cfg["paths"][block_pc].get(block_pc)
    loop = UOp.loop(block_pc) if loop_path is not None else None
    if loop is not None: afters = {b:arg.after(loop) for b,arg in afters.items()}
    branch_cond:UOp|None = None
    for off in block:
      inst = insts[off]
      inst_st = getattr(inst, "op_name", "").lower()
      if inst_st.startswith("s_code_end"): continue
      if inst_st.startswith(("s_getpc", "s_setpc")): raise AssertionError("getpc and setpc are not allowed in ASM_CALL")
      ctx = _Ctx(inst.size(), _wave_size(arch), inst_addr=inst_addr)
      sink = _get_handler(inst)(inst, ctx)
      *_, canonical_name = _canonical_info(inst, ctx, view[off:])
      bufs = sorted((u for u in sink.toposort() if u.op is Ops.PARAM), key=lambda u: u.arg.slot)
      args = [lib_addr+off if b is inst_addr else afters.get(b, b.after(loop) if loop is not None else b) for b in bufs]
      if ctx.branch_cond is not None and loop_path is not None:
        branch_cond = ctx.branch_cond.substitute(dict(zip(bufs, args)), walk=True)
        continue
      body = sink.substitute({b:b.param_like(i, name=b.arg.name) for i,b in enumerate(bufs)})
      call = body.call(*args, name=canonical_name)
      afters.update((b, arg.after(call)) for b, arg in zip(bufs, args) if b is not inst_addr)
    if loop is not None:
      assert branch_cond is not None and loop_path in (COND_TAKEN, COND_NOT_TAKEN)
      if loop_path is COND_NOT_TAKEN: branch_cond = branch_cond.logical_not()
      backedge = UOp.sink(*afters.values()).backedge(loop, branch_cond)
      afters = {b:arg.after(backedge) for b,arg in afters.items()}
    end = UOp.group(*afters.values()).end(wave)
  sink = UOp.sink(end.end(wg), arg=KernelInfo(name=f"emu_{name}", optimized=True))
  sink = graph_rewrite(sink, pm_asm_call, name="pm_asm_call", bottom_up=True, enter_calls=True)
  with Context(CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0):
    return to_program(sink, Device[backend].renderer)
