import itertools, functools, struct
from dataclasses import replace
from tinygrad.viz.serve import amd_code_end, amd_decode, get_cfg, COND_TAKEN, COND_NOT_TAKEN, UNCOND
from tinygrad.uop.ops import UOp, sint, Ops, KernelInfo, PatternMatcher, UPat, rewrite_group, uopfunc
from tinygrad.codegen import to_program, to_program_lower_config, lower_call
from tinygrad.device import Device
from tinygrad.engine.realize import get_runtime, runtime_cache
from tinygrad.renderer.llvmir import CPULLVMRenderer, ldt
from tinygrad.renderer import Renderer
from tinygrad.runtime.support.compiler_llvm import CPULLVMCompiler
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.c import DLL
from tinygrad.runtime.ops_cpu import CPUExecutable
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Context, TracingKey, Target, cpu_profile, to_mv
from tinygrad.runtime.autogen import hsa, libc
from tinygrad.renderer.amd.dsl import Inst, EXEC_LO, ttmp
from test.mockgpu.amd.emu import _Ctx, _get_handler, _wave_size, _canonical_info, SGPR_COUNT, SCRATCH_STRIDE_IDX, F32_INLINE

class CallLLVMModule(CPUExecutable):
  def __init__(self, lib:bytes, functions:dict[str, int]):
    self.functions = functions
    self.exports:dict[str, int] = {}
    super().__init__(lib)

  def _load(self, lib, base=0):
    image = bytearray(super()._load(lib, base))
    _, sections, _ = elf_loader(lib, link_libs=list(DLL._loaded_.values()))
    for section in sections:
      if section.header.sh_type != libc.SHT_SYMTAB: continue
      strings = sections[section.header.sh_link].content
      for symbol in (libc.Elf64_Sym * (section.header.sh_size // section.header.sh_entsize)).from_buffer_copy(section.content):
        if not 0 < symbol.st_shndx < len(sections): continue
        name = strings[symbol.st_name:strings.find(b'\0', symbol.st_name)].decode()
        offset = sections[symbol.st_shndx].header.sh_addr + symbol.st_value
        if name.endswith('_ptr'): struct.pack_into('<Q', image, offset, self.functions[name[:-4]])
        elif libc.ELF64_ST_BIND(symbol.st_info) == libc.STB_GLOBAL and symbol.st_info & 15 == libc.STT_FUNC: self.exports[name] = base + offset
    self.functions.update(self.exports)
    return bytes(image)

class CallLLVMJIT(CPULLVMCompiler):
  def __init__(self, arch:list[str]):
    super().__init__(arch)
    self.cachekey = f"asm_call_obj_{self.cachekey}" if self.cachekey is not None else None
    self.functions:dict[str, int] = {}
    self.modules:dict[bytes, CallLLVMModule] = {}

  def compile_cached(self, src:str) -> bytes:
    lib = super().compile_cached(src)
    if lib not in self.modules: self.modules[lib] = CallLLVMModule(lib, self.functions)
    return lib

  def trampoline(self, lib:bytes, name:str) -> bytes:
    addr = self.modules[lib].exports[name]
    if self.arch == 'x86_64': return b'\x48\xb8'+struct.pack('<Q', addr)+b'\xff\xe0'
    if self.arch == 'arm64': return struct.pack('<IIQ', 0x58000050, 0xd61f0200, addr)
    raise RuntimeError(f'unsupported JIT trampoline architecture {self.arch}')

  def compile(self, src:str) -> bytes:
    with cpu_profile('LLVM JIT compile'): return super().compile(src)

class CallLLVMRenderer(CPULLVMRenderer):
  compiler: CallLLVMJIT
  string_rewrite = PatternMatcher([
    (UPat(Ops.CALL, src=(UPat(Ops.LINEAR, name="body"),), allow_any_len=True, name="x"), lambda ctx,x,body:
     f"  {ctx[x]} = load ptr, ptr @{ctx[body]}_ptr\n  call void {ctx[x]}(" +
     ", ".join(f"{ldt(p.dtype, ptr=p.addrspace != AddrSpace.ALU)} {ctx[x.src[p.arg.slot+1]]}" for p in body.src if p.op is Ops.PARAM) + ")"),
  ]) + CPULLVMRenderer.string_rewrite

  def __init__(self, target:Target):
    super().__init__(target)
    self.compiler = CallLLVMJIT(target.arch.split(','))

  def _render_fn(self, name, args, kernel, prefix=None):
    if name == self.entry_name:
      kernel = [f"  {n} = trunc i64 {n}_arg to {ldt(p.dtype)}" for n,p in args if p.addrspace == AddrSpace.ALU and p.dtype.itemsize < 8]+kernel
      args = [(n+"_arg", p.replace(arg=replace(p.arg, dtype=dtypes.uint64))) if p.addrspace == AddrSpace.ALU and p.dtype.itemsize < 8
              else (n,p) for n,p in args]
    ret = super()._render_fn(name, args, kernel, prefix)
    return ret.replace(" #0", " #1") if name == self.entry_name or name.startswith("asm_block_") else ret

  def render(self, uops:list[UOp]) -> str:
    self.entry_name = next(u.arg.function_name for u in uops if u.op is Ops.SINK)
    def fn_name(b:UOp): return ("asm_block_" if b.arg == "asm_block" else "asm_fn_")+b.key.hex()
    topo = UOp.sink(*uops).toposort(lambda u:u.op is not Ops.LINEAR or fn_name(u) not in self.compiler.functions, enter_calls=True)
    fns = {u.body:fn_name(u.body) for u in topo if u.op is Ops.CALL and u.body.op is Ops.LINEAR}
    def module(us, definition):
      declarations = [f"@{fns[b]}_ptr = hidden global ptr null, section \".data\"" for b in
                      dict.fromkeys(u.body for u in us if u.op is Ops.CALL and u.body.op is Ops.LINEAR)]
      footer = self._render_footer(us, enter_calls=False).replace("alwaysinline", "noinline")
      return "\n".join([*declarations, definition, footer, "attributes #1 = { noinline optnone nounwind }"])
    for body, name in fns.items():
      if name in self.compiler.functions: continue
      definition = self._render_kernel(list(body.src), name=name, fns=fns)[1]
      self.compiler.compile_cached(module(body.src, definition))
    entry = self._render_kernel(uops, fns=fns)
    return module(uops, "\n".join([*entry[0], entry[1]]))

@functools.cache
def llvm_renderer(arch:str):
  return CallLLVMRenderer(Target("CPU", "LLVM", arch))

def backend_renderer(backend:str):
  return llvm_renderer(Device["CPU"].renderer.target.arch) if backend == "CPU" else Device[backend].renderer

def get_call_runtime(backend:str, prg:UOp):
  if backend != "CPU": return get_runtime(backend, prg)
  if (runtime:=runtime_cache.get(key:=(prg.key, backend))) is None:
    obj = prg.to_elf()
    lib = llvm_renderer(Device["CPU"].renderer.target.arch).compiler.trampoline(obj.lib, obj.name)
    runtime_cache[key] = runtime = Device[backend].runtime(replace(obj, lib=lib))
  return runtime

InstructionCall = tuple[UOp, list[UOp]]
instruction_cache:dict[tuple, list[tuple[int, int, tuple[InstructionCall|None, UOp|None]]]] = {}

# this is meant to replace the old emulator
@uopfunc
def init_wave(wg:UOp, wave:UOp, sgpr:UOp, vgpr:UOp, args_ptr:UOp, gx:sint, gy:sint, lx:sint, ly:sint, total_threads:sint, wave_size:int,
              scratch_size:int, rsrc2:int, arch:str="rdna3", user_data_count:int=0):
  # Constant operand encodings are represented in the emulator's SGPR buffer, not writable architectural registers.
  ci = UOp.range(81, 2, dtype=dtypes.int)
  inline_ints = sgpr.index(ci+128).store((ci<65).where(ci, 64-ci).cast(dtypes.uint32)).end(ci)
  fi = UOp.range(len(F32_INLINE), 3, dtype=dtypes.int)
  floats = UOp(Ops.BINARY, arg=struct.pack(f"<{len(F32_INLINE)}I", *F32_INLINE.values())).bitcast(dtypes.uint32)
  inline_floats = sgpr.index(fi+240).store(floats[fi]).end(fi)
  # Only defined launch inputs are initialized; other registers and compute LDS have no initial value guarantee.
  gidx, gidy, gidz = wg%gx, (wg//gx)%gy, wg//(gx*gy)
  n_lanes = (total_threads-wave*wave_size).minimum(wave_size)
  initial:list[tuple[int, sint]] = [(124, 0)] if wave_size == 32 else []  # NULL reads as zero on RDNA; CDNA uses this slot for M0.
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
  return UOp.sink(init_vgpr, *(sgpr.after(inline_ints, inline_floats).index(i).store(UOp.const(v, dtypes.uint32))
                             for i,v in dict(initial).items()))

@uopfunc
def loop_active(routes:UOp, active:UOp, header:UOp, waves:UOp):
  w = UOp.range(waves, 0, dtype=dtypes.int)
  return active[0].store(routes[w].load().eq(header).cast(dtypes.uint32).reduce(w, arg=Ops.MAX).ne(0)).sink()

def lower_body(sink:UOp, renderer:Renderer, name:str) -> tuple[UOp, list[UOp]]:
  bufs = sorted((u for u in sink.toposort() if u.op is Ops.PARAM), key=lambda u: u.arg.slot)
  body = sink.substitute({b:b.param_like(i, name=b.arg.name) for i,b in enumerate(bufs)}, walk=True)
  return lower_call(renderer, body.call(*bufs, name=name)), bufs

def instruction_call(inst:Inst, arch:str, renderer:Renderer) -> tuple[InstructionCall|None, UOp|None]:
  entries = instruction_cache.setdefault((arch, type(inst), inst.size(), type(renderer), renderer.target,
                                          *(x.value for x in to_program_lower_config)), [])
  inst_bytes = inst.to_bytes()
  bits = int.from_bytes(inst_bytes, "little")
  for base, mask, ret in entries:
    if bits & mask == base: return ret
  # Dynamic operand fields share one body; construct and rewrite it only on a canonical cache miss.
  ctx = _Ctx(inst.size(), _wave_size(arch), inst_addr=UOp.param(-1, dtypes.uint64, name="inst", addrspace=AddrSpace.ALU))
  sink = _get_handler(inst)(inst, ctx)
  base, mask, _, name = _canonical_info(inst, ctx, inst_bytes)
  ret = lower_body(sink, renderer, name) if sink.src else None, ctx.branch_cond
  entries.append((base, mask, ret))
  return ret

def instruction_block(instructions:list[tuple[int, InstructionCall]]) -> tuple[UOp, list[UOp]]:
  bufs = sorted({b for _,(_,bs) in instructions for b in bs if b.arg.slot != -1}, key=lambda b:b.arg.slot)
  base = UOp.param(len(bufs), dtypes.uint64, name="code_base", addrspace=AddrSpace.ALU) + \
         UOp.param(len(bufs)+1, dtypes.uint64, name="block_offset", addrspace=AddrSpace.ALU)
  params = {b:b.param_like(i, name=b.arg.name) for i,b in enumerate(bufs)}
  linear:list[UOp] = []
  seen:set[UOp] = set()
  for off,(template,bs) in instructions:
    args = [base+UOp.const(off, dtypes.uint64) if b.arg.slot == -1 else params[b] for b in bs]
    inputs = UOp.sink(*args)
    nodes = inputs.toposort(lambda u:u not in seen)
    nodes.pop(inputs)
    seen.update(nodes)
    linear.extend(nodes)
    linear.append(template.replace(src=(template.body, *args)))
  # The assembly is already linear: keep calls in instruction order and compute their addresses immediately before use.
  return UOp(Ops.LINEAR, src=tuple(linear), arg="asm_block"), bufs

def thread_branches(insts, blocks:dict[int, list[int]], paths:dict[int, dict[int, int]]):
  # Thread jump-only blocks.
  targets:dict[int, int] = {}
  def target(pc:int) -> int:
    seen:set[int] = set()
    while pc not in targets and pc not in seen and len(blocks[pc]) == 1 and getattr(insts[blocks[pc][0]], "op_name", "") == "S_BRANCH":
      seen.add(pc)
      pc = next(iter(paths[pc]))
    pc = targets.get(pc, pc)
    targets.update((src, pc) for src in seen)
    return pc
  for pc, dsts in list(paths.items()): paths[pc] = {target(dst):kind for dst,kind in dsts.items()}

def merge_branches(insts, blocks:dict[int, list[int]], paths:dict[int, dict[int, int]]):
  groups:dict[tuple, list[int]] = {}
  opposite = {"S_CBRANCH_SCC1":"S_CBRANCH_SCC0", "S_CBRANCH_VCCNZ":"S_CBRANCH_VCCZ", "S_CBRANCH_EXECNZ":"S_CBRANCH_EXECZ"}
  for pc, pcs in blocks.items():
    if len(paths[pc]) != 2: continue
    inst = insts[pcs[-1]]
    # Complementary tests with reversed destinations are the same branch.
    branch_dests = tuple(sorted((dst, kind ^ (inst.op_name in opposite)) for dst,kind in paths[pc].items()))
    key = (type(inst), opposite.get(inst.op_name, inst.op_name),
           tuple((n, getattr(inst, n)) for n,_ in inst._fields if n not in {"op", "simm16"}), branch_dests)
    groups.setdefault(key, []).append(pc)
  for group in groups.values():
    if len(group) < 2: continue
    header = blocks[group[0]][-1]
    dests = paths[group[0]]
    for pc in group:
      blocks[pc] = blocks[pc][:-1]
      paths[pc] = {header:UNCOND}
    blocks[header], paths[header] = [header], dests

# A block has an empty body; a loop contains its header block and the rest of its iteration.
CFGRegion = tuple[int, list["CFGRegion"]]
class IrreducibleCFG(RuntimeError): pass

def cfg_regions(paths:dict[int, dict[int, int]], entry:int) -> list[CFGRegion]:
  # Number the reachable blocks in DFS postorder, without recursion on long instruction streams.
  order:dict[int, int] = {}
  seen, pending = set(), [(entry, False)]
  while pending:
    pc, visited = pending.pop()
    if visited: order[pc] = len(order)
    elif pc not in seen:
      seen.add(pc)
      pending.append((pc, True))
      pending.extend((dst, False) for dst in sorted(paths[pc], reverse=True))
  preds:dict[int, set[int]] = {pc:set() for pc in order}
  for pc in order:
    for dst in paths[pc]: preds[dst].add(pc)
  def build(remaining:set[int]) -> list[CFGRegion]:
    regions: list[CFGRegion] = []
    # Reverse postorder and predecessor walks partition this region into SCCs in execution order.
    for pc in sorted(remaining, key=order.__getitem__, reverse=True):
      if pc not in remaining: continue
      component, todo = set(), [pc]
      while todo:
        if (dst:=todo.pop()) not in remaining: continue
        remaining.remove(dst)
        component.add(dst)
        todo.extend(preds[dst])
      body:list[CFGRegion] = []
      if len(component) > 1 or pc in paths[pc]:
        if any(preds[dst] - component for dst in component if dst != pc):
          raise IrreducibleCFG("irreducible control flow is not supported in ASM_CALL")
        # Removing the single entry exposes nested loops; the original DFS order still applies.
        body = [(pc, [])] + build(component - {pc})
      regions.append((pc, body))
    return regions
  return build(set(order))

def decode_cfg(insts:dict[int, Inst]):
  cfg = get_cfg(insts)
  thread_branches(insts, cfg["blocks"], cfg["paths"])
  try: return cfg, cfg_regions(cfg["paths"], 0)
  except IrreducibleCFG:
    # Sharing branch tails can make rotated loops reducible, but adds dispatch blocks otherwise.
    merge_branches(insts, cfg["blocks"], cfg["paths"])
  return cfg, cfg_regions(cfg["paths"], 0)

# ISA operand types and operations are independent of tensor construction and GPU lowering settings.
@Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0,
         DEFAULT_FLOAT="float32", DEFAULT_INT="int32", TRANSCENDENTAL=1)
@rewrite_group(name=lambda *args,ret,**_: TracingKey(f"Lift {(k:=ret.src[0].arg).name}", (("lift", k.function_name),)))
def lift(lib:int, lib_sz:int, lx:int, ly:int, lz:int, rsrc2:int, scratch_size:int, arch:str,
                  user_data_count:int, backend:str) -> UOp:
  view = to_mv(lib, lib_sz)
  dispatch = (lx, ly, lz, rsrc2, scratch_size, user_data_count)
  renderer = backend_renderer(backend)
  key = (lib_sz, arch, backend, dispatch, type(renderer), renderer.target, *(x.value for x in to_program_lower_config))
  entries = dispatch_cache.setdefault(key, [])
  for code, prg in entries:
    if bytes(view[:len(code)]) == code: return prg
  insts = amd_decode(view, arch)
  end = max((pc+inst.size() for pc,inst in insts.items()), default=0)
  # Include the stop marker: an extended instruction stream must not match a shorter cached program.
  code = bytes(view[:end+len(amd_code_end(arch))])
  prg = _lift(insts, arch, backend, dispatch)
  entries.append((code, prg))
  return prg

dispatch_cache:dict[tuple, list[tuple[bytes, UOp]]] = {}

def _lift(insts:dict[int, Inst], arch:str, backend:str, dispatch:tuple) -> UOp:
  renderer = backend_renderer(backend)
  cfg, regions = decode_cfg(insts)
  # Propagate possible wave PCs across acyclic regions; loops retain runtime routing.
  unconditional:set[int] = set()
  possible:set[int]|None = {0}
  for pc, body in regions:
    if body: possible = None
    elif possible is not None:
      if possible == {pc}: unconditional.add(pc)
      possible = (possible-{pc}) | (set(cfg["paths"][pc]) or {-1})
  routed = len(unconditional) != len(regions)
  axes = itertools.count(sum(inst.size() for inst in insts.values()))
  lx, ly, lz, rsrc2, scratch_size, user_data_count = dispatch
  # construct CALL graph
  wave_size, total_threads = _wave_size(arch), lx*ly*lz
  n_waves = (total_threads+wave_size-1)//wave_size
  gx = UOp.variable("groups_x", 1, dtypes.uint32.max, dtypes.uint32)
  gy = UOp.variable("groups_y", 1, dtypes.uint32.max, dtypes.uint32)
  wg = UOp.range(UOp.variable("groups", 0, dtypes.uint64.max, dtypes.uint64), 0, dtype=dtypes.uint64)
  wave = UOp.range(n_waves, 1, dtype=dtypes.int)
  # alloc register and LDS buffers
  sgpr = UOp.alloc((SGPR_COUNT*n_waves,), dtypes.uint32, 0, AddrSpace.REG)
  vgpr = UOp.alloc((256*wave_size*n_waves,), dtypes.uint32, 1, AddrSpace.REG)
  lds_size = ((rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT)*512
  lds = UOp.alloc((max(lds_size//4, 1),), dtypes.uint32, 3, AddrSpace.REG)
  scratch = UOp.alloc((max(scratch_size*wave_size*n_waves, 1),), dtypes.uint8, 4, AddrSpace.REG)
  accvgpr = UOp.alloc((256*wave_size*n_waves,), dtypes.uint32, 5, AddrSpace.REG) if wave_size == 64 else vgpr
  args_ptr = UOp.variable("args_ptr", 0, dtypes.uint64.max, dtypes.uint64)
  code_addr = UOp.variable("lib", 0, dtypes.uint64.max, dtypes.uint64)
  # Keep grid dimensions unsigned: mixing the uint64 workgroup index with int32 promotes division/modulo to floats.
  dims = [gx, gy, *(UOp.const(x, dtypes.int) for x in (lx, ly, total_threads))]
  ctx = _Ctx(4, wave_size)
  buffers = {ctx.sgpr:(sgpr, SGPR_COUNT), ctx.vgpr:(vgpr, 256*wave_size), ctx.vmem:(ctx.vmem, 0),
             ctx.lds:(lds, 0), ctx.scratch:(scratch, scratch_size*wave_size)}
  if wave_size == 64: buffers[ctx.accvgpr] = (accvgpr, 256*wave_size)
  routes = UOp.alloc((n_waves,), dtypes.int, 6, AddrSpace.REG)
  active = UOp.alloc((1,), dtypes.bool, 7, AddrSpace.REG)
  route = UOp.param(6, dtypes.int, (1,), name="route")
  taken, fallthrough = [UOp.param(i, dtypes.int, name=name, addrspace=AddrSpace.ALU)
                        for i,name in enumerate(("taken", "fallthrough"), 7)]
  buffers[route] = (routes, 1)
  init = init_wave(wg, wave, sgpr.index(wave*SGPR_COUNT), vgpr.index(wave*256*wave_size), args_ptr, *dims,
                   wave_size, scratch_size, rsrc2, arch, user_data_count)
  user_vars = [u for u in init.body.toposort() if u.is_variable]
  init = init.replace(src=(init.body.substitute({u:UOp.param(len(init.src)-1+i, u.dtype, name=u.arg.name, addrspace=AddrSpace.ALU)
                                                 for i,u in enumerate(user_vars)}, walk=True), *init.src[1:], *user_vars))
  initialized = (routes.after(init).index(wave).store(0) if routed else UOp.group(init, wave)).end(wave)
  def block_instructions(block_pc:int):
    cond = UOp.const(True)
    branch_pc = block_pc
    instructions = []
    for off in cfg["blocks"][block_pc]:
      inst = insts[off]
      inst_st = getattr(inst, "op_name", "").lower()
      if inst_st.startswith("s_code_end"): continue
      if inst_st.startswith(("s_getpc", "s_setpc")): raise AssertionError("getpc and setpc are not allowed in ASM_CALL")
      call, branch_cond = instruction_call(inst, arch, renderer)
      if branch_cond is not None: cond, branch_pc = branch_cond, off
      if call is not None: instructions.append((off, call))
    values = {}
    if routed:
      dsts = {kind:dst for dst,kind in cfg["paths"][block_pc].items()}
      target = cond.where(taken, fallthrough) if len(dsts) == 2 else taken
      values = {taken:UOp.const(dsts[COND_TAKEN], dtypes.int), fallthrough:UOp.const(dsts[COND_NOT_TAKEN], dtypes.int)} if len(dsts) == 2 \
        else {taken:UOp.const(next(iter(dsts.values()), -1), dtypes.int)}
      instructions.append((branch_pc, lower_body(route[0].store(target).sink(), renderer, "asm_branch")))
    return instructions, values

  def emit_block(block_pc:int, wave:UOp, end:UOp, scopes:tuple[UOp, ...]) -> list[UOp]:
    instructions, values = block_instructions(block_pc)
    if not instructions: return []
    off = instructions[0][0]
    body, bufs = instruction_block([(pc-off, info) for pc,info in instructions])
    args = [values[b] if b.addrspace is AddrSpace.ALU else buffers[b][0].after(end, *scopes).index(wave*buffers[b][1]) for b in bufs]
    return [body.call(*args, code_addr, UOp.const(off, dtypes.uint64), name="asm_block")]
  def emit_dispatch(regions:list[CFGRegion], scopes:tuple[UOp, ...], end:UOp) -> UOp:
    for pc, body in regions:
      if body:
        loop = UOp.loop(next(axes)).replace(src=(UOp(Ops.NOOP).after(end),))
        end = emit_dispatch(body, scopes+(loop,), loop)
        # Waves that exited keep their route while the remaining waves repeat the loop.
        check = loop_active(routes.after(end, *scopes, loop).index(0), active,
                            UOp.const(pc, dtypes.int), UOp.const(n_waves, dtypes.int))
        end = end.backedge(loop, active.after(check, *scopes, loop)[0].load())
      else:
        # Complete the block for every participating wave before entering its successors, including barrier successors.
        w = UOp.range(n_waves, next(axes), dtype=dtypes.int)
        gates = () if pc in unconditional else (UOp.range(routes.after(end)[w].load().eq(pc).cast(dtypes.int).after(*scopes),
                                                      next(axes), dtype=dtypes.int),)
        block_scopes = gates or (*scopes, w)
        end = UOp.group(end, *emit_block(pc, w, end, block_scopes), *block_scopes).end(*gates).end(w)
    return end

  body = emit_dispatch(regions, (wg,), initialized).end(wg)
  # Operand bits are read from the runtime code pointer; identical call graphs can share machine code.
  sink = UOp.sink(body, arg=KernelInfo()).rtag(1)
  return to_program(sink, renderer)
