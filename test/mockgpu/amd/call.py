import ctypes, itertools, functools, struct, re, subprocess, tempfile
from pathlib import Path
from collections.abc import Iterator
from tinygrad.viz.serve import amd_decode, get_cfg, COND_NOT_TAKEN, UNCOND
from tinygrad.uop.ops import UOp, sint, Ops, KernelInfo, PatternMatcher, UPat, graph_rewrite, rewrite_group, uopfunc
from tinygrad.codegen import to_program, to_program_config
from tinygrad.device import Device
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.support.compiler_cpu import ClangCompiler
from tinygrad.dtype import AddrSpace, Invalid, dtypes
from tinygrad.helpers import Context, getenv, TracingKey, dedup, unwrap, Target, cpu_profile
from tinygrad.runtime.autogen import hsa
from tinygrad.renderer.amd.dsl import Inst, EXEC_LO, ttmp
from test.mockgpu.amd.emu import (_Ctx, _get_handler, _wave_size, _canonical_info, PC_LO_IDX, PC_HI_IDX,
                                  SGPR_COUNT, SCRATCH_STRIDE_IDX, F32_INLINE)

class CallClangJIT(ClangCompiler):
  def __init__(self, arch:list[str]):
    super().__init__(arch)
    self.functions:dict[str, int] = {}
    self.modules:list[ctypes.CDLL] = []

  # Function addresses belong to this process and must never enter the disk cache.
  def compile_cached(self, src:str) -> bytes: return self.compile(src)

  def compile(self, src:str) -> bytes:
    definitions = list(re.finditer(r"^__attribute__\(\([^\n]*\)\) void (\w+)\(([^\n]*)\) \{\n.*?^\}", src, re.M|re.S))
    entry = definitions[-1][1]
    new_functions = [m[1] for m in definitions if m[1] not in self.functions]
    for m in reversed(definitions):
      if m[1] in self.functions:
        declaration = f"static void (*{m[1]})({m[2]}) = (void (*)({m[2]})){self.functions[m[1]]}ul;"
        src = src[:m.start()]+declaration+src[m.end():]
    self.add_module(src, new_functions)
    addr = self.functions[entry]
    if self.arch == 'x86_64': return b'\x48\xb8'+struct.pack('<Q', addr)+b'\xff\xe0'
    if self.arch == 'arm64': return struct.pack('<IIQ', 0x58000050, 0xd61f0200, addr)
    raise RuntimeError(f'unsupported JIT trampoline architecture {self.arch}')

  def add_module(self, src:str, names:list[str]):
    with tempfile.TemporaryDirectory() as directory, cpu_profile('Clang JIT compile'):
      path = str(Path(directory)/'module.so')
      subprocess.run([getenv('CC', 'clang'), '-shared', '-x', 'c', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno',
                      '-fno-strict-aliasing', *self.args, '-', '-o', path, '-lm'], input=src.encode(), check=True, capture_output=True)
      module = ctypes.CDLL(path)
    self.modules.append(module)
    self.functions.update((name, unwrap(ctypes.cast(getattr(module, name), ctypes.c_void_p).value)) for name in names)

class CallClangRenderer(ClangRenderer):
  def __init__(self, target:Target):
    super().__init__(target)
    self.compiler = CallClangJIT(target.arch.split(','))

  def render(self, uops:list[UOp]) -> str:
    topo = UOp.sink(*uops).toposort(enter_calls=True)
    self.fn_names = {b:('asm_block_' if b.arg == 'asm_block' else 'asm_fn_')+b.key.hex() for b in topo if b.op is Ops.LINEAR}
    definitions = []
    all_uops = list(uops)
    for body, name in self.fn_names.items():
      _, lines, bufs = self._render(list(body.src))
      params = ', '.join(f"{self.param_type(p)} {n}" for n,(p,_) in bufs)
      attrs = 'noinline, optnone' if body.arg == 'asm_block' else 'noinline'
      definitions.append(f"__attribute__(({attrs})) void {name}({params}) {{\n"+'\n'.join(lines)+"\n}")
      all_uops.extend(body.src)
    name, lines, bufs = self._render(uops)
    entry = super().render_kernel(name, lines, bufs, [], None)
    entry = entry.replace('void '+name, '__attribute__((noinline, optnone)) void '+name, 1)
    return '\n'.join(self._render_defines(all_uops)+definitions+[entry])

@functools.cache
def clang_renderer(arch:str):
  return CallClangRenderer(Target("CPU", "CLANG", arch))

def backend_renderer(backend:str):
  return clang_renderer(Device["CPU"].renderer.target.arch) if backend == "CPU" else Device[backend].renderer

InstructionCall = tuple[UOp, list[UOp], tuple[int, ...], UOp|None]
instruction_cache:dict[tuple, list[tuple[int, int, InstructionCall]]] = {}

# this is meant to replace the old emulator
@uopfunc
def init_wave(wg:UOp, wave:UOp, sgpr:UOp, vgpr:UOp, lds:UOp, args_ptr:UOp, gx:sint, gy:sint, lx:sint, ly:sint, total_threads:sint, wave_size:int,
              lds_size:int, scratch_size:int, rsrc2:int, arch:str="rdna3", user_data:list[int]|None=None, accvgpr:UOp|None=None):
  # define ranges inside a wave
  li = UOp.range((wave.eq(0)).where(max(lds_size//4, 1), 0), 2, dtype=dtypes.int)
  si = UOp.range(SGPR_COUNT, 3, dtype=dtypes.int)
  vi = UOp.range(256*wave_size, 4, dtype=dtypes.int)
  # zero ALLOCs
  zero_lds = lds.index(li).store(0).end(li)
  zero_sgpr = sgpr.after(zero_lds).index(si).store(0).end(si)
  zero_vgpr = vgpr.after(zero_sgpr).index(vi).store(0)
  zero_agpr = unwrap(accvgpr).after(zero_sgpr).index(vi).store(0) if wave_size == 64 else UOp(Ops.NOOP)
  clear_vgpr = UOp.group(zero_vgpr, zero_agpr).end(vi)
  # set RANGE registers
  gidx, gidy, gidz = wg%gx, (wg//gx)%gy, wg//(gx*gy)
  n_lanes = (total_threads-wave*wave_size).minimum(wave_size)
  initial:list[tuple[int, sint]] = [*((128+i, i) for i in range(65)), *((193+i, (-i-1)&0xFFFFFFFF) for i in range(16)), *F32_INLINE.items()]
  initial += [(0, args_ptr.cast(dtypes.uint32)), (1, (args_ptr>>32).cast(dtypes.uint32))][:min(len(user_data), 2) if user_data else 2]
  if user_data: initial += [(i+2,v) for i,v in enumerate(user_data[2:])]
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
  init_vgpr = vgpr.after(clear_vgpr).index(lane.valid(tid<total_threads)).store(
      (((tid//(lx*ly))<<20)|(((tid//lx)%ly)<<10)|(tid%lx)).cast(dtypes.uint32)).end(lane)
  return UOp.sink(init_vgpr, *(sgpr.after(clear_vgpr).index(i).store(UOp.const(v, dtypes.uint32)) for i,v in dict(initial).items()))

def pc_index(idx:int) -> UPat:
  reg, null = UPat.const(idx).cast(), UPat.const(124).cast()
  return UPat.any(reg, reg.ne(null).where(reg, UPat.const(Invalid)))

def move_const_idxs(call:UOp) -> UOp|None:
  indexes = [u for u in call.body.toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and u.src[0].arg.name == "vmem"
             and u.src[1].op is Ops.CONST]
  idxs = dedup(u.src[1] for u in indexes)
  if not idxs: return None
  rep = {idx:UOp.param(len(call.src)-1+i, dtypes.uint64, name=f"inst_{i}", addrspace=AddrSpace.ALU) for i,idx in enumerate(idxs)}
  body = call.body.substitute({u:u.replace(src=(u.src[0], rep[u.src[1]], *u.src[2:])) for u in indexes}, walk=True)
  return call.replace(src=(body, *call.src[1:], *(idx.cast(dtypes.uint64) for idx in idxs)))

pm_asm_call = PatternMatcher([
  # remove PC from CALL body
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat(Ops.PARAM, name="buf").index(UPat.any(pc_index(PC_LO_IDX), pc_index(PC_HI_IDX))),), allow_any_len=True),
   lambda buf: UOp(Ops.NOOP) if buf.arg.name == "sgpr" else None),
  # move CONST outside CALL body
  (UPat(Ops.CALL, src=(UPat(Ops.SINK),), allow_any_len=True, name="call"), move_const_idxs),
])

def instruction_call(inst:Inst, arch:str) -> InstructionCall:
  entries = instruction_cache.setdefault((arch, type(inst), inst.size()), [])
  inst_bytes = inst.to_bytes()
  bits = int.from_bytes(inst_bytes, "little")
  for base, mask, ret in entries:
    if bits & mask == base: return ret
  # Dynamic operand fields share one body; construct and rewrite it only on a canonical cache miss.
  ctx = _Ctx(inst.size(), _wave_size(arch), inst_addr=0)
  sink = _get_handler(inst)(inst, ctx)
  base, mask, _, name = _canonical_info(inst, ctx, inst_bytes)
  bufs = sorted((u for u in sink.toposort() if u.op is Ops.PARAM), key=lambda u: u.arg.slot)
  body = sink.substitute({b:b.param_like(i, name=b.arg.name) for i,b in enumerate(bufs)})
  call = graph_rewrite(body.call(*bufs, name=name), pm_asm_call,
                       name="instruction call", bottom_up=True, enter_calls=True)
  ret = call, bufs, tuple(int(x) for x in call.src[len(bufs)+1:]), ctx.branch_cond
  entries.append((base, mask, ret))
  return ret

block_cache:dict[tuple, tuple[UOp, list[UOp]]] = {}

def instruction_chunks(instructions:list[tuple[int, InstructionCall]], size:int) -> Iterator[list[tuple[int, InstructionCall]]]:
  # Choose boundaries from instruction bodies, so an insertion needn't invalidate every following block.
  start = code_hash = 0
  for i,(_,info) in enumerate(instructions):
    code_hash = ((code_hash << 1) + int.from_bytes(info[0].body.key[:4], "little")) & (size-1)
    length = i-start+1
    if length >= max(size//4, 1) and (code_hash == 0 or length >= size*2):
      yield instructions[start:i+1]
      start = i+1
  if start < len(instructions): yield instructions[start:]

@functools.cache
def has_effects(body:UOp) -> bool:
  return any(u.op in (Ops.STORE, Ops.CALL) for u in body.toposort())

def instruction_block(instructions:list[tuple[int, InstructionCall]]) -> tuple[UOp, list[UOp]]:
  key = tuple((off, template, tuple(bufs), offsets) for off,(template,bufs,offsets,_) in instructions)
  if key not in block_cache:
    bufs = sorted({b for _,(_,bs,_,_) in instructions for b in bs}, key=lambda b:b.arg.slot)
    base = UOp.param(len(bufs), dtypes.uint64, name="code_base", addrspace=AddrSpace.ALU)
    params = {b:b.param_like(i, name=b.arg.name) for i,b in enumerate(bufs)}
    inst = None
    for off,(template,bs,offsets,_) in instructions:
      args = [params[b] for b in bs]
      invoke = template.replace(src=(template.body, *args,
                           *(base+UOp.const(off+i, dtypes.uint64) if off+i else base for i in offsets)))
      if inst is not None:
        invoke = invoke.replace(src=(*invoke.src[:-1], invoke.src[-1].after(inst)))
      inst = invoke
    body = UOp.sink(inst)
    block_cache[key] = body, bufs
  return block_cache[key]

def merge_branches(insts, blocks:dict[int, list[int]], paths:dict[int, dict[int, int]]):
  # Thread jump-only blocks, then share identical conditional branches with the same destinations.
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
        assert all(preds[dst] <= component for dst in component if dst != pc), "irreducible control flow is not supported in ASM_CALL"
        # Removing the single entry exposes nested loops; the original DFS order still applies.
        body = [(pc, [])] + build(component - {pc})
      regions.append((pc, body))
    return regions
  return build(set(order))

@functools.cache
def decode_cfg(lib_bytes:bytes, arch:str):
  insts = amd_decode(lib_bytes, arch)
  cfg = get_cfg(insts, render=False)["data"]
  merge_branches(insts, cfg["blocks"], cfg["paths"])
  return insts, cfg, cfg_regions(cfg["paths"], 0)

@rewrite_group(name=lambda *args,ret,**_: TracingKey(f"Lift {(k:=ret.src[0].arg).name}", (("lift", k.function_name),)))
def lift_dispatch(lib:int, lib_sz:int, lx:int, ly:int, lz:int, rsrc2:int, scratch_size:int, arch:str,
                  user_data:list[int]|None, backend:str) -> UOp:
  lib_bytes = ctypes.string_at(lib, lib_sz)
  dispatch = (lx, ly, lz, rsrc2, scratch_size, tuple(0 if i < 2 else v for i,v in enumerate(user_data)) if user_data else None)
  renderer = backend_renderer(backend)
  key = (lib_bytes, arch, backend, dispatch, type(renderer), renderer.target, *(x.value for x in to_program_config))
  if key not in dispatch_cache:
    dispatch_cache[key] = _lift(lib_bytes, arch, backend, dispatch)
  return dispatch_cache[key]

dispatch_cache:dict[tuple, UOp] = {}

def _lift(lib_bytes:bytes, arch:str, backend:str, dispatch:tuple) -> UOp:
  lib_sz, entry = len(lib_bytes), 0
  insts, cfg, regions = decode_cfg(lib_bytes, arch)
  axes = itertools.count(lib_sz)
  lx, ly, lz, rsrc2, scratch_size, user_data = dispatch
  # construct CALL graph
  wave_size, total_threads = _wave_size(arch), lx*ly*lz
  n_waves = (total_threads+wave_size-1)//wave_size
  gx = UOp.variable("groups_x", 1, dtypes.uint32.max, dtypes.uint32)
  gy = UOp.variable("groups_y", 1, dtypes.uint32.max, dtypes.uint32)
  wg = UOp.range(UOp.variable("groups", 0, dtypes.uint64.max, dtypes.uint64), 0)
  wave = UOp.range(n_waves, 1)
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
  buffers[routes] = (routes, 0)
  init = init_wave(wg, wave, sgpr.index(wave*SGPR_COUNT), vgpr.index(wave*256*wave_size), lds, args_ptr, *dims,
                   wave_size, lds_size, scratch_size, rsrc2, arch, user_data,
                   *([accvgpr.index(wave*256*wave_size)] if wave_size == 64 else []))
  initialized = routes.after(init).index(wave).store(entry).end(wave)
  afters = {b:buf.after(initialized) for b,(buf,_) in buffers.items()}

  def block_instructions(block_pc:int):
    cond = UOp.const(True)
    instructions = []
    for off in cfg["blocks"][block_pc]:
      inst = insts[off]
      inst_st = str(inst)
      if inst_st.startswith("s_code_end"): continue
      if inst_st.startswith(("s_getpc", "s_setpc")): raise AssertionError("getpc and setpc are not allowed in ASM_CALL")
      template, bufs, word_offsets, branch_cond = instruction_call(inst, arch)
      if branch_cond is not None: cond = branch_cond
      if not has_effects(template.body): continue
      instructions.append((off, (template, bufs, word_offsets, branch_cond)))
    return instructions, cond

  chunk_size = 128

  def emit_block(block_pc:int, wave:UOp, scopes:tuple[UOp, ...]) -> UOp:
    instructions, cond = block_instructions(block_pc)
    for chunk in instruction_chunks(instructions, chunk_size):
      off = chunk[0][0]
      body, bufs = instruction_block([((pc-off)//4, info) for pc,info in chunk])
      args = [afters[b].after(*scopes) for b in bufs]
      call = body.call(*args, (code_addr>>2)+off//4, name="asm_block")
      # CALL consumes argument ranges; restore the enclosing control-flow scopes.
      afters.update((b, arg.without_after.after(call, *scopes)) for b, arg in zip(bufs, args))
    return cond.substitute({u:buffers[u.src[0]][0].after(afters[u.src[0]], *scopes).index(wave*buffers[u.src[0]][1]+u.src[1])
                            for u in cond.toposort() if u.op is Ops.INDEX and u.src[0] in buffers}, walk=True)
  def emit_dispatch(regions:list[CFGRegion], scopes:tuple[UOp, ...]):
    for pc, body in regions:
      if body:
        loop = UOp.loop(next(axes)).replace(src=(UOp(Ops.NOOP).after(UOp.group(*afters.values())),))
        emit_dispatch(body, scopes+(loop,))
        # Waves that exited keep their route while the remaining waves repeat the loop.
        w = UOp.range(n_waves, next(axes))
        active = afters[routes].after(*scopes, loop)[w].load().eq(pc).cast(dtypes.uint32).reduce(w, arg=Ops.MAX)
        end = UOp.group(*afters.values()).backedge(loop, active.ne(0))
      else:
        # Complete the block for every participating wave before entering its successors, including barrier successors.
        w = UOp.range(n_waves, next(axes))
        gate = UOp.range(afters[routes][w].load().eq(pc).cast(dtypes.int).after(*scopes), next(axes))
        deps = UOp.group(*afters.values())
        afters.update((b, (buf.index(w*stride) if stride else buf).after(deps))
                      for b,(buf,stride) in buffers.items())
        cond = emit_block(pc, w, scopes+(w, gate))
        dsts = cfg["paths"][pc]
        target = UOp.const(-1, dtypes.int)
        for dst, kind in dsts.items():
          target = UOp.const(dst, dtypes.int) if len(dsts) == 1 else (cond.logical_not() if kind == COND_NOT_TAKEN else cond).where(dst, target)
        end = routes.after(*afters.values()).after(*scopes, w, gate)[w].store(target).end(gate).end(w)
      afters.update((b, buf.after(end)) for b,(buf,_) in buffers.items())

  emit_dispatch(regions, (wg,))
  body = UOp.group(*afters.values()).end(wg)
  # Operand bits are read from the runtime code pointer; identical call graphs can share machine code.
  name = f"asm_call_{arch}_{body.key.hex()[:16]}_{entry}"
  sink = UOp.sink(body, arg=KernelInfo(name=name, opts_to_apply=())).rtag(1)
  with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0):
    renderer = backend_renderer(backend)
    return to_program(sink, renderer)
