from typing import TypeVar, Generic, Callable, Any, Sequence, overload
import functools, os, io, pickle, itertools, threading, warnings, time
from tinygrad.tensor import Tensor, all_tensors
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, BEAM, getenv, JIT, pluralize, VIZ, disable_gc
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.dtype import DType, dtypes
from tinygrad.uop.ops import UOp, PatternMatcher, Variable, Ops, rewrite_group, graph_rewrite, UOpMetaClass
from tinygrad.engine.realize import capturing, compile_linear, link_linear, run_linear
from tinygrad.schedule.memory import memory_plan_rewrite, _collect_bufs
from tinygrad.nn.state import get_parameters
from tinygrad.uop.movement import mop_cleanup
from dataclasses import dataclass

def prune_linear(linear:UOp, needed:set[UOp]) -> tuple[UOp, UOp]:
  kept, onetime = [], []
  for si in linear.src:
    si_bufs = {b for src in si.src[1:] for b in _collect_bufs(src)}
    if not si_bufs.isdisjoint(needed):
      kept.append(si)
      needed |= si_bufs
    else: onetime.append(si)
  return linear.replace(src=tuple(kept)), linear.replace(src=tuple(onetime))

def _copy_input(u:UOp) -> UOp:
  if u.on_disk(): raise JitError("cannot make an independent copy of a written DISK input")
  run_linear(UOp(Ops.LINEAR, src=((new:=UOp.new_buffer(u.device, u.max_numel(), u.dtype)).store_call(u),)))
  return new

@rewrite_group(lambda linear,held_bufs,input_uops,input_vars,scratch=None,ret=(): f"JIT {pluralize('call', len(linear.src))}")
def jit_lower(linear:UOp, held_bufs:set[UOp], input_uops:list[UOp], input_vars:dict[str, int], scratch:dict|None=None) -> UOp:
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View captured linear")

  # parametrize input buffers: map each input buffer UOp to a PARAM with the correct slot index
  linear = linear.substitute({u: UOp.param(i, u.dtype, u.max_numel(), u.device) for i,u in enumerate(input_uops)}, walk=True)
  # unbind input variables, they change every call and are read from var_vals. any other binding was fixed at capture and stays
  linear = linear.substitute({v: v.unbound() for v in linear.toposort() if v.is_bound_var and v.expr in input_vars}, walk=True)
  linear = memory_plan_rewrite(linear, held_bufs, scratch)
  if _forked_capture and scratch is not None and _forked_capture[0].scratch is scratch: _forked_capture[2](scratch)
  # the scratch is freed by free_intermediates, a shared one isn't
  shared = set(scratch.values()) if scratch is not None else set()
  linear = linear.substitute({u: u.rtag("scratch") for u in linear.toposort() if u.op is Ops.BUFFER and u not in held_bufs and u not in shared},
                             walk=True)
  linear = compile_linear(linear, beam=getenv("JITBEAM", BEAM.value), input_uops=input_uops, cache=False)
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View compiled linear")
  return linear

class JitError(Exception): pass

def _check_no_non_tensor_return(ret):
  if ret is None or isinstance(ret, Tensor): return
  if isinstance(ret, (tuple, list, dict)):
    for item in (ret.values() if isinstance(ret, dict) else ret): _check_no_non_tensor_return(item)
    return
  raise JitError(f"JIT return contains non-Tensor value of type {type(ret).__name__}")

ReturnType = TypeVar('ReturnType')
@dataclass
class CapturedJit(Generic[ReturnType]):
  ret: Any  # includes the Tensors or any other returned object
  _linear: UOp
  expected_names: list[int|str]
  expected_input_info: list[tuple[UOp, tuple[Variable, ...], DType, str]]  # (view, variables, dtype, device) per input

  @functools.cached_property
  def linear(self) -> UOp: return link_linear(self._linear, allow_cache=False) # do not cache jit

  def __reduce__(self): return self.__class__, (self.ret, self._linear, self.expected_names, self.expected_input_info)

  @functools.cached_property
  def _jit_bufs(self) -> set[UOp]: return {u for u in self._linear.toposort() if u.op is Ops.BUFFER}

  @functools.cached_property
  def _symbolic_ret(self) -> list[tuple[Tensor, UOp, dict[Variable, int]]]:
    return [(t, *ub) for t in get_parameters(self.ret) if (ub:=t.uop.unbind_all())[1]]

  def __call__(self, input_uops:list[UOp], var_vals:dict[str, int]) -> ReturnType:
    concrete = tuple(_copy_input(u) if u in self._jit_bufs else u for u in input_uops)
    if DEBUG >= 1 and len(self.linear.src) >= 10: print(f"jit execs {len(self.linear.src)} calls")
    run_linear(self.linear, var_vals, input_uops=concrete, jit=True)
    for t,u,vals in self._symbolic_ret: t.uop = u.substitute({v:v.bind(var_vals.get(v.expr, i)) for v,i in vals.items()}, walk=True)
    return self.ret

  def free_intermediates(self):
    bufs = [u.arg.buffer for u in self._jit_bufs if u.tag == "scratch"]
    if not (freed:=[b for b in flatten(b.bufs if isinstance(b, MultiBuffer) else [b] for b in bufs) if b.is_allocated()]): return
    for b in freed: b.deallocate()
    self.__dict__.pop("linear", None) # should be relinked

def _prepare_jit_inputs(args, kwargs):
  input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  # extract tensors from containers (shallow, not recursive to avoid grabbing model weights)
  for x in args + tuple(kwargs.values()):
    it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
    tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
  def get_input_uops() -> list[UOp]: return flatten([[t.uop.src[0]] if t.uop.op is Ops.UNSHARD else [t.uop] for t in tensors])
  if any(u.is_virtual for u in get_input_uops()): raise JitError("JIT inputs must be real buffers; use .clone()")
  if len(unrealized_tensors := [x for x in tensors if not x.uop.is_realized]): Tensor.realize(*unrealized_tensors)
  input_uops = get_input_uops()
  # collect buffer UOps (including MultiBuffer)
  input_buf_uops: list[UOp] = [u.base for u in input_uops if u.base.realized is not None]
  if len(set(input_buf_uops)) != len(input_buf_uops): raise JitError("duplicate inputs to JIT")
  inputs = [(*(u.substitute({u.base:UOp(Ops.NOOP)}, extra_pm=mop_cleanup).unbind_all()), u.dtype, u.device) for u in input_uops]
  _var_vals = merge_dicts([x[1] for x in inputs] + [dict(v.unbind() for v in (args + tuple(kwargs.values())) if isinstance(v, UOp))])
  var_vals = {k.expr:v for k,v in _var_vals.items()}
  expected_input_info = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in inputs]
  return input_buf_uops, var_vals, names, expected_input_info

class _TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]|None, captured:CapturedJit|None=None, prune=False, capture_first=False,
               scratch:dict|None=None):
    # capture_first: capture on the first call, no eager call first (the one time setup of the function is done, e.g. by another jit of it)
    # scratch: the memory of the intermediates, shared by the jits given the same dict (they must never run at the same time)
    assert fxn or captured, "need either a function or a CapturedJit"
    self.fxn = fxn
    self.captured: CapturedJit|None = captured
    self.cnt: int = 2 if self.fxn is None else 1 if capture_first else 0
    self.prune, self.scratch = prune, scratch

  def add_linear(self, linear:UOp): self._linears.append(linear)

  def reset(self):
    assert self.fxn is not None, "can't reset without function"
    self.cnt = 0
    self.captured = None

  def __reduce__(self):
    assert self.captured is not None, "can't pickle an uncaptured JIT"
    return self.__class__, (None, self.captured)

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  @disable_gc()
  def __call__(self, *args, **kwargs) -> ReturnType:
    input_buf_uops, var_vals, names, expected_input_info = _prepare_jit_inputs(args, kwargs)
    if not JIT or self.cnt == 0:
      # jit ignore
      assert self.fxn is not None
      with Context(BEAM=0 if getenv("IGNORE_JIT_FIRST_BEAM") else BEAM.value):
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(*params)
    elif self.cnt == 1:
      # jit capture
      assert self.fxn is not None
      if capturing: raise RuntimeError(f"having TinyJit inside another TinyJit is not supported {len(capturing)=} {capturing=}")
      self._linears: list[UOp] = []
      capturing.append(self)
      try:
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(*params)
      finally: capturing.clear()
      if not len(self._linears): raise JitError("didn't JIT anything!")
      _check_no_non_tensor_return(ret)
      if DEBUG >= 1: print(f"JIT captured {len(self._linears)} linears with {len(input_buf_uops)} inputs")

      # combine all captured linears into one, memory plan, and compile
      big_linear = UOp(Ops.LINEAR, src=tuple(flatten([l.src for l in self._linears])))
      del self._linears

      if self.prune:
        big_linear, onetime_linear = prune_linear(big_linear, set(input_buf_uops))
        if DEBUG >= 1: print(f"pruned from {len(big_linear.src) + len(onetime_linear.src)} -> {len(big_linear.src)} kernels")
        run_linear(onetime_linear, var_vals)
        del onetime_linear

      # hold all buffers with real storage reachable from live Tensors (e.g. lazy .grad created during capture) and all buffers with
      # allocated storage in the captured linear (e.g. constants baked in by copies): the memory planner can't suballocate those
      def _buf_or_none(u:UOp) -> Buffer|MultiBuffer|None: return u.arg.buffer if u.op is Ops.BUFFER else None
      held_bufs = {u for tref in list(all_tensors) if (t:=tref()) is not None for u in t.uop.toposort() if _buf_or_none(u) is not None}
      held_bufs |= {u for u in big_linear.toposort() if (b:=_buf_or_none(u)) is not None and b.is_allocated()}
      linear = jit_lower(big_linear, held_bufs, input_buf_uops, var_vals, self.scratch)
      if _forked_capture and _forked_capture[0] is self: _forked_capture[1](ret, linear, names, expected_input_info) # (doesn't return)
      # drop the pre-planning graph: it keeps the whole capture-time working set allocated (big_linear) or referenced (held_bufs).
      # the planned linear only uses the arena/held buffers, so the intermediates must be freed before linking and first exec
      del big_linear, held_bufs
      self.captured = CapturedJit(ret, linear, names, expected_input_info)
      ret = self.captured(input_buf_uops, var_vals)
    elif self.cnt >= 2:
      # jit exec
      assert self.captured is not None
      if self.captured.expected_names != names: raise JitError(f"args mismatch in JIT: {self.captured.expected_names=} != {names}")
      if self.captured.expected_input_info != expected_input_info:
        raise JitError(f"args mismatch in JIT: {self.captured.expected_input_info=} != {expected_input_info=}")
      ret = self.captured(input_buf_uops, var_vals)

    self.cnt += 1
    return ret

# overload signatures support both @TinyJit and @TinyJit(prune=True) syntax
@overload
def TinyJit(fxn:Callable[..., ReturnType], *, prune:bool=False, capture_first:bool=False, scratch:dict|None=None) -> _TinyJit[ReturnType]: ...
@overload
def TinyJit(fxn:None=None, *, prune:bool=False, capture_first:bool=False,
            scratch:dict|None=None) -> Callable[[Callable[..., ReturnType]], _TinyJit[ReturnType]]: ...
def TinyJit(fxn=None, **kwargs): return (lambda f: _TinyJit(f, **kwargs)) if fxn is None else _TinyJit(fxn, **kwargs)

# ******** parallel capture: jits compiled in forked processes ********

_forked_capture:list[Any] = [] # (the jit a forked process captures, what it does with it)

def _prefork_buffers() -> dict[int, Buffer|MultiBuffer]:
  # the buffers of the uops (and their bases): a forked process refers to them by id, the parent has them at the same ids
  reg:dict[int, Buffer|MultiBuffer] = {}
  stack:list[Buffer|MultiBuffer] = [k[2].buffer for k in list(UOpMetaClass.ucache) if k[0] is Ops.BUFFER and k[2].buffer is not None]
  while stack:
    if id(x:=stack.pop()) in reg: continue
    reg[id(x)] = x
    if isinstance(x, MultiBuffer): stack += x.bufs
    elif x._base is not None: stack.append(x._base)
  return reg

class _Pickler(pickle.Pickler):
  def __init__(self, f, reg:dict[int, Any], idx:dict[UOp, int]|None):
    super().__init__(f, protocol=pickle.HIGHEST_PROTOCOL)
    self.reg, self.idx = reg, idx
  def persistent_id(self, obj):
    if isinstance(obj, (Buffer, MultiBuffer)) and id(obj) in self.reg: return ("b", id(obj))
    if self.idx is not None and isinstance(obj, UOp) and obj in self.idx: return ("u", self.idx[obj])
    if self.idx is not None and isinstance(obj, Tensor): return ("t", self.idx[obj.uop])
    return None

class _Unpickler(pickle.Unpickler):
  def __init__(self, data:bytes, reg:dict[int, Any], out:list[UOp]):
    super().__init__(io.BytesIO(data))
    self.reg, self.out = reg, out
  def persistent_load(self, pid):
    kind, v = pid
    return self.reg[v] if kind == "b" else self.out[v] if kind == "u" else Tensor(self.out[v])

def _send(wfd:int, data:bytes):
  # a message: its length, then it
  view = memoryview(len(data).to_bytes(8, "little") + data)
  while len(view): view = view[os.write(wfd, view):]

def _send_arenas(wfd:int, scratch:dict):
  # the sizes of the arenas first (the parent makes the shared ones, it links the jits as they come)
  _send(wfd, pickle.dumps({k:u.max_numel() for k,u in scratch.items()}))

def _send_capture(wfd:int, reg:dict[int, Any], jit:_TinyJit, ret, linear:UOp, names, info):
  # the compiled jit as uop records in order (the uops by index: no recursion), the buffers of the parent by id, the arenas by size
  roots = [linear, *[t.uop for t in get_parameters(ret)], *[u for x in info for u in (x[0], *x[1])], *(jit.scratch or {}).values()]
  nodes = list(UOp.sink(*roots).toposort(enter_calls=True))[:-1]
  idx = {u:i for i,u in enumerate(nodes)}
  arenas = {k:(idx[u], u.max_numel()) for k,u in (jit.scratch or {}).items() if u in idx}
  recs, rest = io.BytesIO(), io.BytesIO()
  _Pickler(recs, reg, None).dump([(u.op, tuple(idx[x] for x in u.src), u.arg, u.tag) for u in nodes])
  _Pickler(rest, reg, idx).dump((ret, idx[linear], names, info))
  _send(wfd, pickle.dumps((arenas, recs.getvalue(), rest.getvalue()), protocol=pickle.HIGHEST_PROTOCOL))
  os.close(wfd)
  _exit_child(0)

def _exit_child(code:int):
  from tinygrad.engine import worker
  worker.terminate_worker_pool()
  os._exit(code)

def capture_parallel(calls:Sequence[tuple[_TinyJit, Callable[[], Any]]], meanwhile:Callable[[], None]|None=None) -> list[bool]:
  """
  the jits captured and compiled at the same time, each in a forked process: its call (which calls the jit for its capture) runs there, the
  jit is captured and compiled but nothing runs, the parent takes the compiled jit (linked and run on its first call there). the jits share
  their scratch (the biggest arena of each lane). a jit that can't be captured so is left to capture on its call. returns which were
  meanwhile: the parent's work while they compile (e.g. the data the jits read, made in the buffers they use)
  """
  todo = [k for k, (j, _) in enumerate(calls) if j.cnt == 1 and j.captured is None and j.fxn is not None]
  if not todo or not hasattr(os, "fork"):
    if meanwhile is not None: meanwhile()
    return [False] * len(calls)
  every, calls = list(calls), [calls[k] for k in todo]
  st, reg, start = time.perf_counter(), _prefork_buffers(), next(UOp.unique_num)
  children:list[tuple[int, int]] = []
  for i, (jit, call) in enumerate(calls):
    r, w = os.pipe()
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", DeprecationWarning) # (fork with threads: the child runs no other thread's work)
      pid = os.fork()
    if pid == 0:
      os.close(r)
      import tinygrad.device as device_mod
      from tinygrad.engine import worker
      from tinygrad.helpers import PARALLEL
      device_mod.FORKED_COMPILE, worker.worker_pool = True, None
      PARALLEL.value = max(1, PARALLEL.value // len(calls))
      UOp.unique_num = itertools.count(start + (i + 1) * (1 << 40)) # (its new uops never are the parent's or another child's)
      _forked_capture[:] = [jit, functools.partial(_send_capture, w, reg, jit), functools.partial(_send_arenas, w)]
      try: call()
      except BaseException:
        import traceback, sys
        traceback.print_exc(limit=-4 if DEBUG < 2 else None)
        sys.stderr.flush()
      _exit_child(1)
    os.close(w)
    children.append((pid, r))

  # read the pipes at once (a child blocks on a full pipe): the arena sizes, then the jit
  msgs:list[list[bytes]] = [[] for _ in children]
  got = threading.Condition()
  def put(k:int, data:bytes):
    with got:
      msgs[k].append(data)
      got.notify_all()
  def read(k:int):
    with os.fdopen(children[k][1], "rb") as f:
      while len(head:=f.read(8)) == 8: put(k, f.read(int.from_bytes(head, "little")))
    put(k, b"") # (the end)
  threads = [threading.Thread(target=read, args=(k,), daemon=True) for k in range(len(children))]
  for t in threads: t.start()
  if meanwhile is not None: meanwhile()

  # the arenas: the biggest of every lane, shared (made before any jit is linked)
  with got: got.wait_for(lambda: all(len(m) >= 1 for m in msgs))
  sizes:dict[Any, int] = {}
  for m in msgs:
    for k, n in (pickle.loads(m[0]).items() if m[0] else []): sizes[k] = max(sizes.get(k, 0), n)
  for jit, _ in calls:
    if jit.scratch is None: continue
    for k, n in sizes.items():
      if k not in jit.scratch or jit.scratch[k].max_numel() < n: jit.scratch[k] = UOp.new_buffer(k[0], n, dtypes.int8)

  # every jit as it comes: taken and linked
  oks, left = [False] * len(children), set(range(len(children)))
  while left:
    with got: got.wait_for(lambda: any(len(msgs[k]) >= 2 or msgs[k][-1] == b"" for k in left))
    k = next(k for k in left if len(msgs[k]) >= 2 or msgs[k][-1] == b"")
    left.discard(k)
    if os.waitpid(children[k][0], 0)[1] != 0 or len(msgs[k]) < 2 or not msgs[k][1]: continue
    jit = calls[k][0]
    arenas, recs, rest = pickle.loads(msgs[k][1])
    arena_at = {i: (jit.scratch or {})[key] for key, (i, _) in arenas.items()}
    out:list[UOp] = []
    for i, (op, src, arg, tag) in enumerate(_Unpickler(recs, reg, out).load()):
      out.append(arena_at[i] if i in arena_at else UOp(op, tuple(out[j] for j in src), arg, tag))
    ret, lin, names, info = _Unpickler(rest, reg, out).load()
    jit.captured, jit.cnt = CapturedJit(ret, out[lin], names, info), 2
    _ = jit.captured.linear  # linked
    oks[k] = True
    if DEBUG >= 1: print(f"capture_parallel: jit {k} taken in {time.perf_counter()-st:.2f}s")
  return [k in todo and oks[todo.index(k)] for k in range(len(every))]
