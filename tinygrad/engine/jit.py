from typing import TypeVar, Generic, Callable, Any, Sequence, overload, cast
import functools, fnmatch, os, io, pickle, itertools, threading, warnings, time, weakref, sys, heapq, concurrent.futures
from tinygrad.tensor import Tensor, all_tensors
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, BEAM, getenv, JIT, pluralize, VIZ, disable_gc, dedup, unwrap
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.dtype import DType, dtypes
from tinygrad.uop.ops import UOp, PatternMatcher, Variable, Ops, rewrite_group, graph_rewrite, UOpMetaClass
from tinygrad.engine.realize import capturing, compile_linear, link_linear, run_linear, _launch_args, _compile_pickled, _compile_pickled_timed
from tinygrad.engine.realize import startup_mark, startup_marks
from tinygrad.engine.worker import get_worker_pool
from tinygrad.schedule.memory import memory_plan_rewrite, _collect_bufs
from tinygrad.nn.state import get_parameters
from tinygrad.uop.movement import mop_cleanup
from dataclasses import dataclass, field

def prune_linear(linear:UOp, needed:set[UOp]) -> tuple[UOp, UOp]:
  kept, onetime = [], []
  for si in linear.src:
    si_bufs = {b for src in si.src[1:] for b in _collect_bufs(src)}
    if not si_bufs.isdisjoint(needed):
      kept.append(si)
      needed |= si_bufs
    else: onetime.append(si)
  return linear.replace(src=tuple(kept)), linear.replace(src=tuple(onetime))

@rewrite_group(lambda linear,held_bufs,input_uops,input_vars,scratch=None,ret=(): f"JIT {pluralize('call', len(linear.src))}")
def jit_lower(linear:UOp, held_bufs:set[UOp], input_uops:list[UOp], input_vars:dict[str, int], scratch:dict|None=None) -> UOp:
  if VIZ: graph_rewrite(linear, PatternMatcher([]), name="View captured linear")

  # parametrize input buffers: map each input buffer UOp to a PARAM with the correct slot index
  linear = linear.substitute({u: UOp.param(i, u.dtype, u.max_numel(), u.device) for i,u in enumerate(input_uops)}, walk=True)
  # unbind input variables, they change every call and are read from var_vals. any other binding was fixed at capture and stays
  linear = linear.substitute({v: v.unbound() for v in linear.toposort() if v.is_bound_var and v.expr in input_vars}, walk=True)
  startup_mark("captured")
  linear = memory_plan_rewrite(linear, held_bufs, scratch)
  startup_mark("memory planned")
  if _forked_capture and scratch is not None and _forked_capture[0].scratch is scratch: _forked_capture[2](scratch)
  # the intermediates are allocated when the jit is linked (freed by free_intermediates), a shared scratch isn't
  shared = set(scratch.values()) if scratch is not None else set()
  allocs = {u: UOp.alloc((u.max_numel(),), u.dtype, device=u.device) for u in linear.toposort()
            if u.op is Ops.BUFFER and u not in held_bufs and u not in shared}
  linear = linear.substitute(allocs, walk=True)
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
  _linears: dict[tuple[UOp, ...], tuple[UOp, dict[UOp, UOp]]] = field(default_factory=dict)  # aliased inputs -> (linked linear with copies, staging)

  def __reduce__(self): return self.__class__, (self.ret, self._linear, self.expected_names, self.expected_input_info)

  @functools.cached_property
  def _jit_bufs(self) -> set[UOp]: return {u for u in self._linear.toposort() if u.op is Ops.BUFFER}

  @functools.cached_property
  def _symbolic_ret(self) -> list[tuple[Tensor, UOp, dict[Variable, int]]]:
    return [(t, *ub) for t in get_parameters(self.ret) if (ub:=t.uop.unbind_all())[1]]

  def linked(self, aliased:tuple[UOp, ...]=()) -> tuple[UOp, dict[UOp, UOp]]:
    """the linked linear (and the staging of the aliased inputs) for these aliased inputs"""
    if aliased not in self._linears:
      if any(u.on_disk() for u in aliased): raise JitError("cannot make an independent copy of a written DISK input")
      staging = {u: UOp.new_buffer(u.device, u.max_numel(), u.dtype) for u in aliased}
      copies = compile_linear(UOp(Ops.LINEAR, src=tuple(s.store_call(u) for u,s in staging.items())))
      self._linears[aliased] = (link_linear(self._linear.replace(src=copies.src+self._linear.src)), staging)
    return self._linears[aliased]

  def __call__(self, input_uops:list[UOp], var_vals:dict[str, int]) -> ReturnType:
    # aliased inputs
    linear, staging = self.linked(tuple(u for u in input_uops if u in self._jit_bufs))
    if DEBUG >= 1 and len(linear.src) >= 10: print(f"jit execs {len(linear.src)} calls")
    run_linear(linear, var_vals, input_uops=[staging.get(u, u) for u in input_uops], jit=True)
    for t,u,vals in self._symbolic_ret: t.uop = u.substitute({v:v.bind(var_vals.get(v.expr, i)) for v,i in vals.items()}, walk=True)
    return self.ret

  def free_intermediates(self):
    self._linears.clear()
    _launch_args.clear()  # (the relinked calls can be the same: their buffers are allocated again)

  def rebase(self, arenas:dict[UOp, UOp]):
    """the arenas of its intermediates replaced by bigger ones (another jit sharing them needs more): relinked on its next call"""
    if not any(u in arenas for u in self._jit_bufs): return
    self._linear = self._linear.substitute(arenas, walk=True)
    self.__dict__.pop("_jit_bufs", None)
    self.free_intermediates()

_input_infos:weakref.WeakKeyDictionary[UOp, tuple] = weakref.WeakKeyDictionary()
def _input_info(u:UOp) -> tuple[UOp, dict[Variable, int], DType, str|tuple[str, ...]]:
  # the view of an input without its buffer, its bound variables, dtype and device: the same for every call with it (not kept alive by it)
  if (ret:=_input_infos.get(u)) is None:
    ret = _input_infos[u] = (*u.substitute({u.base:UOp(Ops.NOOP)}, extra_pm=mop_cleanup).unbind_all(), u.dtype, u.device)
  return ret

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
  inputs = [_input_info(u) for u in input_uops]
  _var_vals = merge_dicts([x[1] for x in inputs] + [dict(v.unbind() for v in (args + tuple(kwargs.values())) if isinstance(v, UOp))])
  var_vals = {k.expr:v for k,v in _var_vals.items()}
  expected_input_info = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in inputs]
  return input_buf_uops, var_vals, names, expected_input_info

# the jits sharing a scratch (by its id)
_scratch_jits:dict[int, weakref.WeakSet] = {}

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
    if scratch is not None: _scratch_jits.setdefault(id(scratch), weakref.WeakSet()).add(self)

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
    elif self.cnt == 1 and (pending:=self.__dict__.pop("_pending_take", None)) is not None and pending():
      # (capture_parallel's meanwhile: a jit a forked process captures, called here, waited for and taken: it runs)
      assert self.captured is not None
      ret = self.captured(input_buf_uops, var_vals)
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
      before = dict(self.scratch) if self.scratch is not None else {}
      linear = jit_lower(big_linear, held_bufs, input_buf_uops, var_vals, self.scratch)
      # (an arena it needs bigger: the other jits sharing it take the new one, the old one is freed before it's allocated)
      grown = {u: self.scratch[k] for k, u in before.items() if self.scratch is not None and self.scratch.get(k) is not u}
      for j in list(_scratch_jits.get(id(self.scratch), ())) if grown else []:
        if j is not self and j.captured is not None: j.captured.rebase(grown)
      del before, grown
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

    if JIT: self.cnt += 1
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
  _forked_capture.append("arenas sent")

def _send_capture(wfd:int, reg:dict[int, Any], jit:_TinyJit, ret, linear:UOp, names, info):
  # the compiled jit as uop records in order (the uops by index: no recursion), the buffers of the parent by id, the arenas by size
  if "arenas sent" not in _forked_capture: _send_arenas(wfd, {})
  roots = [linear, *[t.uop for t in get_parameters(ret)], *[u for x in info for u in (x[0], *x[1])], *(jit.scratch or {}).values()]
  nodes = list(UOp.sink(*roots).toposort(enter_calls=True))[:-1]
  idx = {u:i for i,u in enumerate(nodes)}
  arenas = {k:(idx[u], u.max_numel()) for k,u in (jit.scratch or {}).items() if u in idx}
  recs, rest = io.BytesIO(), io.BytesIO()
  _Pickler(recs, reg, None).dump([(u.op, tuple(idx[x] for x in u.src), u.arg) for u in nodes])
  _Pickler(rest, reg, idx).dump((ret, idx[linear], names, info))
  _send(wfd, pickle.dumps((arenas, recs.getvalue(), rest.getvalue()), protocol=pickle.HIGHEST_PROTOCOL))
  os.close(wfd)
  if (path:=getenv("STARTUP_STATS", "")):
    import json
    startup_mark("sent")
    with open(f"{path}.forks", "a") as f: f.write(json.dumps({"pid": os.getpid(), "marks": startup_marks}) + "\n")
  _exit_child(0)

def _exit_child(code:int):
  from tinygrad.engine import worker
  worker.terminate_worker_pool()
  os._exit(code)

def _recv(f) -> bytes|None:
  if len(head:=f.read(8)) < 8: return None
  return f.read(int.from_bytes(head, "little"))

def _remote_compile(qw:int, ar:int, tasks:list[tuple[bytes, bytes]]) -> list[UOp]:
  # (in a forked process) the kernels compiled by the parent
  _send(qw, pickle.dumps(tasks))
  with os.fdopen(os.dup(ar), "rb") as f: ret = pickle.loads(unwrap(_recv(f)))
  if isinstance(ret, str): raise RuntimeError(f"compile failed in the parent: {ret}")
  return [pickle.loads(x) for x in ret]

class _Compiles:
  """
  the compiles of the forked processes' kernels: one a key (its bytes), on the workers of a pool, the most urgent first (no more at once than
  the pool has workers): a process's later requests (its jit's host programs: the jit is done then) before first ones, of those the latest
  first (the jits captured last, prefill's, are the longest to finish after their kernels: they don't wait behind the ones captured early)
  """
  def __init__(self, pool):
    self.pool, self.lock, self.results, self.stats = pool, threading.Lock(), cast(dict[bytes, concurrent.futures.Future], {}), cast(list[tuple], [])
    self.queue:list[tuple[tuple, int, bytes, bytes]] = []
    self.seq, self.running, self.timed = itertools.count(), 0, bool(getenv("STARTUP_STATS", ""))
  def _dispatch(self):
    # (lock held) the most urgent waiting compiles started while workers are free
    while self.running < self.pool._processes and self.queue:
      k, t = (x:=heapq.heappop(self.queue))[2], x[3]
      if (r:=self.results[k]).running() or r.done(): continue # (started already: it was asked for again, more urgently)
      r.set_running_or_notify_cancel()
      self.running += 1
      self.pool.apply_async(_compile_pickled_timed if self.timed else _compile_pickled, (t,), callback=functools.partial(self._done, k),
                            error_callback=functools.partial(self._done, k, failed=True))
  def _done(self, k:bytes, res, failed=False):
    with self.lock:
      self.running -= 1
      self._dispatch()
    if failed: self.results[k].set_exception(res)
    else: self.results[k].set_result(res)
  def serve(self, qr:int, aw:int, who:int=-1):
    # (bytes only here: the workers unpickle the kernels and pickle their programs)
    with os.fdopen(qr, "rb") as f:
      for n in itertools.count():
        if (msg:=_recv(f)) is None: break
        t0, tasks = time.time(), pickle.loads(msg)
        with self.lock:
          new = [k for k, _ in tasks if k not in self.results]
          if (dump:=getenv("STARTUP_DUMP", "")):
            # (debug) the new compile tasks of this process, to replay their compiles elsewhere
            with open(f"{dump}.{who}", "ab") as df: pickle.dump([(k, t) for k, t in tasks if k in new], df)
          for k, t in tasks:
            if (r:=self.results.setdefault(k, concurrent.futures.Future())).running() or r.done(): continue
            heapq.heappush(self.queue, ((n == 0, -t0), next(self.seq), k, t))
          self._dispatch()
        try:
          got = [self.results[k].result() for k, _ in tasks]
          if self.timed:
            self.stats.append((who, t0, time.time(), len(tasks), len(new), [g[1:] for g, (k, _) in zip(got, tasks) if k in new]))
            got = [g[0] for g in got]
          _send(aw, pickle.dumps(got))
        # (the worker's traceback: the pool's exception holds it as its cause)
        except Exception as e: _send(aw, pickle.dumps(f"{type(e).__name__}: {e}{chr(10) + str(e.__cause__) if e.__cause__ else ''}"))
    os.close(aw)

def warmup_wanted(name:str) -> bool:
  """
  WARMUP_JITS: the jits a model's warm-up captures and runs, comma separated patterns (fnmatch) of their names (e.g. "step1,prefill192*"; the
  default "all"; "none": no jit, the memory only). the others are captured on their first call (a run that uses a few doesn't wait for all)
  """
  if (sel:=str(getenv("WARMUP_JITS", "all"))) == "all": return True
  return any(fnmatch.fnmatchcase(name, p.strip()) for p in sel.split(",") if p.strip() not in ("", "none"))

def warmup_select(calls:dict[str, tuple[Any, Callable[[], Any]]], par:list[tuple[Any, Callable[[], Any]]]) \
    -> tuple[dict[str, tuple[Any, Callable[[], Any]]], list[tuple[Any, Callable[[], Any]]]]:
  """a warm-up's named calls (name -> (jit, call)) and its jits to capture at once, of the names WARMUP_JITS wants (see warmup_wanted)"""
  if getenv("WARMUP_JITS", "all") == "all": return calls, par
  calls = {n: c for n, c in calls.items() if warmup_wanted(n)}
  # (a call's jit can be a list: the jits of its parts)
  keep = [x for k, _ in calls.values() for x in (k if isinstance(k, list) else [k])]
  return calls, [(j, c) for j, c in par if any(j is k for k in keep)]

def run_as_they_come(take:Callable[..., bool|None], calls:dict[str, tuple[Any, Callable[[], Any]]], ran:set[str]):
  """
  (capture_parallel's meanwhile) the calls (name -> (jit or list of jits, call)) run as their jits come (taken), each once (its name added to
  ran): the warm-up's runs while the jits left compile. WARMUP_EARLY=0: none (the caller runs them after)
  """
  if not getenv("WARMUP_EARLY", 1): return
  while len(ran) < len(calls):
    if (name:=next((n for n, (j, _) in calls.items() if n not in ran and
                    all(take(x, wait=False) is not None for x in (j if isinstance(j, list) else [j]))), None)) is None:
      time.sleep(0.01)
      continue
    ran.add(name)
    calls[name][1]()

def capture_parallel(calls:Sequence[tuple[_TinyJit, Callable[[], Any]]], meanwhile:Callable[[Callable[..., bool|None]], None]|None=None,
                     taken:Callable[[_TinyJit], None]|None=None) -> list[bool]:
  """
  the jits captured and compiled at the same time, each in a forked process: its call (which calls the jit for its capture) runs there, the
  jit is captured and compiled but nothing runs, the parent takes the compiled jit (linked and run on its first call there). the jits share
  their scratch (the biggest arena of each lane). a jit that can't be captured so is left to capture on its call. returns which were
  meanwhile: the parent's work while they compile (e.g. the data the jits read, made in the buffers they use), called with a function that
  takes one of the jits (waits for it: it runs then). taken: called with every jit taken (in the order they come), the parent's work while the
  others compile
  """
  todo = [k for k, (j, _) in enumerate(calls) if j.cnt == 1 and j.captured is None and j.fxn is not None]
  if not todo or not hasattr(os, "fork"):
    if meanwhile is not None: meanwhile(lambda jit, wait=True: False)
    return [False] * len(calls)
  every, calls = list(calls), [calls[k] for k in todo]
  st, reg, start = time.perf_counter(), _prefork_buffers(), next(UOp.unique_num)
  children:list[tuple[int, int]] = []
  # the kernels of all the processes compiled by this one's workers, each once (most are in many jits): a request pipe and a reply pipe each
  pool = get_worker_pool()
  compiles = _Compiles(pool) if pool is not None else None
  for i, (jit, call) in enumerate(calls):
    if pool is not None: (qr, qw), (ar, aw) = os.pipe(), os.pipe()
    r, w = os.pipe()
    sys.stdout.flush(), sys.stderr.flush() # (the child doesn't print the parent's buffered output again)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", DeprecationWarning) # (fork with threads: the child runs no other thread's work)
      pid = os.fork()
    if pid == 0:
      os.close(r)
      import tinygrad.device as device_mod
      from tinygrad.engine import worker
      from tinygrad.helpers import PARALLEL
      device_mod.FORKED_COMPILE, worker.worker_pool = True, None
      if pool is not None:
        import tinygrad.engine.realize as realize
        for fd in (qr, aw): os.close(fd)
        realize.remote_compile = functools.partial(_remote_compile, qw, ar)
        startup_marks[:] = [(f"start {todo[i]}", time.time())]
      # (the cores shared by the processes: more compilers than cores slow the parent's work meanwhile)
      PARALLEL.value = max(1, PARALLEL.value // len(calls))
      UOp.unique_num = itertools.count(start + (i + 1) * (1 << 40)) # (its new uops never are the parent's or another child's)
      _forked_capture[:] = [jit, functools.partial(_send_capture, w, reg, jit), functools.partial(_send_arenas, w)]
      try:
        call()
        print(f"capture_parallel: jit {todo[i]} wasn't captured by its call (cnt {jit.cnt})", file=sys.stderr)
      except BaseException:
        import traceback
        traceback.print_exc(limit=-4 if DEBUG < 2 else None)
      sys.stderr.flush()
      _exit_child(1)
    os.close(w)
    children.append((pid, r))
    if pool is not None:
      for fd in (qw, ar): os.close(fd)
      threading.Thread(target=unwrap(compiles).serve, args=(qr, aw, todo[i]), daemon=True).start()

  # read the pipes at once (a child blocks on a full pipe): the arena sizes, then the jit
  msgs:list[list[bytes]] = [[] for _ in children]
  arrived:list[list[float]] = [[] for _ in children]
  got = threading.Condition()
  def put(k:int, data:bytes):
    with got:
      msgs[k].append(data)
      arrived[k].append(time.perf_counter() - st)
      got.notify_all()
  def read(k:int):
    with os.fdopen(children[k][1], "rb") as f:
      while len(head:=f.read(8)) == 8: put(k, f.read(int.from_bytes(head, "little")))
    put(k, b"") # (the end)
  threads = [threading.Thread(target=read, args=(k,), daemon=True) for k in range(len(children))]
  for t in threads: t.start()
  # the arenas: the biggest of every lane, shared (made before a jit of them is linked)
  arenas_made = [False]
  def make_arenas():
    if arenas_made[0]: return
    with got: got.wait_for(lambda: all(len(m) >= 1 for m in msgs))
    sizes:dict[Any, int] = {}
    for m in msgs:
      for k, n in (pickle.loads(m[0]).items() if m[0] else []): sizes[k] = max(sizes.get(k, 0), n)
    for jit, _ in calls:
      if jit.scratch is None: continue
      for k, n in sizes.items():
        if k not in jit.scratch or jit.scratch[k].max_numel() < n: jit.scratch[k] = UOp.new_buffer(k[0], n, dtypes.int8)
    # (allocated while the processes compile: the links don't)
    for b in dedup(u.buffer for jit, _ in calls for u in (jit.scratch or {}).values()):
      for x in (b.bufs if isinstance(b, MultiBuffer) else [b]): x.ensure_allocated()
    arenas_made[0] = True

  oks, left, took = [False] * len(children), set(range(len(children))), cast(dict[int, tuple[float, float, float]], {})
  def came(k:int) -> bool: return len(msgs[k]) >= 2 or (len(msgs[k]) == 1 and msgs[k][0] == b"")
  @disable_gc()
  def take(k:int):
    # a jit that came: its uops made, linked
    left.discard(k)
    if (status:=os.waitpid(children[k][0], 0)[1]) != 0 or len(msgs[k]) < 2 or not msgs[k][1]:
      if DEBUG >= 1: print(f"capture_parallel: jit {todo[k]} not captured (exit status {status:#x}), captured on its call")
      return
    if calls[k][0].scratch is not None: make_arenas()
    jit, t0 = calls[k][0], time.perf_counter()
    arenas, recs, rest = pickle.loads(msgs[k][1])
    arena_at = {i: (jit.scratch or {})[key] for key, (i, _) in arenas.items()}
    out:list[UOp] = []
    recs_srcs:list[tuple[int, ...]] = []
    for i, (op, src, arg) in enumerate(_Unpickler(recs, reg, out).load()):
      out.append(arena_at[i] if i in arena_at else UOp(op, tuple(out[j] for j in src), arg))
      recs_srcs.append(src)
    ret, lin, names, info = _Unpickler(rest, reg, out).load()
    jit.captured, jit.cnt = CapturedJit(ret, out[lin], names, info), 2
    # its buffers (_jit_bufs: the BUFFERs a toposort of the linear has, not in call bodies) from the records: no walk of the graph on its first call
    reach = bytearray(len(out))
    reach[lin] = 1
    for i in range(lin, -1, -1):
      if reach[i]:
        for j in recs_srcs[i][1 if out[i].op is Ops.CALL else 0:]: reach[j] = 1
    jit.captured.__dict__["_jit_bufs"] = {out[i] for i in range(lin + 1) if reach[i] and out[i].op is Ops.BUFFER}
    t1 = time.perf_counter()
    # (a jit whose memory isn't made yet is linked on its first call)
    if not getattr(jit, "link_later", False): jit.captured.linked()
    oks[k] = True
    took[k] = (t0 - st, t1 - st, time.perf_counter() - st)
    if taken is not None: taken(jit)
    if DEBUG >= 1: print(f"capture_parallel: jit {k} came at {', '.join(f'{t:.2f}' for t in arrived[k])}s, taken at {time.perf_counter()-st:.2f}s "
                         f"({len(out)} uops made in {t1-t0:.2f}s, linked in {time.perf_counter()-t1:.2f}s)")
  def take_jit(jit:_TinyJit, wait:bool=True) -> bool|None:
    # (meanwhile) a jit taken when it comes (waited for, not if wait is False: None if it didn't come yet): it runs now
    if (k:=next((k for k, (j, _) in enumerate(calls) if j is jit), None)) is None: return False # (not one of them)
    if k in left:
      if not wait and not came(k): return None
      with got: got.wait_for(lambda: came(k))
      take(k)
    return oks[k]

  # (meanwhile, a jit called before it's taken waits for it: it isn't captured again)
  for jit, _ in calls: jit.__dict__["_pending_take"] = functools.partial(take_jit, jit)
  try:
    if meanwhile is not None: meanwhile(take_jit)
  finally:
    for jit, _ in calls: jit.__dict__.pop("_pending_take", None)
  if DEBUG >= 1: print(f"capture_parallel: the work meanwhile done at {time.perf_counter()-st:.2f}s")
  # every jit left as it comes: taken and linked
  while left:
    with got: got.wait_for(lambda: any(came(k) for k in left))
    # (the first to have come: one done long ago isn't left behind ones that just came)
    take(min((k for k in left if came(k)), key=lambda k: arrived[k][-1] if arrived[k] else 0))
  if (path:=getenv("STARTUP_STATS", "")):
    import json
    with open(path, "a") as f:
      f.write(json.dumps({"st": st, "end": time.perf_counter() - st, "wall0": time.time() - (time.perf_counter() - st), "todo": todo,
                          "names": [getattr(j.fxn, "__qualname__", str(j.fxn)) for j, _ in calls], "arrived": arrived, "taken": took,
                          "compiles": compiles.stats if compiles is not None else []}) + "\n")
  return [k in todo and oks[todo.index(k)] for k in range(len(every))]
