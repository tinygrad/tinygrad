from __future__ import annotations
from collections import deque
import os, functools, platform, time, re, contextlib, operator, hashlib, pickle, sqlite3, cProfile, pstats, tempfile, pathlib, string, ctypes, sys
import itertools, urllib.request, subprocess, shutil, math
from typing import DefaultDict, Dict, Tuple, Union, List, ClassVar, Optional, Iterable, Any, TypeVar, TYPE_CHECKING, Callable, Sequence
if TYPE_CHECKING:  # TODO: remove this and import TypeGuard from typing once minimum python supported version is 3.10
  from typing_extensions import TypeGuard
  from tinygrad.shape.shapetracker import sint

T = TypeVar("T")
U = TypeVar("U")
# NOTE: it returns int 1 if x is empty regardless of the type of x
def prod(x:Iterable[T]) -> Union[T,int]: return functools.reduce(operator.mul, x, 1)

# NOTE: helpers is not allowed to import from anything else in tinygrad
OSX = platform.system() == "Darwin"
CI = os.getenv("CI", "") != ""

def dedup(x:Iterable[T]): return list(dict.fromkeys(x))   # retains list order
def argfix(*x):
  if x and x[0].__class__ in (tuple, list):
    if len(x) != 1: raise ValueError(f"bad arg {x}")
    return tuple(x[0])
  return x
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def all_same(items:List[T]): return all(x == items[0] for x in items)
def all_int(t: Sequence[Any]) -> TypeGuard[Tuple[int, ...]]: return all(isinstance(s, int) for s in t)
def colored(st, color:Optional[str], background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st  # replace the termcolor library with one line  # noqa: E501
def ansistrip(s:str): return re.sub('\x1b\\[(K|.*?m)', '', s)
def ansilen(s:str): return len(ansistrip(s))
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l:Iterable[Iterable[T]]): return [item for sublist in l for item in sublist]
def fully_flatten(l): return [item for sublist in l for item in (fully_flatten(sublist) if isinstance(sublist, (tuple, list)) else [sublist])]
def fromimport(mod, frm): return getattr(__import__(mod, fromlist=[frm]), frm)
def strip_parens(fst:str): return fst[1:-1] if fst[0] == '(' and fst[-1] == ')' and fst[1:-1].find('(') <= fst[1:-1].find(')') else fst
def round_up(num, amt:int): return (num+amt-1)//amt * amt
def merge_dicts(ds:Iterable[Dict[T,U]]) -> Dict[T,U]:
  assert len(kvs:=set([(k,v) for d in ds for k,v in d.items()])) == len(set(kv[0] for kv in kvs)), f"cannot merge, {kvs} contains different values for the same key"  # noqa: E501
  return {k:v for d in ds for k,v in d.items()}
def partition(lst:List[T], fxn:Callable[[T],bool]) -> Tuple[List[T], List[T]]:
  a:List[T] = []
  b:List[T] = []
  for s in lst: (a if fxn(s) else b).append(s)
  return a,b
def unwrap(x:Optional[T]) -> T:
  assert x is not None
  return x
def unwrap2(x:Tuple[T,Any]) -> T:
  ret, err = x
  assert err is None, str(err)
  return ret
def get_child(obj, key):
  for k in key.split('.'):
    if k.isnumeric(): obj = obj[int(k)]
    elif isinstance(obj, dict): obj = obj[k]
    else: obj = getattr(obj, k)
  return obj

def get_shape(x) -> Tuple[int, ...]:
  if not isinstance(x, (list, tuple)): return ()
  subs = [get_shape(xi) for xi in x]
  if not all_same([sub for sub in subs]): raise ValueError(f"inhomogeneous shape from {x}")
  return (len(subs),) + (subs[0] if subs else ())

# returns the axes to create new_shape if new_shape can be created by combining axis from old_shape
def get_contraction(old_shape:Tuple[sint, ...], new_shape:Tuple[sint, ...]) -> Optional[List[List[int]]]:
  acc_old, acc_new = list(itertools.accumulate(old_shape, operator.mul)), list(itertools.accumulate(new_shape, operator.mul))
  try: split = [acc_old.index(acc)+1 if acc != 1 else 0 for acc in acc_new]
  except ValueError: return None
  return [list(range(st,ed)) for st,ed in zip([0]+split[:-1], split[:-1]+[len(old_shape)])]

@functools.lru_cache(maxsize=None)
def to_function_name(s:str): return ''.join([c if c in (string.ascii_letters+string.digits+'_') else f'{ord(c):02X}' for c in ansistrip(s)])
@functools.lru_cache(maxsize=None)
def getenv(key:str, default=0): return type(default)(os.getenv(key, default))
def temp(x:str) -> str: return (pathlib.Path(tempfile.gettempdir()) / x).as_posix()

class GraphException(Exception): pass

V = TypeVar("V")
def bfs(graph:DefaultDict[T, List[T]], in_degree:DefaultDict[T, int], insert:Callable[[T, Dict[T, V]], None]) -> Dict[T, V]:
  queue = deque(n for n,d in in_degree.items() if d == 0)
  toposort: Dict[T, V] = {}
  while queue:
    v = queue.popleft()
    if v in toposort: continue
    insert(v, toposort)
    for u in graph[v]:
      in_degree[u] -= 1
      if in_degree[u] == 0: queue.append(u)
  return toposort

class Context(contextlib.ContextDecorator):
  stack: ClassVar[List[dict[str, int]]] = [{}]
  def __init__(self, **kwargs): self.kwargs = kwargs
  def __enter__(self):
    Context.stack[-1] = {k:o.value for k,o in ContextVar._cache.items()} # Store current state.
    for k,v in self.kwargs.items(): ContextVar._cache[k].value = v # Update to new temporary state.
    Context.stack.append(self.kwargs) # Store the temporary state so we know what to undo later.
  def __exit__(self, *args):
    for k in Context.stack.pop(): ContextVar._cache[k].value = Context.stack[-1].get(k, ContextVar._cache[k].value)

class ContextVar:
  _cache: ClassVar[Dict[str, ContextVar]] = {}
  value: int
  key: str
  def __new__(cls, key, default_value):
    if key in ContextVar._cache: return ContextVar._cache[key]
    instance = ContextVar._cache[key] = super().__new__(cls)
    instance.value, instance.key = getenv(key, default_value), key
    return instance
  def __bool__(self): return bool(self.value)
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  def __lt__(self, x): return self.value < x

DEBUG, IMAGE, BEAM, NOOPT, JIT = ContextVar("DEBUG", 0), ContextVar("IMAGE", 0), ContextVar("BEAM", 0), ContextVar("NOOPT", 0), ContextVar("JIT", 1)
WINO, THREEFRY, CACHECOLLECTING = ContextVar("WINO", 0), ContextVar("THREEFRY", 0), ContextVar("CACHECOLLECTING", 1)
GRAPH, GRAPHPATH, SAVE_SCHEDULE, RING = ContextVar("GRAPH", 0), getenv("GRAPHPATH", "/tmp/net"), ContextVar("SAVE_SCHEDULE", 0), ContextVar("RING", 1)
MULTIOUTPUT = ContextVar("MULTIOUTPUT", 1)

# **************** global state Counters ****************

class GlobalCounters:
  global_ops: ClassVar[int] = 0
  global_mem: ClassVar[int] = 0
  time_sum_s: ClassVar[float] = 0.0
  kernel_count: ClassVar[int] = 0
  mem_used: ClassVar[int] = 0   # NOTE: this is not reset
  @staticmethod
  def reset(): GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum_s, GlobalCounters.kernel_count = 0,0,0.0,0

# **************** timer and profiler ****************

class Timing(contextlib.ContextDecorator):
  def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
  def __enter__(self): self.st = time.perf_counter_ns()
  def __exit__(self, *exc):
    self.et = time.perf_counter_ns() - self.st
    if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))

def _format_fcn(fcn): return f"{fcn[0]}:{fcn[1]}:{fcn[2]}"
class Profiling(contextlib.ContextDecorator):
  def __init__(self, enabled=True, sort='cumtime', frac=0.2, fn=None, ts=1):
    self.enabled, self.sort, self.frac, self.fn, self.time_scale = enabled, sort, frac, fn, 1e3/ts
  def __enter__(self):
    self.pr = cProfile.Profile()
    if self.enabled: self.pr.enable()
  def __exit__(self, *exc):
    if self.enabled:
      self.pr.disable()
      if self.fn: self.pr.dump_stats(self.fn)
      stats = pstats.Stats(self.pr).strip_dirs().sort_stats(self.sort)
      for fcn in stats.fcn_list[0:int(len(stats.fcn_list)*self.frac)]:    # type: ignore[attr-defined]
        (_primitive_calls, num_calls, tottime, cumtime, callers) = stats.stats[fcn]    # type: ignore[attr-defined]
        scallers = sorted(callers.items(), key=lambda x: -x[1][2])
        print(f"n:{num_calls:8d}  tm:{tottime*self.time_scale:7.2f}ms  tot:{cumtime*self.time_scale:7.2f}ms",
              colored(_format_fcn(fcn), "yellow") + " "*(50-len(_format_fcn(fcn))),
              colored(f"<- {(scallers[0][1][2]/tottime)*100:3.0f}% {_format_fcn(scallers[0][0])}", "BLACK") if len(scallers) else '')

# *** universal database cache ***

_cache_dir: str = getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache"))
CACHEDB: str = getenv("CACHEDB", os.path.abspath(os.path.join(_cache_dir, "tinygrad", "cache.db")))
CACHELEVEL = getenv("CACHELEVEL", 2)

VERSION = 16
_db_connection = None
def db_connection():
  global _db_connection
  if _db_connection is None:
    os.makedirs(CACHEDB.rsplit(os.sep, 1)[0], exist_ok=True)
    _db_connection = sqlite3.connect(CACHEDB)
    if DEBUG >= 7: _db_connection.set_trace_callback(print)
  return _db_connection

def diskcache_clear():
  cur = db_connection().cursor()
  drop_tables = cur.execute("SELECT 'DROP TABLE IF EXISTS ' || quote(name) || ';' FROM sqlite_master WHERE type = 'table';").fetchall()
  cur.executescript("\n".join([s[0] for s in drop_tables]))

def diskcache_get(table:str, key:Union[Dict, str, int]) -> Any:
  if CACHELEVEL == 0: return None
  if isinstance(key, (str,int)): key = {"key": key}
  conn = db_connection()
  cur = conn.cursor()
  try:
    res = cur.execute(f"SELECT val FROM '{table}_{VERSION}' WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}", tuple(key.values()))
  except sqlite3.OperationalError:
    return None  # table doesn't exist
  if (val:=res.fetchone()) is not None: return pickle.loads(val[0])
  return None

_db_tables = set()
def diskcache_put(table:str, key:Union[Dict, str, int], val:Any):
  if CACHELEVEL == 0: return val
  if isinstance(key, (str,int)): key = {"key": key}
  conn = db_connection()
  cur = conn.cursor()
  if table not in _db_tables:
    TYPES = {str: "text", bool: "integer", int: "integer", float: "numeric", bytes: "blob"}
    ltypes = ', '.join(f"{k} {TYPES[type(key[k])]}" for k in key.keys())
    cur.execute(f"CREATE TABLE IF NOT EXISTS '{table}_{VERSION}' ({ltypes}, val blob, PRIMARY KEY ({', '.join(key.keys())}))")
    _db_tables.add(table)
  cur.execute(f"REPLACE INTO '{table}_{VERSION}' ({', '.join(key.keys())}, val) VALUES ({', '.join(['?']*len(key.keys()))}, ?)", tuple(key.values()) + (pickle.dumps(val), ))  # noqa: E501
  conn.commit()
  cur.close()
  return val

def diskcache(func):
  def wrapper(*args, **kwargs) -> bytes:
    table, key = f"cache_{func.__name__}", hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest()
    if (ret:=diskcache_get(table, key)): return ret
    return diskcache_put(table, key, func(*args, **kwargs))
  return wrapper

# *** http support ***

def fetch(url:str, name:Optional[Union[pathlib.Path, str]]=None, subdir:Optional[str]=None,
          allow_caching=not getenv("DISABLE_HTTP_CACHE")) -> pathlib.Path:
  if url.startswith(("/", ".")): return pathlib.Path(url)
  if name is not None and (isinstance(name, pathlib.Path) or '/' in name): fp = pathlib.Path(name)
  else: fp = pathlib.Path(_cache_dir) / "tinygrad" / "downloads" / (subdir or "") / (name if name else hashlib.md5(url.encode('utf-8')).hexdigest())
  if not fp.is_file() or not allow_caching:
    with urllib.request.urlopen(url, timeout=10) as r:
      assert r.status == 200
      total_length = int(r.headers.get('content-length', 0))
      progress_bar = tinytqdm(total=total_length, unit='B', unit_scale=True, desc=f"{url}: ")
      (path := fp.parent).mkdir(parents=True, exist_ok=True)
      with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
        while chunk := r.read(16384): progress_bar.update(f.write(chunk))
        f.close()
        progress_bar.update(close=True)
        if (file_size:=os.stat(f.name).st_size) < total_length: raise RuntimeError(f"fetch size incomplete, {file_size} < {total_length}")
        pathlib.Path(f.name).rename(fp)
  return fp

# *** Exec helpers

def cpu_time_execution(cb, enable):
  if enable: st = time.perf_counter()
  cb()
  if enable: return time.perf_counter()-st

def cpu_objdump(lib):
  with tempfile.NamedTemporaryFile(delete=True) as f:
    pathlib.Path(f.name).write_bytes(lib)
    print(subprocess.check_output(['objdump', '-d', f.name]).decode('utf-8'))

# *** ctypes helpers

# TODO: make this work with read only memoryviews (if possible)
def from_mv(mv:memoryview, to_type=ctypes.c_char):
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents
def to_mv(ptr, sz) -> memoryview: return memoryview(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * sz)).contents).cast("B")
def mv_address(mv:memoryview): return ctypes.addressof(ctypes.c_char.from_buffer(mv))
def to_char_p_p(options: List[bytes], to_type=ctypes.c_char): return (ctypes.POINTER(to_type) * len(options))(*[ctypes.cast(ctypes.create_string_buffer(o), ctypes.POINTER(to_type)) for o in options])  # noqa: E501
@functools.lru_cache(maxsize=None)
def init_c_struct_t(fields: Tuple[Tuple[str, ctypes._SimpleCData], ...]):
  class CStruct(ctypes.Structure):
    _pack_, _fields_ = 1, fields
  return CStruct
def init_c_var(ctypes_var, creat_cb): return (creat_cb(ctypes_var), ctypes_var)[1]
def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes,))

class tinytqdm:
  def __init__(self, iterable=None, desc:str='', disable:bool=False, unit:str='it', unit_scale=False, total:int=-1, rate:int=100):
    self.iter, self.desc, self.dis, self.unit, self.unit_scale, self.rate = iterable, desc, disable, unit, unit_scale, rate
    self.st, self.i, self.n, self.skip, self.t = time.perf_counter(), -1, 0, 1, len(iterable) if total==-1 else total
    self.update(0)
  def __iter__(self):
    try:
      for item in self.iter:
        yield item
        self.update(1)
    finally: self.update(close=True)
  def update(self, n:int=0, close:bool=False):
    self.n, self.i = self.n+n, self.i+1
    if (self.i % self.skip != 0 and not close) or self.dis: return
    prog, dur, term = self.n/self.t if self.t else -1, time.perf_counter()-self.st, shutil.get_terminal_size().columns
    if self.i/dur > self.rate and self.i: self.skip = max(int(self.i/dur)//self.rate,1) if self.i else 1
    def fmt(t): return ':'.join([f'{x:02d}' for x in divmod(int(t), 60)]) if t!=-1 else '?'
    def scl(x): return x/1000**int(math.log(x,1000))
    def fn(x): return (f"{scl(x):.{3-math.ceil(math.log10(scl(x)))}f}"[:4]+(f"{[' ','k','M','G','T','P'][int(math.log(x,1000))]}") if x else '0.00')
    if self.t: unit_text = f"{fn(self.n)}/{fn(self.t)}" if self.unit_scale else f"{self.n}/{self.t}"
    else: unit_text = f"{fn(self.n)}{self.unit}" if self.unit_scale else f"{self.n}{self.unit}"
    it_text = f"{fn(self.n/dur)}" if self.n and self.unit_scale else f"{self.n/dur:5.2f}" if self.n else "?"
    if self.t: suf = f'| {unit_text} [{fmt(dur)}<{fmt(dur/self.n*self.t-dur if self.n else -1)}, {it_text}{self.unit}/s]'
    else: suf = f'{unit_text} [{fmt(dur)}, {it_text}{self.unit}/s]'
    sz = max(term-5-len(suf)-len(self.desc), 1)
    bar = f'\r{self.desc}{round(100*prog):3}%|{"█"*round(sz*prog)}{" "*(sz-round(sz*prog))}{suf}' if self.t else f'\r{self.desc}{suf}{" "*term}'
    print(bar[:term+1],flush=True,end='\n'*close,file=sys.stderr)
