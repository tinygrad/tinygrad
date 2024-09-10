from __future__ import annotations
from typing import Any, DefaultDict, List, Optional, Set, Union, Tuple, Dict, Callable, Sequence
import sys, time, functools, itertools
from collections import defaultdict
from dataclasses import dataclass, field
from tinygrad.dtype import ConstType, DType
from tinygrad.helpers import pretty_print, getenv, all_same
from tinygrad.shape.symbolic import Variable
from tinygrad.ops import UOp, UOps

# ***** pattern matcher *****

def get_location() -> Tuple[str, int]:
  frm = sys._getframe(1)
  # no matchers in ops.py, find the real frame
  while (frm.f_code.co_filename.split('/')[-1] in {"rewrite.py", "ops.py", '<string>'}) and frm.f_back is not None: frm = frm.f_back
  return frm.f_code.co_filename, frm.f_lineno
@functools.lru_cache(None)
def lines(fn) -> List[str]: return open(fn).readlines()

@dataclass(frozen=True, repr=False)  # reuse repr from UOp
class NOp(UOp):
  name: Optional[str] = None
  src: Tuple[NOp, ...] = tuple()
  allow_any_len: bool = False
  location: Tuple[str, int] = field(default_factory=get_location)

  @staticmethod
  @functools.lru_cache(None)
  def var(name:Optional[str]=None, dtype:Optional[DType]=None): return NOp(UOps.NOOP, dtype=dtype, name=name)
  @staticmethod
  @functools.lru_cache(None)
  def cvar(name:Optional[str]=None, dtype:Optional[DType]=None): return NOp(UOps.CONST, dtype=dtype, name=name)

  # this is needed so NOp has a different cache
  @classmethod
  @functools.lru_cache(None)
  def const(cls, dtype:Optional[DType], b:ConstType|Variable): return cls._const(dtype, b)

  @functools.cached_property
  def upat(self:NOp) -> UPat:
    return UPat(name=self.name, dtype=self.dtype, location=self.location) if self.op is UOps.NOOP else \
      UPat(self.op, self.arg, (list if self.commutative() else tuple)([src.upat for src in self.src]) or None, self.name,
           self.dtype, self.allow_any_len, location=self.location)

class UPat:
  def __init__(self, op:Optional[Union[UOps, Set[UOps]]]=None, arg:Any=None, src:Optional[Union[Tuple[UPat, ...], List[UPat], UPat]]=None,
               name:Optional[str]=None, dtype:Optional[Union[DType, Set[DType]]]=None, allow_any_len:bool=False, location=None,
               custom_early_reject:Optional[Set[Tuple[UOps, Any]]]=None):
    self.op: Optional[Tuple[UOps, ...]] = None if op is None else (tuple(op) if isinstance(op, set) else (op,))
    self.dtype: Optional[Tuple[DType, ...]] = None if dtype is None else (tuple(dtype) if isinstance(dtype, set) else (dtype,))
    self.arg, self.name = arg, name
    self.in_src = src
    self.src: Any = None

    # try all permutations if it's a list
    if isinstance(src, list): self.src = list(itertools.permutations(src)) if not all_same(src) else [src]
    # only one if it's a tuple
    elif isinstance(src, tuple): self.src = [src]
    # repeat if it's a UPat
    elif isinstance(src, UPat): self.src = [itertools.repeat(src)]

    self.allowed_len: int = 0 if allow_any_len or isinstance(src, UPat) or src is None else len(src)
    self.location = location or get_location()

    if custom_early_reject is not None: self.early_reject = custom_early_reject
    else:
      upat_match = [self.in_src] if isinstance(self.in_src, UPat) else ([] if self.in_src is None else self.src[0])
      self.early_reject = set((pp.op[0], pp.arg) for pp in upat_match if pp.op is not None and len(pp.op) == 1)

  def printable(self:UPat) -> str:
    try:
      return lines(self.location[0])[self.location[1]-1].strip()
    except FileNotFoundError:
      return "<missing>"
  def __repr__(self):
    def rep(x):
      form = "UPat(%s, %s, name=%s, dtype=%s, allow_any_len=%s, src=%s)"
      return form % (None if x.op is None else ('(%s)'%', '.join(map(str, x.op))), x.arg, repr(x.name),
        set(x.dtype) if x.dtype else None, x.allowed_len == 0, "[%s]" if x.src and len(x.src)>1 else "(%s)")
    return pretty_print(self, rep, srcfn=lambda x:None if x.src is None else [next(x.src[0])] if isinstance(x.src[0], itertools.repeat) else x.src[0])

def _match(uop:UOp, pat:UPat, store:Dict[str, UOp]) -> List[Dict[str, UOp]]:
  if (pat.name is not None and store.setdefault(pat.name, uop) is not uop) or \
     (pat.dtype is not None and uop.dtype not in pat.dtype) or \
     (pat.arg is not None and pat.arg != uop.arg) or \
     (pat.op is not None and uop.op not in pat.op) or \
     (pat.allowed_len != 0 and len(uop.src) != pat.allowed_len): return []
  if pat.src is None: return [store]
  res: List[Dict[str, UOp]] = []
  for vp in pat.src:
    new_stores = [store.copy()]
    for uu, vv in zip(uop.src, vp): new_stores = [rstore for nstore in new_stores for rstore in _match(uu, vv, nstore)]
    res.extend(new_stores)
  return res

class PatternMatcher:
  def __init__(self, patterns:Sequence[Tuple[Union[UPat, NOp], Callable]]):
    self.patterns = [(p.upat if isinstance(p, NOp) else p, fxn) for p,fxn in patterns]
    self.pdict: DefaultDict[Tuple[UOps, Any], List[Tuple[UPat, Callable, Set]]] = defaultdict(list)
    # uop is required, arg is optional
    for p,fxn in self.patterns:
      assert p.op is not None
      for uop in p.op: self.pdict[(uop, p.arg)].append((p, fxn, p.early_reject))

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def __add__(self, more:PatternMatcher): return PatternMatcher(self.patterns+more.patterns)

  def rewrite(self, uop:UOp) -> Optional[UOp]:
    ler = set([(u.op, u.arg) for u in uop.src] + [(u.op, None) for u in uop.src])
    for p,fxn,early_reject in itertools.chain(self.pdict[(uop.op, uop.arg)], self.pdict[(uop.op, None)]):
      if not early_reject.issubset(ler): continue
      if (matches := _match(uop, p, {})) and (ret:=fxn(**matches[0])) is not None: return ret # NOTE: if it returns None, we keep trying to match
    return None

# *** tracking pattern matcher ***

TRACK_MATCH_STATS = getenv("TRACK_MATCH_STATS", 0)
match_stats:Dict[UPat, List[Union[int, float]]] = dict()
class TrackedPattenMatcher(PatternMatcher):
  def __init__(self, patterns:List[Tuple[Union[UPat, NOp], Callable]]):
    super().__init__(patterns)
    for p,_ in self.patterns:
      if p not in match_stats: match_stats[p] = [0,0,0.0,0.0]

  def rewrite(self, uop:UOp) -> Optional[UOp]:
    ret = None
    ler = set([(u.op, u.arg) for u in uop.src] + [(u.op, None) for u in uop.src])
    for p,fxn,early_reject in itertools.chain(self.pdict[(uop.op, uop.arg)], self.pdict[(uop.op, None)]):
      st = time.perf_counter()
      if not early_reject.issubset(ler):
        match_stats[p][2] += time.perf_counter()-st
        continue
      match_stats[p][1] += 1
      if (matches := _match(uop, p, {})) and (ret:=fxn(**matches[0])) is not None:
        match_stats[p][0] += 1
        match_stats[p][2] += (et:=time.perf_counter()-st)
        match_stats[p][3] += et
        if TRACK_MATCH_STATS >= 2: print(f"{et*1e6:7.2f} us -- ", p.printable())
        return ret # NOTE: if it returns None, we keep trying to match
      match_stats[p][2] += time.perf_counter()-st
    return None

if TRACK_MATCH_STATS:
  PatternMatcher = TrackedPattenMatcher  # type: ignore
  import atexit
  @atexit.register
  def print_match_stats():
    ret = [0,0,0.0,0.0]
    for k,v in sorted(list(match_stats.items()), key=lambda x: x[1][2]):
      loc_str = f"{k.location[0].split('/')[-1]}:{k.location[1]}"
      print(f"{v[0]:6d} / {v[1]:7d} -- {v[3]*1000.:9.2f} / {v[2]*1000.:9.2f} ms -- {loc_str:15s}", k.printable())
      ret = [x+y for x,y in zip(ret, v)]
    print(f"{ret[0]:6d} / {ret[1]:7d} -- {ret[3]*1000.:9.2f} / {ret[2]*1000.:9.2f} ms -- TOTAL")

# *** simple graph rewrite engine ***

def graph_rewrite(sink:UOp, pm:PatternMatcher) -> UOp:
  nodes: Dict[Tuple, UOp] = {}
  replace: Dict[UOp, UOp] = {}
  def __inner_rewrite(n:UOp) -> UOp:
    if rn := replace.get(n): return rn
    replace_source = (n.op, n.dtype, new_src:=tuple(__inner_rewrite(y) for y in n.src), n.arg)
    if found := nodes.get(replace_source): replace[n] = found
    else:
      x = UOp(*replace_source) if new_src != n.src else n
      nodes[replace_source] = replace[n] = found = __inner_rewrite(new_x) if (new_x := pm.rewrite(x)) else x
    return found
  return __inner_rewrite(sink)
