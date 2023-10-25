from typing import Callable, List, Tuple, Any, Dict, cast, Union, Optional, Set
from weakref import ref
from collections import defaultdict
import functools, itertools
from tinygrad.helpers import DEBUG, DType, merge_dicts, ImageDType, getenv
from tinygrad.ops import RawBuffer, Device, BasicBatchExecutor, ASTRunner
from tinygrad.tensor import Tensor
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable

JIT_SUPPORTED_DEVICE = ["GPU", "CLANG", "METAL", "CUDA", "HIP", "WEBGPU", "LLVM"]

class TinyJit:
  def __init__(self, fxn:Callable):
    self.fxn: Callable = fxn
    self.cnt: int = 0
    self.jit_cache: List[Tuple[Any, List[Optional[RawBuffer]], Dict[Variable, int]]] = []
    self.ret: Any = None
    self.input_replace: Dict[Tuple[int, int], Tuple[Union[int, str], ShapeTracker, DType]]= {}   # (kernel_number, buffer_number) -> (input_name, expected_shapetracker, expected_type)
    self.batch_executor: Any = None
    self.updatable_entries: Dict[int, List[int]] = defaultdict(list) # (kernel_number) -> list(argument id). These are buffers from input + variables.

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> Any:
    if Device.DEFAULT.split(":")[0] not in JIT_SUPPORTED_DEVICE: return self.fxn(*args, **kwargs)  # only jit on supported device
    # NOTE: this cast is needed since although we know realize will create a ".realized" RawBuffer, the type checker doesn't
    input_rawbuffers: Dict[Union[int, str], Tuple[RawBuffer, ShapeTracker]] = {cast(Union[int, str], k):(cast(RawBuffer, v.realize().lazydata.realized), v.lazydata.st) for k,v in itertools.chain(enumerate(args), kwargs.items()) if v.__class__ is Tensor}
    assert len(input_rawbuffers) != 0, "no inputs to JIT"
    assert len(set(input_rawbuffers.values())) == len(input_rawbuffers), "duplicate inputs to JIT"
    if self.cnt >= 2:
      try: var_vals: Dict[Variable, int] = kwargs["jit_ctx"]
      except KeyError: var_vals = merge_dicts([arg.lazydata.st.var_vals for arg in args if arg.__class__ is Tensor])
      if len(var_vals) > 1: var_vals = dict(sorted(var_vals.items(), key=lambda kv: kv[0].key))
      for (j,i),(input_name, expected_st, expected_type) in self.input_replace.items():
        assert input_rawbuffers[input_name][0].dtype == expected_type, f"type mismatch in JIT, {input_rawbuffers[input_name][0].dtype} != {expected_type}"
        # NOTE: if we pass jit_ctx instead of using reshape to update the var_vals, we cannot compare the shapetracker directly
        if "jit_ctx" not in kwargs: assert input_rawbuffers[input_name][1].unbind() == expected_st, f"ShapeTracker mismatch in JIT, {input_rawbuffers[input_name][1].unbind()} != {expected_st}"
        self.jit_cache[j][1][i] = input_rawbuffers[input_name][0]
      for j in self.updatable_entries.keys():
        for k in self.jit_cache[j][2].keys():
          try: self.jit_cache[j][2][k] = var_vals[k]
          except KeyError: pass
      self.batch_executor.exec(self.jit_cache, self.updatable_entries)
      for (j,i) in self.input_replace.keys(): self.jit_cache[j][1][i] = None
    elif self.cnt == 1:
      CacheCollector.start()
      self.ret = self.fxn(*args, **kwargs)
      self.jit_cache = CacheCollector.finish()
      assert len(self.jit_cache) != 0, "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")

      # get the inputs for replacement
      for j_,cache in enumerate(self.jit_cache): # type: Tuple[int, Tuple[Callable, List[Optional[RawBuffer]], Dict[Variable, int]]]
        for i,a in enumerate(cache[1]):
          if a in [v[0] for v in input_rawbuffers.values()]:
            self.input_replace[(j_,i)] = [(k, v[1].unbind(), v[0].dtype) for k,v in input_rawbuffers.items() if v[0] == a][0]
            self.updatable_entries[j_].append(i)
        for i in range(len(cache[2])): self.updatable_entries[j_].append(len(cache[1])+i)
        #if prg.local_size is None: prg.local_size = prg.optimize_local_size(args, preserve_output=True)  # the JIT can optimize local
      assert set([x[0] for x in self.input_replace.values()]) == set(input_rawbuffers.keys()), "some input tensors not found"
      self.batch_executor = self.jit_cache[0][0].batch_exec(self.jit_cache) if hasattr(self.jit_cache[0][0], 'batch_exec') else BasicBatchExecutor(self.jit_cache)
      for (j,i) in self.input_replace.keys(): self.jit_cache[j][1][i] = None
    elif self.cnt == 0:
      self.ret = self.fxn(*args, **kwargs)
    self.cnt += 1
    return self.ret

class _CacheCollector:
  class _Placeholder:
    def __init__(self, buf): self.size, self.dtype, self._device, self.ref, self.buftype = buf.size, buf.dtype, getattr(buf, '_device', None), ref(buf), type(buf)
    def alloc_rawbuf(self): return self.buftype(self.size, self.dtype, **({'device':self._device} if self._device is not None else dict()))

  def __init__(self):
    self.cache: Optional[List[Tuple[Callable, List[Any], Dict[Any,Any]]]] = None
    self.placeholders: Dict[ref[RawBuffer], _CacheCollector._Placeholder] = {} # Output rawbufs are replaced with placeholders to allow freeing of the real buffer while collecting cache.
    self.circular_signatures: Set[Any] = set()
  def start(self): self.cache, self.placeholders, self.circular_signatures = [], {}, set()
  def add(self, prg, rawbufs, var_vals):
    if self.cache is None: return
    # Substitute output buffers with placeholders to find the most optimal reusage.
    if ref(rawbufs[0]) not in self.placeholders: self.placeholders[ref(rawbufs[0])] = _CacheCollector._Placeholder(rawbufs[0])
    cached_rawbufs = [self.placeholders.get(ref(buf), buf) if isinstance(prg, ASTRunner) and isinstance(buf, RawBuffer) and ref(buf) not in self.circular_signatures else buf for buf in rawbufs]
    self.cache.append((prg, cached_rawbufs, var_vals))
  def finish(self):
    if self.cache is None: return []

    rawbuf_pool: List[Tuple[RawBuffer, List[Tuple[int, int]]]] = []
    buf_usage_bounds: Dict[_CacheCollector._Placeholder, Tuple[int, int]] = {}
    buf_map: Dict[_CacheCollector._Placeholder, RawBuffer] = {}

    for j,(_,bufs,_) in enumerate(self.cache):
      for buf in bufs:
        if buf.__class__ is not _CacheCollector._Placeholder: continue
        if buf.ref() is not None: buf_map[buf] = buf.ref() # rawbufs that are referenced are not replaced but are used as is.
        else: buf_usage_bounds[buf] = buf_usage_bounds.get(buf, (j, j))[0], j

    # The query list contains a query for every placeholder that should be replaced with the actual rawbuffer. Queries are served from the largest to the smallest.
    # For each query, find any rawbuffer that is free within the query timeframe or allocate a new one.
    query_list = sorted([(buf.size*buf.dtype.itemsize, buf_usage_bounds[buf][0], buf_usage_bounds[buf][1], buf) for buf in buf_usage_bounds.keys()], key=lambda x: x[0], reverse=True)
    for _, start, end, buf in query_list:
      pool_idx = next((i for i,(with_buf, usages) in enumerate(rawbuf_pool) if self._can_substitute(buf, with_buf) and self._no_intersect(start,end,usages)), -1)
      if pool_idx == -1: rawbuf_pool.append((buf.alloc_rawbuf(), []))
      buf_map[buf] = rawbuf_pool[pool_idx][0]
      rawbuf_pool[pool_idx][1].append((start, end))

    cache_result = [(p, [buf_map.get(buf, buf) for buf in cached_bufs], var_vals) for p, cached_bufs, var_vals in self.cache]
    self.cache = None
    return cache_result
  def _no_intersect(self, start:int, end:int, usages:List[Tuple[int, int]]): return all(en < start or end < st for st, en in usages)
  def _can_substitute(self, buf, with_buf):
    if getenv("NO_BUFFER_REUSE"): return False
    return buf._device==with_buf._device and (buf.size*buf.dtype.itemsize<=with_buf.size*with_buf.dtype.itemsize if not isinstance(buf.dtype, ImageDType) and not isinstance(with_buf.dtype, ImageDType) else buf.size==with_buf.size and buf.dtype==with_buf.dtype and buf.dtype.shape==with_buf.dtype.shape)
  def _mark_output_buffer(self, output_buffer): self.circular_signatures.add(ref(output_buffer))
CacheCollector = _CacheCollector()
