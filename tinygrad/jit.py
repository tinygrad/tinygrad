from typing import Callable, List, Tuple, Any, Dict, cast, Union, Optional, Set
from weakref import ref
from collections import defaultdict
import functools, itertools
from tinygrad.helpers import DEBUG, DType, merge_dicts, ImageDType
from tinygrad.ops import RawBuffer, Device, BasicBatchExecutor
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
    if Device.DEFAULT not in JIT_SUPPORTED_DEVICE: return self.fxn(*args, **kwargs)  # only jit on supported device
    # NOTE: this cast is needed since although we know realize will create a ".realized" RawBuffer, the type checker doesn't
    input_rawbuffers: Dict[Union[int, str], Tuple[RawBuffer, ShapeTracker]] = {cast(Union[int, str], k):(cast(RawBuffer, v.realize().lazydata.realized), v.lazydata.st) for k,v in itertools.chain(enumerate(args), kwargs.items()) if v.__class__ is Tensor}
    assert len(input_rawbuffers) != 0, "no inputs to JIT"
    assert len(set(input_rawbuffers.values())) == len(input_rawbuffers), "duplicate inputs to JIT"
    if self.cnt >= 2:
      try: var_vals: Dict[Variable, int] = kwargs["jit_ctx"]
      except KeyError: var_vals = merge_dicts([arg.lazydata.var_vals for arg in args if arg.__class__ is Tensor])
      if len(var_vals) > 1: var_vals = dict(sorted(var_vals.items(), key=lambda kv: kv[0].key))
      for (j,i),(input_name, expected_st, expected_type) in self.input_replace.items():
        assert input_rawbuffers[input_name][0].dtype == expected_type, f"type mismatch in JIT, {input_rawbuffers[input_name][0].dtype} != {expected_type}"
        # NOTE: if we pass jit_ctx instead of using reshape to update the var_vals, we cannot compare the shapetracker directly
        if "jit_ctx" not in kwargs: assert input_rawbuffers[input_name][1].views == expected_st.views, f"ShapeTracker.views mismatch in JIT, {input_rawbuffers[input_name][1].views} != {expected_st.views}"
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
            self.input_replace[(j_,i)] = [(k, v[1], v[0].dtype) for k,v in input_rawbuffers.items() if v[0] == a][0]
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
    def alive(self): return self.ref() is not None
    def alloc_rawbuf(self): return self.buftype(self.size, self.dtype, **({'device':self._device} if self._device is not None else dict()))

  def __init__(self):
    self.cache: Optional[List[Tuple[Callable, List[Any], Dict[Any,Any]]]] = None
    self.placeholders: Dict[RawBuffer, _CacheCollector._Placeholder] = {} # Rawbuffers are replaced with placeholders to allow freeing of the real buffer while collecting cache.
    self.last_buftype: Dict[Tuple[int,...], int] = {} # Last index of the cached entry where a buffer with the shape (shape is a key) is used as input to the prog.
    self.last_placeholder_index: Dict[_CacheCollector._Placeholder, int] = {} # Last index where the placeholder is used as output. This allows tracking when we need to stick to the original buffer if it is still alive.
    self.freed_placeholders: Dict[Tuple[int,...], List[_CacheCollector._Placeholder]] = defaultdict(list)
    self.circular_signatures: Set[Any] = set()
  def start(self):
    self.cache, self.placeholders, self.last_buftype, self.last_placeholder_index, self.freed_buffers, self.circular_signatures = [], {}, {}, {}, defaultdict(list), set()
  def add(self, prg, rawbufs, var_vals):
    if self.cache is None: return
    cached_rawbufs = []
    if ref(rawbufs[0]) not in self.placeholders: self.placeholders[ref(rawbufs[0])] = _CacheCollector._Placeholder(rawbufs[0])
    for buf in rawbufs:
      if not isinstance(buf, RawBuffer) or ref(buf) in self.circular_signatures: cached_rawbufs.append(buf)
      else: cached_rawbufs.append(self.placeholders.get(ref(buf), buf))
    self.cache.append((prg, cached_rawbufs, var_vals))
  def finish(self):
    if self.cache is None: return []
    # print("FINISH")

    buf_pool = []
    buf_map = {}
    buf_first_use, buf_last_use, last_buftype = {}, {}, {}
    for j,(p,cached_bufs,var_vals) in enumerate(self.cache):
      for buf in cached_bufs:
        if isinstance(buf, RawBuffer): last_buftype[self._buftype_key(buf)] = j
    for j,(p,cached_bufs,var_vals) in enumerate(self.cache):
      for buf in cached_bufs:
        if buf.__class__ is not _CacheCollector._Placeholder: continue
        if buf.alive():
          buf_pool.append((buf.ref(), [(-1, last_buftype.get(self._buftype_key(buf), -1)), (j, len(self.cache)+1)]))
          buf_map[buf] = buf.ref()
        else:
          if buf not in buf_first_use: buf_first_use[buf] = j
          buf_last_use[buf] = j

    query_list = sorted([(buf.size*buf.dtype.itemsize, buf_first_use[buf], buf_last_use[buf], buf) for buf in buf_first_use.keys()], key=lambda x: x[0], reverse=True)
    for size,start,end,buf in query_list:
      buf_pool_i = -1
      for i,(with_buf,usages) in enumerate(buf_pool):
        if not self._can_replace(buf, with_buf): continue
        bad = False
        for st,en in usages:
          if end < st or en < start: pass
          else:
            bad = True
            break
        if not bad: buf_pool_i = i
      if buf_pool_i == -1:
        buf_pool.append((buf.alloc_rawbuf(), []))
        buf_pool_i = len(buf_pool) - 1
      # else:
        # print("Reuse", buf_pool_i)
      buf_map[buf] = buf_pool[buf_pool_i][0]
      buf_pool[buf_pool_i][1].append((start,end))

    # print(buf_pool)

    cache_result = []
    for j,(p,cached_bufs,var_vals) in enumerate(self.cache):
      cache_result.append((p, [buf_map.get(buf, buf) for buf in cached_bufs], var_vals))
    self.cache = None
    return cache_result

    # for j,(p,cached_bufs,var_vals) in enumerate(self.cache): pass
    placeholder_mapper, cache_result = {}, []
    for j,(p,cached_bufs,var_vals) in enumerate(self.cache):
      if cached_bufs[0].__class__ is _CacheCollector._Placeholder:
        if cached_bufs[0].alive():
          # Since the placeholder is alive (someone holds refed RawBuffer) to avoid hazards when this output buffer could be used as input on the other launch (e.g., LSTM),
          # we allocate a backing buffer and and use it until the penultimate entry (the last entry is 100% safe to use the original RawBuffer).
          if self.last_buftype.get(self._buftype_key(cached_bufs[0]), -1) < j or self.last_placeholder_index[cached_bufs[0]] == j:
            # Safe to use the original buffer when all inputs buffers of the same size and dtype are behind or this is the last usage of this buffer as output.
            placeholder_mapper[cached_bufs[0]] = cached_bufs[0].ref()
          elif cached_bufs[0] not in placeholder_mapper:
            placeholder_mapper[cached_bufs[0]] = cached_bufs[0].alloc_rawbuf() # Allocating a backing buffer.
        elif cached_bufs[0] not in placeholder_mapper:
          placeholder_mapper[cached_bufs[0]] = cached_bufs[0].alloc_rawbuf()
      cache_result.append((p, [placeholder_mapper.get(buf, buf) for buf in cached_bufs], var_vals))
    self.cache, self.placeholders, self.last_buftype, self.last_placeholder_index, self.freed_buffers, self.circular_signatures = None, {}, {}, {}, defaultdict(list), set()
    return cache_result

  def _can_replace(self, buf, with_buf): 
    return buf._device==with_buf._device and (buf.size*buf.dtype.itemsize<=with_buf.size*with_buf.dtype.itemsize if not isinstance(buf.dtype, ImageDType) and not isinstance(with_buf.dtype, ImageDType) else buf.size==with_buf.size and buf.dtype==with_buf.dtype)
  def _mark_output_buffer(self, output_buffer): self.circular_signatures.add(ref(output_buffer))
  def _on_buf_free(self, underlying_buf):
    if underlying_buf not in self.placeholders: return
    self.freed_placeholders[self._buftype_key(self.placeholders[underlying_buf])].append(self.placeholders[underlying_buf])
    self.placeholders.pop(underlying_buf)
  def _get_signature(self, buf): return buf._buf if getattr(buf, '_buf', None) is not None and getattr(buf, '_allocator', None) is not None else buf
  def _buftype_key(self, buf): return (buf.size, buf.dtype, buf._device, buf.dtype.shape if hasattr(buf.dtype, 'shape') else None)
CacheCollector = _CacheCollector()
