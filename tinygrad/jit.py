from typing import Callable, List, Tuple, Any, Dict, cast, Union, Optional, Set
from weakref import ref
from collections import defaultdict
import functools, itertools
from tinygrad.helpers import DEBUG, DType, merge_dicts
from tinygrad.ops import Device
from tinygrad.tensor import Tensor
from tinygrad.ops import RawBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable

JIT_SUPPORTED_DEVICE = ["GPU", "CLANG", "METAL", "CUDA", "HIP", "WEBGPU", "LLVM"]

class TinyJit:
  def __init__(self, fxn:Callable):
    self.fxn: Callable = fxn
    self.cnt: int = 0
    self.jit_cache: List[Tuple[Callable, List[Optional[RawBuffer]], Dict[Variable, int]]] = []
    self.ret: Any = None
    self.input_replace: Dict[Tuple[int, int], Tuple[Union[int, str], ShapeTracker, DType]]= {}   # (kernel_number, buffer_number) -> (input_name, expected_shapetracker, expected_type)

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> Any:
    if Device.DEFAULT not in JIT_SUPPORTED_DEVICE: return self.fxn(*args, **kwargs)  # only jit on supported device
    # NOTE: this cast is needed since although we know realize will create a ".realized" RawBuffer, the type checker doesn't
    input_rawbuffers: Dict[Union[int, str], Tuple[RawBuffer, ShapeTracker]] = {cast(Union[int, str], k):(cast(RawBuffer, v.realize().lazydata.realized), v.lazydata.st) for k,v in itertools.chain(enumerate(args), kwargs.items()) if v.__class__ is Tensor}
    # assert len(input_rawbuffers) != 0, "no inputs to JIT"
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
      for prg, pargs, variables in self.jit_cache: # type: Callable, List[Optional[RawBuffer]], Dict[Variable, int]
        for k in variables.keys():
          try: variables[k] = var_vals[k]
          except KeyError: pass
        prg(pargs, variables, jit=True)
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
        #if prg.local_size is None: prg.local_size = prg.optimize_local_size(args, preserve_output=True)  # the JIT can optimize local
      assert set([x[0] for x in self.input_replace.values()]) == set(input_rawbuffers.keys()), "some input tensors not found"
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
    # When we got buffers with the same signature, we can use just 1(max 2, see cycle avoidance below) buffer insted of all of them.
    # Current implementation of a signature is an underlying buffer, because if 2 or more different RawBuffers shares the same, all but the very last are dead.
    for buf in rawbufs[1:]:
      # Check if the input matches any of placeholder to determine if it's existing or newly created input.
      # In case of newly created input remove placeholder and capture the whole buffer.
      if isinstance(buf, RawBuffer) and self._get_signature(buf) in self.placeholders and self.placeholders[self._get_signature(buf)].ref != ref(buf):
        self.placeholders.pop(self._get_signature(buf))
      if isinstance(buf, RawBuffer) and self._get_signature(buf) not in self.placeholders:
        self.last_buftype[self._buftype_key(buf)] = len(self.cache)

    # Creating/updating a placeholder for the current output buffer. If we update output, set the ref to point to the new RawBuffer,
    # since the previous RawBuffer is dead (overwise we won't get a new RawBuffer with the same signature). Do not care about dead buffers, they 100% could be replaced with any other buffer.
    if self._get_signature(rawbufs[0]) in self.placeholders: self.placeholders[self._get_signature(rawbufs[0])].ref = ref(rawbufs[0])
    else:
      # This is a new output buffer. Try to reuse any freed placeholders with the same type to "merge" these buffers.
      # If this output buffer is "output_buffer", reusage of the output buffer is scary.
      plh = self.freed_placeholders[self._buftype_key(rawbufs[0])].pop() if self._get_signature(rawbufs[0]) not in self.circular_signatures and self.freed_placeholders[self._buftype_key(rawbufs[0])] else _CacheCollector._Placeholder(rawbufs[0])
      self.placeholders.setdefault(self._get_signature(rawbufs[0]), plh).ref = ref(rawbufs[0])
    self.last_placeholder_index[self.placeholders[self._get_signature(rawbufs[0])]] = len(self.cache)
    self.cache.append((prg,[self.placeholders.get(self._get_signature(x), x) for x in rawbufs],var_vals))
  def finish(self):
    if self.cache is None: return []
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
  def _mark_output_buffer(self, output_buffer): self.circular_signatures.add(self._get_signature(output_buffer))
  def _on_buf_free(self, underlying_buf):
    if underlying_buf not in self.placeholders: return
    self.freed_placeholders[self._buftype_key(self.placeholders[underlying_buf])].append(self.placeholders[underlying_buf])
    self.placeholders.pop(underlying_buf)
  def _get_signature(self, buf): return buf._buf if getattr(buf, '_buf', None) is not None and getattr(buf, '_allocator', None) is not None else buf
  def _buftype_key(self, buf): return (buf.size, buf.dtype, buf._device, buf.dtype.shape if hasattr(buf.dtype, 'shape') else None)
CacheCollector = _CacheCollector()