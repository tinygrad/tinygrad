from typing import Any, Optional, Dict, List, Callable, Tuple
from weakref import ref
from tinygrad.runtime.lib import RawBuffer

class _CacheCollector:
  class _Placeholder:
    def __init__(self, buf): self.size, self.dtype, self.device, self.ref, self.buftype = buf.size, buf.dtype, getattr(buf, '_device', None), ref(buf), type(buf)
    def alive(self): return self.ref() is not None
    def alloc_rawbuf(self): return self.buftype(self.size, self.dtype, **({'device':self.device} if self.device is not None else dict()))
    def get_rawbuf(self): # NOTE: If ref is valid, it means the actual buffer is still in use (not even reused from the cache). To keep it valid, we use this buffer directly.
      return self.ref() if self.alive() else self.alloc_rawbuf()

  def __init__(self):
    self.cache: Optional[List[Tuple[Callable, List[Any], Dict[Any,Any]]]] = None
    self.placeholders: Dict[RawBuffer, _CacheCollector._Placeholder] = {} # Rawbuffers are replaced with placeholders to allow freeing of the real buffer while collecting cache.
    self.last_buftype: Dict[Tuple[int,...], int] = {} # Last index of the cached entry where a buffer with the shape (shape is a key) is used as input to the prog.
    self.last_placeholder_index: Dict[_CacheCollector._Placeholder, int] = {} # Last index where the placeholder is used as output. This allows tracking when we need to stick to the original buffer if it is still alive.
  def start(self):
    self.cache, self.placeholders, self.last_buftype, self.last_placeholder_index = [], {}, {}, {}
  def add(self, prg, rawbufs, var_vals):
    if self.cache is None: return
    def get_signature(buf): return buf._buf if getattr(buf, '_buf', None) is not None and getattr(buf, '_allocator', None) is not None else buf # Buffers are considered to be mergable when their signatures match.

    # NOTE: Check if the input matches any of placeholder to determine if it's existing or newly created input. In case of newly created input remove placeholder and capture the whole buffer.
    [self.placeholders.pop(sig) for buf in rawbufs[1:] if isinstance(buf, RawBuffer) and (sig:=get_signature(buf)) in self.placeholders and self.placeholders[sig].ref != ref(buf)]
    self.last_buftype.update({self._buftype_key(buf): len(self.cache) for buf in rawbufs[1:] if isinstance(buf, RawBuffer) and get_signature(buf) not in self.placeholders})
    self.placeholders.setdefault(get_signature(rawbufs[0]), _CacheCollector._Placeholder(rawbufs[0])).ref = ref(rawbufs[0]) # Updating placeholder ref to point to reallocated buffer(or wrapper).
    self.last_placeholder_index[self.placeholders[get_signature(rawbufs[0])]] = len(self.cache)
    self.cache.append((prg,[self.placeholders.get(get_signature(x), x) for x in rawbufs],var_vals))
  def finish(self):
    if self.cache is None: return []
    placeholder_mapper, cache_result = {}, []
    for j,(p,cached_bufs,var_vals) in enumerate(self.cache):
      if cached_bufs[0].__class__ is _CacheCollector._Placeholder and cached_bufs[0].alive():
        # NOTE: To avoid hazards when output could be used as input (e.g., LSTM), we use a backing buffer to handle ambiguous situations where a usage of a single buffer may not be safe. Need to track this only for alive placeholders.
        if self.last_buftype.get(self._buftype_key(cached_bufs[0]), -1) < j or self.last_placeholder_index[cached_bufs[0]] == j: placeholder_mapper[cached_bufs[0]] = cached_bufs[0].get_rawbuf()
        elif cached_bufs[0] not in placeholder_mapper: placeholder_mapper[cached_bufs[0]] = cached_bufs[0].alloc_rawbuf() # Allocate a backing buffer and use it until the penultimate entry.
      placeholder_mapper.update({buf: buf.get_rawbuf() for buf in cached_bufs if buf.__class__ is _CacheCollector._Placeholder and buf not in placeholder_mapper})
      cache_result.append((p, [placeholder_mapper.get(buf, buf) for buf in cached_bufs], var_vals))
    self.cache, self.placeholders, self.last_buftype, self.last_placeholder_index = None, {}, {}, {}
    return cache_result
  def _buftype_key(self, buf): return (buf.size, buf.dtype, buf.dtype.shape if hasattr(buf.dtype, 'shape') else None)
CacheCollector = _CacheCollector()