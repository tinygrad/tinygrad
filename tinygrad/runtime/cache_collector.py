from typing import Any, Optional, Dict, List, Callable, Tuple
from weakref import ref
from tinygrad.runtime.lib import RawBuffer, is_buffer_wrapped

class _CacheCollector:
  class _Placeholder:
    def __init__(self, buf):
      self.alloc_clb = buf._allocator if is_buffer_wrapped(buf) else type(buf) # Get allocator if this is a AllocatedBufferWrapper or the buftype.
      self.size, self.dtype, self.ref = buf.size, buf.dtype, ref(buf)
    def alive(self): return self.ref() is not None
    def alloc_rawbuf(self): return self.alloc_clb(self.size, self.dtype)
    def get_rawbuf(self): # NOTE: If ref is valid, it means the actual buffer is still in use (not even reused from the cache). To keep it valid, we use this buffer directly.
      return self.ref() if self.alive() else self.alloc_rawbuf()

  def __init__(self):
    self.cache: Optional[List[Tuple[Callable, Any]]] = None
    self.placeholders: Dict[RawBuffer, _CacheCollector._Placeholder] = {} # Rawbuffers are replaced with placeholders to allow freeing of the real buffer while collecting cache.
    self.last_buftype: Dict[Any, int] = {} # Last index of the cached entry where a buffer with the shape (shape is a key) is used as input to the prog.
    self.last_placeholder_index: Dict[_CacheCollector._Placeholder, int] = {} # Last index where the placeholder is used as output. This allows tracking when we need to stick to the original buffer if it is still alive.
  def start(self):
    self.cache, self.placeholders, self.last_buftype, self.last_placeholder_index = [], {}, {}, {}
  def add(self, prg, rawbufs):
    if self.cache is None: return
    def get_ref(buf): return ref(buf._wrapped_buffer) if is_buffer_wrapped(buf) else ref(buf)
    self.last_buftype.update({self._buftype_key(buf): len(self.cache) for buf in rawbufs[1:] if get_ref(buf) not in self.placeholders})
    self.placeholders.setdefault(get_ref(rawbufs[0]), _CacheCollector._Placeholder(rawbufs[0])).ref = ref(rawbufs[0]) # Updating placeholder ref to point to reallocated buffer(or wrapper).
    self.last_placeholder_index[self.placeholders[get_ref(rawbufs[0])]] = len(self.cache)
    self.cache.append((prg,[(self.placeholders[get_ref(x)] if get_ref(x) in self.placeholders else x) for x in rawbufs]))
  def finish(self):
    if self.cache is None: return []
    placeholder_mapper, cache_result = {}, []
    for j,(p,cached_bufs) in enumerate(self.cache):
      if cached_bufs[0].__class__ is _CacheCollector._Placeholder and cached_bufs[0].alive():
        # NOTE: To avoid hazards when output could be used as input (e.g., LSTM), we use a backing buffer to handle ambiguous situations
        # where a single buffer may not be safe. Need to track this only for alive placeholders (buffers still held by someone outside).
        if self.last_buftype.get(self._buftype_key(cached_bufs[0]), -1) < j or self.last_placeholder_index[cached_bufs[0]] == j: placeholder_mapper[cached_bufs[0]] = cached_bufs[0].get_rawbuf()
        elif cached_bufs[0] not in placeholder_mapper: placeholder_mapper[cached_bufs[0]] = cached_bufs[0].alloc_rawbuf() # Allocate a backing buffer and use it until the penultimate entry.
      placeholder_mapper.update({buf: buf.get_rawbuf() for buf in cached_bufs if buf.__class__ is _CacheCollector._Placeholder and buf not in placeholder_mapper})
      cache_result.append((p, [placeholder_mapper.get(buf, buf) for buf in cached_bufs]))
    self.cache = None
    return cache_result
  def _buftype_key(self, buf): return (buf.size, buf.dtype, buf.dtype.shape if hasattr(buf.dtype, 'shape') else None)
CacheCollector = _CacheCollector()