from weakref import KeyedRef, ref
from _weakref import _remove_dead_weakref # type: ignore

# Stripped down version of a WeakValueDictionary
class LightWeakValueDictionary:
  __slots__ = 'data', '_remove', '__weakref__'
  def __init__(self):
    def remove(wr, selfref=ref(self), _atomic_removal=_remove_dead_weakref):
      self = selfref()
      if self: _atomic_removal(self.data, wr.key)
    self._remove = remove
    self.data = {}

  def __getitem__(self, key):
    o = self.data[key]()
    if o is None: raise KeyError(key)
    else: return o

  def __len__(self): return len(self.data)
  def __delitem__(self, key): del self.data[key]
  def __setitem__(self, key, value): self.data[key] = KeyedRef(value, self._remove, key)
  def __contains__(self, key): return key in self.data
