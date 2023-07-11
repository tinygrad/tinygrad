from weakref import ref

# Stripped down version of a WeakSet
class LightWeakSet:
  __slots__ = 'data', '_remove', '__weakref__'
  def __init__(self):
    self.data = set()
    def _remove(item, selfref=ref(self)):
      self = selfref()
      if self: self.data.discard(item)
    self._remove = _remove

  def __len__(self): return len(self.data)
  def add(self, item): self.data.add(ref(item, self._remove))
  def discard(self, item): self.data.discard(ref(item))
