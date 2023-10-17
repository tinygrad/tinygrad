__all__ = ["nn", "codegen", "renderer", "runtime", "shape", "features"]

def __getattr__(name):
  if name in __all__:
    return __import__(f"tinygrad.{name}")
  raise AttributeError(f"module 'tinygrad' has no attribute {name}")

# this supports dir(tinygrad), but still uses lazy loading with getattr above
def __dir__():
  return sorted(__all__ + list(globals().keys()))
