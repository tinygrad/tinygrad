import weakref

def support_weakref(x): return x

@support_weakref
class Hello(object):
  def __init__(self):
    self.a = "bob"

def create_ref(h):
  return weakref.ref(h)

