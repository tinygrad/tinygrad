import ctypes, ctypes.util, functools, sys

class id_(ctypes.c_void_p):
  retain: bool = False
  # This prevents ctypes from converting response to plain int, and dict.fromkeys() can use it to dedup
  def __hash__(self): return hash(self.value)
  def __eq__(self, other): return self.value == other.value
  def __del__(self):
    if self.retain and not sys.is_finalizing(): msg("release")(self)
  def retained(self): return (setattr(self, 'retain', True), self)[1]

def returns_retained(f): return functools.wraps(f)(lambda *args, **kwargs: f(*args, **kwargs).retained())

libobjc = ctypes.CDLL(ctypes.util.find_library('objc'))
libobjc.sel_registerName.restype = id_
getsel = functools.cache(libobjc.sel_registerName)
libobjc.objc_getClass.restype = id_

def msg(sel:str, restype=id_, argtypes=[], retain=False, clsmeth=False):
  # Using attribute access returns a new reference so setting restype is safe
  (sender:=libobjc["objc_msgSend"]).restype, sender.argtypes = restype, argtypes
  return (returns_retained if retain else lambda x:x)(lambda ptr, *args: sender(ptr._objc_class_ if clsmeth else ptr, getsel(sel.encode()), *args))

class MetaSpec(type(id_)):
  def __new__(mcs, name, bases, dct):
    for m in dct.pop("_methods_", []): dct[m[0]] = msg(*m)
    for cm in dct.pop("_classmethods_", []): dct[cm[0]] = classmethod(msg(*cm, clsmeth=True))
    return super().__new__(mcs, name, bases, {'_objc_class_': libobjc.objc_getClass(name.encode()), **dct})

class Spec(id_, metaclass=MetaSpec): pass
