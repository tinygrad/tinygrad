from typing import Dict, Union, Any
import functools, ctypes, ctypes.util
import tinygrad.runtime.autogen.objc as libobjc
# note: The Objective-C runtime does not expose enough information to provide completely automatic bindings of all APIs. source: https://pyobjc.readthedocs.io/en/latest/metadata/index.html

def convert_arg(arg, arg_type):
  if isinstance(arg, ObjcInstance): assert not arg.released, f"use after free ({arg})"
  if isinstance(arg, str) and arg_type is ctypes.c_char_p: return arg.encode()
  if isinstance(arg, str) and arg_type is ctypes.c_void_p: return NSString.stringWithUTF8String_(arg)
  if isinstance(arg, list) or isinstance(arg, tuple) and arg_type is ctypes.c_void_p: return (ctypes.c_void_p * len(arg))(*arg)
  return arg

@functools.lru_cache(maxsize=None)
def sel_registerName(sel: str) -> libobjc.SEL:
  return libobjc.sel_registerName(sel.encode())

def objc_msgSend(obj: ctypes.c_void_p, sel: str, *args, restype=None, argtypes=[]):
  base_argtypes = [ctypes.c_void_p, ctypes.c_void_p]
  encoded_args = [convert_arg(a, t) for a, t in zip(args, argtypes)]
  # print(f"Sending {sel}(restype:{restype} argtypes:{argtypes}) to ptr:{obj} with args:{args}")
  libobjc.objc_msgSend.restype, libobjc.objc_msgSend.argtypes = restype, ((base_argtypes + argtypes) if argtypes else base_argtypes)
  return libobjc.objc_msgSend(obj, sel_registerName(sel), *encoded_args)

libc = ctypes.CDLL(None)
libc.free.argtypes = [ctypes.c_void_p]

def dump_objc_methods(clz: libobjc.Class):
  method_count = ctypes.c_uint()
  methods_ptr = libobjc.class_copyMethodList(clz, ctypes.byref(method_count))
  assert methods_ptr is not None, f"Failed to get methods for class {clz}"

  methods = {}
  for i in range(method_count.value):
    method = methods_ptr[i]
    sel_name = ctypes.string_at(libobjc.sel_getName(libobjc.method_getName(method))).decode('ascii')
    return_type_p = libobjc.method_copyReturnType(method)
    return_type = ctypes.string_at(return_type_p).decode('ascii')
    argtypes_ps = tuple(libobjc.method_copyArgumentType(method, j) for j in range(libobjc.method_getNumberOfArguments(method)))

    methods[sel_name] = {"restype": return_type, "argtypes": tuple(ctypes.string_at(arg).decode('ascii') for arg in argtypes_ps)}

    [libc.free(p) for p in argtypes_ps]
    libc.free(ctypes.cast(return_type_p, ctypes.c_void_p))
  libc.free(methods_ptr)
  return methods

SIMPLE_TYPES = {
    'c': ctypes.c_char,
    'i': ctypes.c_int,
    's': ctypes.c_short,
    'l': ctypes.c_long,
    'q': ctypes.c_longlong,
    'C': ctypes.c_uint8,
    'I': ctypes.c_uint,
    'S': ctypes.c_ushort,
    'L': ctypes.c_ulong,
    'Q': ctypes.c_ulonglong,
    'f': ctypes.c_float,
    'd': ctypes.c_double,
    'B': ctypes.c_bool,
    'v': None,
    '*': ctypes.c_char_p,
    '@': ctypes.c_void_p,
    '#': 'Class',
    ':': 'SEL',
    '?': '<unknown-type>',
}

@functools.lru_cache(maxsize=None)
def get_methods_rec(c: int):
  p = ctypes.cast(ctypes.c_void_p(c), libobjc.Class)
  methods = {}
  while p:
    methods.update(dump_objc_methods(p))
    p = libobjc.class_getSuperclass(p)
  return methods

def objc_type_to_ctype(t: str):
  if len(t) == 1:
    return SIMPLE_TYPES[t]
  elif t[0] == '^':
    return ctypes.POINTER(objc_type_to_ctype(t[1:]))
  elif t[0] == 'r':
    return objc_type_to_ctype(t[1:])
  elif t[0] == "V":
    return objc_type_to_ctype(t[1:])
  elif t.startswith("{") and "=" in t and t.endswith("}"):
    return ctypes.Structure  # wooo! safety is out the window now
  else:
    raise ValueError(f"Unknown type {t}")

def wrapper_return_objc_instance(f):
  def _wrapper(*args, **kwargs):
    res = f(*args, **kwargs)
    if res: return ObjcInstance(res)
    return None
  return _wrapper

def wrapper_arg_error(f):
  def _wrapper(*args, **kwargs):
    err = ctypes.c_void_p()
    res = f(*args[:-1], ctypes.byref(err), **kwargs)
    return (res, err if err.value else None)
  return _wrapper

@functools.lru_cache(maxsize=None)
def build_method(name, sel_name, restype, argtypes):
  """hashable args for lru_cache, this should only be ran once for each called method name"""
  # print(f"Building method {name} with sel_name {sel_name} restype {restype} argtypes {argtypes}")
  def f(p):
    _f = functools.partial(objc_msgSend, p, sel_name, restype=objc_type_to_ctype(restype),
          argtypes=[objc_type_to_ctype(t) for t in argtypes[2:]])
    if restype == "@": _f = wrapper_return_objc_instance(_f)
    if name.endswith("error_"): _f = wrapper_arg_error(_f)
    return _f
  return f


class ObjcClass(ctypes.c_void_p):
  def __init__(self, name:str):
    p: int | None = ctypes.cast(libobjc.objc_getClass(name.encode()), ctypes.c_void_p).value
    super().__init__(p)
    assert self.value, f"Class {name} not found"
    _metaclass_ptr = libobjc.object_getClass(ctypes.cast(ctypes.c_void_p(p), libobjc.id))
    self.methods_info: Dict[str, Dict[str, Any]] = get_methods_rec(ctypes.cast(_metaclass_ptr, ctypes.c_void_p).value)

  def __repr__(self) -> str:
    return f"<{ctypes.string_at(libobjc.object_getClassName(ctypes.cast(ctypes.c_void_p(self.value), libobjc.id))).decode()} at 0x{self.value:x}>"

  def __hash__(self) -> int:
    return self.value

  def __getattr__(self, name:str) -> Any:
    sel_name = name.replace("_", ":")
    if sel_name in self.methods_info:
      method_info = self.methods_info[sel_name]
      restype, argtypes = method_info["restype"], method_info["argtypes"]
      return build_method(name, sel_name, restype, argtypes)(self)  # use cached method

    raise AttributeError(f"Method {name} not found on {self.__class__.__name__}")


class ObjcInstance(ObjcClass):
  def __init__(self, ptr: Union[int, ctypes.c_void_p, None]):
    v: int | None = ptr.value if isinstance(ptr, ctypes.c_void_p) else ptr
    assert v, "Can't create ObjcInstance with null ptr"
    super(ctypes.c_void_p, self).__init__(v)
    c = libobjc.object_getClass(ctypes.cast(ctypes.c_void_p(v), libobjc.id))
    self.methods_info = get_methods_rec(ctypes.cast(c, ctypes.c_void_p).value)
    self.auto_release = True
    self.released = False

  def __del__(self):
    if self.auto_release:
      # print(f"Releasing {self}")
      self.released = True
      self.release()

NSString: Any = ObjcClass("NSString")

def nsstring_to_str(nsstring) -> str:
  return ctypes.string_at(nsstring.UTF8String(), size=nsstring.length()).decode()
