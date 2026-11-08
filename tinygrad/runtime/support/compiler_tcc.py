import ctypes, functools, os, tempfile
from typing import Any
from tinygrad.device import Compiler, CompileError

@functools.cache
def libtcc() -> ctypes.CDLL|None:
  """libtcc (the Tiny C Compiler as a library), None if it isn't installed"""
  try: lib = ctypes.CDLL("libtcc.so")
  except OSError: return None
  lib.tcc_new.restype, vp, cp = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p
  args:dict[str, list[Any]] = {"tcc_delete": [vp], "tcc_set_error_func": [vp, vp, vp], "tcc_set_options": [vp, cp],
                               "tcc_set_output_type": [vp, ctypes.c_int], "tcc_compile_string": [vp, cp], "tcc_output_file": [vp, cp]}
  for f, a in args.items(): getattr(lib, f).argtypes = a
  return lib

TCC_OUTPUT_OBJ = 3
_ERROR_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p)

class TCCCompiler(Compiler):
  """
  C to an ELF object in this process by libtcc: no optimizations, milliseconds for the large straight line programs (the HCQ host programs,
  where LLVM's take seconds)
  """
  def __init__(self, cachekey="compile_tcc"):
    if libtcc() is None: raise RuntimeError("libtcc.so is not installed")
    super().__init__(cachekey)

  def compile(self, src:str) -> bytes:
    lib, errors = libtcc(), []
    assert lib is not None
    on_error = _ERROR_FUNC(lambda _, msg: errors.append(msg.decode()))
    s = lib.tcc_new()
    try:
      lib.tcc_set_error_func(s, None, ctypes.cast(on_error, ctypes.c_void_p))
      lib.tcc_set_options(s, b"-nostdlib -w -fno-asynchronous-unwind-tables")  # (no .eh_frame: nothing unwinds them)
      lib.tcc_set_output_type(s, TCC_OUTPUT_OBJ)
      if lib.tcc_compile_string(s, src.encode()) != 0: raise CompileError("tcc: " + "\n".join(errors))
      # (the object to a file: libtcc writes no other output. a tmpfs one)
      with tempfile.NamedTemporaryFile(suffix=".o", dir="/dev/shm" if os.path.isdir("/dev/shm") else None) as f:
        if lib.tcc_output_file(s, f.name.encode()) != 0: raise CompileError("tcc: " + "\n".join(errors))
        with open(f.name, "rb") as obj: return obj.read()
    finally: lib.tcc_delete(s)
