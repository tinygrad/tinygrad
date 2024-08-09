#!/usr/bin/env python3
import os, ctypes, time
import llvmlite.binding as llvm
from tinygrad.helpers import getenv
from tinygrad.runtime.support.elf import elf_loader
from hexdump import hexdump
if getenv("IOCTL"): import run # noqa: F401 # pylint: disable=unused-import

adsp = ctypes.CDLL(ctypes.util.find_library("adsprpc"))
import adsprpc

if __name__ == "__main__":
  from tinygrad.runtime.ops_clang import ClangCompiler
  cc = ClangCompiler(["--target=hexagon", "-mcpu=hexagonv65", "-fuse-ld=lld", "-nostdlib"])

  obj = cc.compile("""
    struct remote_buf { void *pv; unsigned int len; };
    int entry(unsigned long long handle, unsigned int sc, struct remote_buf* pra) {
      if (sc>>24 == 1) {
        ((char*)pra[0].pv)[0] = 55;
        // NOTE: you have to return 0 for outbufs to work
        return 0;
      }
      return 0;
    }
  """)
  with open("/tmp/swag.so", "wb") as f: f.write(obj)

  handle = ctypes.c_int64(-1)
  adsp.remote_handle64_open(ctypes.create_string_buffer(b"file:////tmp/swag.so?entry&_modver=1.0&_dom=cdsp"), ctypes.byref(handle))
  print(handle.value)
  print(adsp.remote_handle64_invoke(handle, 0, None))

  arg_2 = ctypes.c_int64(-1)
  pra = (adsprpc.union_remote_arg64 * 3)()
  pra[0].buf.pv = ctypes.addressof(arg_2)
  pra[0].buf.len = 8
  print(adsp.remote_handle64_invoke(handle, (1<<24) | (1<<8), pra))
  print(hex(arg_2.value), arg_2.value)
