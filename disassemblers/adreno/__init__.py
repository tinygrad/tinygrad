import ctypes
import os
import pathlib
from hexdump import hexdump

fxn = None
def disasm(buf):
  global fxn
  if fxn is None:
    shared = pathlib.Path(__file__).parent / "disasm.so"
    if not os.path.isfile(shared):
      os.system(f'cd {pathlib.Path(__file__).parent} && gcc -shared disasm-a3xx.c -o disasm.so')
    fxn = ctypes.CDLL(shared)['disasm']
  #hexdump(buf)
  END = b"\x00\x00\x00\x00\x00\x00\x00\x03"
  buf = buf[0x510:]  # this right?
  buf = buf.split(END)[0] + END
  fxn(buf, len(buf))
