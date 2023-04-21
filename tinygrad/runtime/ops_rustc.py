import os, time, ctypes, hashlib, subprocess, platform
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.codegen.cstyle import RustCodegen # TODO copy cstyle to rustcstyle and revert changes to cstyle.py

class RustcProgram:
  def __init__(self, name:str, prg:str):
    # this might not even be needed
    prg = "#![crate_type = \"dylib\"]\n" + prg

    prg += "fn pow(a: f32, b: f32) -> f32 { a.powf(b) }\n"
    prg += "fn log(a: f32) -> f32 { a.log10() }\n"
    prg += "fn exp(a: f32) -> f32 { a.exp() }\n"
    prg += "type float = f32;\n"

    # TODO: is there a way to not write this to disk?
    fn = f"/tmp/rustc_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{'dylib' if platform.system() == 'Darwin' else 'so'}"

    if not os.path.exists(fn):
      subprocess.check_output(['rustc', '--crate-type=dylib', '-Copt-level=3', '-o', fn+".tmp", "-"], input=prg.encode('utf-8'))
      os.rename(fn+".tmp", fn)
    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name+"_c"]

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn(*[x._buf for x in args])
    if wait: return time.monotonic()-st

RustcBuffer = Compiled(RawMallocBuffer, RustCodegen, RustcProgram)
