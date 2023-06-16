import time, ctypes, hashlib, subprocess, platform
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.runtime.ops_clang import ClangCodegen

class X86Program:
  def __init__(self, name:str, prg:str):

    prg = f"""
.section .text 
.globl {name}
{name}:
  enter $8, $0
  movq $0, -8(%rsp)

.L0:
    mov -8(%rsp), %r8

    movd 0(%rsi, %r8, 4), %xmm0
    addss 0(%rdx, %r8, 4), %xmm0
    movd %xmm0, 0(%rdi, %r8, 4)

    mov -8(%rsp), %r15
    inc %r15
    mov %r15, -8(%rsp)
    cmp $5, %r15
    jle .L0

    mov %rsi, %rax
    leave
    ret
"""

    fn = f"/tmp/clang_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.{'dylib' if platform.system() == 'Darwin' else 'so'}"
    print(subprocess.run(["as", "-o", "Add.o"], input=prg.encode('utf-8')))
    print(subprocess.run(["ld", "-shared", "Add.o", "-o", fn]))

    self.lib = ctypes.CDLL(fn)
    self.fxn = self.lib[name]

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait: st = time.monotonic()
    self.fxn.argtypes = [ctypes.c_uint64 for i in range(len(args))]
    self.fxn.restype = ctypes.c_uint64
    out = self.fxn(*[ctypes.addressof(x._buf) for x in args])
    print("out", out)
    if wait: return time.monotonic()-st

X86Buffer = Compiled(RawMallocBuffer, ClangCodegen, X86Program)
