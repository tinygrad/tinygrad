from enum import Enum
from typing import List, NamedTuple, Tuple, Union, cast
from typing_extensions import Dict
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.ops import BinaryOps, PatternMatcher, UOp, UOps, UPat
from tinygrad.renderer import Renderer
import platform, os

class X86R(Enum): # CPU register
  RAX = 0; RCX = 1; RDX = 2; RBX = 3; RBP = 5; RSI = 6; RDI = 7
  R8 = 8; R9 = 9; R10 = 10; R11 = 11; R12 = 12; R13 = 13; R14 = 14; R15 = 15
class X86FR(Enum): # Floating point register
  XMM0 = 0; XMM1 = 1; XMM2 = 2; XMM3 = 3; XMM4 = 4; XMM5 = 5; XMM6 = 6; XMM7 = 7; XMM8 = 8
  XMM9 = 9; XMM10 = 10; XMM11 = 11; XMM12 = 12; XMM13 = 13; XMM14 = 14; XMM15 = 15;
  YMM0 = 0; YMM1 = 1; YMM2 = 2; YMM3 = 3; YMM4 = 4; YMM5 = 5; YMM6 = 6; YMM7 = 7; YMM8 = 8
  YMM9 = 9; YMM10 = 10; YMM11 = 11; YMM12 = 12; YMM13 = 13; YMM14 = 14; YMM15 = 15;
  ZMM0 = 0; ZMM1 = 1; ZMM2 = 2; ZMM3 = 3; ZMM4 = 4; ZMM5 = 5; ZMM6 = 6; ZMM7 = 7; ZMM8 = 8
  ZMM9 = 9; ZMM10 = 10; ZMM11 = 11; ZMM12 = 12; ZMM13 = 13; ZMM14 = 14; ZMM15 = 15;

X86_REGISTERS_THAT_NEED_TO_BE_SAVED = [X86R.RBX, X86R.R12, X86R.R13, X86R.R14, X86R.R15]
X86_ARG_REGISTERS = [X86R.RDI, X86R.RSI, X86R.RDX, X86R.RCX, X86R.R8, X86R.R9]

X86_64BIT_PREFIX = 0b01001000; X86_REX_W = 1 << 3; X86_REX_R = 1 << 2; X86_REX_X = 1 << 1; X86_REX_B = 1;

def prefix_byte_register_to_register(op1: X86R, op2: X86R): return X86_64BIT_PREFIX | (X86_REX_R if op1.value >= X86R.R8.value else 0) | (X86_REX_B if op2.value >= X86R.R8.value else 0)
def modrm_register_to_register(op1: X86R, op2: X86R): return 0b11000000 | (op1.value & 0b111) << 3 | (op2.value & 0b111)

class X8664ASMRenderer(Renderer):
  def __init__(self, avx=False, avx2=False, avx512=False): self.avx, self.avx2, self.avx512, self.code = avx, avx2, avx512, bytearray()
  def alloc_register(self) -> X86R: 
    reg = filter(lambda r: self.free_registers[r], X86R).__next__()
    self.free_registers[reg] = False
    return reg
  def dealloc_register(self, reg): assert not self.free_registers[reg]; self.free_registers[reg] = True
  def alloc_fpregister(self) -> X86FR:
    reg = filter(lambda r: self.free_fpu_registers[r], X86FR).__next__()
    self.free_fpu_registers[reg] = False
    return reg
  def dealloc_fpregister(self, reg): assert not self.free_fpu_registers[reg]; self.free_fpu_registers[reg] = True
  def alloc_argument_register(self, dtype, buf): 
    if isinstance(dtype, PtrDType):
      reg = filter(lambda r: self.free_registers[r], X86_ARG_REGISTERS).__next__()
      self.free_registers[reg] = False; self.buffer_registers[buf] = (reg, dtype)
      return reg
    assert False, "something that wasnt a buffer in arguments"

  def emit(self, code: bytearray): self.code = self.code + code
  def tell(self): return len(self.code) - 1

  def xor(self, op1: X86R, op2: X86R): self.emit(bytearray([prefix_byte_register_to_register(op1, op2), 0x31, modrm_register_to_register(op1, op2)]))

  def reset(self):
    self.free_registers = { r: True for r in X86R }
    self.free_fpu_registers = { r: True for r in X86FR }
    self.code = bytearray()
    self.buffer_registers: Dict[int, Tuple[X86R, DType]] = {}

  def render_recursive(self, uops: List[UOp], off=0):
    i, max = 0, len(uops)-off
    while i < max:
      u = uops[i + off]
      uop,dtype,src,args = u.op,u.dtype,u.src,u.arg
      print(u.render(False))

      if uop == UOps.DEFINE_GLOBAL: self.alloc_argument_register(dtype, args)
      elif uop == UOps.CONST: pass
      elif uop == UOps.RANGE: 
        # counter_register = self.alloc_register()
        # self.emit()
        # loop_begin = i
        # self.render_recursive(uops, i+1)
        # loop_ends_at =
        continue
        
      elif uop == UOps.ENDRANGE: return
      else: assert False, f"op {uop} not implemented" 

      i+=1

    return i

  def render(self, name: str, uops: List[UOp]) -> str:
    instructions_parsed = self.render_recursive(uops)
    print(instructions_parsed, self.code)
    return str(self.code)

class ASMRenderer(Renderer):
  device = "ASM"
  global_max = None
  has_local = False

  def __init__(self) -> None:
    if platform.machine() == 'x86_64':
      with open("/proc/cpuinfo", "r") as f:
        cpuinfo = f.read()
        # TODO: probably not a good way to do this.
        avx = cpuinfo.find("avx ") != -1; avx2 = cpuinfo.find("avx2 ") != -1; avx512 = cpuinfo.find("avx512 ") != -1
        self.renderer = X8664ASMRenderer(avx, avx2, avx512)
    else: raise RuntimeError(f'Architecture {platform.machine()} not supported for assembly backend.')

  def render(self, name: str, uops: List[UOp]) -> str:
    return self.renderer.render(name, uops)

if __name__ == "__main__":
  renderer = X8664ASMRenderer()
  renderer.xor(X86R.R9, X86R.RAX)
  renderer.xor(X86R.RAX, X86R.RAX)
  print(renderer.code.hex())
