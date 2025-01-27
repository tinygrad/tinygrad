import ctypes, platform
from tinygrad.device import Compiled, Compiler, MallocAllocator, CPUProgram
from tinygrad.helpers import getenv, capstone_flatdump
from tinygrad.renderer.llvmir import LLVMRenderer
from tinygrad.runtime.support.llvm import LLVM_VER
import tinygrad.runtime.autogen.llvm as llvm
from tinygrad.runtime.support.elf import jit_loader

def cerr(): return ctypes.pointer(ctypes.pointer(ctypes.c_char()))

def expect(x, err, ret=None):
  if x: raise RuntimeError(llvm.string_cast(err.contents) if not isinstance(err, str) else err)
  return ret

HOST_ARCH = {'arm64': 'AArch64', 'aarch64': 'AArch64', 'x86_64': 'X86', 'AMD64': 'X86'}[platform.machine()]
REQUIRED_COMPONENTS = ['Target', 'TargetInfo', 'TargetMC', 'AsmPrinter']

class LLVMCompiler(Compiler):
  def __init__(self, target_machine, opt):
    self.pbo = llvm.LLVMCreatePassBuilderOptions()
    if opt:
      self.passes = b'default<O2>'
      llvm.LLVMPassBuilderOptionsSetLoopUnrolling(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetLoopVectorization(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetSLPVectorization(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetVerifyEach(self.pbo, True)
    else:
      self.passes = b'default<O0>'
    self.target_machine, self.opt = target_machine, opt
    super().__init__(f"compile_llvm_{LLVM_VER}_jit{'_opt' if opt else ''}")

  def __del__(self):
    llvm.LLVMDisposePassBuilderOptions(self.pbo)

  def compile(self, src:str) -> bytes:
    src_buf = llvm.LLVMCreateMemoryBufferWithMemoryRangeCopy(ctypes.create_string_buffer(src_bytes:=src.encode()), len(src_bytes), b'src')
    mod = expect(llvm.LLVMParseIRInContext(llvm.LLVMGetGlobalContext(), src_buf, ctypes.pointer(m:=llvm.LLVMModuleRef()), err:=cerr()), err, m)
    expect(llvm.LLVMVerifyModule(mod, llvm.LLVMReturnStatusAction, err:=cerr()), err)
    expect(llvm.LLVMRunPasses(mod, self.passes, self.target_machine, self.pbo), 'failed to run passes')
    obj_buf = expect(llvm.LLVMTargetMachineEmitToMemoryBuffer(self.target_machine, mod, llvm.LLVMObjectFile, err:=cerr(),
                                                              ctypes.pointer(buf:=llvm.LLVMMemoryBufferRef())), err, buf)
    obj = ctypes.string_at(llvm.LLVMGetBufferStart(obj_buf), llvm.LLVMGetBufferSize(obj_buf))
    llvm.LLVMDisposeModule(mod)
    llvm.LLVMDisposeMemoryBuffer(obj_buf)
    return jit_loader(obj)

  def disassemble(self, lib:bytes): capstone_flatdump(lib)

class LLVMDevice(Compiled):
  def __init__(self, device:str):
    for component in REQUIRED_COMPONENTS:
      getattr(llvm, f'LLVMInitialize{HOST_ARCH}{component}')()

    triple = f'{platform.machine()}-none-unknown-elf'.encode()
    target = expect(llvm.LLVMGetTargetFromTriple(triple, ctypes.pointer(tgt:=llvm.LLVMTargetRef()), err:=cerr()), err, tgt)
    features = b'+reserve-x18' if platform.machine() == 'arm64' else b''
    target_machine = llvm.LLVMCreateTargetMachine(target, triple, b'', features, llvm.LLVMCodeGenLevelDefault, llvm.LLVMRelocPIC,
                                                  llvm.LLVMCodeModelDefault)

    super().__init__(device, MallocAllocator, LLVMRenderer(), LLVMCompiler(target_machine, getenv("LLVMOPT")), CPUProgram)
