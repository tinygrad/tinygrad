import ctypes, platform
from tinygrad.device import Compiled, CompileError, Compiler, MallocAllocator, CPUProgram
from tinygrad.helpers import OSX, getenv, amdgpu_disassemble, capstone_flatdump, DEBUG
from tinygrad.renderer.llvmir import LLVMRenderer
import tinygrad.runtime.autogen.llvm as llvm
from tinygrad.runtime.support.elf import jit_loader

def cerr(): return ctypes.pointer(ctypes.pointer(ctypes.c_char()))

def expect(x, err, ret=None):
  if x: raise RuntimeError(llvm.string_cast(err.contents) if not isinstance(err, str) else err)
  return ret

class LLVMCompiler(Compiler):
  def __init__(self, target_arch:str, triple:bytes, cpu:bytes, feats:bytes, jit:bool=False, cache_key:str="llvm"):
    for component in ['Target', 'TargetInfo', 'TargetMC', 'AsmParser', 'AsmPrinter']: getattr(llvm, f'LLVMInitialize{target_arch}{component}')()

    target = expect(llvm.LLVMGetTargetFromTriple(triple, ctypes.pointer(tgt:=llvm.LLVMTargetRef()), err:=cerr()), err, tgt)
    if DEBUG >= 2: print(f"LLVM init for {cpu!r} with {feats!r}")
    self.target_machine = llvm.LLVMCreateTargetMachine(target, triple, cpu, feats,
                                                       llvm.LLVMCodeGenLevelDefault, llvm.LLVMRelocPIC, llvm.LLVMCodeModelDefault)

    self.pbo = llvm.LLVMCreatePassBuilderOptions()
    if (opt:=bool(getenv("LLVMOPT", "1"))):
      self.passes = b'default<O2>'
      llvm.LLVMPassBuilderOptionsSetLoopUnrolling(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetLoopVectorization(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetSLPVectorization(self.pbo, True)
      llvm.LLVMPassBuilderOptionsSetVerifyEach(self.pbo, True)
    else:
      self.passes = b'default<O0>'

    self.jit = jit
    super().__init__(f"compile_{cache_key}{'_opt' if opt else ''}")

  def __del__(self): llvm.LLVMDisposePassBuilderOptions(self.pbo)

  def compile(self, src:str) -> bytes:
    src_buf = llvm.LLVMCreateMemoryBufferWithMemoryRangeCopy(ctypes.create_string_buffer(src_bytes:=src.encode()), len(src_bytes), b'src')
    mod = expect(llvm.LLVMParseIRInContext(llvm.LLVMGetGlobalContext(), src_buf, ctypes.pointer(m:=llvm.LLVMModuleRef()), err:=cerr()), err, m)
    expect(llvm.LLVMVerifyModule(mod, llvm.LLVMReturnStatusAction, err:=cerr()), err)
    expect(llvm.LLVMRunPasses(mod, self.passes, self.target_machine, self.pbo), 'failed to run passes')
    if DEBUG >= 7: print(ctypes.string_at(llvm.LLVMPrintModuleToString(mod)).decode())
    obj_buf = expect(llvm.LLVMTargetMachineEmitToMemoryBuffer(self.target_machine, mod, llvm.LLVMObjectFile, err:=cerr(),
                                                              ctypes.pointer(buf:=llvm.LLVMMemoryBufferRef())), err, buf)
    llvm.LLVMDisposeModule(mod)
    obj = ctypes.string_at(llvm.LLVMGetBufferStart(obj_buf), llvm.LLVMGetBufferSize(obj_buf))
    llvm.LLVMDisposeMemoryBuffer(obj_buf)
    return jit_loader(obj) if self.jit else obj

  def disassemble(self, lib:bytes): capstone_flatdump(lib)

class HostLLVMCompiler(LLVMCompiler):
  def __init__(self):
    host_arch = {'arm64': 'AArch64', 'aarch64': 'AArch64', 'x86_64': 'X86', 'AMD64': 'X86'}[platform.machine()]
    triple = {'AArch64': b'aarch64', 'X86': b'x86_64'}[host_arch] + b'-none-unknown-elf'
    # +reserve-x18 here does the same thing as -ffixed-x18 in ops_cpu.py, see comments there for why it's needed on arm osx
    cpu, feats = ctypes.string_at(llvm.LLVMGetHostCPUName()), (b'+reserve-x18,' if OSX else b'') + ctypes.string_at(llvm.LLVMGetHostCPUFeatures())
    super().__init__(host_arch, triple, cpu, feats, jit=True, cache_key="llvm_jit")

class AMDGPULLVMCompiler(LLVMCompiler):
  def __init__(self, arch: str):
    self.arch = arch
    super().__init__("AMDGPU", "amdgcn-amd-amdhsa", self.arch, "+cumode", jit=False, cache_key="llvm_amdgpu")
  def __reduce__(self): return (AMDGPULLVMCompiler, (self.arch,))
  def compile(self, src:str) -> bytes:
    try: return super().compile(src)
    except RuntimeError as e:
      if "undefined value '@llvm.amdgcn." in str(e): raise CompileError(str(e) + "AMD with LLVM backend requires LLVM >= 18") from e
      raise CompileError(e) from e
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)

class LLVMDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, MallocAllocator, LLVMRenderer(), HostLLVMCompiler(), CPUProgram)
