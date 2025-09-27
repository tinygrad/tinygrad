import base64, ctypes, gzip, pathlib, tempfile, hashlib, subprocess
from typing import Tuple
from tinygrad.device import Compiler
from tinygrad.helpers import cpu_objdump, round_up
from tinygrad.runtime.autogen.nir import nir_shader_compiler_options
from tinygrad.runtime.support.compiler_cpu import cerr, expect
import tinygrad.runtime.autogen.lvp as lvp
import tinygrad.runtime.autogen.nak as nak
try: import tinygrad.runtime.autogen.llvm as llvm
except (ImportError, FileNotFoundError): llvm = None #type:ignore[assignment]

class LVPCompiler(Compiler):
  def __init__(self, cache_key="lvp"):
    # FIXME: this is wrong if mesa is compiled using ORCJIT
    self.ctx = lvp.lp_context_ref(ctypes.cast(llvm.LLVMContextCreate(), ctypes.POINTER(lvp.struct_LLVMOpaqueContext)), True)
    # see extra/mesa/lvp_nir_options.sh
    self.nir_options = ctypes.pointer(nir_shader_compiler_options.from_buffer_copy(gzip.decompress(base64.b64decode(
      "H4sIAAAAAAAAA5WMsREAIAwCnw0cwU1czdE9JbHwtJAiEEhAEjCnTGbCxIIUikvLTYmZZcmJi/WCqJfHxQV67K3t8KPbGIq2g3b4AAAA%"))))
    super().__init__(f"compile_{cache_key}")

  def compile(self, src) -> bytes:
    # import os
    # input(f"pid: {os.getpid()}")
    blobreader = lvp.struct_blob_reader()
    lvp.blob_reader_init(blobreader, src, len(src))
    shader = lvp.nir_deserialize(None, ctypes.cast(self.nir_options, ctypes.POINTER(lvp.nir_shader_compiler_options)), blobreader)

    gallivm = lvp.gallivm_create(None, self.ctx, None)
    module = ctypes.cast(gallivm.contents.module, ctypes.POINTER(llvm.struct_LLVMOpaqueModule))

    params = lvp.struct_lp_build_tgsi_params(lvp.struct_lp_type(floating=True, sign=True, width=32, length=4),
      resources_type=lvp.lp_build_jit_resources_type(gallivm), mask=ctypes.pointer(lvp.struct_lp_build_mask_context()))

    ctx = ctypes.cast(gallivm.contents.context, ctypes.POINTER(llvm.struct_LLVMOpaqueContext))
    builder = ctypes.cast(gallivm.contents.builder, ctypes.POINTER(llvm.struct_LLVMOpaqueBuilder))
    pt = llvm.LLVMPointerType(ctypes.cast(params.resources_type, ctypes.POINTER(llvm.struct_LLVMOpaqueType)), 0)
    fn = llvm.LLVMAddFunction(module, b"aaa", llvm.LLVMFunctionType(llvm.LLVMVoidTypeInContext(ctx), pt, 1, 0))
    llvm.LLVMPositionBuilderAtEnd(builder, llvm.LLVMAppendBasicBlockInContext(ctx, fn, b"entry"))

    params.consts_ptr = lvp.lp_build_struct_get_ptr2(gallivm, params.resources_type,
      ctypes.cast(llvm.LLVMGetParam(fn, 0), ctypes.POINTER(lvp.struct_LLVMOpaqueValue)), lvp.LP_JIT_RES_CONSTANTS, b"constants")
    lvp.lp_build_mask_begin(params.mask, gallivm, params.type, lvp.lp_build_one(gallivm, params.type))
    lvp.lp_build_mask_end(params.mask)

    lvp.lp_build_nir_soa(gallivm, shader, params, None)
    llvm.LLVMBuildRetVoid(builder)
    lvp.gallivm_verify_function(gallivm, ctypes.cast(fn, ctypes.POINTER(lvp.struct_LLVMOpaqueValue)))
    lvp.gallivm_compile_module(gallivm)
    t = llvm.LLVMGetExecutionEngineTargetMachine(ctypes.cast(gallivm.contents.engine, ctypes.POINTER(llvm.struct_LLVMOpaqueExecutionEngine)))
    obj_buf = expect(llvm.LLVMTargetMachineEmitToMemoryBuffer(t, module, llvm.LLVMObjectFile, e:=cerr(),
                                                              ctypes.pointer(b:=llvm.LLVMMemoryBufferRef())), e, b)
    return ctypes.string_at(llvm.LLVMGetBufferStart(obj_buf), llvm.LLVMGetBufferSize(obj_buf))

  def disassemble(self, lib:bytes): cpu_objdump(lib)

class NAKCompiler(Compiler):
  def __init__(self, dev, cache_key="nak"):
    self.arch = dev.arch
    self.cc = nak.nak_compiler_create(nak.struct_nv_device_info(sm=int(dev.arch[3:]), max_warps_per_mp=dev.max_warps_per_sm))
    self.nir_options = ctypes.cast(nak.nak_nir_options(self.cc), ctypes.POINTER(nir_shader_compiler_options))
    nak.glsl_type_singleton_init_or_ref()
    super().__init__(f"compile_{cache_key}_{dev.arch}")

  def __del__(self): nak.glsl_type_singleton_decref()

  def compile(self, src) -> bytes:
    blobreader = nak.struct_blob_reader()
    nak.blob_reader_init(blobreader, src, len(src))
    shader = nak.nir_deserialize(None, ctypes.cast(self.nir_options, ctypes.POINTER(nak.nir_shader_compiler_options)), blobreader)
    nak.nak_preprocess_nir(shader, self.cc)
    return nak.nak_compile_shader(shader, False, self.cc, 0, None).contents
  def disassemble(self, lib: bytes):
    try:
      fn = (pathlib.Path(tempfile.gettempdir()) / f"tinynak_{hashlib.md5(lib).hexdigest()}").as_posix()
      with open(fn, "wb") as f: f.write(parse_nak_shader(lib)[0])
      print(subprocess.check_output(['nvdisasm', "-b", f"SM{self.arch[3:]}", fn]).decode('utf-8'))
    except Exception as e: print("Failed to generate SASS", str(e), "Make sure your PATH contains nvdisasm binary of compatible version.")

def parse_nak_shader(shader:bytes) -> Tuple[memoryview, int, int, int]:
  sb = nak.struct_nak_shader_bin.from_buffer(shader)
  return (memoryview(ctypes.cast(sb.code, ctypes.POINTER(ctypes.c_char * sb.code_size)).contents), sb.info.num_gprs,
          round_up(sb.info.cs.smem_size, 0x80), round_up(sb.info.slm_size, 0x10))

