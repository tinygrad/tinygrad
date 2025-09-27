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

def deserialize(lib, enc_src, opts):
  blobreader = lib.struct_blob_reader()
  lib.blob_reader_init(blobreader, src:=base64.b64decode(enc_src), len(src))
  return lib.nir_deserialize(None, ctypes.cast(opts, ctypes.POINTER(lib.nir_shader_compiler_options)), blobreader)

class LVPCompiler(Compiler):
  def __init__(self, cache_key="lvp"):
    # FIXME: this is wrong if mesa is compiled using ORCJIT
    self.ctx = lvp.lp_context_ref(ctypes.cast(llvm.LLVMContextCreate(), ctypes.POINTER(lvp.struct_LLVMOpaqueContext)), True)
    # see extra/mesa/lvp_nir_options.sh
    self.nir_options = ctypes.pointer(nir_shader_compiler_options.from_buffer_copy(gzip.decompress(base64.b64decode(
      "H4sIAAAAAAAAA5WMsREAIAwCnw0cwU1czdE9JbHwtJAiEEhAEjCnTGbCxIIUikvLTYmZZcmJi/WCqJfHxQV67K3t8KPbGIq2g3b4AAAA%"))))
    super().__init__(f"compile_{cache_key}")

  def __del__(self): llvm.LLVMContextDispose(ctypes.cast(self.ctx.ref, llvm.LLVMContextRef))

  def compile(self, src) -> bytes:
    shader = deserialize(lvp, src, self.nir_options)

    gallivm = lvp.gallivm_create(None, self.ctx, None)
    module = ctypes.cast(gallivm.contents.module, llvm.LLVMModuleRef)
    ctx, builder = ctypes.cast(gallivm.contents.context, llvm.LLVMContextRef), ctypes.cast(gallivm.contents.builder, llvm.LLVMBuilderRef)

    params = lvp.struct_lp_build_tgsi_params(lvp.struct_lp_type(floating=True, sign=True, width=32, length=4),
      resources_type=lvp.lp_build_jit_resources_type(gallivm), mask=ctypes.pointer(lvp.struct_lp_build_mask_context()))

    pt = llvm.LLVMPointerType(ctypes.cast(params.resources_type, llvm.LLVMTypeRef), 0)
    fn = llvm.LLVMAddFunction(module, shader.contents.info.name, llvm.LLVMFunctionType(llvm.LLVMVoidTypeInContext(ctx), pt, 1, 0))
    llvm.LLVMPositionBuilderAtEnd(builder, llvm.LLVMAppendBasicBlockInContext(ctx, fn, b"entry"))

    params.consts_ptr = lvp.lp_build_struct_get_ptr2(gallivm, params.resources_type,
      ctypes.cast(llvm.LLVMGetParam(fn, 0), lvp.LLVMValueRef), lvp.LP_JIT_RES_CONSTANTS, b"constants")
    lvp.lp_build_mask_begin(params.mask, gallivm, params.type, lvp.lp_build_one(gallivm, params.type))
    lvp.lp_build_mask_end(params.mask)

    lvp.lp_build_nir_soa(gallivm, shader, params, None)
    llvm.LLVMBuildRetVoid(builder)
    lvp.gallivm_verify_function(gallivm, ctypes.cast(fn, lvp.LLVMValueRef))
    lvp.gallivm_compile_module(gallivm)
    t = llvm.LLVMGetExecutionEngineTargetMachine(ctypes.cast(gallivm.contents.engine, llvm.LLVMExecutionEngineRef))
    obj_buf = expect(llvm.LLVMTargetMachineEmitToMemoryBuffer(t, module, llvm.LLVMObjectFile, e:=cerr(),
                                                              ctypes.pointer(b:=llvm.LLVMMemoryBufferRef())), e, b)
    ret = ctypes.string_at(llvm.LLVMGetBufferStart(obj_buf), llvm.LLVMGetBufferSize(obj_buf))
    lvp.gallivm_destroy(gallivm)
    llvm.LLVMDisposeMemoryBuffer(obj_buf)
    lvp.ralloc_free(shader)
    return ret

  def disassemble(self, lib:bytes): cpu_objdump(lib)

class NAKCompiler(Compiler):
  def __init__(self, dev, cache_key="nak"):
    self.arch = dev.arch
    self.cc = nak.nak_compiler_create(nak.struct_nv_device_info(sm=int(dev.arch[3:]), max_warps_per_mp=dev.max_warps_per_sm))
    self.nir_options = ctypes.cast(nak.nak_nir_options(self.cc), ctypes.POINTER(nir_shader_compiler_options))
    nak.glsl_type_singleton_init_or_ref()
    super().__init__(f"compile_{cache_key}_{dev.arch}")

  def __del__(self):
    nak.nak_compiler_destroy(self.cc)
    nak.glsl_type_singleton_decref()

  def compile(self, src) -> bytes:
    shader = deserialize(nak, src, self.nir_options)
    nak.nak_preprocess_nir(shader, self.cc)
    ret = nak.nak_compile_shader(shader, False, self.cc, 0, None).contents
    nak.ralloc_free(shader)
    return ret

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

