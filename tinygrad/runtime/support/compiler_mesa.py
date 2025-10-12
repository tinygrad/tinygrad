import base64, ctypes, pathlib, tempfile, hashlib, subprocess
from tinygrad.device import Compiler
from tinygrad.helpers import cpu_objdump
import tinygrad.runtime.autogen.mesa as mesa
try: import tinygrad.runtime.autogen.llvm as llvm
except (ImportError, FileNotFoundError): llvm = None #type:ignore[assignment]

def deserialize(enc_src, opts):
  blobreader = mesa.struct_blob_reader()
  mesa.blob_reader_init(blobreader, src:=base64.b64decode(enc_src), len(src))
  return mesa.nir_deserialize(None, ctypes.cast(opts, ctypes.POINTER(mesa.nir_shader_compiler_options)), blobreader)

class NIRCompiler(Compiler):
  def __init__(self, cache_key):
    mesa.glsl_type_singleton_init_or_ref()
    super().__init__(cache_key)
  def __del__(self): mesa.glsl_type_singleton_decref()

class LVPCompiler(NIRCompiler):
  def __init__(self, cache_key="lvp"): super().__init__(f"compile_{cache_key}")

  def compile(self, src) -> bytes:
    shader, ctx, cache = deserialize(src, mesa.lvp_nir_options), llvm.LLVMGetGlobalContext(), mesa.struct_lp_cached_code()
    gallivm = mesa.gallivm_create(None, mesa.lp_context_ref(ctypes.cast(ctx, ctypes.POINTER(mesa.struct_LLVMOpaqueContext)), True), cache)
    module, builder = ctypes.cast(gallivm.contents.module, llvm.LLVMModuleRef), ctypes.cast(gallivm.contents.builder, llvm.LLVMBuilderRef)

    params = mesa.struct_lp_build_tgsi_params(mesa.struct_lp_type(floating=True, sign=True, width=32, length=4),
      resources_type=mesa.lp_build_jit_resources_type(gallivm), mask=ctypes.pointer(mesa.struct_lp_build_mask_context()))

    pt = llvm.LLVMPointerType(ctypes.cast(params.resources_type, llvm.LLVMTypeRef), 0)
    fn = llvm.LLVMAddFunction(module, shader.contents.info.name, llvm.LLVMFunctionType(llvm.LLVMVoidTypeInContext(ctx), pt, 1, 0))
    llvm.LLVMPositionBuilderAtEnd(builder, llvm.LLVMAppendBasicBlockInContext(ctx, fn, b"entry"))

    params.consts_ptr = mesa.lp_build_struct_get_ptr2(gallivm, params.resources_type,
      ctypes.cast(llvm.LLVMGetParam(fn, 0), mesa.LLVMValueRef), mesa.LP_JIT_RES_CONSTANTS, b"constants")
    mesa.lp_build_mask_begin(params.mask, gallivm, params.type, mesa.lp_build_one(gallivm, params.type))
    mesa.lp_build_mask_end(params.mask)

    mesa.lp_build_nir_soa(gallivm, shader, params, None)
    llvm.LLVMBuildRetVoid(builder)
    mesa.gallivm_verify_function(gallivm, ctypes.cast(fn, mesa.LLVMValueRef))
    mesa.gallivm_compile_module(gallivm)
    mesa.gallivm_jit_function(gallivm, ctypes.cast(fn, mesa.LLVMValueRef), shader.contents.info.name)
    ret = ctypes.string_at(cache.data, cache.data_size)
    mesa.gallivm_destroy(gallivm)
    mesa.ralloc_free(shader)
    return ret

  def disassemble(self, lib:bytes): cpu_objdump(lib)

class NAKCompiler(NIRCompiler):
  def __init__(self, arch, warps_per_sm, cache_key="nak"):
    self.arch, self.warps_per_sm = arch, warps_per_sm
    self.cc = mesa.nak_compiler_create(mesa.struct_nv_device_info(sm=int(arch[3:]), max_warps_per_mp=warps_per_sm))
    self.nir_options = bytes(mesa.nak_nir_options(self.cc).contents)
    super().__init__(f"compile_{cache_key}_{arch}")

  def __del__(self):
    mesa.nak_compiler_destroy(self.cc)
    super().__del__()

  def __reduce__(self): return NAKCompiler, (self.arch, self.warps_per_sm)

  def compile(self, src) -> bytes:
    shader = deserialize(src, self.nir_options)
    mesa.nak_preprocess_nir(shader, self.cc)
    ret = bytearray(bytes((out:=mesa.nak_compile_shader(shader, False, self.cc, 0, None).contents).info) + ctypes.string_at(out.code, out.code_size))
    mesa.nak_shader_bin_destroy(out)
    mesa.ralloc_free(shader)
    return ret

  def disassemble(self, lib: bytes):
    try:
      fn = (pathlib.Path(tempfile.gettempdir()) / f"tinynak_{hashlib.md5(lib).hexdigest()}").as_posix()
      with open(fn, "wb") as f: f.write(lib[ctypes.sizeof(mesa.struct_nak_shader_info):])
      print(subprocess.check_output(['nvdisasm', "-b", f"SM{self.arch[3:]}", fn]).decode('utf-8'))
    except Exception as e: print("Failed to generate SASS", str(e), "Make sure your PATH contains nvdisasm binary of compatible version.")
