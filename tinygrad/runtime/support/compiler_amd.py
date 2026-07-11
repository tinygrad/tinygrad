import ctypes, hashlib, tempfile, subprocess, pathlib, shutil
from tinygrad.helpers import system, getenv, diskcache_get, diskcache_get_batch, diskcache_put_batch
from tinygrad.runtime.autogen import comgr
try:
  comgr.amd_comgr_get_version(ctypes.byref(major:=ctypes.c_uint64()), ctypes.byref(minor:=ctypes.c_uint64()))
  if major.value >= 3:
    # in comgr 3 the values of enums in headers were changed: https://github.com/ROCm/llvm-project/issues/272
    import tinygrad.runtime.autogen.comgr_3 as comgr # type: ignore[no-redef]
    assert comgr.AMD_COMGR_LANGUAGE_HIP == 3
except AttributeError: pass  # ignore if ROCm isn't installed
from tinygrad.device import Compiler, CompileError
from tinygrad.runtime.support.compiler_cpu import LLVMCompiler
from tinygrad.runtime.support import c
from tinygrad.helpers import OSX, to_char_p_p

def _find_llvm_objdump():
  if OSX: return '/opt/homebrew/opt/llvm/bin/llvm-objdump'
  # Try ROCm path first, then versioned, then unversioned
  for p in ['/opt/rocm/llvm/bin/llvm-objdump', 'llvm-objdump-21', 'llvm-objdump-20', 'llvm-objdump']:
    if shutil.which(p): return p
  raise FileNotFoundError("llvm-objdump not found")

def amdgpu_disassemble(lib:bytes):
  asm = system(f"{_find_llvm_objdump()} -d -", input=lib).splitlines()
  while asm and ("s_nop 0" in asm[-1] or "s_code_end" in asm[-1]): asm.pop()
  print("\n".join(asm))

def check(status):
  if status != 0:
    comgr.amd_comgr_status_string(status, ctypes.byref(status_str := ctypes.POINTER(ctypes.c_char)()))
    raise RuntimeError(f"comgr fail {status}, {ctypes.string_at(status_str).decode()}")

def _get_comgr_data(data_set, data_type):
  check(comgr.amd_comgr_action_data_get_data(data_set, data_type, 0, ctypes.byref(data_exec := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz := ctypes.c_uint64()), None))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz), (dat := ctypes.create_string_buffer(sz.value))))
  check(comgr.amd_comgr_release_data(data_exec))
  return bytes(dat)

# amd_comgr_action_info_set_options was deprecated
def set_options(action_info, options:bytes):
  # TODO: this type should be correct in the autogen stub
  @comgr.dll.bind(comgr.amd_comgr_status_t, comgr.amd_comgr_action_info_t, c.POINTER[c.POINTER[ctypes.c_char]], comgr.size_t)
  def amd_comgr_action_info_set_option_list(ai, o, c) -> comgr.amd_comgr_status_t: pass # type: ignore[empty-body]
  return amd_comgr_action_info_set_option_list(action_info, to_char_p_p(options_list:=options.split(b' ')), len(options_list))

# AMD_COMGR_SAVE_TEMPS=1 AMD_COMGR_REDIRECT_LOGS=stdout AMD_COMGR_EMIT_VERBOSE_LOGS=1
def compile_hip(prg:str, arch="gfx1100", asm=False, use_device_libs=True, backend_opt=3) -> bytes:
  check(comgr.amd_comgr_create_action_info(ctypes.byref(action_info := comgr.amd_comgr_action_info_t())))
  check(comgr.amd_comgr_action_info_set_language(action_info, comgr.AMD_COMGR_LANGUAGE_HIP))
  check(comgr.amd_comgr_action_info_set_isa_name(action_info, b"amdgcn-amd-amdhsa--" + arch.encode()))
  check(comgr.amd_comgr_action_info_set_logging(action_info, True))

  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_src := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_bc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_reloc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_exec := comgr.amd_comgr_data_set_t())))

  check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(data_src := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_set_data(data_src, len(rprg := prg.encode()), rprg))

  if asm:
    check(comgr.amd_comgr_set_data_name(data_src, b"<null>.s"))
    check(comgr.amd_comgr_data_set_add(data_set_src, data_src))
    status = comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, action_info, data_set_src, data_set_reloc)
    if status != 0:
      print(_get_comgr_data(data_set_reloc, comgr.AMD_COMGR_DATA_KIND_LOG).decode())
      raise RuntimeError("assemble failed")
  else:
    check(comgr.amd_comgr_set_data_name(data_src, b"<null>"))
    check(comgr.amd_comgr_data_set_add(data_set_src, data_src))
    # -include hiprtc_runtime.h was removed
    frontend_opt = getenv("AMD_FRONTEND_OPT", 1)
    options = [
      f"-O{frontend_opt}", "-mcumode", "--hip-version=6.0.32830", "-DHIP_VERSION_MAJOR=6", "-DHIP_VERSION_MINOR=0", "-DHIP_VERSION_PATCH=32830",
      "-D__HIPCC_RTC__", "-std=c++14", "-nogpuinc", "-Wno-gnu-line-marker", "-Wno-missing-prototypes", f"--offload-arch={arch}",
      "-I/opt/rocm/include", "-Xclang -disable-llvm-passes", "-Xclang -aux-triple", "-Xclang x86_64-unknown-linux-gnu"]
    check(set_options(action_info, ' '.join(options).encode()))
    compile_action = comgr.AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC if use_device_libs else comgr.AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC
    status = comgr.amd_comgr_do_action(compile_action, action_info, data_set_src, data_set_bc)
    if status != 0:
      print(_get_comgr_data(data_set_bc, comgr.AMD_COMGR_DATA_KIND_LOG).decode())
      raise RuntimeError("compile failed")
    check(set_options(action_info, f"-O{backend_opt} -mllvm -vectorize-loops=false "
                                     "-mllvm -vectorize-slp=false -mllvm -unroll-threshold=0".encode()))
    check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, action_info, data_set_bc, data_set_reloc))

  check(set_options(action_info, b""))
  check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action_info, data_set_reloc, data_set_exec))
  ret = _get_comgr_data(data_set_exec, comgr.AMD_COMGR_DATA_KIND_EXECUTABLE)
  check(comgr.amd_comgr_release_data(data_src))
  for x in [data_set_src, data_set_bc, data_set_reloc, data_set_exec]: check(comgr.amd_comgr_destroy_data_set(x))
  check(comgr.amd_comgr_destroy_action_info(action_info))
  return ret

class HIPCompiler(Compiler):
  def __init__(self, arch:str):
    assert comgr.dll.nm in c.DLL._loaded_, f"comgr not available: {comgr.dll.emsg}"
    self.arch, self.frontend_opt, self.generic_opt = arch, getenv("AMD_FRONTEND_OPT", 1), getenv("AMD_GENERIC_COMPILE_OPT", 1)
    super().__init__(f"compile_hip_{self.arch}_F{self.frontend_opt}G{self.generic_opt}")
  def compile(self, src:str) -> bytes:
    try: return compile_hip(src, self.arch, src.split('\n', 1)[0].strip() == '.text', use_device_libs="__ocml_" in src)
    except RuntimeError as e: raise CompileError(e) from e
  def compile_cached_batch(self, srcs:list[tuple[str, str]]) -> list[tuple[str, str, bytes]]:
    batch_size = getenv("AMD_COMPILE_BATCH_SIZE", 256)
    batch_opt = getenv("AMD_COMPILE_OPT", 3)
    generic_opt = getenv("AMD_GENERIC_COMPILE_OPT", 1)
    frontend_opt = getenv("AMD_FRONTEND_OPT", 1)
    if batch_size <= 1 or any(src.split('\n', 1)[0].strip() == '.text' for _,src in srcs): return super().compile_cached_batch(srcs)

    renamed = []
    for name,src in srcs:
      new_name = f"{name}_{hashlib.sha256(src.encode()).hexdigest()[:8]}"
      renamed.append((new_name, src.replace(f" {name}(", f" {new_name}(", 1)))

    ret:list[tuple[str, str, bytes]|None] = [None] * len(srcs)
    missing:dict[tuple[int, bool], list[int]] = {}
    cache_writes:list[tuple[str, tuple[str, str]]] = []
    module_writes:list[tuple[str, bytes]] = []
    opts = [batch_opt if name.startswith("coop_") else generic_opt for name,_ in renamed]
    cache_srcs = [f"batchF{frontend_opt}O{opts[i]}\n{src}" for i,(_,src) in enumerate(renamed)]
    cached_sources = diskcache_get_batch(self.cachekey, cache_srcs) if self.cachekey is not None else {}
    refs = {v[1] for v in cached_sources.values() if isinstance(v, tuple) and v[0] == "batch"}
    module_keys = {key:f"__batch__{key}" for key in refs}
    same_table_modules = diskcache_get_batch(self.cachekey, list(module_keys.values())) if self.cachekey is not None else {}
    cached_modules = {key:same_table_modules[module_keys[key]] for key in refs if module_keys[key] in same_table_modules}
    if self.cachekey is not None and (old_refs:=refs-cached_modules.keys()):
      cached_modules.update(diskcache_get_batch(f"{self.cachekey}_batch", list(old_refs)))
    for i,(name,src) in enumerate(renamed):
      opt = opts[i]
      cached = cached_sources.get(cache_srcs[i])
      if isinstance(cached, tuple) and cached[0] == "batch": cached = cached_modules.get(cached[1])
      if cached is not None: ret[i] = (name, src, cached)
      else: missing.setdefault((opt, "__ocml_" in src), []).append(i)
    for (opt,use_device_libs),indices in missing.items():
      for start in range(0, len(indices), batch_size):
        batch = indices[start:start+batch_size]
        preamble = dict.fromkeys(line for i in batch for line in
                                 renamed[i][1].split('extern "C" __attribute__((global))', 1)[0].splitlines())
        kernels = []
        for i in batch:
          _, kernel = renamed[i][1].split('extern "C" __attribute__((global))', 1)
          kernels.append('extern "C" __attribute__((global))'+kernel)
        combined = '\n'.join((*preamble, *kernels))
        module_key = hashlib.sha256(f"F{frontend_opt}O{opt}\n{combined}".encode()).hexdigest()
        lib = diskcache_get(self.cachekey, f"__batch__{module_key}") if self.cachekey is not None else None
        if lib is None and self.cachekey is not None: lib = diskcache_get(f"{self.cachekey}_batch", module_key)
        if lib is None:
          try: lib = compile_hip(combined, self.arch, use_device_libs=use_device_libs, backend_opt=opt)
          except RuntimeError as e: raise CompileError(e) from e
          if self.cachekey is not None: module_writes.append((f"__batch__{module_key}", lib))
        for i in batch:
          name,src = renamed[i]
          if self.cachekey is not None: cache_writes.append((f"batchF{frontend_opt}O{opt}\n{src}", ("batch", module_key)))
          ret[i] = (name, src, lib)
    if self.cachekey is not None:
      diskcache_put_batch(self.cachekey, module_writes+cache_writes)
    assert all(x is not None for x in ret)
    return [x for x in ret if x is not None]
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)

class HIPCCCompiler(Compiler):
  def __init__(self, arch:str, extra_options:list[str]=[]):
    self.arch, self.extra_options = arch, extra_options
    super().__init__(f"compile_hipcc_{self.arch}_{hashlib.sha256(' '.join(extra_options).encode()).hexdigest()[:8]}")
  def compile(self, src:str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".cpp") as srcf, tempfile.NamedTemporaryFile(suffix=".bc") as bcf:
      with tempfile.NamedTemporaryFile(suffix=".hsaco") as libf:
        srcf.write(src.encode())
        srcf.flush()

        rocm_path = getenv("ROCM_PATH", "/opt/rocm")
        subprocess.run(["hipcc", "-c", "-emit-llvm", "--cuda-device-only", "-O3", "-mcumode",
                        f"--offload-arch={self.arch}", f"-I{rocm_path}/include/hip", "-o", bcf.name, srcf.name] + self.extra_options, check=True)
        subprocess.run(["hipcc", "-target", "amdgcn-amd-amdhsa", f"-mcpu={self.arch}",
                        "-O3", "-mllvm", "-amdgpu-internalize-symbols", "-c", "-o", libf.name, bcf.name] + self.extra_options, check=True)

        return pathlib.Path(libf.name).read_bytes()
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)

class AMDLLVMCompiler(LLVMCompiler):
  jit = False
  def __init__(self, arch: str):
    self.arch = arch
    super().__init__("AMDGPU", self.arch, "+cumode")
  def __reduce__(self): return (AMDLLVMCompiler, (self.arch,))
  def compile(self, src:str) -> bytes:
    try: return super().compile(src)
    except RuntimeError as e:
      if "undefined value '@llvm.amdgcn." in str(e): raise CompileError(str(e) + "AMD with LLVM backend requires LLVM >= 18") from e
      raise CompileError(e) from e
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)
