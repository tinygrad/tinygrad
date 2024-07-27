import subprocess, hashlib, tempfile, ctypes, ctypes.util, re
from pathlib import Path
from tinygrad.helpers import to_char_p_p, colored, init_c_var, getenv
import tinygrad.runtime.autogen.nvrtc as nvrtc
from tinygrad.device import Compiler, CompileError

PTX = getenv("PTX")  # this shouldn't be here, in fact, it shouldn't exist

def _get_bytes(arg, get_str, get_sz, check) -> bytes:
  sz = init_c_var(ctypes.c_size_t(), lambda x: check(get_sz(arg, ctypes.byref(x))))
  return ctypes.string_at(init_c_var(ctypes.create_string_buffer(sz.value), lambda x: check(get_str(arg, x))), size=sz.value)

def nvrtc_check(status, ctx=None):
  if status != 0:
    err_log = _get_bytes(ctx, nvrtc.nvrtcGetProgramLog, nvrtc.nvrtcGetProgramLogSize, lambda _: None).decode() if ctx else ""
    raise CompileError(f"Nvrtc Error {status}, {ctypes.string_at(nvrtc.nvrtcGetErrorString(status)).decode()}\n{err_log}")

def jitlink_check(status, ctx=None):
  if status != 0:
    err_log = _get_bytes(ctx, nvrtc.nvJitLinkGetErrorLog, nvrtc.nvJitLinkGetErrorLogSize, lambda _: None).decode() if ctx else ""
    raise CompileError(f"NvJitLink Error {status}, {nvrtc.nvJitLinkResult__enumvalues.get(status, 'Unknown')}\n{err_log}")

def pretty_ptx(s):
  # all expressions match `<valid_before><expr><valid_after>` and replace it with `<valid_before>color(<expr>)<valid_after>`
  s = re.sub(r'([!@<\[\s,\+\-;\n])((?:[_%$][\w%\$_]+(?:\.[xyz])?\:?)|(?:buf\d+))([<>\]\s,\+\-;\n\)])', lambda m:m[1]+colored(m[2], "blue")+m[3], s, flags=re.M) # identifiers  # noqa: E501
  s = re.sub(r'(.)((?:b|s|u|f)(?:8|16|32|64)|pred)([\.\s])', lambda m:m[1]+colored(m[2], "green")+m[3], s, flags=re.M) # types
  s = re.sub(r'^(\s*)([\w]+)(.*?;$)', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # instructions
  s = re.sub(r'([<>\[\]\s,\+\-;])((?:0[fF][0-9a-fA-F]{8})|(?:[0-9]+)|(?:0[xX][0-9a-fA-F]+))([<>\[\]\s,\+\-;])', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # numbers  # noqa: E501
  s = re.sub(r'(\.)(param|reg|global)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # space
  s = re.sub(r'(\.)(version|target|address_size|visible|entry)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # derivatives
  return s

def cuda_disassemble(lib, arch):
  try:
    fn = (Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
    with open(fn + ".ptx", "wb") as f: f.write(lib)
    subprocess.run(["ptxas", f"-arch={arch}", "-o", fn, fn+".ptx"], check=True)
    print(subprocess.check_output(['nvdisasm', fn]).decode('utf-8'))
  except Exception as e: print("Failed to generate SASS", str(e), "Make sure your PATH contains ptxas/nvdisasm binary of compatible version.")

def nv_disassemble(lib):
  try:
    fn = (Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
    with open(fn + ".cubin", "wb") as f: f.write(lib)
    print(subprocess.check_output(["nvdisasm", fn+".cubin"]).decode('utf-8'))
  except Exception as e: print("Failed to disasm cubin:", str(e), "Make sure your PATH contains nvdisasm binary of compatible version.")

class PTXCompiler(Compiler):
  def __init__(self, arch:str):
    self.arch = arch
    self.version = "7.8" if arch >= "sm_89" else "7.5"
    super().__init__(f"compile_ptx_{self.arch}")
  def compile(self, src:str) -> bytes: return src.replace("TARGET", self.arch).replace("VERSION", self.version).encode()

class CUDACompiler(Compiler):
  def __init__(self, arch:str):
    self.arch = arch
    nvrtc_check(nvrtc.nvrtcVersion((nvrtcMajor := ctypes.c_int()), (nvrtcMinor := ctypes.c_int())))
    self.compile_options = [f'--gpu-architecture={arch}', "-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include/"]
    if (nvrtcMajor.value, nvrtcMinor.value) >= (12, 4): self.compile_options.append("--minimal")
    super().__init__(f"compile_cuda_{self.arch}")
  def compile(self, src:str) -> bytes:
    nvrtc_check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog := nvrtc.nvrtcProgram()), src.encode(), "<null>".encode(), 0, None, None))
    nvrtc_check(nvrtc.nvrtcCompileProgram(prog, len(self.compile_options), to_char_p_p([o.encode() for o in self.compile_options])), prog)
    return _get_bytes(prog, nvrtc.nvrtcGetPTX, nvrtc.nvrtcGetPTXSize, nvrtc_check)

class NVCompiler(Compiler):
  def __init__(self, arch:str):
    self.arch, self.compile_options = arch, [f'--gpu-architecture={arch}', "-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include/"]
    nvrtc_check(nvrtc.nvrtcVersion((nvrtcMajor := ctypes.c_int()), (nvrtcMinor := ctypes.c_int())))
    if (nvrtcMajor.value, nvrtcMinor.value) >= (12, 4): self.compile_options.append("--minimal")
    super().__init__(f"compile_nv_{self.arch}")
  def compile(self, src:str) -> bytes:
    nvrtc_check(nvrtc.nvrtcCreateProgram(ctypes.byref(prog := nvrtc.nvrtcProgram()), src.encode(), "<null>".encode(), 0, None, None))
    nvrtc_check(nvrtc.nvrtcCompileProgram(prog, len(self.compile_options), to_char_p_p([o.encode() for o in self.compile_options])), prog)
    return _get_bytes(prog, nvrtc.nvrtcGetCUBIN, nvrtc.nvrtcGetCUBINSize, nvrtc_check)

class NVPTXCompiler(NVCompiler):
  def compile(self, src:str) -> bytes:
    ptxsrc = src.replace("TARGET", self.arch).replace("VERSION", "7.8" if self.arch >= "sm_89" else "7.5")
    jitlink_check(nvrtc.nvJitLinkCreate(handle := nvrtc.nvJitLinkHandle(), 1, to_char_p_p([f'-arch={self.arch}'.encode()])), handle)
    jitlink_check(nvrtc.nvJitLinkAddData(handle, nvrtc.NVJITLINK_INPUT_PTX, ptxsrc.encode(), len(ptxsrc), "<null>".encode()), handle)
    jitlink_check(nvrtc.nvJitLinkComplete(handle), handle)
    return _get_bytes(handle, nvrtc.nvJitLinkGetLinkedCubin, nvrtc.nvJitLinkGetLinkedCubinSize, jitlink_check)
