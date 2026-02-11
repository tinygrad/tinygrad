from __future__ import annotations
import ctypes, functools, platform, queue, re, struct, time
from dataclasses import dataclass
from typing import Any, cast
from tinygrad.device import Compiler, CompilerSet
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.helpers import DEBUG, getenv, to_mv
from tinygrad.renderer.cstyle import ClangJITRenderer
from tinygrad.runtime.ops_cpu import CPUAllocator, CPUComputeQueue, CPUProgram, CPUSignal, CPUWorker
from tinygrad.runtime.support.compiler_cpu import ClangJITCompiler
from tinygrad.runtime.support.hcq import HCQBuffer, HCQCompiled

_RKNN_META_MAGIC = b"RKNNJIT1"
_RKNN_META = struct.Struct("<8sBIII")

RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_NPU_CORE_AUTO = 0
RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256


@dataclass(frozen=True)
class MatmulHint:
  m: int
  n: int
  k: int


def _parse_matmul_hint(src:str) -> MatmulHint|None:
  m = re.search(r"void\\s+r_(\\d+)_(\\d+)_(\\d+)\\s*\\(([^)]*)\\)", src)
  if m is None: return None
  params = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\\s*\\*\\s*restrict\\s+data\\d+_\\d+", m.group(4))
  if len(params) < 3 or params[:3] != ["float", "float", "float"]: return None
  m_sz, n_sz, k_sz = [int(m.group(i)) for i in (1, 2, 3)]
  return MatmulHint(m_sz, n_sz, k_sz)


def _pack_rknn_lib(lib:bytes, hint:MatmulHint|None) -> bytes:
  if hint is None: return _RKNN_META.pack(_RKNN_META_MAGIC, 0, 0, 0, 0) + lib
  return _RKNN_META.pack(_RKNN_META_MAGIC, 1, hint.m, hint.n, hint.k) + lib


def _unpack_rknn_lib(lib:bytes) -> tuple[MatmulHint|None, bytes]:
  if len(lib) < _RKNN_META.size: return None, lib
  magic, flags, m_sz, n_sz, k_sz = _RKNN_META.unpack(lib[:_RKNN_META.size])
  if magic != _RKNN_META_MAGIC: return None, lib
  hint = MatmulHint(m_sz, n_sz, k_sz) if (flags & 0x1) else None
  return hint, lib[_RKNN_META.size:]


class RKNNClangCompiler(Compiler):
  def __init__(self):
    super().__init__("compile_rknn_clang_jit")
    self.base = ClangJITCompiler(cachekey=None)

  def compile(self, src:str) -> bytes:
    return _pack_rknn_lib(self.base.compile(src), _parse_matmul_hint(src))

  def disassemble(self, lib:bytes):
    self.base.disassemble(_unpack_rknn_lib(lib)[1])


class RKNNRenderer(ClangJITRenderer):
  device = "RKNN"

  def __init__(self):
    super().__init__()
    self.compiler = RKNNClangCompiler()


class RknnTensorMem(ctypes.Structure):
  _fields_ = [
    ("virt_addr", ctypes.c_void_p),
    ("phys_addr", ctypes.c_uint64),
    ("fd", ctypes.c_int32),
    ("offset", ctypes.c_int32),
    ("size", ctypes.c_uint32),
    ("flags", ctypes.c_uint32),
    ("priv_data", ctypes.c_void_p),
  ]


class RknnMatmulTensorAttr(ctypes.Structure):
  _fields_ = [
    ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
    ("n_dims", ctypes.c_uint32),
    ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
    ("size", ctypes.c_uint32),
    ("type", ctypes.c_int32),
  ]


class RknnMatmulIoAttr(ctypes.Structure):
  _fields_ = [("A", RknnMatmulTensorAttr), ("B", RknnMatmulTensorAttr), ("C", RknnMatmulTensorAttr)]


class RknnMatmulInfo(ctypes.Structure):
  _fields_ = [
    ("M", ctypes.c_int32),
    ("K", ctypes.c_int32),
    ("N", ctypes.c_int32),
    ("type", ctypes.c_int32),
    ("native_layout", ctypes.c_int32),
    ("perf_layout", ctypes.c_int32),
  ]


class RKNNRuntime:
  def __init__(self):
    self.lib_path = getenv("RKNN_LIB", "librknnrt.so")
    self.offload = getenv("RKNN_OFFLOAD", 0) == 1
    self.require_lib = getenv("RKNN_REQUIRE_LIB", 0) == 1
    self.stats = {
      "matmul_attempts": 0,
      "matmul_run_calls": 0,
      "matmul_success": 0,
      "matmul_failures": 0,
      "fallback_calls": 0,
    }
    self._tried = False
    self._ready = False
    self._error: str|None = None
    self.lib: Any = None
    self.rknn_matmul_create: Any = None
    self.rknn_matmul_set_io_mem: Any = None
    self.rknn_matmul_set_core_mask: Any = None
    self.rknn_matmul_run: Any = None
    self.rknn_matmul_destroy: Any = None
    self.rknn_create_mem: Any = None
    self.rknn_destroy_mem: Any = None
    if self.require_lib: self.require_ready()

  def _host_ok(self) -> bool:
    return platform.system() == "Linux" and platform.machine().lower() in {"aarch64", "arm64"}

  def _load_sym(self, name:str, required=True):
    try: return getattr(self.lib, name)
    except AttributeError:
      if required: raise
      return None

  def _bind(self):
    self.rknn_matmul_create = self._load_sym("rknn_matmul_create")
    self.rknn_matmul_create.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(RknnMatmulInfo), ctypes.POINTER(RknnMatmulIoAttr)]
    self.rknn_matmul_create.restype = ctypes.c_int

    self.rknn_matmul_set_io_mem = self._load_sym("rknn_matmul_set_io_mem")
    self.rknn_matmul_set_io_mem.argtypes = [ctypes.c_uint64, ctypes.POINTER(RknnTensorMem), ctypes.POINTER(RknnMatmulTensorAttr)]
    self.rknn_matmul_set_io_mem.restype = ctypes.c_int

    self.rknn_matmul_set_core_mask = self._load_sym("rknn_matmul_set_core_mask", required=False)
    if self.rknn_matmul_set_core_mask is not None:
      self.rknn_matmul_set_core_mask.argtypes = [ctypes.c_uint64, ctypes.c_int]
      self.rknn_matmul_set_core_mask.restype = ctypes.c_int

    self.rknn_matmul_run = self._load_sym("rknn_matmul_run")
    self.rknn_matmul_run.argtypes = [ctypes.c_uint64]
    self.rknn_matmul_run.restype = ctypes.c_int

    self.rknn_matmul_destroy = self._load_sym("rknn_matmul_destroy")
    self.rknn_matmul_destroy.argtypes = [ctypes.c_uint64]
    self.rknn_matmul_destroy.restype = ctypes.c_int

    self.rknn_create_mem = self._load_sym("rknn_create_mem")
    self.rknn_create_mem.argtypes = [ctypes.c_uint64, ctypes.c_uint32]
    self.rknn_create_mem.restype = ctypes.POINTER(RknnTensorMem)

    self.rknn_destroy_mem = self._load_sym("rknn_destroy_mem")
    self.rknn_destroy_mem.argtypes = [ctypes.c_uint64, ctypes.POINTER(RknnTensorMem)]
    self.rknn_destroy_mem.restype = ctypes.c_int

  def _ensure_ready(self) -> bool:
    if self._ready: return True
    if self._tried: return False
    self._tried = True

    if not (self.offload or self.require_lib):
      self._error = "RKNN offload disabled"
      return False

    if not self._host_ok():
      self._error = "RKNN offload requires Linux on AArch64"
      return False

    try: self.lib = ctypes.CDLL(self.lib_path)
    except OSError as e:
      self._error = f"unable to load {self.lib_path}: {e}"
      return False

    try: self._bind()
    except Exception as e:
      self._error = f"failed to bind RKNN symbols: {e}"
      return False

    self._ready = True
    return True

  def require_ready(self):
    if self._ensure_ready(): return
    raise RuntimeError(self._error or "RKNN runtime unavailable")

  def can_offload(self) -> bool:
    return self.offload and self._ensure_ready()

  def _destroy_mem(self, ctx:ctypes.c_uint64, mem:ctypes.POINTER(RknnTensorMem)|None):
    if mem is not None: self.rknn_destroy_mem(ctx.value, mem)

  def try_matmul_f32(self, out:HCQBuffer, a:HCQBuffer, b:HCQBuffer, hint:MatmulHint) -> bool:
    if not self.can_offload(): return False

    import numpy as np

    self.stats["matmul_attempts"] += 1
    success = False
    ctx = ctypes.c_uint64(0)
    io_attr = RknnMatmulIoAttr()
    info = RknnMatmulInfo(M=hint.m, K=hint.k, N=hint.n, type=RKNN_TENSOR_FLOAT16, native_layout=0, perf_layout=0)

    a_mem: ctypes.POINTER(RknnTensorMem)|None = None
    b_mem: ctypes.POINTER(RknnTensorMem)|None = None
    c_mem: ctypes.POINTER(RknnTensorMem)|None = None

    try:
      if self.rknn_matmul_create(ctypes.byref(ctx), ctypes.byref(info), ctypes.byref(io_attr)) != 0: return False
      if self.rknn_matmul_set_core_mask is not None: self.rknn_matmul_set_core_mask(ctx.value, RKNN_NPU_CORE_AUTO)

      a_mem = self.rknn_create_mem(ctx.value, io_attr.A.size)
      b_mem = self.rknn_create_mem(ctx.value, io_attr.B.size)
      c_mem = self.rknn_create_mem(ctx.value, io_attr.C.size)
      if a_mem is None or b_mem is None or c_mem is None: return False

      a_src = np.frombuffer(to_mv(int(a.va_addr), a.size), dtype=np.float32, count=hint.m*hint.k).astype(np.float16, copy=False).tobytes()
      b_src = np.frombuffer(to_mv(int(b.va_addr), b.size), dtype=np.float32, count=hint.k*hint.n).astype(np.float16, copy=False).tobytes()

      if len(a_src) > a_mem.contents.size or len(b_src) > b_mem.contents.size: return False

      a_dst = to_mv(cast(int, a_mem.contents.virt_addr), a_mem.contents.size).cast('B')
      b_dst = to_mv(cast(int, b_mem.contents.virt_addr), b_mem.contents.size).cast('B')
      a_dst[:len(a_src)] = a_src
      b_dst[:len(b_src)] = b_src
      if len(a_src) < len(a_dst): a_dst[len(a_src):] = b"\x00" * (len(a_dst) - len(a_src))
      if len(b_src) < len(b_dst): b_dst[len(b_src):] = b"\x00" * (len(b_dst) - len(b_src))

      if self.rknn_matmul_set_io_mem(ctx.value, a_mem, ctypes.byref(io_attr.A)) != 0: return False
      if self.rknn_matmul_set_io_mem(ctx.value, b_mem, ctypes.byref(io_attr.B)) != 0: return False
      if self.rknn_matmul_set_io_mem(ctx.value, c_mem, ctypes.byref(io_attr.C)) != 0: return False

      self.stats["matmul_run_calls"] += 1
      if self.rknn_matmul_run(ctx.value) != 0: return False

      out_dst = to_mv(int(out.va_addr), out.size).cast('B')
      c_src = to_mv(cast(int, c_mem.contents.virt_addr), c_mem.contents.size).cast('B')
      out_dst[:min(len(out_dst), len(c_src))] = c_src[:min(len(out_dst), len(c_src))]
      self.stats["matmul_success"] += 1
      success = True
      return True
    except Exception as e:
      if DEBUG >= 2: print(f"RKNN matmul offload failed: {e}")
      return False
    finally:
      self._destroy_mem(ctx, a_mem)
      self._destroy_mem(ctx, b_mem)
      self._destroy_mem(ctx, c_mem)
      if ctx.value != 0: self.rknn_matmul_destroy(ctx.value)
      if not success: self.stats["matmul_failures"] += 1


class RKNNProgram(CPUProgram):
  def __init__(self, dev:RKNNDevice, name:str, lib:bytes, buf_dtypes=(), runtimevars:dict[str, int]|None=None, **kwargs):
    self.dev: RKNNDevice = dev
    self.buf_dtypes = tuple(buf_dtypes)
    self.matmul_hint, lib = _unpack_rknn_lib(lib)
    super().__init__(dev, name, lib, runtimevars=runtimevars, **kwargs)

  def _eligible_matmul(self, bufs:tuple[HCQBuffer, ...]) -> bool:
    if self.matmul_hint is None or len(bufs) != 3 or len(self.buf_dtypes) < 3: return False
    if not all(isinstance(dt, PtrDType) for dt in self.buf_dtypes[:3]): return False
    if any(cast(PtrDType, dt).base != dtypes.float for dt in self.buf_dtypes[:3]): return False
    out_sz, a_sz, b_sz = bufs[0].size, bufs[1].size, bufs[2].size
    m, n, k = self.matmul_hint.m, self.matmul_hint.n, self.matmul_hint.k
    return out_sz == m*n*4 and a_sz == m*k*4 and b_sz == k*n*4

  def __call__(self, *bufs:HCQBuffer, global_size:tuple[int, int, int]=(1, 1, 1), local_size:tuple[int, int, int]=(1, 1, 1),
               vals:tuple[int|None, ...]=(), wait=False) -> float|None:
    if self._eligible_matmul(cast(tuple[HCQBuffer, ...], bufs)):
      st = time.perf_counter()
      if self.dev.rknn.try_matmul_f32(bufs[0], bufs[1], bufs[2], cast(MatmulHint, self.matmul_hint)):
        return (time.perf_counter() - st) if wait else None
      self.dev.rknn.stats["fallback_calls"] += 1
    return super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)


class RKNNDevice(HCQCompiled):
  def __init__(self, device:str=""):
    self.tasks: queue.Queue = queue.Queue()
    CPUWorker(self, self.tasks, thread_id=0).start()
    self.rknn = RKNNRuntime()
    if self.rknn.require_lib: self.rknn.require_ready()
    super().__init__(device, CPUAllocator(self), CompilerSet([(RKNNRenderer, None)]), functools.partial(RKNNProgram, self), CPUSignal,
                     CPUComputeQueue)
