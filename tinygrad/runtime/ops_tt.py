# Tenstorrent Blackhole backend for tinygrad
# Renderer generates C source for RISC-V dataflow cores, runtime dispatches via ttnn.generic_op
from __future__ import annotations
import re, math, functools, struct
from tinygrad.dtype import dtypes, PtrDType, DType
from tinygrad.device import Compiled, Compiler, LRUAllocator, BufferSpec, CompilerSet, CompilerPair, Buffer
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import Ops, UOp
from tinygrad.helpers import DEBUG, LRU

TILE_HW = 32
TILE_ELEMS = TILE_HW * TILE_HW  # 1024 elements per 32x32 tile
TILE_BYTES = TILE_ELEMS * 4      # 4096 bytes per float32 tile

def _flatten_list(nested) -> list[float]:
  """Recursively flatten a nested list from ttnn.Tensor.to_list() into a flat float list."""
  if isinstance(nested, (int, float)): return [float(nested)]
  out: list[float] = []
  for item in nested: out.extend(_flatten_list(item))
  return out

_KERNEL_PREAMBLE = r'''#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#pragma GCC push_options
#pragma GCC optimize("no-fast-math")
#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
#ifndef NAN
#define NAN (__builtin_nanf(""))
#endif
#pragma GCC pop_options
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-variable"
static inline void tile_coords(uint32_t j, uint32_t t, uint32_t *row, uint32_t *col) {
  uint32_t face = j >> 8, fr = (j & 0xFF) >> 4, fc = j & 0xF;
  *row = (face >= 2 ? 16 : 0) + fr;
  *col = t * 32 + (face & 1 ? 16 : 0) + fc;
}
static inline float __f16_round(float x) { return (float)(_Float16)x; }
template<typename T> static inline float _to_tile(T v) { float f; __builtin_memcpy(&f, &v, sizeof(float)); return f; }
template<> inline float _to_tile<float>(float v) { return v; }
template<typename T> static inline T _from_tile(float f) { T v; __builtin_memcpy(&v, &f, sizeof(T)); return v; }
'''

class TTRenderer(CStyleLanguage):
  device = "TT"
  has_local = False
  has_shared = False
  has_aux = True
  supports_float4 = False
  global_max = (0,) * 3
  local_max = (0,) * 3
  infinity = "__builtin_inff()"
  nan = '__builtin_nanf("")'
  type_map = {dtypes.half: "float", dtypes.bool: "_Bool"}
  def render_cast(self, dt, val): return f"__f16_round({val})" if dt == dtypes.half else super().render_cast(dt, val)
  code_for_op = {k:v for k,v in CStyleLanguage.code_for_op.items() if k not in [Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.TRUNC]}
  code_for_op[Ops.TRUNC] = lambda x,dtype: f"__builtin_truncf({x})"
  code_for_op[Ops.SIN] = lambda x,dtype: f"__builtin_sinf({x})"  # sin decomposition needs int64 for Payne-Hanek; TT only has int32

  def aux(self, uops: list[UOp]): return (tuple(u.dtype for u in uops if u.op == Ops.DEFINE_GLOBAL),)

  def render_kernel(self, function_name: str, kernel: list[str], bufs: list[tuple[str, tuple]], uops: list[UOp], prefix=None) -> str:
    # Extract buffer metadata from bufs - separate DEFINE_GLOBAL (PtrDType) from DEFINE_VAR (int)
    buf_meta = []  # (name, sz, mutable, ctype)
    for name, (dtype, mutable) in bufs:
      if not isinstance(dtype, PtrDType): continue
      base_dt = dtype.base
      ctype = "int" if base_dt in (dtypes.int, dtypes.int32, dtypes.int64, dtypes.uint, dtypes.uint32) else \
              ("bool" if base_dt == dtypes.bool else "float")
      buf_meta.append((name, dtype.size, mutable, ctype))

    out_name, out_n, out_ctype = None, 1, "float"
    in_bufs = []
    extra_mut_indices = []
    for name, sz, mut, ctype in buf_meta:
      if out_name is None and mut:
        out_name, out_n, out_ctype = name, sz, ctype
      else:
        if mut: extra_mut_indices.append(len(in_bufs))
        in_bufs.append((name, sz, ctype))
    if out_name is None: out_name = "data0_1"
    extra_mut_set = set(extra_mut_indices)
    immutable_indices = [i for i in range(len(in_bufs)) if i not in extra_mut_set]
    var_names = [u.arg[0] for u in uops if u.op is Ops.DEFINE_VAR]
    c_body = "\n".join(kernel)

    # Build the kernel source
    n_out_tiles = max(math.ceil(out_n / TILE_ELEMS), 1)
    out_width = n_out_tiles * TILE_HW

    src = _KERNEL_PREAMBLE
    src += '\nvoid kernel_main() __attribute__((optimize("no-fast-math")));\n'
    src += 'void kernel_main() {\n'

    # Runtime args (all inputs get addr+tiles, even immutable -- must consume in order)
    arg_idx = 0
    src += f'  uint32_t out_addr = get_arg_val<uint32_t>({arg_idx});\n'
    arg_idx += 1
    src += f'  uint32_t tile_start = get_arg_val<uint32_t>({arg_idx});\n'
    arg_idx += 1
    src += f'  uint32_t tile_end = get_arg_val<uint32_t>({arg_idx});\n'
    arg_idx += 1
    for i in range(len(in_bufs)):
      src += f'  uint32_t in{i}_addr = get_arg_val<uint32_t>({arg_idx});\n'
      arg_idx += 1
      src += f'  uint32_t in{i}_tiles = get_arg_val<uint32_t>({arg_idx});\n'
      arg_idx += 1
    for vname in var_names:
      src += f'  int {vname} = (int)get_arg_val<uint32_t>({arg_idx});\n'
      arg_idx += 1

    # Compile-time tensor accessor args
    src += '  constexpr auto out_taa = TensorAccessorArgs<0>();\n'
    for i in range(len(in_bufs)):
      src += f'  constexpr auto in{i}_taa = TensorAccessorArgs<{i + 1}>();\n'

    # Constants
    src += '  constexpr uint32_t TILE_BYTES = 4096;\n'
    src += '  constexpr uint32_t TILE_ELEMS = 1024;\n'
    src += f'  constexpr uint32_t OUT_N = {out_n};\n'
    src += f'  constexpr uint32_t OUT_W = {out_width};\n'
    for i, (name, sz, _) in enumerate(in_bufs):
      in_tiles_i = max(math.ceil(sz / TILE_ELEMS), 1)
      # Extra mutables need IN_N for bounds check, all inputs need IN_W for tile indexing
      if i in extra_mut_set: src += f'  constexpr uint32_t IN{i}_N = {sz};\n'
      src += f'  constexpr uint32_t IN{i}_W = {in_tiles_i * TILE_HW};\n'

    # L1 memory layout: all tile-sized (4K) caches -- eliminates L1 overflow for large buffers
    src += '  uint32_t l1_base = get_write_ptr(0);\n'
    offset = 0
    for i in immutable_indices:
      src += f'  float* in{i}_cache = (float*)(l1_base + {offset});\n'
      offset += TILE_BYTES
    src += f'  float* out_cache = (float*)(l1_base + {offset});\n'
    offset += TILE_BYTES
    for i in extra_mut_indices:
      src += f'  float* mut{i}_cache = (float*)(l1_base + {offset});\n'
      offset += TILE_BYTES

    # Tile cache: read-only for immutable inputs
    for i in immutable_indices:
      src += f'  uint32_t in{i}_cached = UINT32_MAX;\n'
      src += f'  auto in{i}_ta = TensorAccessor(in{i}_taa, in{i}_addr, TILE_BYTES);\n'
    for i in immutable_indices:
      src += f'  auto load_in{i} = [&](uint32_t idx) -> float {{\n'
      src += f'    uint32_t row = idx / IN{i}_W, col = idx % IN{i}_W;\n'
      src += '    uint32_t tile = col / 32;\n'
      src += f'    if (tile != in{i}_cached) {{\n'
      src += f'      noc_async_read_page(tile, in{i}_ta, (uint32_t)in{i}_cache);\n'
      src += '      noc_async_read_barrier();\n'
      src += f'      in{i}_cached = tile;\n'
      src += '    }\n'
      src += '    uint32_t lc = col % 32;\n'
      src += '    uint32_t face = (row >= 16 ? 2 : 0) + (lc >= 16 ? 1 : 0);\n'
      src += f'    return in{i}_cache[face * 256 + (row % 16) * 16 + lc % 16];\n'
      src += '  };\n'

    # Tile cache: write-back for output (zero-fill on first access, read-back after flush)
    src += f'  constexpr uint32_t N_OUT_TILES = {n_out_tiles};\n'
    src += '  uint32_t out_cached = UINT32_MAX;\n'
    src += '  uint8_t out_flushed[N_OUT_TILES];\n'
    src += '  for (uint32_t i = 0; i < N_OUT_TILES; i++) out_flushed[i] = 0;\n'
    src += '  auto out_ta = TensorAccessor(out_taa, out_addr, TILE_BYTES);\n'
    src += '  auto out_flush = [&]() {\n'
    src += '    if (out_cached != UINT32_MAX && out_cached >= tile_start && out_cached < tile_end) {\n'
    src += '      noc_async_write_page(out_cached, out_ta, (uint32_t)out_cache);\n'
    src += '      noc_async_write_barrier();\n'
    src += '      out_flushed[out_cached] = 1;\n'
    src += '    }\n'
    src += '  };\n'
    src += '  auto out_load = [&](uint32_t tile) {\n'
    src += '    out_flush();\n'
    src += '    if (out_flushed[tile]) {\n'
    src += '      noc_async_read_page(tile, out_ta, (uint32_t)out_cache);\n'
    src += '      noc_async_read_barrier();\n'
    src += '    } else {\n'
    src += '      for (uint32_t i = 0; i < TILE_ELEMS; i++) out_cache[i] = 0.0f;\n'
    src += '    }\n'
    src += '    out_cached = tile;\n'
    src += '  };\n'
    src += '  auto out_write = [&](uint32_t flat, float val) {\n'
    src += '    uint32_t row = flat / OUT_W, col = flat % OUT_W;\n'
    src += '    uint32_t tile = col / 32;\n'
    src += '    if (tile != out_cached) out_load(tile);\n'
    src += '    uint32_t lc = col % 32;\n'
    src += '    uint32_t face = (row >= 16 ? 2 : 0) + (lc >= 16 ? 1 : 0);\n'
    src += '    out_cache[face * 256 + (row % 16) * 16 + lc % 16] = val;\n'
    src += '  };\n'
    src += '  auto out_read = [&](uint32_t flat) -> float {\n'
    src += '    uint32_t row = flat / OUT_W, col = flat % OUT_W;\n'
    src += '    uint32_t tile = col / 32;\n'
    src += '    if (tile != out_cached) out_load(tile);\n'
    src += '    uint32_t lc = col % 32;\n'
    src += '    uint32_t face = (row >= 16 ? 2 : 0) + (lc >= 16 ? 1 : 0);\n'
    src += '    return out_cache[face * 256 + (row % 16) * 16 + lc % 16];\n'
    src += '  };\n'

    # Tile cache: read-write for extra mutable buffers (read from DRAM on miss, core 0 writes back)
    for i in extra_mut_indices:
      src += f'  uint32_t mut{i}_cached = UINT32_MAX;\n'
      src += f'  auto mut{i}_ta = TensorAccessor(in{i}_taa, in{i}_addr, TILE_BYTES);\n'
      src += f'  auto mut{i}_flush = [&]() {{\n'
      src += f'    if (mut{i}_cached != UINT32_MAX && tile_start == 0) {{\n'
      src += f'      noc_async_write_page(mut{i}_cached, mut{i}_ta, (uint32_t)mut{i}_cache);\n'
      src += '      noc_async_write_barrier();\n'
      src += '    }\n'
      src += '  };\n'
      src += f'  auto mut{i}_load = [&](uint32_t tile) {{\n'
      src += f'    mut{i}_flush();\n'
      src += f'    noc_async_read_page(tile, mut{i}_ta, (uint32_t)mut{i}_cache);\n'
      src += '    noc_async_read_barrier();\n'
      src += f'    mut{i}_cached = tile;\n'
      src += '  };\n'
      src += f'  auto mut{i}_write = [&](uint32_t flat, float val) {{\n'
      src += f'    uint32_t row = flat / IN{i}_W, col = flat % IN{i}_W;\n'
      src += '    uint32_t tile = col / 32;\n'
      src += f'    if (tile != mut{i}_cached) mut{i}_load(tile);\n'
      src += '    uint32_t lc = col % 32;\n'
      src += '    uint32_t face = (row >= 16 ? 2 : 0) + (lc >= 16 ? 1 : 0);\n'
      src += f'    mut{i}_cache[face * 256 + (row % 16) * 16 + lc % 16] = val;\n'
      src += '  };\n'
      src += f'  auto mut{i}_read = [&](uint32_t flat) -> float {{\n'
      src += f'    uint32_t row = flat / IN{i}_W, col = flat % IN{i}_W;\n'
      src += '    uint32_t tile = col / 32;\n'
      src += f'    if (tile != mut{i}_cached) mut{i}_load(tile);\n'
      src += '    uint32_t lc = col % 32;\n'
      src += '    uint32_t face = (row >= 16 ? 2 : 0) + (lc >= 16 ? 1 : 0);\n'
      src += f'    return mut{i}_cache[face * 256 + (row % 16) * 16 + lc % 16];\n'
      src += '  };\n'

    # Post-process computation body: rewrite buffer accesses to use tile cache functions
    def _replace_buf_refs(body, buf_name, write_fn, read_fn):
      """Replace *(buf+expr) patterns with function calls, handling arbitrary paren nesting."""
      prefix = f'*({buf_name}+'
      result, i = [], 0
      while i < len(body):
        pos = body.find(prefix, i)
        if pos == -1:
          result.append(body[i:])
          break
        result.append(body[i:pos])
        j, depth = pos + len(prefix), 1
        while j < len(body) and depth > 0:
          if body[j] == '(': depth += 1
          elif body[j] == ')': depth -= 1
          j += 1
        expr = body[pos + len(prefix):j - 1]
        wm = re.match(r'\s*=\s*([^;]+);', body[j:]) if write_fn else None
        if wm:
          result.append(write_fn(expr, wm.group(1)))
          i = j + wm.end()
        else:
          result.append(read_fn(expr))
          i = j
      return ''.join(result)

    # 1. Immutable inputs -> tile cache load calls (with type cast if non-float)
    for i in immutable_indices:
      name, _, ctype = in_bufs[i]
      if ctype == "int":
        def rfn(e, _i=i): return f'_from_tile<int>(load_in{_i}({e}))'
      elif ctype != "float":
        def rfn(e, _i=i, ct=ctype): return f'({ct})load_in{_i}({e})'
      else:
        def rfn(e, _i=i): return f'load_in{_i}({e})'
      c_body = _replace_buf_refs(c_body, name, None, rfn)
    # 2. Fix unsigned int constants that overflow float32
    def _fix_large_int(m):
      val = int(m.group(1))
      return str(val - 4294967296) if val > 2147483647 else m.group(0)
    c_body = re.sub(r'(?<!-)(\b\d{10,})u?\b', _fix_large_int, c_body)
    # 3. Output -> tile cache write/read calls (with type cast if non-float)
    if out_ctype == "int":
      c_body = _replace_buf_refs(c_body, out_name, lambda e, v: f'out_write({e}, _to_tile({v}));',
                                 lambda e: f'_from_tile<int>(out_read({e}))')
    elif out_ctype != "float":
      c_body = _replace_buf_refs(c_body, out_name, lambda e, v: f'out_write({e}, (float)({v}));',
                                 lambda e: f'({out_ctype})out_read({e})')
    else:
      c_body = _replace_buf_refs(c_body, out_name, lambda e, v: f'out_write({e}, {v});', lambda e: f'out_read({e})')
    # 4. Extra mutable -> tile cache write/read calls (with type cast if non-float)
    for i in extra_mut_indices:
      name, _, ctype = in_bufs[i]
      if ctype == "int":
        c_body = _replace_buf_refs(c_body, name, lambda e, v, _i=i: f'mut{_i}_write({e}, _to_tile({v}));',
                                   lambda e, _i=i: f'_from_tile<int>(mut{_i}_read({e}))')
      elif ctype != "float":
        c_body = _replace_buf_refs(c_body, name, lambda e, v, _i=i: f'mut{_i}_write({e}, (float)({v}));',
                                   lambda e, _i=i, ct=ctype: f'({ct})mut{_i}_read({e})')
      else:
        c_body = _replace_buf_refs(c_body, name, lambda e, v, _i=i: f'mut{_i}_write({e}, {v});',
                                   lambda e, _i=i: f'mut{_i}_read({e})')

    src += c_body + '\n'

    # Flush all write-back caches
    src += '  out_flush();\n'
    for i in extra_mut_indices:
      src += f'  mut{i}_flush();\n'

    src += '}\n'
    # CB = total tile cache buffers, always small (eliminates L1 overflow)
    cb_total = max(offset, 2 * TILE_BYTES)
    return f'// TT_CB={cb_total}\n' + src

class TTCompiler(Compiler):
  def compile(self, src: str) -> bytes: return src.encode()

class TTProgram:
  def __init__(self, dev: TTDevice, name: str, lib: bytes, buf_dtypes=(), **kwargs):
    self.dev, self.name = dev, name
    self.src = lib.decode()
    # buf_dtypes from aux(): tuple of PtrDType, one per DEFINE_GLOBAL, ordered by arg index
    # buf_dtypes[0] is always output, buf_dtypes[1:] are inputs
    self.buf_dtypes = buf_dtypes
    self.out_n = buf_dtypes[0].size if buf_dtypes else 1
    self.in_sizes = [dt.size for dt in buf_dtypes[1:]] if len(buf_dtypes) > 1 else []
    # Parse CB requirement from kernel source (embedded by render_kernel)
    m = re.search(r'TT_CB=(\d+)', self.src)
    self.cb_total = int(m.group(1)) if m else None

  def _tag_buf_dtypes(self, bufs):
    allocator = self.dev.allocator
    for i, buf in enumerate(bufs):
      if i < len(self.buf_dtypes):
        allocator._buf_dtypes[buf[0].buffer_address()] = self.buf_dtypes[i].base

  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
    self._tag_buf_dtypes(bufs)
    return self._run_generic(bufs, vals)

  def _run_generic(self, bufs, vals):
    import ttnn, time
    st = time.perf_counter()
    device = self.dev.ttnn_device
    out_n = self.out_n
    n_out_tiles = max(math.ceil(out_n / TILE_ELEMS), 1)
    out_t = bufs[0][0]

    # Multi-core: distribute output tiles across all Tensix cores
    grid = device.compute_with_storage_grid_size()
    max_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    (_, core_grid, grp1, grp2, work1, work2) = ttnn.split_work_to_cores(max_cores, n_out_tiles)

    cbs = [ttnn.CBDescriptor(total_size=self.cb_total, core_ranges=core_grid,
           format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=TILE_BYTES)])]

    ct_args = list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args())
    io_tensors = []
    for i in range(len(self.in_sizes)):
      in_t = bufs[i + 1][0]
      ct_args.extend(list(ttnn.TensorAccessorArgs(in_t).get_compile_time_args()))
      io_tensors.append(in_t)

    # Base runtime args (same for all cores): out_addr, then per-input addr+tiles, then var vals
    out_addr = out_t.buffer_address()
    input_rt = []
    for i, sz in enumerate(self.in_sizes):
      in_t = bufs[i + 1][0]
      input_rt.extend([in_t.buffer_address(), max(math.ceil(sz / TILE_ELEMS), 1)])
    for v in vals:
      if v is not None: input_rt.append(int(v))

    # Per-core runtime args with tile ranges
    writer_rt = ttnn.RuntimeArgs()
    tile_idx = 0
    for cr in grp1.ranges():
      for x in range(cr.start.x, cr.end.x + 1):
        for y in range(cr.start.y, cr.end.y + 1):
          writer_rt[x][y] = [out_addr, tile_idx, tile_idx + work1] + input_rt
          tile_idx += work1
    for cr in grp2.ranges():
      for x in range(cr.start.x, cr.end.x + 1):
        for y in range(cr.start.y, cr.end.y + 1):
          writer_rt[x][y] = [out_addr, tile_idx, tile_idx + work2] + input_rt
          tile_idx += work2

    if not io_tensors: io_tensors.append(out_t)
    io_tensors.append(out_t)

    writer_k = ttnn.KernelDescriptor(kernel_source=self.src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                     core_ranges=core_grid, compile_time_args=ct_args, runtime_args=writer_rt,
                                     config=ttnn.WriterConfigDescriptor())

    prog = ttnn.ProgramDescriptor(kernels=[writer_k], semaphores=[], cbs=cbs)
    if DEBUG >= 3: print(f"TT: generic n_in={len(self.in_sizes)} out_n={out_n} tiles={n_out_tiles} cores={grid.x}x{grid.y}")
    if DEBUG >= 5: print(f"TT: source:\n{self.src}")
    ttnn.generic_op(io_tensors, prog)

    return time.perf_counter() - st

class TTDevice(Compiled):
  device_handle = None
  _buffer_patched = False

  def __init__(self, device: str):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    if TTDevice.device_handle is None:
      import ttnn
      TTDevice.device_handle = ttnn.open_device(device_id=self.device_id)
      if DEBUG >= 1: print(f"TT: opened device {self.device_id}")
    self.ttnn_device = TTDevice.device_handle
    # Patch Buffer to pass dtype to TTAllocator via side channel (only once)
    if not TTDevice._buffer_patched:
      _orig_copyin, _orig_copyout, _orig_allocate = Buffer.copyin, Buffer.copyout, Buffer.allocate
      def _tt_allocate(buf_self, opaque=None, external_ptr=None):
        if buf_self.device.startswith("TT"):
          from tinygrad.device import Device as _Dev
          _Dev[buf_self.device].allocator._alloc_dtype = buf_self.dtype.base
        return _orig_allocate(buf_self, opaque, external_ptr)
      def _tt_copyin(buf_self, mv):
        if buf_self.device.startswith("TT"): buf_self.allocator._copyin_dtype = buf_self.dtype.base
        return _orig_copyin(buf_self, mv)
      def _tt_copyout(buf_self, mv):
        if buf_self.device.startswith("TT"): buf_self.allocator._copyout_dtype = buf_self.dtype.base
        return _orig_copyout(buf_self, mv)
      Buffer.allocate, Buffer.copyin, Buffer.copyout = _tt_allocate, _tt_copyin, _tt_copyout
      TTDevice._buffer_patched = True
    super().__init__(device, TTAllocator(self), CompilerSet([CompilerPair(TTRenderer, TTCompiler)]),
                     functools.partial(TTProgram, self))

  def synchronize(self):
    import ttnn
    ttnn.synchronize_device(self.ttnn_device)

  def finalize(self):
    if TTDevice.device_handle is not None:
      import ttnn
      ttnn.close_device(TTDevice.device_handle)
      TTDevice.device_handle = None
      if DEBUG >= 1: print(f"TT: closed device {self.device_id}")

class TTAllocator(LRUAllocator['TTDevice']):
  def __init__(self, dev):
    super().__init__(dev)
    self._buf_dtypes: dict[int, 'DType'] = {}  # buffer_address -> base dtype
    self._alloc_dtype: DType|None = None  # side channel: set by patched Buffer.allocate before _alloc
    self._copyin_dtype: DType|None = None  # side channel: set by patched Buffer.copyin before _copyin
    self._copyout_dtype: DType|None = None  # side channel: set by patched Buffer.copyout before _copyout

  def alloc(self, size: int, options: BufferSpec|None=None):
    buf_dt = self._alloc_dtype or dtypes.float32
    n_elements = max(size // buf_dt.itemsize, 1)
    if len(c := self.cache[(n_elements, options)]): return c.pop()
    try: return self._alloc(size, options if options is not None else self.default_buffer_spec)
    except (RuntimeError, MemoryError):
      self.free_cache()
      return self._alloc(size, options if options is not None else self.default_buffer_spec)

  def free(self, opaque, size: int, options: BufferSpec|None=None):
    if LRU and (options is None or not options.nolru): self.cache[(opaque[2], options)].append(opaque)
    else: self._free(opaque, options if options is not None else self.default_buffer_spec)

  def _alloc(self, size: int, options: BufferSpec):
    import ttnn
    buf_dt = self._alloc_dtype or dtypes.float32
    self._alloc_dtype = None
    n_elements = max(size // buf_dt.itemsize, 1)
    num_tiles = max(math.ceil(n_elements / TILE_ELEMS), 1)
    shape = [1, 1, TILE_HW, num_tiles * TILE_HW]
    t = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.float32, ttnn.TILE_LAYOUT, self.dev.ttnn_device, ttnn.DRAM_MEMORY_CONFIG)
    return (t, size, n_elements)

  def _free(self, opaque, options: BufferSpec):
    import ttnn
    self._buf_dtypes.pop(opaque[0].buffer_address(), None)
    ttnn.deallocate(opaque[0])

  def _copyin(self, dest, src: memoryview):
    import ttnn
    t_dev, size, n_elements = dest
    num_tiles = max(math.ceil(n_elements / TILE_ELEMS), 1)
    raw_bytes = bytes(src)
    nbytes = len(raw_bytes)
    total = num_tiles * TILE_ELEMS
    addr = t_dev.buffer_address()

    # Get dtype from side channel (set by patched Buffer.copyin), kernel aux tag, or default to float
    buf_dt = self._copyin_dtype or self._buf_dtypes.get(addr, dtypes.float32)
    self._copyin_dtype = None
    self._buf_dtypes[addr] = buf_dt

    if buf_dt == dtypes.bool:
      float_vals = [float(b) for b in raw_bytes]
      if len(float_vals) > total: float_vals = float_vals[:total]
    elif buf_dt == dtypes.half:
      n_f16 = nbytes // 2
      float_vals = [float(v) for v in struct.unpack(f'<{n_f16}e', raw_bytes[:n_f16 * 2])]
    elif buf_dt in (dtypes.int, dtypes.int32, dtypes.int64, dtypes.uint, dtypes.uint32):
      n_i32 = nbytes // 4
      float_vals = list(struct.unpack(f'<{n_i32}f', raw_bytes[:n_i32 * 4]))
    else:
      n_f32 = nbytes // 4
      float_vals = list(struct.unpack(f'<{n_f32}f', raw_bytes[:n_f32 * 4])) if n_f32 > 0 else []

    padded = float_vals + [0.0] * (total - len(float_vals))
    host_t = ttnn.Tensor(padded, [1, 1, TILE_HW, num_tiles * TILE_HW], ttnn.float32, ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(host_t, t_dev)

  def _copyout(self, dest: memoryview, src):
    import ttnn
    t_dev, size, n_elements = src
    addr = t_dev.buffer_address()
    buf_dt = self._copyout_dtype or self._buf_dtypes.get(addr, dtypes.float32)
    self._copyout_dtype = None
    host_t = t_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    flat = _flatten_list(host_t.to_list())

    if buf_dt == dtypes.bool:
      bool_bytes = bytes(1 if v > 0.5 else 0 for v in flat[:size])
      dest[:] = bool_bytes[:len(dest)]
    elif buf_dt == dtypes.half:
      n_half = len(dest) // 2
      vals = flat[:n_half]
      dest[:] = struct.pack(f'<{len(vals)}e', *vals)
    elif buf_dt in (dtypes.int, dtypes.int32, dtypes.int64, dtypes.uint, dtypes.uint32):
      n_vals = min(len(flat), n_elements)
      dest[:] = struct.pack(f'<{n_vals}f', *flat[:n_vals])[:len(dest)]
    else:
      dest[:] = struct.pack(f'<{len(flat[:n_elements])}f', *flat[:n_elements])[:len(dest)]
