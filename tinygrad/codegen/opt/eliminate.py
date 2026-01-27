# pure elimination kernel solver for Metal - stack constraints until |R| = 1
from dataclasses import dataclass
from typing import Iterator
from tinygrad.helpers import getenv

@dataclass(frozen=True)
class MetalHardware:
  simd_width: int = 32
  max_threads_per_tg: int = 1024
  shared_memory_bytes: int = 32768
  max_registers_per_simd: int = 128
  memory_bandwidth_gbps: float = 200.0
  peak_tflops: float = 2.6
  @property
  def bytes_per_flop(self) -> float: return (self.memory_bandwidth_gbps * 1e9) / (self.peak_tflops * 1e12)

@dataclass(frozen=True)
class MatmulConfig:
  tile_m: int
  tile_n: int
  tile_k: int
  threads_m: int
  threads_n: int
  @property
  def threads_total(self) -> int: return self.threads_m * self.threads_n
  @property
  def elements_per_thread(self) -> int: return (self.tile_m // self.threads_m) * (self.tile_n // self.threads_n)
  @property
  def shared_memory_bytes(self) -> int: return (self.tile_m * self.tile_k + self.tile_k * self.tile_n) * 4
  @property
  def registers_per_thread(self) -> int: return self.elements_per_thread + 4
  @property
  def arithmetic_intensity(self) -> float:
    flops = 2 * self.tile_m * self.tile_n * self.tile_k
    bytes_loaded = (self.tile_m * self.tile_k + self.tile_k * self.tile_n) * 4
    return flops / bytes_loaded if bytes_loaded > 0 else 0

def generate_omega(max_tile: int = 128, tile_step: int = 8) -> Iterator[MatmulConfig]:
  tiles = list(range(tile_step, max_tile + 1, tile_step))
  thread_counts = [1, 2, 4, 8, 16, 32]
  for tile_m in tiles:
    for tile_n in tiles:
      for tile_k in tiles:
        for threads_m in thread_counts:
          for threads_n in thread_counts:
            if tile_m % threads_m != 0 or tile_n % threads_n != 0: continue
            yield MatmulConfig(tile_m, tile_n, tile_k, threads_m, threads_n)

# eliminators return True if config survives
def elim_threads(c: MatmulConfig, hw: MetalHardware) -> bool:
  if c.threads_total > hw.max_threads_per_tg: return False
  if c.threads_total % hw.simd_width != 0 and c.threads_total > hw.simd_width: return False
  return True

def elim_shared_mem(c: MatmulConfig, hw: MetalHardware) -> bool: return c.shared_memory_bytes <= hw.shared_memory_bytes
def elim_registers(c: MatmulConfig, hw: MetalHardware) -> bool: return c.registers_per_thread <= hw.max_registers_per_simd // 2
def elim_parallelism(c: MatmulConfig, hw: MetalHardware) -> bool: return c.threads_total >= hw.simd_width * 2
def elim_ai(c: MatmulConfig, hw: MetalHardware) -> bool: return c.arithmetic_intensity >= 1.0 / hw.bytes_per_flop
def elim_simd(c: MatmulConfig, hw: MetalHardware) -> bool: return c.threads_n == hw.simd_width or c.threads_m >= 4
def elim_balance(c: MatmulConfig, hw: MetalHardware) -> bool: return max(c.tile_m, c.tile_n) / min(c.tile_m, c.tile_n) <= 4
def elim_tiny(c: MatmulConfig, hw: MetalHardware) -> bool: return c.tile_m >= 16 and c.tile_n >= 16 and c.tile_k >= 8
def elim_ilp(c: MatmulConfig, hw: MetalHardware) -> bool: return c.elements_per_thread <= 64
def elim_coalesce(c: MatmulConfig, hw: MetalHardware) -> bool: return c.threads_n == hw.simd_width
def elim_tile_k(c: MatmulConfig, hw: MetalHardware) -> bool: return c.tile_k <= 16
def elim_vec(c: MatmulConfig, hw: MetalHardware) -> bool: return c.tile_k % 4 == 0
def elim_occupancy(c: MatmulConfig, hw: MetalHardware) -> bool: return 4 <= c.elements_per_thread <= 32
def elim_pow2_threads(c: MatmulConfig, hw: MetalHardware) -> bool:
  return (c.threads_m & (c.threads_m-1)) == 0 and (c.threads_n & (c.threads_n-1)) == 0
def elim_thread_count(c: MatmulConfig, hw: MetalHardware) -> bool: return c.threads_total in [256, 512]
def elim_square(c: MatmulConfig, hw: MetalHardware) -> bool: return max(c.tile_m, c.tile_n) / min(c.tile_m, c.tile_n) <= 2
def elim_tile_k_8(c: MatmulConfig, hw: MetalHardware) -> bool: return c.tile_k == 8
def elim_pow2_tiles(c: MatmulConfig, hw: MetalHardware) -> bool: return (c.tile_m & (c.tile_m-1)) == 0 and (c.tile_n & (c.tile_n-1)) == 0
def elim_canonical(c: MatmulConfig, hw: MetalHardware) -> bool: return c.threads_m == 16 and c.threads_n == 32
def elim_high_ai(c: MatmulConfig, hw: MetalHardware) -> bool: return c.arithmetic_intensity >= 30
def elim_layout(c: MatmulConfig, hw: MetalHardware) -> bool: return c.tile_n >= c.tile_m

ELIMINATORS = [
  elim_threads, elim_shared_mem, elim_registers, elim_parallelism, elim_coalesce,  # correctness
  elim_ai, elim_tiny, elim_tile_k, elim_vec, elim_pow2_threads, elim_thread_count, elim_pow2_tiles,  # quality
  elim_simd, elim_balance, elim_ilp, elim_occupancy, elim_square, elim_tile_k_8, elim_canonical, elim_high_ai, elim_layout,  # preference
]

def solve_matmul(M: int, N: int, K: int, hw: MetalHardware = MetalHardware(), verbose: bool = False) -> list[MatmulConfig]:
  survivors = list(generate_omega())
  for elim in ELIMINATORS:
    survivors = [c for c in survivors if elim(c, hw)]
    if not survivors: return []
  # filter by matrix dimensions
  survivors = [c for c in survivors if (M % c.tile_m == 0 or c.tile_m <= M) and
               (N % c.tile_n == 0 or c.tile_n <= N) and (K % c.tile_k == 0 or c.tile_k <= K)]
  return sorted(survivors, key=lambda x: -x.arithmetic_intensity)

def config_to_opts(config: MatmulConfig, use_tc: bool = True) -> list:
  from tinygrad.codegen.opt import Opt, OptOps
  opts = []
  if use_tc: opts.append(Opt(op=OptOps.TC, axis=0, arg=(0, 1, getenv("TC", 1))))
  opts.append(Opt(op=OptOps.UPCAST, axis=0, arg=4))
  opts.append(Opt(op=OptOps.UPCAST, axis=1, arg=4))
  opts.append(Opt(op=OptOps.LOCAL, axis=1, arg=4))
  return opts

def is_matmul_kernel(shape: tuple) -> bool:
  return len(shape) == 3 and all(isinstance(s, int) and s >= 16 for s in shape)

def is_conv2d_kernel(shape: tuple) -> bool:
  if len(shape) != 6 or not all(isinstance(s, int) for s in shape): return False
  Co, Ho, Wo, Ci, kH, kW = shape
  return kH <= 11 and kW <= 11 and Ho >= kH and Wo >= kW

def is_conv1x1_kernel(shape: tuple) -> bool:
  if len(shape) != 4 or not all(isinstance(s, int) for s in shape): return False
  Co, Ho, Wo, Ci = shape
  return Co >= 4 and Ci >= 4 and Ho >= 4 and Wo >= 4

def is_reduce_kernel(shape: tuple, axis_types: list) -> bool:
  from tinygrad.codegen.opt.postrange import AxisType
  return any(at in (AxisType.REDUCE, AxisType.GROUP_REDUCE) for at in axis_types)

def conv1x1_opts(shape: tuple, hw: MetalHardware) -> list:
  from tinygrad.codegen.opt import Opt, OptOps
  Co, Ho, Wo, Ci = shape
  opts = [Opt(op=OptOps.TC, axis=0, arg=(-1, 0, getenv("TC", 1)))]
  if Co >= 4 and Co % 4 == 0: opts.append(Opt(op=OptOps.UPCAST, axis=0, arg=4))
  if Ho * Wo >= 32: opts.append(Opt(op=OptOps.LOCAL, axis=0, arg=2))
  return opts

def reduce_opts(shape: tuple, axis_types: list, hw: MetalHardware) -> list:
  from tinygrad.codegen.opt import Opt, OptOps
  from tinygrad.codegen.opt.postrange import AxisType
  reduce_dim = next((s for s, at in zip(shape, axis_types) if at in (AxisType.REDUCE, AxisType.GROUP_REDUCE)), None)
  if reduce_dim is None or not isinstance(reduce_dim, int): return []
  for arg in [16, 8, 4, 2]:
    if reduce_dim % arg == 0 and reduce_dim >= arg: return [Opt(op=OptOps.GROUPTOP, axis=0, arg=arg)]
  return []

def conv2d_opts(shape: tuple, hw: MetalHardware) -> list:
  from tinygrad.codegen.opt import Opt, OptOps
  Co, Ho, Wo, Ci, kH, kW = shape
  opts = []
  def find_div(n, targets):
    for t in sorted(targets, reverse=True):
      if n % t == 0 and n // t > 1: return t
    return 1
  if Wo >= 4 and Wo % 4 == 0:
    opts.append(Opt(op=OptOps.UPCAST, axis=2, arg=4))
    Wo = Wo // 4
  if Co >= 4 and Co % 4 == 0:
    opts.append(Opt(op=OptOps.UPCAST, axis=0, arg=4))
    Co = Co // 4
  if Wo <= 28: opts.append(Opt(op=OptOps.UNROLL, axis=2, arg=0))
  if Ho <= 56: opts.append(Opt(op=OptOps.UNROLL, axis=1, arg=0))
  if (l0 := find_div(Co, [32, 16, 8, 4, 2])) > 1: opts.append(Opt(op=OptOps.LOCAL, axis=0, arg=l0))
  if (l1 := find_div(Ho, [16, 8, 4, 2])) > 1: opts.append(Opt(op=OptOps.LOCAL, axis=1, arg=l1))
  if (l2 := find_div(Wo, [8, 4, 2])) > 1: opts.append(Opt(op=OptOps.LOCAL, axis=2, arg=l2))
  return opts

def eliminate_optimize(s, hw: MetalHardware = MetalHardware()) -> list | None:
  from tinygrad.codegen.opt.postrange import Scheduler
  if not isinstance(s, Scheduler): return None
  shape, axis_types = s.full_shape, s.axis_types
  if is_matmul_kernel(shape):
    survivors = solve_matmul(*shape, hw, verbose=False)
    if not survivors: return None
    return config_to_opts(survivors[0])
  if is_conv1x1_kernel(shape): return conv1x1_opts(shape, hw)
  if is_conv2d_kernel(shape): return conv2d_opts(shape, hw)
  if is_reduce_kernel(shape, axis_types): return reduce_opts(shape, axis_types, hw)
  return None
