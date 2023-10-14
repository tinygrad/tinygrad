from __future__ import annotations
from typing import NamedTuple, Optional, List, Tuple, cast, Dict
from copy import deepcopy
import itertools
from tinygrad.ops import LazyOp, FlopCounter, get_lazyop_info, ReduceOps, MemBuffer, BufferOps, Device, Compiled
from tinygrad.helpers import dedup, dtypes, colored, ImageDType, DType, all_int, ansilen
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import sint
from tinygrad.shape.view import strides_for_shape
from dataclasses import dataclass

@dataclass(frozen=True)
class TensorCore:
  device: str
  dims: List[int]
  dtype_in: DType
  dtype_out: DType
  threads: List[int]
  thread_local_aliases: List[List[List[int]]]
  thread_local_sizes: List[int]
  arch: Optional[str] = None
  def __str__(self): return f"tensor_core<{self.device}, {self.dims}, {self.dtype_in}, {self.dtype_out}>"

# TODO(TC): doesn't belong here!!!
tensor_cores: Dict[str, List[TensorCore]] = {
  "METAL": [
    TensorCore(device="METAL", dims=[8,8,8], dtype_in=dtypes.float, dtype_out=dtypes.float, threads=[2,4,2,2], thread_local_sizes=[2,2,2], thread_local_aliases=[ [[-1, 1, 3], [0], [2, 4]], [[2, 4], [-1, 1, 3], [0]], [[0], [-1, 1, 3], [2, 4]] ], arch="arm64"),
    TensorCore(device="METAL", dims=[8,8,8], dtype_in=dtypes.half,  dtype_out=dtypes.half,  threads=[2,4,2,2], thread_local_sizes=[2,2,2], thread_local_aliases=[ [[-1, 1, 3], [0], [2, 4]], [[2, 4], [-1, 1, 3], [0]], [[0], [-1, 1, 3], [2, 4]] ], arch="arm64")
  ],
  "HIP": [
    TensorCore(device="HIP", dims=[16,16,16], dtype_in=dtypes.half, dtype_out=dtypes.float, threads=[16,2], thread_local_sizes=[16,16,8], thread_local_aliases=[ [[-1], [0], [1]], [[-1], [1], [0]], [[0], [1], [2, -1]] ]),
  ]
}

class LocalBuffer(NamedTuple):
  name: str
  size: int
  dtype: DType = dtypes.float32
  realized: None = None
  def __str__(self): return f"localbuffer<{self.name}[{self.size}]>"

class LinearizerOptions(NamedTuple):
  device: str = ""
  # TODO: make this generic with a list of supported types
  supports_float4: bool = True
  supports_float4_alu: bool = True
  has_local: bool = True
  has_shared: bool = True
  # NOTE: these two should be in z,y,x(reversed) order for cstyle backends, they are flipped when kernel is rendered
  global_max: Optional[List[int]] = None
  local_max: Optional[List[int]] = None

class Kernel:
  def __init__(self, ast:LazyOp, opts:Optional[LinearizerOptions]=None):
    self.opts = opts if opts else (cast(Compiled, Device[Device.DEFAULT]).linearizer_opts if isinstance(Device[Device.DEFAULT], Compiled) else LinearizerOptions())
    self.ast = ast

    # fetch lazyop info
    self.info: FlopCounter = get_lazyop_info(cast(LazyOp, self.ast))

    # there's only allowed to be one reduceop
    reduceops = [x for x in self.ast.get_lazyops() if x.op in ReduceOps]
    assert len(dedup(reduceops)) <= 1, "max one reduce op in an ast"
    self.reduceop = reduceops[0] if reduceops else None

    # create new shapetrackers inside this kernel, we will permute them
    self.bufs = [MemBuffer(0, self.info.dtype, ShapeTracker.from_shape(self.info.shape))] + dedup([x.arg for x in self.ast.get_lazyops() if x.op in BufferOps])
    self.sts: List[ShapeTracker] = [x.st for x in self.bufs]

    self.mem_estimate: int = sum(x.dtype.itemsize*x.st.size() for x in self.bufs)

    # get earlybufs, before the one reduce op
    self.earlybufs = [x.arg for x in self.reduceop.get_lazyops() if x.op in BufferOps] if self.reduceop else []
    self.full_buf_index: int = self.bufs.index(self.earlybufs[0]) if self.earlybufs else 0

    # parameters
    self.group_for_reduce: List[int] = []
    self.upcasted: int = 0
    self.local_dims: int = 0
    self.local_alias: Dict[int, LocalBuffer] = {}
    self.use_tensor_cores: bool = False
    self.exclude_local_upcast: int = 0
    self.reverse_upcast_dir: bool = False

    self.global_size: Optional[List[int]] = None
    self.local_size: Optional[List[int]] = None

  def copy(self):
    return deepcopy(self)

  @property
  def membufs(self) -> List[MemBuffer]: return [x for x in self.bufs if isinstance(x, MemBuffer)]

  def has_variable_shape(self) -> bool:
    for b in self.bufs:
      if not all_int(b.st.views[-1].shape): return True
    return False

  def shape_offsets(self, i): return itertools.product(*[list(range(s)) for s in self.sts[i].shape[self.shape_len-self.upcasted:][::-1]]) if self.upcasted > 0 else [tuple()]
  def float4_axis(self, i): return [x-(self.shape_len-self.upcasted) for x in self.sts[i].unit_stride_axes() if x >= self.shape_len-self.upcasted and self.sts[i].shape[x]%4 == 0]

  def upcasted_axis(self, i):
    return list(zip(self.sts[i].shape[self.shape_len-self.upcasted:],
                    self.sts[i].real_strides()[self.shape_len-self.upcasted:],
                    [x!=y for x,y in zip(self.sts[0].shape[self.shape_len-self.upcasted:], self.full_shape[self.shape_len-self.upcasted:])]))

  # TODO: is there a better way to write this?
  def acc_offsets(self, i):
    if self.upcasted == 0: return [0]
    upcasted_i = self.upcasted_axis(i)
    acc_strides = [x*(1-upcasted_i[::-1][i][2]) for i,x in enumerate(strides_for_shape(tuple(1 if r else s for s,_,r in upcasted_i[::-1])))]
    return [sum(t) for t in itertools.product(*[[y*acc_strides[i] for y in range(x[0])] for i,x in enumerate(upcasted_i[::-1])])]

  def get_upcast_dim(self, i) -> List[int]:
    should_upcast = self.opts.supports_float4 and (self.bufs[i].dtype in [dtypes.float32, dtypes.float16] or isinstance(self.bufs[i].dtype, ImageDType))
    return [x for x in self.sts[i].unit_stride_axes() if should_upcast and x >= self.shape_len-self.upcasted and self.sts[i].shape[x] > 1]

  @property
  def first_reduce(self) -> int: return [x!=y for x,y in zip(self.sts[0].shape[:self.shape_len-self.upcasted]+(0,), self.full_shape[:self.shape_len-self.upcasted]+(1,))].index(True)

  @property
  def output_shape(self) -> Tuple[sint, ...]: return self.sts[0].shape

  @property
  def full_shape(self) -> Tuple[sint, ...]: return self.sts[self.full_buf_index].shape

  @property
  def full_unupcasted_shape(self) -> Tuple[sint, ...]: return self.full_shape[:self.shape_len-self.upcasted]

  @property
  def shape_len(self) -> int: return len(self.sts[0].shape)

  @property
  def upcast_in_mid_reduce_axes(self) -> List[int]: return [j for j in range(self.first_reduce, self.first_reduce+len(self.group_for_reduce)) if self.full_shape[j] == self.sts[0].shape[j]]

  @property
  def global_dims(self) -> int: return self.first_reduce-self.local_dims

  # there's eight chunks of the shape
  # blue   -- global dims
  # CYAN   -- excluded local dims (non-warp)
  # cyan   -- local dims
  #  *** self.first_reduce
  # green  -- reduce-local dims
  # white  -- reduce-late upcasted dim (self.upcast_in_mid_reduce_axes)
  # red    -- reduce loops
  #  *** self.upcasted
  # purple -- reduce upcasted
  # yellow -- normal upcasted dimensions
  def colors(self) -> List[str]:
    # first non local non reduce dims are global (blue)
    colors = ["blue"] * self.global_dims
    # some special local_dims are excluded from the local upcast
    colors += ["CYAN"] * self.exclude_local_upcast
    # except the local_dims, these are non-reduce locals (cyan)
    colors += ["cyan"] * (self.local_dims - self.exclude_local_upcast)
    # between first_reduce and first_reduce + group_for_reduce, they are either local (cyan), or late upcasted (green)
    colors += ["white" if i in self.upcast_in_mid_reduce_axes else "green" for i in range(self.first_reduce, self.first_reduce + len(self.group_for_reduce))]
    # between first_reduce + group_for_reduce and upcasted, they are reduce (red)
    colors += ["red"] * ((self.shape_len-self.upcasted) - (self.first_reduce + len(self.group_for_reduce)))
    # upcasted dimensions are reduce (magenta) or normal (yellow)
    colors += ["magenta" if self.full_shape[i] != self.sts[0].shape[i] else "yellow" for i in range(self.shape_len-self.upcasted, self.shape_len)]
    assert len(colors) == self.shape_len, "colors size mismatch"
    return colors

  def colored_shape(self, pad=None) -> str:
    ret = ' '.join(colored(s, color) for s,color in zip([f"{s:4d}" if isinstance(s, int) else s for s in self.full_shape], self.colors()))
    if pad: ret += ' '*(pad-ansilen(ret))
    return ret
  def printbufs(self, prefix=""):
    for i,st in enumerate(self.sts):
      print(prefix, f"{i:3d} {str(self.bufs[i]):47s}", st.views)
    print(self.colored_shape())

