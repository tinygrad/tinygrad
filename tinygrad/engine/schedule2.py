from typing import Dict, List, Tuple, Optional, cast
from dataclasses import dataclass
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.engine.lazy import LazyBuffer
from tinygrad.ops import MetaOps, PatternMatcher, UOp, UOps, UPat, graph_rewrite, track_rewrites
from tinygrad.helpers import Metadata

METAOPS = {MetaOps.CUSTOM:UOps.CUSTOM, MetaOps.COPY:UOps.COPY, MetaOps.EMPTY:UOps.EMPTY, MetaOps.VIEW:UOps.BUFFER_VIEW}

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: Tuple[Buffer, ...]
  metadata: Optional[Tuple[Metadata, ...]]
  @property
  def outputs(self) -> Tuple[Buffer, ...]:
    """Read/write or write only buffers in the schedule."""
    return self.bufs[:len(self.ast.src)] if self.ast.op is UOps.SINK else self.bufs[0:1]
  @property
  def inputs(self) -> Tuple[Buffer, ...]:
    """Read only buffers in the schedule."""
    return self.bufs[len(self.ast.src):] if self.ast.op is UOps.SINK else self.bufs[1:]

def _to_uop(x:LazyBuffer, buf_uops:Dict[Buffer, UOp]) -> UOp:
  ubuf = buf_uops.setdefault(x.buffer, UOp(UOps.BUFFER, x.buffer.dtype.ptr(), (), (len(buf_uops), (x.buffer.device, x.buffer.size, x.buffer.dtype))))
  if x.realized is not None: return ubuf
  src = tuple(_to_uop(y, buf_uops) for y in x.srcs)
  if x.op in MetaOps: return UOp(METAOPS[cast(MetaOps, x.op)], x.dtype, (ubuf, *src), x.arg).sink()
  val = UOp(UOps.ALU, x.dtype, src, x.op)
  return UOp(UOps.STORE, dtypes.void, (ubuf, x.st.to_uop(), val))

lazy = PatternMatcher([])

@track_rewrites
def _graph(k, outs:List[LazyBuffer]) -> List[ScheduleItem]:
  buf_uops: Dict[Buffer, UOp] = {}
  sink = UOp.sink(*(_to_uop(x, buf_uops) for x in outs))
  sink = graph_rewrite(sink, lazy)
  raise Exception(sink)
  ret: List[ScheduleItem] = []
  return ret
