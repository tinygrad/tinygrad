import functools
from tinygrad.ops import Variable
from tinygrad.engine.jit import GraphRunner
from tinygrad.engine.realize import CompiledRunner, BufferXfer, ExecItem
from tinygrad.device import Device, Buffer
from tinygrad.runtime.ops_cloud import CloudDevice, Transfer, GraphComputeItem, GraphAlloc, GraphFree, GraphExec
from tinygrad.helpers import unwrap, flatten, dedup
from typing import cast

class CloudGraph(GraphRunner):
  def __init__(self, jit_cache: list[ExecItem], rawbufs: list[Buffer], var_vals: dict[Variable, int]):
    super().__init__(jit_cache, rawbufs, var_vals)
    self.devices = dedup(flatten([[Device[unwrap(buf).device] for buf in ji.bufs] for ji in jit_cache]))
    self.iids = sorted(self.input_replace.values())
    def _process_ji(ji: ExecItem):
      match ji.prg:
        case CompiledRunner():
          return GraphComputeItem(ji.prg.dev.idx, ji.prg._prg.name, ji.prg._prg.datahash, tuple(unwrap(buf)._buf for buf in ji.bufs),
                                  tuple(ji.prg.p.vars), tuple(ji.prg.p.outs), tuple(ji.prg.p.ins),
                                  tuple(ji.prg.p.global_size) if ji.prg.p.global_size is not None else None,
                                  tuple(ji.prg.p.local_size) if ji.prg.p.local_size is not None else None)
        case BufferXfer():
          dest, src = ji.bufs[0:2]
          assert dest is not None and src is not None, ji
          return Transfer(cast(CloudDevice, Device[dest.device]).idx, dest._buf, cast(CloudDevice, Device[src.device]).idx, src._buf)
        case _: raise RuntimeError(ji)
    self.graph_num = self.devices[0].graph_num
    self.devices[0].graph_num += 1
    jis = tuple(_process_ji(ji) for ji in jit_cache)
    bufs = tuple((cast(CloudDevice, Device[rawbufs[i].device]).idx, rawbufs[i]._buf) for i in self.iids)
    self.devices[0].conn.req.q(GraphAlloc(self.devices[0].idx, self.graph_num, jis, bufs, var_vals))

  def __del__(self):
    self.devices[0].conn.req.q(GraphFree(self.devices[0].idx, self.graph_num))

  def __call__(self, rawbufs: list[Buffer], var_vals: dict[Variable, int], wait=False):
    bufs = tuple((cast(CloudDevice, Device[rawbufs[i].device]).idx, rawbufs[i]._buf) for i in self.iids)
    self.devices[0].conn.req.q(GraphExec(self.devices[0].idx, self.graph_num, bufs, var_vals, wait))
    if wait: return float(self.devices[0].conn.batch_submit())

  # Creates a new type per-host because different hosts can't share the same CloudGraph
  @staticmethod
  @functools.cache
  def construct(host: str, graph_multi: bool):
    return type('CloudGraph', (CloudGraph,), {'host': host, 'graph_multi': graph_multi})
