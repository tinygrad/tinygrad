import numpy as np
import unittest

from tinygrad.device import Device, Buffer
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import run_linear, compile_linear
from tinygrad.uop.ops import UOp, Ops


np.random.seed(1337)
Tensor.manual_seed(1337)
BUF_SIZE = 4096
RUN_CNT = 5

# cache AST by (device, num_inputs)
cached_asts: dict[tuple[str, int], UOp] = {}
def get_ast(device:str, num_inputs:int) -> UOp:
  if (device, num_inputs) not in cached_asts:
    with Context(DEBUG=0):
      fst = [Tensor.randn(BUF_SIZE, dtype=dtypes.int).realize() for _ in range(num_inputs)]
      s = fst[0]
      for i in range(1, num_inputs): s = s.bitwise_xor(fst[i])
      cached_asts[(device, num_inputs)] = s.schedule_linear().src[-1].src[0]
  return cached_asts[(device, num_inputs)]

def make_buffer(device, size=BUF_SIZE, fill=False):
  buf = Buffer(device, size, dtypes.int).ensure_allocated()
  if fill:
    with Context(DEBUG=0):
      buf.copy_from(Tensor(np.random.randint(-10000, 10000, size=size, dtype=np.int32)).realize().uop.base.realized)
  return buf

def get_buf_uop(buf:Buffer, cache:dict[Buffer,UOp]) -> UOp:
  if buf not in cache:
    cache[buf] = UOp.from_buffer(buf)
  return cache[buf]

def make_graph(graph_cls, calls:list[UOp]):
  linear = compile_linear(UOp(Ops.LINEAR, src=tuple(calls)))
  cf = UOp(Ops.CUSTOM_FUNCTION, src=(linear,), arg="graph")
  return graph_cls(cf, [])

def run_schedule(calls:list[UOp]):
  run_linear(UOp(Ops.LINEAR, src=tuple(calls)))

def zero_bufs(bufs):
  for b in bufs: b.copy_from(Buffer("PYTHON", b.size, b.dtype, opaque=memoryview(bytearray(b.nbytes))))

@unittest.skipUnless(Device[Device.DEFAULT].graph is not None, "graph support required")
class TestGraph(unittest.TestCase):
  def skip_if_no_offset(self):
    if Device.DEFAULT in {"WEBGPU", "CL"}: self.skipTest("device does not support _offset")

  def test_order_2_writes_to_same_buf(self):
    d0 = Device.DEFAULT
    b = [make_buffer(d0, fill=True) for _ in range(5)]
    c: dict[Buffer,UOp] = {}

    calls = [
      get_ast(d0, 2).call(get_buf_uop(b[0],c), get_buf_uop(b[1],c), get_buf_uop(b[2],c)),
      get_ast(d0, 2).call(get_buf_uop(b[0],c), get_buf_uop(b[3],c), get_buf_uop(b[4],c)),
    ]

    zero_bufs([b[0]])
    run_schedule(calls)
    expected = [np.frombuffer(x.as_memoryview(), np.int32).copy() for x in b]

    for _ in range(RUN_CNT):
      zero_bufs([b[0]])
      make_graph(Device[d0].graph, calls)([], {})
      for i, buf in enumerate(b): np.testing.assert_equal(expected[i], np.frombuffer(buf.as_memoryview(), np.int32))

  def test_order_read_write_same_buf(self):
    d0 = Device.DEFAULT
    b = [make_buffer(d0, fill=True) for _ in range(5)]
    c: dict[Buffer,UOp] = {}

    calls = [
      get_ast(d0, 2).call(get_buf_uop(b[0],c), get_buf_uop(b[1],c), get_buf_uop(b[2],c)),
      get_ast(d0, 2).call(get_buf_uop(b[1],c), get_buf_uop(b[3],c), get_buf_uop(b[4],c)),
    ]

    zero_bufs([b[0], b[1]])
    run_schedule(calls)
    expected = [np.frombuffer(x.as_memoryview(), np.int32).copy() for x in b]

    for _ in range(RUN_CNT):
      zero_bufs([b[0], b[1]])
      make_graph(Device[d0].graph, calls)([], {})
      for i, buf in enumerate(b): np.testing.assert_equal(expected[i], np.frombuffer(buf.as_memoryview(), np.int32))

  def test_order_write_read_same_buf(self):
    d0 = Device.DEFAULT
    b = [make_buffer(d0, fill=True) for _ in range(5)]
    c: dict[Buffer,UOp] = {}

    calls = [
      get_ast(d0, 2).call(get_buf_uop(b[0],c), get_buf_uop(b[1],c), get_buf_uop(b[2],c)),
      get_ast(d0, 2).call(get_buf_uop(b[1],c), get_buf_uop(b[0],c), get_buf_uop(b[4],c)),
    ]

    zero_bufs([b[0], b[1]])
    run_schedule(calls)
    expected = [np.frombuffer(x.as_memoryview(), np.int32).copy() for x in b]

    for _ in range(RUN_CNT):
      zero_bufs([b[0], b[1]])
      make_graph(Device[d0].graph, calls)([], {})
      for i, buf in enumerate(b): np.testing.assert_equal(expected[i], np.frombuffer(buf.as_memoryview(), np.int32))

  def test_read_write_several_graphs(self):
    d0 = Device.DEFAULT
    b = [make_buffer(d0, fill=True) for _ in range(8)]
    c: dict[Buffer,UOp] = {}

    calls1 = [get_ast(d0, 2).call(get_buf_uop(b[3],c), get_buf_uop(b[1],c), get_buf_uop(b[2],c))]
    calls2 = [get_ast(d0, 2).call(get_buf_uop(b[4],c), get_buf_uop(b[1],c), get_buf_uop(b[3],c))]
    calls3 = [get_ast(d0, 2).call(get_buf_uop(b[5],c), get_buf_uop(b[4],c), get_buf_uop(b[2],c))]

    out = [b[3], b[4], b[5]]
    zero_bufs(out)
    run_schedule(calls1 + calls2 + calls3)
    expected = [np.frombuffer(x.as_memoryview(), np.int32).copy() for x in b]

    for _ in range(RUN_CNT):
      zero_bufs(out)
      make_graph(Device[d0].graph, calls1)([], {})
      make_graph(Device[d0].graph, calls2)([], {})
      make_graph(Device[d0].graph, calls3)([], {})
      for i, buf in enumerate(b): np.testing.assert_equal(expected[i], np.frombuffer(buf.as_memoryview(), np.int32))

if __name__ == '__main__':
  unittest.main()
