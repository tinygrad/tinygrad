import graphlib, unittest
from typing import Any, Dict, List, Union
from tinygrad.codegen.kernel import LocalBuffer
from tinygrad.codegen.uops import UOp, UOps
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.features.graph import print_tree
from tinygrad.ops import BinaryOps, BufferOps, LazyOp, MemBuffer, ReduceOps
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.shape.shapetracker import ShapeTracker
from verify import verify

def panic(x:Any=""): raise Exception(f"explicit panic: {x}")

def create_graph(outs: List[LazyOp]):
  ts = graphlib.TopologicalSorter()
  def _recursive_add(op: LazyOp):
    for src in op.src: _recursive_add(src)
    ts.add(op, *op.src)
  for out in outs: _recursive_add(out)
  return tuple(ts.static_order())

def linearize(ast) -> List[UOp]:
  loaded_bufs: Dict[Union[MemBuffer,LocalBuffer], UOp] = {}
  buf_pointers: Dict[Union[MemBuffer,LocalBuffer], UOp] = {}
  uops: List[UOp] = []

  def const(val):
    existing = [u for u in uops if u.uop == UOps.CONST and u.arg == val]
    if len(existing) != 0: return existing[0]
    uop = UOp(UOps.CONST, dtype=dtypes.int32, arg=val)
    uops.append(uop)
    return uop

  def _lower_op(op: LazyOp) -> UOp:
    if op.op == BufferOps.LOAD: return loaded_bufs[op.arg]
    if op.op in ReduceOps:
      min, max = const(0), const(3)
      loop = UOp(UOps.LOOP, dtype=dtypes.int, vin=(min,max))
      src = UOp(UOps.LOAD, dtype=(buf:=op.src[0].arg).dtype, vin=(buf_pointers[buf],loop))
      acc = UOp(UOps.DEFINE_ACC, dtype=src.dtype, arg=0)
      reduce_alu = UOp(UOps.ALU, dtype=src.dtype, vin=(acc,src), arg=BinaryOps.ADD if op.op == ReduceOps.SUM else BinaryOps.MAX)
      ret = UOp(UOps.PHI, dtype=src.dtype, vin=(acc,reduce_alu,loop))
      uops.extend([acc, loop, src, reduce_alu, ret])
      return ret
    srcs = tuple(_lower_op(src) for src in op.src)
    ret = UOp(UOps.ALU, vin=srcs, dtype=srcs[-1].dtype, arg=op.op)
    uops.append(ret)
    return ret

  for op in ast:
    if op.op in BufferOps and isinstance(buf:=op.arg, MemBuffer):
      if buf not in buf_pointers:
        buf_pointers[buf] = UOp(UOps.DEFINE_GLOBAL, dtype=PtrDType(buf.dtype), arg=f"data{buf.idx}")
        uops.append(buf_pointers[buf])
      if op.op == BufferOps.LOAD and buf not in loaded_bufs:
        loaded_bufs[buf] = UOp(UOps.LOAD, dtype=buf.dtype, vin=(buf_pointers[buf],const(0)))
        uops.append(loaded_bufs[buf])
      else:
        ret = _lower_op(op.src[0])
        if ret.uop == UOps.PHI: uops.append(UOp(UOps.ENDLOOP, vin=(ret.vin[-1],)))
        uops.append(UOp(UOps.STORE, dtype=ret.dtype, vin=(buf_pointers[buf],const(0),ret)))

  return uops


class TestLinearizer2(unittest.TestCase):
  def test_multi_output_simple(self):
    a = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=0, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    b = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    c = LazyOp(BinaryOps.ADD, src=(a,b))
    out0 = LazyOp(BufferOps.STORE, src=(LazyOp(BinaryOps.ADD, src=(a,c)),), arg=MemBuffer(idx=2, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    out1 = LazyOp(BufferOps.STORE, src=(LazyOp(BinaryOps.MUL, src=(a,b)),), arg=MemBuffer(idx=3, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    graph = create_graph([out0, out1])
    uops = linearize(graph)
    alloc_data, init_outputs, prg, get_outputs = verify(uops)
    a, b = alloc_data([1]), alloc_data([2])
    outs = init_outputs([1,1])
    prg(a, b, *outs, global_size=(1,1,1), local_size=(1,1,1))
    assert get_outputs(outs) == [[4], [2]]

  def test_multi_output_multi_reduce(self):
    a = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=0, dtype=dtypes.int, st=ShapeTracker.from_shape((3,))))
    b = LazyOp(BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.int, st=ShapeTracker.from_shape((3,))))
    c = LazyOp(BinaryOps.ADD, src=(a,b))
    out0 = LazyOp(BufferOps.STORE, src=(LazyOp(BinaryOps.ADD, src=(a,c)),), arg=MemBuffer(idx=2, dtype=dtypes.int, st=ShapeTracker.from_shape((3,))))
    out1 = LazyOp(BufferOps.STORE, src=(LazyOp(BinaryOps.MUL, src=(a,b)),), arg=MemBuffer(idx=3, dtype=dtypes.int, st=ShapeTracker.from_shape((3,))))
    out2 = LazyOp(BufferOps.STORE, src=(LazyOp(ReduceOps.SUM, src=(a,)),), arg=MemBuffer(idx=4, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    out3 = LazyOp(BufferOps.STORE, src=(LazyOp(ReduceOps.MAX, src=(b,)),), arg=MemBuffer(idx=5, dtype=dtypes.int, st=ShapeTracker.from_shape((1,))))
    graph = create_graph([out0, out1, out2, out3])
    uops = linearize(graph)
    alloc_data, init_outputs, prg, get_outputs = verify(uops)
    a, b = alloc_data([4,3,4]), alloc_data([10,20,3])
    outs = init_outputs([1,1,1,1])
    prg(a, b, *outs, global_size=(1,1,1), local_size=(1,1,1))
    assert get_outputs(outs) == [[18], [40], [11], [20]]
