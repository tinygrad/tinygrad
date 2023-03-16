from __future__ import annotations
from typing import Tuple, Any, ClassVar, Optional, Callable, Dict
import functools
from tinygrad.helpers import DType, dtypes, prod, GlobalCounters, DEBUG
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.ops import DeviceBuffer, LazyOp, get_buffers, map_buffers, Op, FusedOps, UnaryOps, MovementOps, ReduceOps, BinaryOps

# this is a quick "buffer" class for flop tracking and getting the output shape
class GenericShape:
  def __init__(self, shape:Tuple[int, ...], dtype:DType=dtypes.float32, flops:int=0): self.shape, self.dtype, self.flops = shape, dtype, flops
  def consume_flops(self):
    self.flops, ret = 0, self.flops
    return ret
shape_fxn_for_op: Dict[Op, Callable] = {
  **{op:lambda self: GenericShape(self.shape, self.dtype, self.consume_flops() + prod(self.shape)) for op in UnaryOps},
  **{op:lambda self,y: GenericShape(self.shape, max(self.dtype, y.dtype), self.consume_flops() + y.consume_flops() + prod(self.shape)) for op in BinaryOps},
  **{op:lambda self,new_shape: GenericShape(new_shape, self.dtype, self.consume_flops() + prod(self.shape)) for op in ReduceOps},
  **{op:functools.partial(lambda mop,self,arg: GenericShape(ShapeTracker(self.shape).movement_op(mop, arg).shape, self.dtype, self.consume_flops()), op) for op in MovementOps}}

# this runs the LazyOp and gives you the output shape/dtype and flop count
def get_lazyop_info(ast:LazyOp) -> GenericShape: return InterpretedBuffer.exec_ast(map_buffers({x:InterpretedBuffer(GenericShape(x.shape, x.dtype)) for x in get_buffers(ast)}, ast))._buf

# used in CPUBuffer and TorchBuffer
class InterpretedBuffer(DeviceBuffer):  # pylint: disable=abstract-method
  fxn_for_op: ClassVar = shape_fxn_for_op
  def __init__(self, lbuf:Any):
    self._buf: Any = lbuf
    self.shape: Tuple[int, ...] = tuple(lbuf.shape)
    self.dtype: DType = self.to_tinygrad_dtype() if hasattr(self, 'to_tinygrad_dtype') else lbuf.dtype
    # NOTE: this is overcounting the memory used, as reshapes and stuff are aliases
    self._memsz = (prod(self.shape) * self.dtype.itemsize) if not isinstance(self, InterpretedBuffer) else 0
    GlobalCounters.mem_used += self._memsz
  def __del__(self): GlobalCounters.mem_used -= self._memsz
  def contiguous(self): return type(self).exec_ast(LazyOp(op=UnaryOps.NOOP, src=(self,)))
  def movement_op(self, op:MovementOps, arg=None): return type(self)(self.fxn_for_op[op](self._buf, arg)) if op in self.fxn_for_op else type(self)(getattr(self._buf, op.name.lower())(arg))
  @classmethod
  def exec_ast(cls, ast:LazyOp, output_buffer:Optional[InterpretedBuffer]=None, context=None):
    if FusedOps.MULACC in cls.fxn_for_op and ast.op == ReduceOps.SUM and isinstance(ast.src[0], LazyOp) and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(FusedOps.MULACC, ast.src[0].src, ast.arg)
    created_context = context is None
    if context is None: context = dict()
    if not created_context and ast in context: return context[ast]
    srcs = [cls.exec_ast(x, context=context) if isinstance(x, LazyOp) else (x.realized if not isinstance(x, InterpretedBuffer) else x) for x in ast.src]
    if ast.op in BinaryOps: assert srcs[0].shape == srcs[1].shape, f"BinaryOps shape mismatch {srcs[0].shape} != {srcs[1].shape}"
    if ast.op in ReduceOps: assert all(r == n or n == 1 for r,n in zip(srcs[0].shape, ast.arg)), f"ReduceOps can't reduce {srcs[0].shape} -> {ast.arg}"
    if ast.op in MovementOps: ret = srcs[0].movement_op(ast.op, ast.arg)
    else: ret = cls(cls.fxn_for_op[ast.op](*([x._buf for x in srcs] + ([ast.arg] if ast.arg is not None else []))))
    if DEBUG >= 4 or (not isinstance(cls, InterpretedBuffer) and DEBUG >= 3):
      print(f"*** {'exec' if created_context else '    '} {GlobalCounters.mem_used/1e9:5.2f} GB op: {ast.op:20s} out({ret.dtype.name}): {str(ret.shape):30s} in({len(srcs)}):", list(set(x.shape for x in srcs)), ast.arg if ast.arg is not None else "")
    if not created_context: context[ast] = ret
    return ret

    """
    if output_buffer is not None:
      assert output_buffer.shape == ret.shape, output_buffer.dtype == ret.dtype
      output_buffer._buf = ret._buf
      return output_buffer
    else:
      return ret
    """