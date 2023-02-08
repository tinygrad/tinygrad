import operator
import numpy as np
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, GenericExecAST
from tinygrad.helpers import shape_to_axis

base_fxn_for_op = {
  UnaryOps.NOOP: lambda x: x[:], UnaryOps.NEG: lambda x: -x, UnaryOps.GT0: lambda x: operator.gt(x, 0.0), UnaryOps.RECIPROCAL: lambda x: 1.0/x,
  BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.MUL: operator.mul, BinaryOps.DIV: operator.truediv, BinaryOps.POW: operator.pow,
  ReduceOps.SUM: lambda x, new_shape: x.sum(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
  ReduceOps.MAX: lambda x, new_shape: (x.amax if hasattr(x, 'amax') else x.max)(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
  MovementOps.SHRINK: lambda x, arg: x[tuple(slice(p[0], p[1], None) for p in arg)],
}

class CPUBuffer(GenericExecAST):
  fxn_for_op = (lambda d: d.update(base_fxn_for_op) or d)({
    UnaryOps.RELU: lambda x: np.maximum(x, 0), UnaryOps.EXP: lambda x: np.exp(x), UnaryOps.LOG: lambda x: np.log(x), BinaryOps.CMPEQ: lambda x,y: (x==y).astype(np.float32),
    MovementOps.FLIP: lambda x, axis: np.flip(x, axis), MovementOps.PERMUTE: lambda x, order: x.transpose(order),
    MovementOps.PAD: lambda x, padding: np.pad(x, padding), MovementOps.EXPAND: lambda x, new_shape: np.broadcast_to(x, new_shape),
    MovementOps.STRIDED: lambda x, arg: np.lib.stride_tricks.as_strided(x.ravel().reshape(x.shape), shape=[y[0] for y in arg], strides=[y[1]*x.dtype.itemsize for y in arg])
  })

  def __init__(self, lbuf:np.ndarray): self.buf, self.shape = lbuf, tuple(lbuf.shape)

  @staticmethod
  def fromCPU(x): return CPUBuffer(x)
  def toCPU(x): return x.buf

  def contiguous(x): return x.unary_op(UnaryOps.NOOP)
  def unary_op(x, op): return type(x)(x.fxn_for_op[op](x.buf))
  def binary_op(x, op, y): return type(x)(x.fxn_for_op[op](x.buf, y.buf))
  def reduce_op(x, op, new_shape): return type(x)(x.fxn_for_op[op](x.buf, new_shape))
  def movement_op(x, op, arg=None): return type(x)(x.fxn_for_op[op](x.buf, arg)) if op in x.fxn_for_op else type(x)(getattr(x.buf, op.name.lower())(arg))

  def processing_op(x,op,w,C):
    assert op == ProcessingOps.CONV, f"{op} isn't supported"
    assert C.px == 0 and C.px_ == 0 and C.py == 0 and C.py_ == 0, "padding in conv is not supported"
    tx = x.movement_op(MovementOps.STRIDED, (
      (C.bs, C.groups*C.cin*x.shape[2]*x.shape[3]), (C.groups, C.cin*x.shape[2]*x.shape[3]),
      (C.oy, C.sy*x.shape[3]), (C.ox, C.sx), (C.cin, x.shape[2]*x.shape[3]), (C.H, C.dy*x.shape[3]), (C.W, C.dx)))
    tw = w.movement_op(MovementOps.RESHAPE, (C.groups, C.rcout, C.cin, C.H, C.W))
    out = np.einsum("nGhwCHW, GkCHW -> nGkhw", tx.buf.ravel().reshape(tx.shape), tw.buf.ravel().reshape(tw.shape))
    return CPUBuffer(out.reshape(C.bs, C.groups*C.rcout, C.oy, C.ox))