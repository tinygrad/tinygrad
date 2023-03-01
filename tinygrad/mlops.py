from tinygrad.helpers import prod, argsort
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, MovementOps
from tinygrad.tensor import Function

class Contiguous(Function):
  def forward(self, x): return x.contiguous()
  def backward(self, grad_output): return grad_output

# ************* unary ops *************

class Log(Function):
  def forward(self, x):
    self.save_for_backward(x)
    return x.unary_op(UnaryOps.LOG)

  def backward(self, grad_output):
    return grad_output.binary_op(BinaryOps.DIV, self.saved_tensors[0])

class Exp(Function):
  def forward(self, x):
    ret = x.unary_op(UnaryOps.EXP)
    self.save_for_backward(ret)
    return ret

  def backward(self, grad_output):
    return self.saved_tensors[0].binary_op(BinaryOps.MUL, grad_output)

# ************* reduce ops *************

class Sum(Function):
  def forward(self, x, new_shape):
    self.input_shape = x.shape
    return x.reduce_op(ReduceOps.SUM, new_shape)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.EXPAND, self.input_shape)

class Max(Function):
  def forward(self, x, new_shape):
    ret = x.reduce_op(ReduceOps.MAX, new_shape)
    self.save_for_backward(x, ret)
    return ret

  def backward(self, grad_output):
    x, ret = self.saved_tensors

    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = x.binary_op(BinaryOps.CMPEQ, ret.movement_op(MovementOps.EXPAND, x.shape))

    # sum of locations, averaged
    div = max_is_1s.reduce_op(ReduceOps.SUM, grad_output.shape).movement_op(MovementOps.EXPAND, x.shape)
    max_is_amount = max_is_1s.binary_op(BinaryOps.DIV, div)

    grad_output_expanded = grad_output.movement_op(MovementOps.EXPAND, x.shape)
    return max_is_amount.binary_op(BinaryOps.MUL, grad_output_expanded)

# ************* binary ops *************

class Maximum(Function):
  def forward(self, x, y):
    ret = x.binary_op(BinaryOps.MAX, y)
    self.save_for_backward(y, ret)
    return ret

  def backward(self, grad_output):
    mask = self.saved_tensors[0].binary_op(BinaryOps.CMPEQ, self.saved_tensors[1])
    return grad_output.binary_op(BinaryOps.MUL, mask.unary_op(UnaryOps.NOT)) if self.needs_input_grad[0] else None, \
           grad_output.binary_op(BinaryOps.MUL, mask) if self.needs_input_grad[1] else None

class Add(Function):
  def forward(self, x, y):
    return x.binary_op(BinaryOps.ADD, y)

  def backward(self, grad_output):
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output if self.needs_input_grad[1] else None

class Sub(Function):
  def forward(self, x, y):
    return x.binary_op(BinaryOps.SUB, y)

  def backward(self, grad_output):
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output.unary_op(UnaryOps.NEG) if self.needs_input_grad[1] else None

class Mul(Function):
  def forward(self, x, y):
    self.save_for_backward(x, y)
    return x.binary_op(BinaryOps.MUL, y)

  def backward(self, grad_output):
    return self.saved_tensors[1].binary_op(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None, \
           self.saved_tensors[0].binary_op(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None

class Pow(Function):
  def forward(self, x, y):
    ret = x.binary_op(BinaryOps.POW, y)
    self.save_for_backward(x, y, ret)
    return ret

  def backward(self, grad_output):
    x,y,powxy = self.saved_tensors
    return grad_output.binary_op(BinaryOps.MUL, y.binary_op(BinaryOps.MUL, powxy.binary_op(BinaryOps.DIV, x))) if self.needs_input_grad[0] else None, \
           grad_output.binary_op(BinaryOps.MUL, x.unary_op(UnaryOps.LOG).binary_op(BinaryOps.MUL, powxy)) if self.needs_input_grad[1] else None

class Div(Function):
  def forward(self, x, y):
    self.save_for_backward(x, y)
    return x.binary_op(BinaryOps.DIV, y)

  def backward(self, grad_output):
    x, y = self.saved_tensors
    return grad_output.binary_op(BinaryOps.DIV, y) if self.needs_input_grad[0] else None, \
           grad_output.unary_op(UnaryOps.NEG).binary_op(BinaryOps.MUL, x).binary_op(BinaryOps.DIV, y.binary_op(BinaryOps.MUL, y)) if self.needs_input_grad[1] else None

# ************* movement ops *************

# NOTE: this is sum in reverse
class Expand(Function):
  def forward(self, x, shape):
    self.input_shape = x.shape
    return x.movement_op(MovementOps.EXPAND, shape)

  def backward(self, grad_output):
    return grad_output.reduce_op(ReduceOps.SUM, self.input_shape)

class Reshape(Function):
  def forward(self, x, shape):
    assert len(shape) > 0 and all(x != 0 for x in shape), f"zeros not allowed in shape {shape}"
    self.input_shape = x.shape
    shape = tuple(-prod(x.shape) // prod(shape) if s == -1 else s for s in shape)
    return x.movement_op(MovementOps.RESHAPE, shape)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.RESHAPE, self.input_shape)

class Permute(Function):
  def forward(self, x, order=(1,0)):
    self.input_order = order
    return x.movement_op(MovementOps.PERMUTE, order)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.PERMUTE, tuple(argsort(self.input_order)))

class Slice(Function):
  def forward(self, x, arg=None):
    self.narg = tuple((0-p[0], x.shape[i]-p[0]) for i,p in enumerate(arg))
    return x.slice(tuple(arg))

  def backward(self, grad_output):
    return grad_output.slice(self.narg)

class Flip(Function):
  def forward(self, x, axis):
    self.axis = axis
    return x.movement_op(MovementOps.FLIP, axis)

  def backward(self, grad_output):
    return grad_output.movement_op(MovementOps.FLIP, self.axis)
