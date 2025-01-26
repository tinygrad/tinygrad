"""This is where the forwards and backwards passes live."""
import math
from tinygrad.dtype import DType
from tinygrad.ops import Ops, sint, UOp
from tinygrad.tensor import Function

class Contiguous(Function):
  def forward(self, x:UOp) -> UOp: return x.contiguous()

class ContiguousBackward(Function):
  def forward(self, x:UOp) -> UOp: return x.contiguous_backward()

class Cast(Function):
  def forward(self, x:UOp, dtype:DType, bitcast:bool=False) -> UOp: return x.bitcast(dtype) if bitcast else x.cast(dtype)

# ************* unary ops *************

class Reciprocal(Function):
  def forward(self, x:UOp) -> UOp: return x.reciprocal()

class Sin(Function):
  def forward(self, x:UOp) -> UOp: return x.sin()

class Relu(Function):
  def forward(self, x:UOp) -> UOp: return (x>0).where(x, 0)

class Log(Function):
  def forward(self, x:UOp) -> UOp: return x.log2() * math.log(2)

class Exp(Function):
  def forward(self, x:UOp) -> UOp: return (x * (1/math.log(2))).exp2()

class Sqrt(Function):
  def forward(self, x:UOp) -> UOp: return x.sqrt()

class Sign(Function):
  # NOTE: the x*0 is to match torch behavior without function.py
  def forward(self, x:UOp) -> UOp: return x.ne(0).where((x<0).where(x.const_like(-1), x.const_like(1)), x.const_like(0)) + x*0

# ************* binary ops *************

class Less(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x<y

class Neq(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x.ne(y)

class Xor(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x^y

class BitwiseAnd(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x&y

class BitwiseOr(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x|y

class Threefry(Function):
  def forward(self, x:UOp, seed:UOp) -> UOp: return x.threefry(seed)

class Add(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x+y

class Mul(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x * y

class IDiv(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x // y

class Mod(Function):
  def forward(self, x:UOp, y:UOp) -> UOp: return x % y

# ************* ternary ops *************

class Where(Function):
  def forward(self, x:UOp, y:UOp, z:UOp) -> UOp: return x.where(y, z)


# ************* reduce ops *************

class Sum(Function):
  def forward(self, x:UOp, axis:tuple[int, ...]) -> UOp: return x.r(Ops.ADD, axis)

class Prod(Function):
  def forward(self, x:UOp, axis:tuple[int, ...]) -> UOp: return x.r(Ops.MUL, axis)

class Max(Function):
  def forward(self, x:UOp, axis:tuple[int, ...]) -> UOp: return x.r(Ops.MAX, axis)

# ************* movement ops *************

# NOTE: this is sum in reverse
class Expand(Function):
  def forward(self, x:UOp, shape:tuple[int, ...]) -> UOp: return x.expand(shape)

class Reshape(Function):
  def forward(self, x:UOp, shape:tuple[int, ...]) -> UOp: return x.reshape(shape)

class Permute(Function):
  def forward(self, x:UOp, order:tuple[int, ...]) -> UOp: return x.permute(order)

class Pad(Function):
  def forward(self, x:UOp, arg:tuple[tuple[int, int], ...]) -> UOp: return x.pad(arg)

class Shrink(Function):
  def forward(self, x:UOp, arg:tuple[tuple[sint, sint], ...]) -> UOp: return x.shrink(arg)

class Flip(Function):
  def forward(self, x:UOp, axis:tuple[int, ...]) -> UOp: return x.stride(tuple([-1 if i in axis else 1 for i in range(len(x.shape))]))
