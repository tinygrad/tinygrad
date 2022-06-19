# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from tinygrad.helpers import prod
from functools import cached_property, reduce
from itertools import chain

def divmodidx(acc, d, mod=True):
  lr = f"(idx//{acc})" if acc != 1 else "idx"
  return f"({lr}%{d})" if mod else lr  # don't mod the top shape dimension

class View:
  def __init__(self, shape, strides, offset:int=0):
    assert len(shape) == len(strides)
    self.shape, self.offset = shape, offset

    self.shape_strides = [(shape[0], strides[0])]
    for i in range(1, len(shape)):
      if (strides[i] != 0 and self.shape_strides[-1][1]//strides[i] == shape[i]) or (strides[i] == 0 and self.shape_strides[-1][1] == 0):
        self.shape_strides[-1] = (self.shape_strides[-1][0] * shape[i], strides[i])
      else:
        self.shape_strides.append((shape[i], strides[i]))

  @cached_property
  def expr(self):
    ret = [f"{self.offset}"] if self.offset != 0 else []
    acc = 1
    for i,(d,s) in enumerate(self.shape_strides[::-1]):
      if d != 1 and s != 0:
        lr = divmodidx(acc, d, i != len(self.shape_strides)-1)
        lr = f"({lr}*{s})" if s != 1 else lr
        ret.append(lr)
      acc *= d
    return 'idx=' + ('+'.join(ret) if len(ret) > 0 else "0")

class ZeroView:
  def __init__(self, old_shape, arg):
    expr = ['valid']
    self.shape = []
    acc = 1
    for s,(x,y) in list(zip(old_shape, arg))[::-1]:
      self.shape = [y-x] + self.shape
      base = divmodidx(acc, self.shape[0], len(self.shape) != len(old_shape)) + f"+{x}"
      if x < 0: expr.append(f"(({base}) >= 0)")
      if y > s: expr.append(f"(({base}) < {s})")
      acc *= self.shape[0]
    self.expr = 'valid=' + ' && '.join(expr)

def strides_for_shape(shape):
  strides = [1]
  for d in shape[::-1][:-1]:
    strides = [d*strides[0]] + strides
  return strides

# TODO: support "contiguous" property and "simple_strided" property and make them work
# TODO: support simplification across views
class ShapeTracker:
  def __init__(self, *shape, strides=None):
    assert all([isinstance(x, int) for x in shape])
    if len(shape) == 0: shape = (1,)
    self.views = [View(shape, strides_for_shape(shape) if strides == None else strides)]
    self.contiguous = True

  @property
  def shape(self): return tuple(self.views[-1].shape)

  def expr(self): return ';'.join([v.expr for v in self.views[::-1] if v.expr != 'idx=idx' and v.expr != 'valid=valid'])
  def movement_op(self, op, arg): getattr(self, str(op).split(".")[1].lower())(*arg); return self
  def needs_valid(self): return any(isinstance(v, ZeroView) for v in self.views)

  def __getitem__(self, val):
    locals = {"idx": val, "valid": 1}
    exec(self.expr(), None, locals)
    return locals["idx"] if locals["valid"] else -1

  def reshape(self, *new_shape):
    assert all([isinstance(x, int) for x in new_shape])
    assert prod(self.shape) == prod(new_shape)
    if self.shape == new_shape: return
    self.views.append(View(new_shape, strides_for_shape(new_shape)))

  def permute(self, *axis):
    assert all([isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis])
    assert len(set(axis)) == len(axis) and len(axis) == len(self.shape)
    if tuple(range(len(axis))) == axis: return
    self.contiguous = False   # TODO: some permutes don't break this. ex: (100, 1) -> (1, 100)
    strides = strides_for_shape(self.shape)
    self.views.append(View([self.shape[a] for a in axis], [strides[a] for a in axis]))

  # TODO: this is a special case of slice with strides, remove it
  # though it's nice that it can't change size
  def flip(self, *axis):
    self.stride(*[-1 if i in axis else 1 for i in range(len((self.shape)))])

  # *** under this line are not invertible ***

  def slice(self, *arg):
    assert len(arg) == len(self.shape)
    if all([(x,y) == (0,s) for s,(x,y) in zip(self.shape, arg)]): return
    self.contiguous = False
    strides = strides_for_shape(self.shape)
    offset = sum([strides[i]*x for i,(x,_) in enumerate(arg)])
    zeroview = ZeroView(self.shape, arg)
    self.views.append(View([y-x for x,y in arg], strides, offset))
    if zeroview.expr != "valid=valid": self.views.append(zeroview)

  def expand(self, *new_shape):
    assert all([isinstance(x, int) for x in new_shape])
    assert all([x == y or x == 1 for x,y in zip(self.shape, new_shape)])
    if self.shape == new_shape: return
    self.contiguous = False
    strides = [s if x == y else 0 for s,(x,y) in zip(strides_for_shape(self.shape), zip(self.shape, new_shape))]
    self.views.append(View(new_shape, strides))

  # TODO: combine with slice? this doesn't require a ZeroView, though slice shouldn't always either
  def stride(self, *mul):
    assert all([isinstance(x, int) for x in mul])
    if all([x==1 for x in mul]): return
    self.contiguous = False
    old_strides = strides_for_shape(self.shape)
    strides = [z*m for z,m in zip(old_strides, mul)]
    new_shape = [(s+(abs(m)-1))//abs(m) for s,m in zip(self.shape, mul)]
    offset = sum([(s-1)*z for s,z,m in zip(self.shape,old_strides,mul) if m < 0])
    self.views.append(View(new_shape, strides, offset))

