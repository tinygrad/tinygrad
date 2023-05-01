# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import math, functools, itertools
import numpy as np
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence
from tinygrad.helpers import prod, argfix, make_pair, getenv, IMAGE, DEBUG, flatten, DType, dtypes, LazyNumpyArray
from tinygrad.lazy import Device, LazyBuffer

# An instantiation of the Function is the Context
class Function:
  def __init__(self, device:str, *tensors:Tensor):
    self.device, self.parents = device, tensors
    self.needs_input_grad = [t.requires_grad for t in self.parents]
    self.requires_grad = True if any(self.needs_input_grad) else (None if any(x is None for x in self.needs_input_grad) else False)

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

  @classmethod
  def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
    ctx = fxn(x[0].device, *x)
    ret = Tensor(ctx.forward(*[t.lazydata for t in x], **kwargs), device=ctx.device, requires_grad=ctx.requires_grad)
    if ctx.requires_grad and not Tensor.no_grad: ret._ctx = ctx    # used by autograd engine
    return ret

import tinygrad.mlops as mlops

# **** start with two base classes, Tensor and Function ****

class Tensor:
  __deletable__ = ('_ctx',)
  training: ClassVar[bool] = False
  no_grad: ClassVar[bool] = False
  default_type: ClassVar[DType] = dtypes.float32

  def __init__(self, data:Union[list, LazyBuffer, LazyNumpyArray, np.ndarray], device=Device.DEFAULT, dtype:Optional[DType]=None, requires_grad:Optional[bool]=None):
    if isinstance(data, list):
      data = np.array(data, dtype=(dtype if dtype is not None else Tensor.default_type).np)
    elif isinstance(data, LazyBuffer) and data.device != device:
      # TODO: this has to realize, it shouldn't have to
      data = data.realize().toCPU()

    # all ndarrays are lazy now
    if isinstance(data, np.ndarray): data = LazyNumpyArray(data, data.shape, data.dtype)

    # by here, it's either LazyNumpyArray or LazyBuffer
    # TODO: it should all be LazyBuffer I think
    if isinstance(data, LazyNumpyArray):
      data = data if data.shape else data.reshape((1,))
      lazydata = LazyBuffer.fromCPU(data.astype(dtype.np) if dtype is not None else data, device)
    elif isinstance(data, LazyBuffer):
      assert dtype is None or dtype == data.dtype, "dtype doesn't match, and casting isn't supported"
      lazydata = data
    else:
      raise RuntimeError(f"can't create Tensor from {data}")

    # this is set once we are here
    self.lazydata: LazyBuffer = lazydata

    # tensors have gradients, buffers do not
    self.grad: Optional[Tensor] = None

    # NOTE: this can be in three states. False and None: no gradient, True: gradient
    # None (the default) will be updated to True if it's put in an optimizer
    self.requires_grad: Optional[bool] = requires_grad

    # internal variables used for autograd graph construction
    self._ctx: Optional[Function] = None

  def __repr__(self):
    return f"<Tensor {self.lazydata if self.lazydata.realized is None else self.lazydata.realized!r} with grad {(self.grad.lazydata if self.grad else None)!r}>"

  # Python has a non moving GC, so this should be okay
  def __hash__(self): return id(self)

  @property
  def device(self) -> str: return self.lazydata.device

  @property
  def shape(self) -> Tuple[int, ...]: return self.lazydata.shape

  @property
  def dtype(self) -> DType: return self.lazydata.dtype

  # ***** data handlers ****

  def realize(self) -> Tensor:
    self.lazydata.realize()
    return self

  def assign(self, x) -> Tensor:
    if not isinstance(x, Tensor): x = Tensor(x)
    assert self.shape == x.shape, f"assign shape mismatch {self.shape} != {x.shape}"
    assert not x.requires_grad  # self requires_grad is okay?
    if DEBUG >= 4: print(f"assign {self.lazydata} <- {x.lazydata}")
    if self.lazydata.realized is not None and not getenv("DISALLOW_ASSIGN"): x.lazydata.output_buffer = self.lazydata.realized
    self.lazydata = x.lazydata
    return self

  def detach(self): return Tensor(self.lazydata, device=self.device, requires_grad=False)
  def numpy(self) -> np.ndarray: return self.lazydata.toCPU()

  # TODO: if things are realized this won't work
  def to_(self, device:str):
    assert self.lazydata.realized is None
    self.lazydata.device = device
    if self.grad:
      self.grad.lazydata.device = device

  def to(self, device:str):
    ret = Tensor(self.lazydata, device)
    if self.grad:
      ret.grad = self.grad.to(device)
    return ret

  # ***** creation helper functions *****

  @staticmethod
  def zeros(*shape, **kwargs): return Tensor([0], **kwargs).reshape([1]*len(shape)).expand(shape).contiguous()

  @staticmethod
  def ones(*shape, **kwargs): return Tensor([1], **kwargs).reshape([1]*len(shape)).expand(shape).contiguous()

  @staticmethod
  def zeros_like(tensor, **kwargs): return Tensor.zeros(*tensor.shape, **kwargs)

  @staticmethod
  def empty(*shape, **kwargs): return Tensor.zeros(*shape, **kwargs)

  @staticmethod
  def eye(dim, **kwargs): return Tensor([1], **kwargs).slice(((0,dim+1),)).reshape(1, dim+1).expand(dim, dim+1).reshape(dim*(dim+1)).slice(((0,dim*dim),)).reshape(dim, dim)

  # TODO: below line, remove use of numpy here and make lazy
  # TODO: requires cumsum to remove numpy
  @staticmethod
  def arange(stop, start=0, step=1, **kwargs): return Tensor(np.arange(start=start, stop=stop, step=step, dtype=np.float32), **kwargs)

  # ***** (numpy) rng helper functions *****
  # TODO: move randomness generation out of numpy

  _rng: ClassVar[np.random.Generator] = np.random.default_rng()
  @staticmethod
  def manual_seed(seed=None): Tensor._rng = np.random.default_rng(seed=seed)

  @staticmethod
  def rand(*shape, **kwargs) -> Tensor: return Tensor(LazyNumpyArray(lambda lna: Tensor._rng.random(size=lna.shape, dtype=lna.dtype), shape, np.float32), **kwargs)

  # TODO: replace with a transformation from uniform -> gaussian
  @staticmethod
  def randn(*shape, **kwargs) -> Tensor: return Tensor(LazyNumpyArray(lambda lna: Tensor._rng.standard_normal(size=lna.shape, dtype=lna.dtype), shape, np.float32), **kwargs)

  # ***** rng hlops *****

  @staticmethod
  def uniform(*shape, **kwargs) -> Tensor: return Tensor.rand(*shape, **kwargs) * 2 - 1

  @staticmethod
  def scaled_uniform(*shape, **kwargs) -> Tensor: return Tensor.uniform(*shape, **kwargs).mul(prod(shape)**-0.5)

  # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
  @staticmethod
  def glorot_uniform(*shape, **kwargs) -> Tensor: return Tensor.uniform(*shape, **kwargs).mul((6/(shape[0]+prod(shape[1:])))**0.5)

  # ***** toposort and backward pass *****

  def deepwalk(self):
    def _deepwalk(node, visited, nodes):
      visited.add(node)
      if node._ctx:
        for i in node._ctx.parents:
          if i not in visited: _deepwalk(i, visited, nodes)
        nodes.append(node)
      return nodes
    return _deepwalk(self, set(), [])

  def backward(self):
    assert self.shape == (1,)

    # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
    # this is "implicit gradient creation"
    self.grad = Tensor([1], device=self.device, requires_grad=False)

    for t0 in reversed(self.deepwalk()):
      if not any(x.requires_grad for x in t0._ctx.parents):
        del t0._ctx  # TODO: does it help to delete this here ever?
        continue
      assert (t0.grad is not None)
      grads = t0._ctx.backward(t0.grad.lazydata)
      grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
        for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
      for t, g in zip(t0._ctx.parents, grads):
        if g is not None and t.requires_grad:
          assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
          t.grad = g if t.grad is None else (t.grad + g)
      del t0._ctx

  # ***** movement mlops *****

  def reshape(self, shape, *args) -> Tensor:
    new_shape = argfix(shape, *args)
    assert len(new_shape) > 0 and all(x != 0 for x in new_shape), f"zeros not allowed in shape {new_shape}"
    return mlops.Reshape.apply(self, shape=tuple(-prod(self.shape) // prod(new_shape) if s == -1 else s for s in new_shape))
  def expand(self, shape, *args) -> Tensor: return mlops.Expand.apply(self, shape=tuple(x if x != -1 else s for s,x in zip(self.shape, argfix(shape, *args))))
  def permute(self, order, *args) -> Tensor: return mlops.Permute.apply(self, order=argfix(order, *args))
  def flip(self, axis, *args) -> Tensor: return mlops.Flip.apply(self, axis=[x if x >= 0 else x+len(self.shape) for x in argfix(axis, *args)])
  def pad(self, arg:Tuple[Tuple[int, int], ...]) -> Tensor: return mlops.Pad.apply(self, arg=arg) if any(x != (0,0) for x in arg) else self
  def shrink(self, arg:Tuple[Tuple[int, int], ...]) -> Tensor: return mlops.Shrink.apply(self, arg=arg) if any(x != (0,s) for x,s in zip(arg, self.shape)) else self

  # ***** movement hlops *****

  # NOTE: using slice is discouraged and things should migrate to pad and shrink
  def slice(self, arg:Sequence[Optional[Tuple[int, int]]]) -> Tensor:
    arg_ = tuple(a if a is not None else (0,s) for s,a in zip(self.shape, arg))
    padding = tuple((max(0, -p[0]), max(0, p[1]-self.shape[i])) for i,p in enumerate(arg_))
    return self.pad(padding).shrink(tuple((p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg_)))

  # Tensors mostly follow the normal python indexing / slicing behavior for sequences
  # - Negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
  # - A slice i:j returns the elements with indices in [i, j)
  #   - If omitted, i and j will default to 0 and N, respectively, where N is the length of the sequence
  #   - Negative values for i and j are taken relative to the end of the sequence
  #   - Both i and j will be clamped to the range (-N, N], where N in the length of the sequence
  # - Indexing with np.newaxis or None on a given axis will add a new dimension of size one before that axis
  # - Empty slices are not allowed
  # - Strides other than 1 are not allowed
  def __getitem__(self, val):
    def slcfix(i, sz, default): return default if i is None else max(0, min(sz, sz+i if i < 0 else i))  # Fix negative idxs, clamp to [0,N]
    new_slice, new_shape = [], []
    val = [val] if not isinstance(val, (list, tuple)) else val
    assert sum(s is not None for s in val) <= len(self.shape)
    assert all(s.step is None or s.step == 1 for s in val if isinstance(s, slice))
    for i,(sz,s) in enumerate(zip(self.shape, [v for v in val if v is not None])):  # Slicing only depends on ints + slices
      if isinstance(s, int) and not (-sz <= s < sz):
        raise IndexError(f"index {s} is out of bounds for dimension {i} with size {sz}")
      new_slice.append((s%sz, s%sz+1) if isinstance(s, int) else (slcfix(s.start, sz, 0), slcfix(s.stop, sz, sz)))
    for s,sz in zip(val, [self.shape[i-1] for i in itertools.accumulate([int(s is not None) for s in val])]):  # Shape depends on slices + positions of Nones
      if not isinstance(s, int):
        new_shape.append(1 if s is None else slcfix(s.stop, sz, sz) - slcfix(s.start, sz, 0))
    new_shape += [self.shape[i] for i in range(len(new_slice), len(self.shape))]
    new_slice += [(0,self.shape[i]) for i in range(len(new_slice), len(self.shape))]
    return self.slice(new_slice).reshape(new_shape if len(new_shape) else (1,))

  def cat(self, *args, dim=0):
    dim = (dim + len(self.shape)) if dim < 0 else dim
    for y in args:
      assert len(y.shape) == len(self.shape) and all(y.shape[i] == s for i,s in enumerate(self.shape) if i != dim)
    catargs = [self] + list(args)
    shape_cumsum = [0, *itertools.accumulate([y.shape[dim] for y in catargs])]
    slc = [[(0, s) for s in self.shape] for _ in catargs]
    for s,k in zip(slc, shape_cumsum):
      s[dim] = (-k, shape_cumsum[-1]-k)
    return functools.reduce(Tensor.__add__, [arg.slice(s) for arg,s in zip(catargs, slc)])

  @staticmethod
  def stack(tensors, dim=0):
    first = tensors[0].unsqueeze(dim)
    unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors[1:]]
    # checks for shapes and number of dimensions delegated to cat
    return first.cat(*unsqueezed_tensors, dim=dim)

  def repeat(self, repeats):
    ndim = len(self.shape)
    base_shape = self.shape
    if len(repeats) > ndim:
      base_shape = (1,) * (len(repeats) - ndim) + base_shape
    new_shape = [x for i in range(len(base_shape)) for x in [1, base_shape[i]]]
    expand_shape = [x for r,s in zip(repeats, base_shape) for x in [r,s]]
    final_shape = [r*s for r,s in zip(repeats, base_shape)]
    return self.reshape(new_shape).expand(expand_shape).reshape(final_shape)

  # TODO: make this nicer with syntactic sugar in slice
  def chunk(self, num, dim):
    slice_params = [[(0, s) for s in self.shape] for _ in range(num)]
    for i,k in enumerate(range(0, self.shape[dim], self.shape[dim]//num)):
      slice_params[i][dim] = (k, min(self.shape[dim], k+self.shape[dim]//num))
    return [self.slice(p) for p in slice_params]

  def unsqueeze(self, dim):
    if dim < 0: dim = len(self.shape) + dim + 1
    return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

  # (padding_left, padding_right, padding_top, padding_bottom)
  def pad2d(self, padding:Union[List[int], Tuple[int, ...]]): return self.slice(((0,self.shape[0]), (0,self.shape[1]), (-padding[2],self.shape[2]+padding[3]), (-padding[0],self.shape[3]+padding[1])))
  @property
  def T(self) -> Tensor: return self.transpose()
  def transpose(self, ax1=1, ax2=0) -> Tensor:
    order = list(range(len(self.shape)))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return self.permute(order)
  def flatten(self, start_dim=0): return self.reshape(shape=tuple(list(self.shape[0:start_dim]) + [-1]))

  # ***** reduce ops *****

  def _reduce(self, fxn:Type[Function], axis:Optional[Union[int, Tuple[int, ...]]]=None, keepdim=False):
    axis_: List[int] = list(range(len(self.shape))) if axis is None else ([axis] if isinstance(axis, int) else list(axis))
    axis_ = [x if x >= 0 else x+len(self.shape) for x in axis_]
    shape = [self.shape[i] for i in range(len(self.shape)) if i not in axis_]
    ret = fxn.apply(self, new_shape=tuple(1 if i in axis_ else self.shape[i] for i in range(len(self.shape))))
    return ret if keepdim else ret.reshape(shape=[1] if shape == [] else shape)

  def sum(self, axis=None, keepdim=False): return self._reduce(mlops.Sum, axis, keepdim)
  def max(self, axis=None, keepdim=False): return self._reduce(mlops.Max, axis, keepdim)
  def min(self, axis=None, keepdim=False): return -((-self).max(axis=axis, keepdim=keepdim))

  def mean(self, axis=None, keepdim=False):
    out = self.sum(axis=axis, keepdim=keepdim)
    return out * (prod(out.shape)/prod(self.shape))

  def _softmax(self, axis):
    m = self - self.max(axis=axis, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis=-1):
    _, e, ss = self._softmax(axis)
    return e.div(ss)

  def log_softmax(self, axis=-1):
    m, _, ss = self._softmax(axis)
    return m - ss.log()

  # ***** processing ops *****

  def _pool(self, k_:Tuple[int, ...], stride:Union[Tuple[int, ...], int]=1, dilation:Union[Tuple[int, ...], int]=1, _insert_dims=tuple()) -> Tensor:
    assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
    s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
    assert len(k_) == len(s_) and len(k_) == len(d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    slc_prefix, prefix, i_ = [(0,x) for x in self.shape[0:-len(k_)]], self.shape[0:-len(k_)], self.shape[-len(k_):]
    if any(k > s for k,s in zip(k_, s_)) or any(d != 1 for d in d_):
      o_ = [(i - d * (k-1) - 1)//s + 1 for i,d,k,s in zip(i_, d_, k_, s_)]
      e_ = [math.ceil(k*(i+d) / i) for k,i,d in zip(k_, i_, d_)]  # expands such that we don't need padding
      xup = self.reshape(*prefix, *([1]*len(_insert_dims)), *flatten((1,i) for i in i_)).expand(*prefix, *_insert_dims, *flatten((e,i) for e,i in zip(e_, i_))).reshape(*prefix, *_insert_dims, *[e*i for e,i in zip(e_, i_)])
      # NOTE: _insert_dims is required because reduces can't be merged (yet)
      prefix += _insert_dims
      slc_prefix += [(0,x) for x in _insert_dims]
      # slide by dilation
      xup = xup.slice(slc_prefix + [(0,k*(i+d)) for k,i,d in zip(k_, i_, d_)])
      xup = xup.reshape(*prefix, *flatten((k,i+d) for k,i,d in zip(k_, i_, d_)))
      xup = xup.slice(slc_prefix + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_, o_, s_)))
      # handle stride, and permute to move reduce to the end
      xup = xup.reshape(*prefix, *flatten((k,o,s) for k,o,s in zip(k_, o_, s_)))
      xup = xup.slice(slc_prefix + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_, o_)))
      xup = xup.reshape(*prefix, *flatten((k,o) for k,o in zip(k_, o_)))
      return xup.permute(*range(len(prefix)), *[len(prefix)+i*2+1 for i in range(len(k_))], *[len(prefix)+i*2 for i in range(len(k_))])
    else:
      # TODO: once the shapetracker can optimize well, remove this alternative implementation. or not if the CPU implementation doesn't use ShapeTracker
      o_ = [(i+(s-k))//s for i,s,k in zip(i_, s_, k_)]
      xup = self.slice(slc_prefix + [(0,o*s) for o,s in zip(o_, s_)])
      xup = xup.reshape(*prefix, *([1]*len(_insert_dims)), *flatten(((o, s) for o,s in zip(o_, s_))))
      if len(_insert_dims):
        xup = xup.expand(*prefix, *_insert_dims, *flatten(((o, s) for o,s in zip(o_, s_))))
        prefix += _insert_dims
        slc_prefix += [(0,x) for x in _insert_dims]
      xup = xup.slice(slc_prefix + flatten(((0,o), (0,k)) for o,k in zip(o_, k_)))
      return xup.permute(*range(len(prefix)), *[len(prefix)+i*2 for i in range(len(k_))], *[len(prefix)+i*2+1 for i in range(len(k_))])

  # NOTE: these work for more than 2D
  def avg_pool2d(self, kernel_size=(2,2), stride=None): return self._pool(make_pair(kernel_size), stride if stride is not None else kernel_size).mean(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))
  def max_pool2d(self, kernel_size=(2,2), stride=None): return self._pool(make_pair(kernel_size), stride if stride is not None else kernel_size).max(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))

  def conv2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0) -> Tensor:
    (bs,cin_,_,_), (cout,cin,H,W) = self.shape, weight.shape
    assert groups*cin == cin_, f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"
    padding_ = [padding]*4 if isinstance(padding, int) else (padding if len(padding) == 4 else [padding[1], padding[1], padding[0], padding[0]])

    # conv2d is a pooling op (with padding)
    x = self.pad2d(padding_)._pool((H,W), stride, dilation)   # (bs, groups*cin, oy, ox, H, W)
    rcout, oy, ox = cout//groups, x.shape[2], x.shape[3]
    x = x.reshape(bs, groups, cin, 1, oy, ox, H, W).expand(bs, groups, cin, rcout, oy, ox, H, W).permute(0,1,3,4,5,2,6,7)

    # expand the channels with the pool
    # TODO: this reduces the number of kernels, but it's slower!
    #x = self.pad2d(padding_)._pool((H,W), stride, dilation, _insert_dims=(cout//groups,))   # (bs, groups*cin, rcout, oy, ox, H, W)
    #rcout, oy, ox = x.shape[2:5]
    #x = x.reshape(bs, groups, cin, rcout, oy, ox, H, W).permute(0,1,3,4,5,2,6,7)

    # conv! broadcasted to (bs, groups, rcout, oy, ox, cin, H, W)
    ret = (x * weight.reshape(1, groups, rcout, 1, 1, cin, H, W)).sum((-3, -2, -1), keepdim=True).reshape(bs, cout, oy, ox)
    return ret if bias is None else ret.add(bias.reshape(1, -1, 1, 1))

  def dot(self, w:Tensor) -> Tensor:
    x = self.reshape(*self.shape[0:-1], 1, self.shape[-1])
    w = w.reshape(*w.shape[0:-2], 1, w.shape[-2], w.shape[-1]).transpose(-1, -2)
    return (x*w).sum(-1).reshape(*x.shape[0:-2], -1)

  # ***** mlops (unary) *****

  def contiguous(self): return mlops.Contiguous.apply(self)
  def log(self): return mlops.Log.apply(self)
  def exp(self): return mlops.Exp.apply(self)
  def relu(self): return mlops.Relu.apply(self)

  # ***** math functions (unary) *****

  def __neg__(self): return 0.0-self
  def sqrt(self): return self.pow(0.5)
  def rsqrt(self): return self.pow(-0.5)
  def square(self): return self*self
  def clip(self, min_, max_): return ((self-min_).relu()+min_) - (self-max_).relu()
  def abs(self): return self.relu() + (-self).relu()
  def sign(self): return self / (self.abs() + 1e-10)
  def reciprocal(self): return 1.0/self

  # ***** activation functions (unary) *****

  def sigmoid(self): return (1.0 + (-self).exp()).reciprocal()
  def elu(self, alpha=1.0): return self.relu() - alpha*(1-self.exp()).relu()
  def celu(self, alpha=1.0): return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)
  def swish(self): return self * self.sigmoid()
  def silu(self): return self.swish()   # The SiLU function is also known as the swish function.
  def relu6(self): return self.relu() - (self-6).relu()
  def hardswish(self): return self * (self+3).relu6() * (1/6)
  def tanh(self): return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  def hardtanh(self, min_val=-1, max_val=1): return self.clip(min_val, max_val)
  def gelu(self): return 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())
  def quick_gelu(self): return self * (self * 1.702).sigmoid()
  def leakyrelu(self, neg_slope=0.01): return self.relu() - (-neg_slope*self).relu()
  def mish(self): return self * self.softplus().tanh()
  def softplus(self, beta=1): return (1/beta) * (1 + (self*beta).exp()).log()
  def softsign(self): return self / (1 + self.abs())

  # ***** broadcasted binary mlops *****

  def _broadcasted(self, fxn:Type[Function], other:Union[Tensor, float], reverse:bool=False) -> Tensor:
    x,y = [Tensor([t], device=self.device, requires_grad=False) if not isinstance(t, Tensor) else t for t in ([other,self] if reverse else [self,other])]
    x,y = [t.reshape([1]*(max(len(x.shape), len(y.shape))-len(t.shape)) + list(t.shape)) for t in [x,y]]
    shape_ret = tuple(max(sx, sy) for sx,sy in zip(x.shape, y.shape))
    return fxn.apply(x.expand(shape_ret), y.expand(shape_ret))

  def add(self, x:Union[Tensor, float], reverse=False) -> Tensor: return self._broadcasted(mlops.Add, x, reverse) if isinstance(x, Tensor) or x != 0.0 else self
  def sub(self, x:Union[Tensor, float], reverse=False) -> Tensor: return self._broadcasted(mlops.Sub, x, reverse) if isinstance(x, Tensor) or x != 0.0 or reverse else self
  def mul(self, x:Union[Tensor, float], reverse=False) -> Tensor: return self._broadcasted(mlops.Mul, x, reverse) if isinstance(x, Tensor) or x != 1.0 else self
  def pow(self, x:Union[Tensor, float], reverse=False) -> Tensor: return self._broadcasted(mlops.Pow, x, reverse) if isinstance(x, Tensor) or x != 1.0 or reverse else self
  def div(self, x:Union[Tensor, float], reverse=False) -> Tensor: return self._broadcasted(mlops.Div, x, reverse) if isinstance(x, Tensor) or reverse else self.mul(1/x)
  def matmul(self, x:Tensor, reverse=False) -> Tensor: return x.dot(self) if reverse else self.dot(x)

  def maximum(self, x:Union[Tensor, float]) -> Tensor: return self._broadcasted(mlops.Maximum, x)
  def minimum(self, x:Union[Tensor, float]) -> Tensor: return -((-self).maximum(-x))
  def eq(self, x) -> Tensor: return self._broadcasted(mlops.Equal, x, False)

  # ***** binary op wrappers (18 wasted lines to make the typechecker happy) *****

  # NOTE: __pow__ and friends are broken in mypyc with the ** operator
  def __add__(self, x) -> Tensor: return self.add(x)
  def __sub__(self, x) -> Tensor: return self.sub(x)
  def __mul__(self, x) -> Tensor: return self.mul(x)
  def __pow__(self, x) -> Tensor: return self.pow(x)
  def __truediv__(self, x) -> Tensor: return self.div(x)
  def __matmul__(self, x) -> Tensor: return self.matmul(x)

  def __radd__(self, x) -> Tensor: return self.add(x, True)
  def __rsub__(self, x) -> Tensor: return self.sub(x, True)
  def __rmul__(self, x) -> Tensor: return self.mul(x, True)
  def __rpow__(self, x) -> Tensor: return self.pow(x, True)
  def __rtruediv__(self, x) -> Tensor: return self.div(x, True)
  def __rmatmul__(self, x) -> Tensor: return self.matmul(x, True)

  def __iadd__(self, x) -> Tensor: return self.assign(self.add(x))
  def __isub__(self, x) -> Tensor: return self.assign(self.sub(x))
  def __imul__(self, x) -> Tensor: return self.assign(self.mul(x))
  def __ipow__(self, x) -> Tensor: return self.assign(self.pow(x))
  def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x))
  def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x))

  def __ge__(self, x) -> Tensor: return self.maximum(x).eq(self)
  def __le__(self, x) -> Tensor: return self.maximum(x).eq(x)
  def __lt__(self, x) -> Tensor: return 1.0-(self>=x)
  def __gt__(self, x) -> Tensor: return 1.0-(self<=x)
  def __eq__(self, x) -> Tensor: return self.eq(x)  # type: ignore # mypy things this should be a bool

  # ***** functional nn ops *****

  def linear(self, weight:Tensor, bias:Optional[Tensor]=None):
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
    return x.add(bias) if bias is not None else x

  def sequential(self, ll:List[Callable[[Tensor], Tensor]]): return functools.reduce(lambda x,f: f(x), ll, self)

  def layernorm(self, axis=-1, eps:float=1e-5) -> Tensor:
    y = (self - self.mean(axis, keepdim=True))
    return y.mul((y*y).mean(axis, keepdim=True).add(eps).rsqrt())

  def batchnorm(self, weight:Optional[Tensor], bias:Optional[Tensor], mean:Tensor, invstd:Tensor) -> Tensor:
    x = (self - mean.reshape(shape=[1, -1, 1, 1]))
    if weight: x = x * weight.reshape(shape=[1, -1, 1, 1])
    ret = x.mul(invstd.reshape(shape=[1, -1, 1, 1]) if len(invstd.shape) == 1 else invstd)
    return (ret + bias.reshape(shape=[1, -1, 1, 1])) if bias else ret

  def dropout(self, p=0.5) -> Tensor:
    if not Tensor.training: return self
    # TODO: why is this going through numpy?
    _mask: np.ndarray = np.asarray(Tensor._rng.binomial(1, 1.0-p, size=self.shape), dtype=np.float32)
    return self * Tensor(_mask, requires_grad=False, device=self.device) * (1/(1.0 - p))

  # ***** cast ops *****

  def cast(self, dtype:DType) -> Tensor: return mlops.Cast.apply(self, dtype=dtype) if self.dtype != dtype else self
  def float(self) -> Tensor: return self.cast(dtypes.float32)
  def half(self) -> Tensor: return self.cast(dtypes.float16)

# register functions to move between devices
for device in Device._buffers:
  setattr(Tensor, f"{device.lower()}", functools.partialmethod(Tensor.to, device))
  setattr(Tensor, f"{device.lower()}_", functools.partialmethod(Tensor.to_, device))

# if IMAGE>0 we install these replacement functions in Tensor (hack!)
from tinygrad.nn.image import image_conv2d, image_dot
if IMAGE:
  setattr(Tensor, "conv2d", image_conv2d)
  setattr(Tensor, "dot", image_dot)
