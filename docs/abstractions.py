"""
Welcome to the tinygrad documentation
=================

this file will take you on a whirlwind journey from a Tensor all the way down
tinygrad has been aggressively refactored in the 2.5 years it's been worked on.
what you see here is a refined library (with more refining to go still!)

the whole tinygrad is ~2300 lines, so while it's readable in an evening or two,
this documentation will help with entry points and understanding the abstraction stack
"""

# %%
# == Boilerplate imports for typing ==
from __future__ import annotations
from typing import Optional, Tuple, Union, Any, Dict, Callable, Type, List, ClassVar
from enum import Enum, auto
from abc import ABC

# %%
# == Example: Tensor 2+3 ==
# let's trace an addition down through the layers of abstraction.

# we will be using the clang backend
from tinygrad.lazy import Device
Device.DEFAULT = "CLANG"

# first, 2+3 as a Tensor, the highest level
from tinygrad.tensor import Tensor
a = Tensor([2])
b = Tensor([3])
result = a + b
print(f"{a.numpy()} + {b.numpy()} = {result.numpy()}")
assert result.numpy()[0] == 5.

# %%
# == Tensor (in tinygrad/tensor.py, code 8/10) ==
# it's worth reading tinygrad/tensor.py. it's pretty beautiful
import tinygrad.mlops as mlops

# this is the good old familiar Tensor class
class Tensor:
  # these two are pretty straightforward
  grad: Optional[Tensor]
  requires_grad: Optional[bool]

  # this is the graph for the autograd engine
  _ctx: Optional[Function]

  # this is where the data (and other tensor properties) actually live
  lazydata: LazyBuffer

  # high level ops (hlops) are defined on this class. example: relu
  def relu(self): return self.maximum(0)

  # log is an mlop, this is the wrapper function in Tensor
  def log(self): return mlops.Log.apply(self)

# all the definitions of the derivatives are subclasses of Function (like mlops.Log)
# there's only 18 mlops for derivatives for everything (in tinygrad/mlops.py, code 9/10)
# if you read one file, read mlops.py. if you read two files, also read tinygrad/tensor.py
# you can differentiate the world using the chain rule
class Function:
  # example types of forward and backward
  def forward(self, x:LazyBuffer) -> LazyBuffer: pass
  def backward(self, x:LazyBuffer) -> LazyBuffer: pass

# %%
# == LazyBuffer (in tinygrad/lazy.py, code 5/10) ==
from tinygrad.helpers import DType

# this is where the properties live that you thought were a part of Tensor
# LazyBuffer is like a Tensor without derivatives, at the mlop layer
class LazyBuffer:
  # these three define the "type" of the buffer, and they are returned as Tensor properties
  device: str
  shape: Tuple[int, ...]
  dtype: DType

  # a ShapeTracker is used to track things like reshapes and permutes
  # all MovementOps are zero copy in tinygrad!
  # the ShapeTracker specifies how the data in the RawBuffer matches to the shape
  # we'll come back to this later
  st: ShapeTracker

  # if the LazyBuffer is realized, it has a RawBuffer
  # we will come back to RawBuffers later
  realized: Optional[RawBuffer]

  # if the lazybuffer is unrealized, it has a LazyOp
  # this LazyOp describes the computation needed to realize this LazyBuffer
  op: Optional[LazyOp]

# LazyOp (in tinygrad/ops.py, code 4/10)
# in a tree they form an Abstract Syntax Tree for a single GPU kernel
class LazyOp:
  op: Op                                       # the type of the compute
  src: Tuple[Union[LazyOp, LazyBuffer], ...]   # the sources
  arg: Optional[Any] = None                    # and an optional static argument

# there's currently 20 Ops you have to implement for an accelerator.
class UnaryOps(Enum):    NOOP = auto(); EXP = auto(); LOG = auto(); NEG = auto(); NOT = auto()
class BinaryOps(Enum):   ADD = auto();  SUB = auto(); MUL = auto(); DIV = auto(); POW = auto(); CMPEQ = auto(); MAX = auto()
class ReduceOps(Enum):   SUM = auto();  MAX = auto()
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto()
class LoadOps(Enum):     FROMCPU = auto()
# NOTE: if you have a CompiledBuffer(DeviceBuffer)
#       you do not need to implement the MovementOps
#       as they are handled by the ShapeTracker(in tinygrad/shape/shapetracker.py, code 7/10)
Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps]

# most of tinygrad/lazy.py is concerned with fusing Ops into LazyOps ASTs that map to GPUKernels
# it's beyond the scope of this tutorial, but you can read the file if interested

# %%
# == Example: LazyBuffer for 2+3 ==

from tinygrad.tensor import Tensor
from tinygrad.ops import LazyOp, BinaryOps, LoadOps

# the 2+3 from before
result = Tensor([2]) + Tensor([3])
print(type(result.lazydata), result.lazydata)  # let's look at the lazydata of result

# you'll see it has a LazyOp
# the op type is BinaryOps.ADD
# and it has two sources, the 2 and the 3
lazyop: LazyOp = result.lazydata.op
assert lazyop.op == BinaryOps.ADD
assert len(lazyop.src) == 2

# the first source is the 2, it comes from the CPU
# the source is a LazyBuffer, since FROMCPU cannot be folded into LazyOp ASTs
# again, a LazyOp AST is like a GPU kernel. you have to copy the data on the device first
print(lazyop.src[0].op)
assert lazyop.src[0].op.op == LoadOps.FROMCPU
assert lazyop.src[0].op.arg == [2], "the arg of the FROMCPU LazyOP is the [2.]"
assert result.lazydata.realized is None, "the LazyBuffer is not realized yet"

# now we realize the LazyBuffer
result.lazydata.realize()
assert result.lazydata.realized is not None, "the LazyBuffer is realized!"
# this brings us nicely to DeviceBuffer, of which the realized ClangBuffer is a subclass
assert 'RawMallocBuffer' in str(type(result.lazydata.realized))
# getting ahead of ourselves, but we can copy the DeviceBuffer toCPU
assert result.lazydata.realized.toCPU()[0] == 5, "when put in numpy with toCPU, it's 5"

# %%
# == Union[Interpreted, Compiled] (in tinygrad/ops.py, code 5/10) ==

# Now you have a choice, you can either write a "Interpreted" backend or "Compiled" backend

# Interpreted backends are very simple (example: CPU and TORCH)
class Interpreted:
  # they have a backing RawBuffer
  buffer: Type[RawBuffer]

  # and they have a lookup table to functions for the Ops
  fxn_for_op: Dict[Op, Callable] = {
    UnaryOps.EXP: lambda x: np.exp(x),
    BinaryOps.ADD: lambda x,y: x+y}

# Compiled backends take a little more (example: GPU and LLVM)
class Compiled:
  # they also have a backing RawBuffer
  buffer: Type[RawBuffer]

  # a code generator, which compiles the AST
  codegen: Type[ASTKernel]

  # and a runtime, which runs the generated code
  runtime: Type[Runtime]

# Runtime is what actually runs the kernels for a compiled backend
class Runtime(ABC):
  # `name` is the name of the function, and `prg` is the code
  # the constructor compiles the code
  def __init__(self, name:str, prg:str): pass
  # call runs the code on the bufs. NOTE: the output is always bufs[0], but this is just a convention
  def __call__(self, global_size:Optional[List[int]], local_size:Optional[List[int]], *bufs:List[RawBuffer]): pass

# %%
# == RawBuffer (in tinygrad/runtime/lib.py, code 5/10) ==
import numpy as np

# RawBuffer is where the data is actualy held. it's pretty close to just memory
class RawBuffer(ABC):
  # create an empty rawbuffer that holds `size` elements of type `dtype`
  # `buf` is an opaque container class
  def __init__(self, size:int, dtype:DType, buf:Any): raise NotImplementedError("must be implemented")

  # fromCPU is classmethod that creates a RawBuffer, it's a classmethod since some runtimes are 0 copy
  @classmethod
  def fromCPU(cls:RawBuffer, x:np.ndarray) -> RawBuffer: raise NotImplementedError("must be implemented")

  # toCPU converts the RawBuffer to a numpy array with shape (size,). many backends are 0 copy here
  def toCPU(self) -> np.ndarray: raise NotImplementedError("must be implemented")

# RawNumpyBuffer is a RawBuffer example for numpy. It's very simple
class RawNumpyBuffer(RawBuffer):
  # NOTE: the "np.ndarray" is stored in the opaque container
  def __init__(self, buf:np.ndarray):
    super().__init__(buf.size, dtypes.from_np(buf.dtype), buf)
  @classmethod
  def fromCPU(cls, x): return cls(x)
  def toCPU(self): return self._buf

# %%
# == Example: 2+3 in raw clang ==

# RawMallocBuffer is the simplest concrete version of RawBuffer (in tinygrad/ops.py)
# it's used for the CLANG and LLVM backends
# it's just malloc(size * dtype.itemsize)
from tinygrad.runtime.lib import RawMallocBuffer

# ClangProgram is the simplest runtime (in tinygrad/runtime/ops_clang.py, code 7/10)
# __init__ calls clang, and __call__ calls the function in the *.so outputted by clang
# in CLANG, global_size and local_size are ignored
from tinygrad.runtime.ops_clang import ClangProgram

# a concrete example looks like this, this adds two size 1 RawBuffer
# first we create two numpy buffers containing 2 and 3
# then we copy the numpy in to RawMallocBuffers
# last, we create an empty output buffer
from tinygrad.helpers import dtypes
numpy_a, numpy_b = np.array([2], dtype=np.float32), np.array([3], dtype=np.float32)
input_a, input_b = RawMallocBuffer.fromCPU(numpy_a), RawMallocBuffer.fromCPU(numpy_b)
output = RawMallocBuffer(1, dtypes.float32)

# compile the program, run it, and 2+3 does indeed equal 5
program = ClangProgram("add", "void add(float *a, float *b, float *c) { *a = *b + *c; }")
program(None, None, output, input_a, input_b)  # NOTE: the None are for global_size and local_size
print(output.toCPU())
assert output.toCPU()[0] == 5, "it's still 5"
np.testing.assert_allclose(output.toCPU(), numpy_a+numpy_b)

# %%
# == ASTKernel (in tinygrad/codegen/ast.py, code 2/10) ==

# but we are nowhere near done!
# we wrote the code above by hand
# we need the LazyOp ASTs to be automatically turned into code
# the current class looks roughly like this, but this will change and we will update the docs
# this stuff is in the terrible 528 lines of (tinygrad/codegen/*, code 2/10 aka turd quality)
class ASTKernel:
  # create the kernel with the AST
  # NOTE: the AST contains the CompiledBuffers themselves as the root nodes. this will change
  def __init__(self, ast:LazyOp): pass
  def codegen(self) -> ASTRunner: pass

# we return a class that runs code on LazyBuffers, which are all expected to be realized
class ASTRunner:  # (from tinygrad/ops.py)
  def __init__(self, name, prg, global_size:Optional[List[int]], local_size:Optional[List[int]]): pass
  def build(self, runtime:Runtime): pass
  def exec(self, bufs:List[LazyBuffer]): pass

# that hides a lot of complexity that will be refactored, but that's the basic idea of code generation

# %%
# == Example: 2+3 autogenerated clang code ==

from tinygrad.tensor import Tensor
result = Tensor([2]) + Tensor([3])

# we have a global cache used by the JIT
# from there, we can see the generated clang code
from tinygrad.helpers import GlobalCounters
GlobalCounters.cache = []    # enables the cache
result.realize()             # create the program and runs it
cache_saved = GlobalCounters.cache
GlobalCounters.cache = None  # disable the cache

# there's one ASTRunner in the cache
assert len(cache_saved) == 1
prg, bufs = cache_saved[0]

# print the C Program :)
print(prg.prg)

# after some formatting (the compiler doesn't care)
# NOTE: the 2 and 3 are constant folded
"""
void E_1(float* data0) {
  for (int idx0 = 0; idx0 < 1; idx0++) {
    data0[0] = (2.0f) + (3.0f);
  }
}
"""

# %%
# == Example: ShapeTracker (in tinygrad/shape/shapetracker.py, code 7/10) ==

# remember how I said you don't have to write the MovementOps for CompiledBuffers?
# that's all thanks to ShapeTracker!
# ShapeTracker tracks the indices into the RawBuffer
from tinygrad.shape.shapetracker import ShapeTracker

# create a virtual (10, 10) Tensor. this is just a shape, there's no actual tensor
a = ShapeTracker((10, 10))

# you'll see it has one view. the (10, 1 are the strides)
print(a) # ShapeTracker(shape=(10, 10), views=[View((10, 10), (10, 1), 0)])

# we can permute it, and the strides change
a.permute((1,0))
print(a) # ShapeTracker(shape=(10, 10), views=[View((10, 10), (1, 10), 0)])

# we can then reshape it, and the strides change again
# note how the permute stays applied
a.reshape((5,2,5,2))
print(a) # ShapeTracker(shape=(5, 2, 5, 2), views=[View((5, 2, 5, 2), (2, 1, 20, 10), 0)])

# now, if we were to reshape it to a (100,) shape tensor, we have to create a second view
a.reshape((100,))
print(a) # ShapeTracker(shape=(100,), views=[
         #   View((5, 2, 5, 2), (2, 1, 20, 10), 0),
         #   View((100,), (1,), 0)])

# Views stack on top of each other, to allow zero copy for any number of MovementOps
# we can render a Python expression for the index at any time
idx, _ = a.expr_idxs()
print(idx.render())  # (((idx0%10)*10)+(idx0//10))

# of course, if we reshape it back, the indexes get simple again
a.reshape((10,10))
idx, _ = a.expr_idxs()
print(idx.render())  # ((idx1*10)+idx0)

# the ShapeTracker still has two views though...
print(a) # ShapeTracker(shape=(10, 10), views=[
         #   View((5, 2, 5, 2), (2, 1, 20, 10), 0),
         #   View((10, 10), (10, 1), 0)])

# ...until we simplify it!
a.simplify()
print(a) # ShapeTracker(shape=(10, 10), views=[View((10, 10), (1, 10), 0)])

# and now we permute it back
a.permute((1,0))
print(a) # ShapeTracker(shape=(10, 10), views=[View((10, 10), (10, 1), 0)])

# and it's even contiguous
assert a.contiguous == True

# %%
# == Example: Variable (in tinygrad/shape/symbolic.py, code 6/10) ==

# Under the hood, ShapeTracker is powered by a small symbolic algebra library
from tinygrad.shape.symbolic import Variable

# Variable is the basic class from symbolic
# it's created with a name and a min and max (inclusive)
a = Variable("a", 0, 10)
b = Variable("b", 0, 10)

# some math examples
print((a*10).min, (a*10).max)  # you'll see a*10 has a min of 0 and max of 100
print((a+b).min, (a+b).max)    # 0 20, you get the idea

# but complex expressions are where it gets fun
expr = (a + b*10) % 10
print(expr.render())   # (a%10)
# as you can see, b is gone!

# one more
expr = (a*40 + b) // 20
print(expr.render())       # (a*2)
print(expr.min, expr.max)  # 0 20
# this is just "(a*2)"
# since b only has a range from 0-10, it can't affect the output

# %%
