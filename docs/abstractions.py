"""
Welcome to the tinygrad documentation
=================

this file will take you on a whirlwind journey from a Tensor all the way down
tinygrad has been aggressively refactored in the 3 years it's been worked on.
what you see here is a refined library (with more refining to go still!)

the whole tinygrad is < 5000 lines, so while it's readable in an evening or two,
this documentation will help with entry points and understanding the abstraction stack
"""

# %%
# == Boilerplate imports for typing ==
from __future__ import annotations
from typing import Optional, Tuple, Union, Any, Dict, Callable, Type, List
from enum import Enum, auto
from abc import ABC

# %%
# == Example: Tensor 2+3 ==
# let's trace an addition down through the layers of abstraction.

# we will be using the clang backend
from tinygrad import Device
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
from tinygrad.dtype import DType

# this is where the properties live that you thought were a part of Tensor
# LazyBuffer is like a Tensor without derivatives, at the mlop layer
class LazyBuffer:
  # these three define the "type" of the buffer, and they are returned as Tensor properties
  device: str
  shape: Tuple[int, ...]
  dtype: DType

  # a ShapeTracker is used to track things like reshapes and permutes
  # all shapetracker operations are zero copy in tinygrad!
  # the ShapeTracker specifies how the data in the RawBuffer matches to the shape
  # we'll come back to this later
  st: ShapeTracker

  # if the LazyBuffer is realized, it has a Buffer
  # we will come back to Buffer later
  realized: Optional[Buffer]

  # if the lazybuffer is unrealized, it has a LazyOp
  # this LazyOp describes the computation needed to realize this LazyBuffer
  op: Optional[LazyOp]

# LazyOp (in tinygrad/ops.py, code 5/10)
# in a tree they form an Abstract Syntax Tree for a single GPU kernel
class LazyOp:
  op: Op                                       # the type of the compute
  src: Tuple[LazyOp, ...]                      # the sources
  arg: Any = None                              # and an optional static argument

# there's currently 26 Ops you have to implement for an accelerator.
class UnaryOps(Enum):    EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto();   SQRT = auto()
class BinaryOps(Enum):   ADD = auto();  SUB = auto();  MUL = auto();  DIV = auto();  CMPLT = auto(); MAX = auto()
class ReduceOps(Enum):   SUM = auto();  MAX = auto()
class TernaryOps(Enum):  WHERE = auto()
class LoadOps(Enum):     EMPTY = auto(); CONST = auto(); COPY = auto(); CONTIGUOUS = auto(); CUSTOM = auto()
Op = Union[UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps]

# most of tinygrad/lazy.py is concerned with fusing Ops into LazyOps ASTs that map to kernels
# it's beyond the scope of this tutorial, but you can read the file if interested

# %%
# == Example: LazyBuffer for 2+3 ==

from tinygrad.tensor import Tensor
from tinygrad.ops import LazyOp, BinaryOps, LoadOps
from tinygrad.lazy import LazyBuffer
from tinygrad.device import Buffer

# the 2+3 from before
result = Tensor([2]) + Tensor([3])
print(type(result.lazydata), result.lazydata)  # let's look at the lazydata of result

# the op type is BinaryOps.ADD
# and it has two sources, the 2 and the 3
lazyop: LazyBuffer = result.lazydata
assert lazyop.op == BinaryOps.ADD
assert len(lazyop.srcs) == 2

# the first source is the 2, it comes from the CPU
# the source is a LazyBuffer that is a "CPU" Tensor
# again, a LazyOp AST is like a GPU kernel. you have to copy the data on the device first
assert lazyop.srcs[0].op == LoadOps.COPY
assert lazyop.srcs[0].srcs[0].device == "EXT"
assert lazyop.srcs[0].srcs[0].realized._buf[0][0] == 2, "the src of the COPY LazyOP is a LazyBuffer on the CPU holding [2]"
assert result.lazydata.base.realized is None, "the LazyBuffer is not realized yet"

# now we realize the LazyBuffer
result.realize()
assert result.lazydata.base.realized is not None, "the LazyBuffer is realized!"
# this brings us nicely to Buffer
assert isinstance(result.lazydata.base.realized, Buffer)
assert result.lazydata.base.realized.device == "CLANG"
# getting ahead of ourselves, but we can move the Buffer to CPU
out = result.lazydata.base.realized.as_buffer().cast('I')
assert out[0] == 5, "when put in numpy, it's 5"

# %%
# == Compiled (in tinygrad/device.py, code 6/10) ==

# Now you can write a Compiled backend (example: GPU, LLVM or PYTHON)

class Compiled:
  # a code generator, which compiles the AST
  codegen: Type[Linearizer]

  # and a runtime, which runs the generated code
  runtime: Type[Runtime]

# Runtime is what actually runs the kernels for a compiled backend
class Runtime(ABC):
  # `name` is the name of the function, and `prg` is the code
  # the constructor compiles the code
  def __init__(self, name:str, prg:str): pass
  # call runs the code on the bufs. NOTE: the output is always bufs[0], but this is just a convention
  def __call__(self, *bufs:List[Buffer], global_size:Optional[List[int]], local_size:Optional[List[int]]): pass

# %%
# == Buffer (in tinygrad/device.py, code 6/10) ==
import numpy as np

# Buffer is where the data is actually held. it's pretty close to just memory
class Buffer(ABC):
  # create an empty rawbuffer that holds `size` elements of type `dtype`
  # `opaque` is an opaque container class
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None): pass

# %%
# == Example: 2+3 in raw clang ==

# MallocAllocator is the simplest concrete version of Allocator (in tinygrad/device.py)
# it's used for the CLANG and LLVM backends
# it's just malloc(size * dtype.itemsize)
from tinygrad.device import MallocAllocator

# ClangProgram is the simplest runtime (in tinygrad/runtime/ops_clang.py, code 7/10)
# __init__ calls clang, and __call__ calls the function in the *.so outputted by clang
# in CLANG, global_size and local_size are ignored
from tinygrad.runtime.ops_clang import ClangProgram, ClangCompiler

# a concrete example looks like this, this adds two size 1 RawBuffer
# first we create two numpy buffers containing 2 and 3
# then we copy the numpy in to RawMallocBuffers
# last, we create an empty output buffer
input_a, input_b = MallocAllocator.alloc(4), MallocAllocator.alloc(4)
output = MallocAllocator.alloc(4)

# now we copy in the values
numpy_a, numpy_b = np.array([2], dtype=np.float32), np.array([3], dtype=np.float32)
MallocAllocator.copyin(input_a, numpy_a.data.cast("B"))
MallocAllocator.copyin(input_b, numpy_b.data.cast("B"))

# compile the program, run it, and 2+3 does indeed equal 5
program = ClangProgram("add", ClangCompiler().compile(f"void add(float *a, float *b, float *c) {{ *a = *b + *c; }}"))
program(output, input_a, input_b)
numpy_out = np.empty(1, dtype=np.float32)
MallocAllocator.copyout(numpy_out.data.cast("B"), output)
assert numpy_out[0] == 5, "it's still 5"
np.testing.assert_allclose(numpy_out, numpy_a+numpy_b)

# %%
# == Linearizer (in tinygrad/codegen/linearizer.py, code 4/10) ==

# in the above example, we wrote the code by hand
# normally while using tinygrad you don't do that
# the first step of transforming an AST into code is to "linearize" it, think like toposort on the AST
# for that, we use the Linearizer, which turns an AST into a list of (linear) UOps

class UOps(Enum): LOOP = auto(); DEFINE_LOCAL = auto(); LOAD = auto(); ALU = auto(); CONST = auto(); ENDLOOP = auto(); STORE = auto();

class UOp:
  uop: UOps
  dtype: Optional[DType]
  vin: Tuple[UOp, ...]
  arg: Any

class Linearizer:
  # create the kernel with the AST
  # NOTE: the AST contains the CompiledBuffers themselves as the root nodes. this will change
  def __init__(self, ast:LazyOp): pass
  def linearize(self): pass

  # when linearize is run, it fills in this list
  uops: List[UOp]

from tinygrad.tensor import Tensor
result = Tensor(2.0).realize() + Tensor(3.0).realize()

# use the real Linearizer to linearize 2+3
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.realize import create_schedule
sched = create_schedule([result.lazydata])
linearizer = Linearizer(sched[-1].ast, opts=ClangCompiler.linearizer_opts)
linearizer.linearize()

# print the uops
for uop in linearizer.uops: print(uop)

# output:
"""
   0 UOps.DEFINE_GLOBAL  : ptr.dtypes.float          []                               data0
   1 UOps.CONST          : dtypes.float              []                               2.0
   2 UOps.CONST          : dtypes.float              []                               3.0
   3 UOps.ALU            : dtypes.float              [1, 2]                           BinaryOps.ADD
   4 UOps.CONST          : dtypes.int                []                               0
   5 UOps.STORE          :                           [0, 4, 3]                        None
"""

# %%
# == Example: 2+3 autogenerated clang code ==
# to generate clang code, the Linearizer is wrapped with CStyleCodegen
# here, we have an example where we fetch the generated code from the JIT

from tinygrad.tensor import Tensor
result = Tensor(2.0) + Tensor(3.0)

# we have a global cache used by the JIT
# from there, we can see the generated clang code
from tinygrad.features.jit import CacheCollector
CacheCollector.start()       # enables the cache
result.realize()             # create the program and runs it
cache_saved = CacheCollector.finish()  # disable the cache

# there's one ASTRunner in the cache
assert len(cache_saved) == 1

# print the C Program :)
print(cache_saved[0].prg.prg)

# NOTE: the 2 and 3 are constant folded
"""
void E_n2(float* restrict data0) {
  data0[0] = (2.0f+3.0f);
}
"""

# %%
# == Example: ShapeTracker (in tinygrad/shape/shapetracker.py, code 7/10) ==

# ShapeTracker tracks the indices into the RawBuffer
from tinygrad.shape.shapetracker import ShapeTracker

# create a virtual (10, 10) Tensor. this is just a shape, there's no actual tensor
a = ShapeTracker.from_shape((10, 10))

# you'll see it has one view
print(a) # ShapeTracker(views=(View(shape=(10, 10), strides=(10, 1))))

# we can permute it, and the strides change
a = a.permute((1,0))
print(a) # ShapeTracker(views=(View(shape=(10, 10), strides=(1, 10))))

# we can then reshape it, and the strides change again
# note how the permute stays applied
a = a.reshape((5,2,5,2))
print(a) # ShapeTracker(views=(View(shape=(5, 2, 5, 2), strides=(2, 1, 20, 10))))

# now, if we were to reshape it to a (100,) shape tensor, we have to create a second view
a = a.reshape((100,))
print(a) # ShapeTracker(views=(
         #   View(shape=(5, 2, 5, 2), strides=(2, 1, 20, 10)),
         #   View(shape=(100,), strides=(1,))))

# Views stack on top of each other, to allow zero copy for any number of shapetracker ops
# we can render a Python expression for the index at any time
idx, _ = a.expr_idxs()
print(idx.render())  # (((idx0%10)*10)+(idx0//10))

# of course, if we reshape it back, the indexes get simple again
a = a.reshape((10,10))
idx, _ = a.expr_idxs()
print(idx.render())  # ((idx1*10)+idx0)

# the ShapeTracker still has two views though...
print(a) # ShapeTracker(views=(
         #   View(shape=(5, 2, 5, 2), strides=(2, 1, 20, 10),
         #   View(shape=(10, 10), strides=(10, 1))))

# ...until we simplify it!
a = a.simplify()
print(a) # ShapeTracker(views=(View(shape=(10, 10), strides=(1, 10), offset=0)))

# and now we permute it back
a = a.permute((1,0))
print(a) # ShapeTracker(views=(View(shape=(10, 10), strides=(10, 1), offset=0)))

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
