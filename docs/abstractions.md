# Abstraction Stack

This file will take you on a whirlwind journey from a tensor all the way down. Tinygrad has been aggressively refactored in the 2.5 years it has been worked on. What you see here is a refined library, with more refining still to come!

The whole tinygrad is \~2300 lines, so while it's readable in an evening or two, this documentation will help with entry points and understanding the abstraction stack.

> If you only read one file, read [tinygrad/mlops.py](/tinygrad/mlops.py). It's also worth reading [tinygrad/tensor.py](/tinygrad/tensor.py). It's pretty beautiful.

#### Boilerplate imports for typing

```python
from __future__ import annotations
from typing import Optional, Tuple, Union, Any, Dict, Callable, Type, List, ClassVar
from enum import Enum, auto
from abc import ABC
```

### Example: Tensor 2+3

Let's trace an addition down through the layers of abstraction.

We will be using the clang backend.

```python
from tinygrad.lazy import Device
Device.DEFAULT = "CLANG"
```

#### First, 2+3 as a Tensor, the highest level.

```python
from tinygrad.tensor import Tensor
a = Tensor([2])
b = Tensor([3])
result = a + b
print(f"{a.numpy()} + {b.numpy()} = {result.numpy()}")
assert result.numpy()[0] == 5.
```

#### Tensor (in [tinygrad/tensor.py](/tinygrad/tensor.py), code 8/10)

```python
import tinygrad.mlops as mlops
```

This is the good old familiar Tensor class.

```python
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
```

All the definitions of the derivatives are subclasses of Function (like mlops.Log). There's only 18 mlops for derivatives for everything (in [tinygrad/mlops.py](/tinygrad/mlops.py), code 9/10). If you read one file, read mlops.py. If you read two files, also read [tinygrad/tensor.py](/tinygrad/tensor.py). You can differentiate the world using the chain rule.

```python
class Function:
  # example types of forward and backward
  def forward(self, x:LazyBuffer) -> LazyBuffer: pass
  def backward(self, x:LazyBuffer) -> LazyBuffer: pass
```

#### LazyBuffer (in [tinygrad/lazy.py](/tinygrad/lazy.py), code 5/10)

```python
from tinygrad.helpers import DType
```

This is where the properties live that you thought were a part of Tensor. LazyBuffer is like a Tensor without derivatives, at the mlop layer.

```python
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
```

LazyOp (in [tinygrad/ops.py](/tinygrad/ops.py), code 4/10). In a tree they form an Abstract Syntax Tree for a single GPU kernel.

```python
class LazyOp:
  op: Op                                       # the type of the compute
  src: Tuple[Union[LazyOp, LazyBuffer], ...]   # the sources
  arg: Optional[Any] = None                    # and an optional static argument
```

There are currently 27 Ops you have to implement for an accelerator.

```python
class UnaryOps(Enum):    NOOP = auto(); EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto()
class BinaryOps(Enum):   ADD = auto();  SUB = auto();  MUL = auto();  DIV = auto();  POW = auto(); CMPEQ = auto(); MAX = auto()
class ReduceOps(Enum):   SUM = auto();  MAX = auto()
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto()
class FusedOps(Enum):    MULACC = auto()
class LoadOps(Enum):     EMPTY = auto(); RAND = auto(); CONST = auto(); FROM = auto(); CONTIGUOUS = auto(); CUSTOM = auto()
# NOTE: if you have a CompiledBuffer(DeviceBuffer)
#       you do not need to implement the MovementOps
#       as they are handled by the ShapeTracker(in tinygrad/shape/shapetracker.py, code 7/10)
Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, FusedOps, LoadOps]
```

Most of [tinygrad/lazy.py](/tinygrad/lazy.py) is concerned with fusing Ops into LazyOps ASTs that map to GPUKernels. It's beyond the scope of this tutorial, but you can read the file if interested.

#### Example: LazyBuffer for 2+3

```python
from tinygrad.tensor import Tensor
from tinygrad.ops import LazyOp, BinaryOps, LoadOps
```

The 2+3 from before.

```python
result = Tensor([2]) + Tensor([3])
```

Let's look at the lazydata of result.

```python
print(type(result.lazydata), result.lazydata)
```

You'll see it has a LazyOp. The op type is BinaryOps.ADD and it has two sources, the 2 and the 3.

```python
lazyop: LazyOp = result.lazydata.op
assert lazyop.op == BinaryOps.ADD
assert len(lazyop.src) == 2
```

The first source is the `2`. It comes from the CPU. The source is a LazyBuffer, holding the data as an ndarray. Again, a LazyOp AST is like a GPU kernel. You have to copy the data on the device first.

```python
print(lazyop.src[0].op)
assert lazyop.src[0].op.op == LoadOps.FROM
assert lazyop.src[0].op.src[0].realized.toCPU()[0] == 2, "the arg of the FROM LazyOP is a LazyBuffer holding [2.]"
assert result.lazydata.realized is None, "the LazyBuffer is not realized yet"
```

Now we realize the LazyBuffer.

```python
result.lazydata.realize()
assert result.lazydata.realized is not None, "the LazyBuffer is realized!"
```

This brings us nicely to DeviceBuffer, of which the realized ClangBuffer is a subclass.

```python
assert 'RawMallocBuffer' in str(type(result.lazydata.realized))
```

Getting ahead of ourselves, but we can copy the DeviceBuffer toCPU.

```python
assert result.lazydata.realized.toCPU()[0] == 5, "when put in numpy with toCPU, it's 5"
```

#### Union\[Interpreted, Compiled] (in [tinygrad/ops.py](/tinygrad/ops.py), code 5/10)

Now you have a choice, you can either write an "Interpreted" backend or a "Compiled" backend.

Interpreted backends are very simple (example: CPU and TORCH)

```python
class Interpreted:
  # they have a backing RawBuffer
  buffer: Type[RawBuffer]

  # and they have a lookup table to functions for the Ops
  fxn_for_op: Dict[Op, Callable] = {
    UnaryOps.EXP2: lambda x: np.exp2(x),
    BinaryOps.ADD: lambda x,y: x+y}
```

Compiled backends take a little more (example: GPU and LLVM)

```python
class Compiled:
  # they also have a backing RawBuffer
  buffer: Type[RawBuffer]

  # a code generator, which compiles the AST
  codegen: Type[ASTKernel]

  # and a runtime, which runs the generated code
  runtime: Type[Runtime]
```

Runtime is what actually runs the kernels for a compiled backend.

```python
class Runtime(ABC):
  # `name` is the name of the function, and `prg` is the code
  # the constructor compiles the code
  def __init__(self, name:str, prg:str): pass
  # call runs the code on the bufs. NOTE: the output is always bufs[0], but this is just a convention
  def __call__(self, global_size:Optional[List[int]], local_size:Optional[List[int]], *bufs:List[RawBuffer]): pass
```

#### RawBuffer (in [tinygrad/runtime/lib.py](/tinygrad/runtime/lib.py), code 5/10)

```python
import numpy as np
```

RawBuffer is where the data is actually held. It's pretty close to just memory.

```python
class RawBuffer(ABC):
  # create an empty rawbuffer that holds `size` elements of type `dtype`
  # `buf` is an opaque container class
  def __init__(self, size:int, dtype:DType, buf:Any): raise NotImplementedError("must be implemented")

  # fromCPU is classmethod that creates a RawBuffer, it's a classmethod since some runtimes are 0 copy
  @classmethod
  def fromCPU(cls:RawBuffer, x:np.ndarray) -> RawBuffer: raise NotImplementedError("must be implemented")

  # toCPU converts the RawBuffer to a numpy array with shape (size,). many backends are 0 copy here
  def toCPU(self) -> np.ndarray: raise NotImplementedError("must be implemented")
```

RawNumpyBuffer is a RawBuffer example for numpy. It's very simple.

```python
class RawNumpyBuffer(RawBuffer):
  # NOTE: the "np.ndarray" is stored in the opaque container
  def __init__(self, buf:np.ndarray):
    super().__init__(buf.size, dtypes.from_np(buf.dtype), buf)
  @classmethod
  def fromCPU(cls, x): return cls(x)
  def toCPU(self): return self._buf
```

#### Example: 2+3 in raw clang

RawMallocBuffer is the simplest concrete version of RawBuffer (in [tinygrad/ops.py](/tinygrad/ops.py)). It is used for the CLANG and LLVM backends. It is just `malloc(size * dtype.itemsize)`.

```python
from tinygrad.runtime.lib import RawMallocBuffer
```

ClangProgram is the simplest runtime (in [tinygrad/runtime/ops_clang.py](/tinygrad/runtime/ops_clang.py), code 7/10).

`__init__` calls clang, and `__call__` calls the function in the `*.so` outputted by clang.

In CLANG, `global_size` and `local_size` are ignored.

```python
from tinygrad.runtime.ops_clang import ClangProgram, ClangCodegen
```

A concrete example looks like this. This code adds two size 1 RawBuffer.

First, we create two numpy buffers containing 2 and 3.

```python
from tinygrad.helpers import dtypes
numpy_a, numpy_b = np.array([2], dtype=np.float32), np.array([3], dtype=np.float32)
```

Then, we copy the numpy into RawMallocBuffers.

```python
input_a, input_b = RawMallocBuffer.fromCPU(numpy_a), RawMallocBuffer.fromCPU(numpy_b)
```

Last, we create an empty output buffer.

```python
output = RawMallocBuffer(1, dtypes.float32)
```

Compile the program, run it, and 2+3 does indeed equal 5.

```python
program = ClangProgram("add", f"{ClangCodegen.lang.kernel_prefix} void add(float *a, float *b, float *c) {{ *a = *b + *c; }}")
program(None, None, output, input_a, input_b)  # NOTE: the None are for global_size and local_size
print(output.toCPU())
assert output.toCPU()[0] == 5, "it's still 5"
np.testing.assert_allclose(output.toCPU(), numpy_a+numpy_b)
```

#### ASTKernel (in [tinygrad/codegen/\*](/tinygrad/codegen), code 2/10)

We are nowhere near done! We wrote the code above by hand. We need the LazyOp ASTs to be automatically turned into code. The current class looks roughly like this, but this will change and we will update the docs. This stuff is in the terrible 528 lines of ([tinygrad/codegen/\*](/tinygrad/codegen), code 2/10 aka turd quality).

Create the kernel with the AST.

```python
# NOTE: the AST contains the CompiledBuffers themselves as the root nodes. this will change
class ASTKernel:
  def __init__(self, ast:LazyOp): pass
  def codegen(self) -> ASTRunner: pass
```

We return a class that runs code on LazyBuffers, which are all expected to be realized.

```python
class ASTRunner:  # (from tinygrad/ops.py)
  def __init__(self, name, prg, global_size:Optional[List[int]], local_size:Optional[List[int]]): pass
  def build(self, runtime:Runtime): pass
  def exec(self, bufs:List[LazyBuffer]): pass
```

The above hides a lot of complexity that will be refactored, but that's the basic idea of code generation.

#### Example: 2+3 autogenerated clang code

```python
from tinygrad.tensor import Tensor
result = Tensor([2]) + Tensor([3])
```

We have a global cache used by the JIT. From there, we can see the generated clang code.

```python
from tinygrad.helpers import GlobalCounters
GlobalCounters.cache = []    # enables the cache
result.realize()             # create the program and runs it
cache_saved = GlobalCounters.cache
GlobalCounters.cache = None  # disable the cache
```

There's one ASTRunner in the cache.

```python
assert len(cache_saved) == 1
prg, bufs = cache_saved[0]
```

Print the C Program :)

```python
print(prg.prg)
```

After some formatting (the compiler doesn't care). Note that the 2 and 3 are constant folded.

```python
"""
void E_1(float* data0) {
  for (int idx0 = 0; idx0 < 1; idx0++) {
    data0[0] = (2.0f) + (3.0f);
  }
}
"""
```

#### Example: ShapeTracker (in [tinygrad/shape/shapetracker.py](/tinygrad/shape/shapetracker.py), code 7/10)

Remember how we said you don't have to write the MovementOps for CompiledBuffers? That's all thanks to ShapeTracker! ShapeTracker tracks the indices into the RawBuffer.

```python
from tinygrad.shape.shapetracker import ShapeTracker
```

Create a virtual (10,10) Tensor. This is just a shape, there's no actual Tensor. You'll see it has one view. The (10, 1) are the strides.

```python
a = ShapeTracker((10, 10))
print(a) # ShapeTracker(shape=(10, 10), views=[View((10, 10), (10, 1), 0)])
```

We can permute it, and the strides change.

```python
a.permute((1,0))
print(a) # ShapeTracker(shape=(10, 10), views=[View((10, 10), (1, 10), 0)])
```

We can then reshape it, and the strides change again. Note how the permute stays applied.

```python
a.reshape((5,2,5,2))
print(a) # ShapeTracker(shape=(5, 2, 5, 2), views=[View((5, 2, 5, 2), (2, 1, 20, 10), 0)])
```

Now, if we were to reshape it to a (100,) shape tensor, we have to create a second view.

```python
a.reshape((100,))
print(a) # ShapeTracker(shape=(100,), views=[
         #   View((5, 2, 5, 2), (2, 1, 20, 10), 0),
         #   View((100,), (1,), 0)])
```

Views stack on top of each other. To allow zero copy for any number of MovementOps we can render a Python expression for the index at any time.

```python
idx, _ = a.expr_idxs()
print(idx.render())  # (((idx0%10)*10)+(idx0//10))
```

Of course, if we reshape it back, the indexes get simple again.

```python
a.reshape((10,10))
idx, _ = a.expr_idxs()
print(idx.render())  # ((idx1*10)+idx0)
```

The ShapeTracker still has two views though...

```python
print(a) # ShapeTracker(shape=(10, 10), views=[
         #   View((5, 2, 5, 2), (2, 1, 20, 10), 0),
         #   View((10, 10), (10, 1), 0)])
```

We can go from two views to one view by simplifying it.

```python
a.simplify()
print(a) # ShapeTracker(shape=(10, 10), views=[View((10, 10), (1, 10), 0)])
```

And now we permute it back.

```python
a.permute((1,0))
print(a) # ShapeTracker(shape=(10, 10), views=[View((10, 10), (10, 1), 0)])
```

It is even contiguous!

```python
assert a.contiguous == True
```

#### Example: Variable (in [tinygrad/shape/symbolic.py](/tinygrad/shape/symbolic.py), code 6/10)

Under the hood, ShapeTracker is powered by a small symbolic algebra library.

```python
from tinygrad.shape.symbolic import Variable
```

Variable is the basic class from symbolic. It's created with a name and a min and max (inclusive).

```python
a = Variable("a", 0, 10)
b = Variable("b", 0, 10)
```

Here are some math examples.

```python
print((a*10).min, (a*10).max)  # you'll see a*10 has a min of 0 and max of 100
print((a+b).min, (a+b).max)    # 0 20, you get the idea
```

But complex expressions are where it gets fun.

```python
expr = (a + b*10) % 10
print(expr.render())   # (a%10)
```

As you can see, `b` is gone!

One more example.

```python
expr = (a*40 + b) // 20
print(expr.render())       # (a*2)
print(expr.min, expr.max)  # 0 20
```

This is just `(a*2)`. Since b only has a range from 0-10, it can't affect the output.

