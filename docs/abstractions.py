from __future__ import annotations
from typing import Optional, Tuple, Union, Any, Dict, Callable, Type, List
from enum import Enum, auto
from abc import ABC
import numpy as np
import torch

# tinygrad has grown a lot since any docs were last written.
# It's now a terribly large 2300 lines!!
# Let's trace an addition down through the layers of abstraction:
# (note: this is a Python file for syntax highlighting)

# Some of this is documentation from the future, meaning this is how things will be refactored to be
# Though this is pretty close, and is cleaner than what's actually there.

# Tensor (in tinygrad/tensor.py, code 8/10)
# this is the good old familiar Tensor class
class Function: pass
class Tensor:
  # these two are pretty striaghtforward
  grad: Optional[Tensor]
  requires_grad: Optional[bool]
  # this is the graph for the autograd engine
  _ctx: Optional[Function]

  # this is where the data (and other tensor properties) actually live
  lazydata: LazyBuffer

# all the definitions of the derivatives are superclasses of Function
# (in tinygrad/mlops.py, code 9/10)
# they have forward and backward, and operate on LazyBuffers
# Function.apply is responsible for lowering the Tensors to LazyBuffers and running forward

# LazyBuffer (in tinygrad/lazy.py, code 5/10)
# this is where the properties live that you thought were a part of Tensor
from tinygrad.helpers import DType
class LazyBuffer:
  # these three define the "type" of the buffer, and they are proxied through Tensor
  device:str
  shape:Tuple[int, ...]
  dtype:DType

  # if the lazybuffer is unrealized, it has a LazyOp
  # this LazyOp describes the computation needed to realize this LazyBuffer
  op: Optional[LazyOp]

  # if the LazyBuffer is realized, it has a DeviceBuffer
  # we will come back to DeviceBuffers later, first we'll explore the LazyOp
  realized: Optional[DeviceBuffer]

# LazyOp (in tinygrad/ops.py, code 4/10)
# it's an AST node, that defines the type of the compute, the sources, and an optional static argument
# they form an Abstract Syntax Tree
class LazyOp:
  op: Op
  src: Tuple[Union[LazyOp, LazyBuffer], ...]
  arg: Optional[Any] = None

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

# which reminds me, we should get back to DeviceBuffer
# it's an abstract class to be implemented for each backend
class DeviceBuffer(ABC):
  # these two are straightforward. no need for device, since that's contained in the concrete type
  shape: Tuple[int, ...]
  dtype: DType

  # this is the magic method that "fills" a DeviceBuffer and does all the math in tinygrad
  # NOTE: fromCPU no longer exists here, it's just a one LoadOps AST, LoadOps.FROMCPU
  def exec_ast(self, ast:LazyOp): raise NotImplementedError("must be implemented")

  # however, toCPU still exists. it will raise a RuntimeException if exec_ast has never been called
  # it copies out the underlying to the CPU, and will do any sync operations
  def toCPU(self) -> np.ndarray: raise NotImplementedError("must be implemented")

# DeviceBuffers come in two flavors, InterpretedBuffer and CompiledBuffer
# InterpretedBuffers are a lot simpler than CompiledBuffers, and are used to implement the CPU(numpy) and TORCH backends
class InterpretedBuffer(DeviceBuffer):
  # this is where the data actually lives
  # finally some classes you recognize!
  _buf: Union[np.ndarray, torch.Tensor]

  # the compute itself is defined here. these functions are called with _buf
  # here's a UnaryOp and BinaryOp from CPUBuffer(InterpretedBuffer)
  fxn_for_op: Dict[Op, Callable] = {UnaryOps.EXP: lambda x: np.exp(x), BinaryOps.ADD: lambda x,y: x+y}

  # NOTE: exec_ast should not need to be overridden!
  # The actual method lives in tinygrad/ops.py, and it walks the LazyOp tree and calls fxn_for_op as appropriate

# ********** NOTE: for the CPU and TORCH backends, we are done and you can stop reading here **********

# however, all the magic of tinygrad will come from CompiledBuffer
# this is used for the GPU(opencl), CUDA, METAL, CLANG, and LLVM backends
class CompiledBuffer(DeviceBuffer):
  # this is where the data actually lives, same as InterpretedBuffer
  # a RawBuffer is just raw (typed) memory on the Device in question
  _buf: RawBuffer

  # introducing...ShapeTracker! all MovementOps are zero copy in tinygrad
  # the ShapeTracker specifies how the data in the RawBuffer matches to the shape
  # we'll come back to this later
  st: ShapeTracker

  # NOTE: exec_ast should not need to be overridden!
  # instead you need three classes, explained below
  raw_buffer: Type[RawBuffer]
  runtime: Type[Runtime]
  codegen: Type[ASTKernel]

# for completeness, we include RawBuffer. it's very boring and exactly what you expect
class RawBuffer(ABC):
  # create an empty rawbuffer that holds `size` elements of type `dtype`
  def __init__(self, size:int, dtype:DType): raise NotImplementedError("must be implemented")

  # fromCPU is classmethod that creates a RawBuffer, it's a classmethod since some runtimes are 0 copy
  @classmethod
  def fromCPU(cls:RawBuffer, x:np.ndarray) -> RawBuffer: raise NotImplementedError("must be implemented")

  # toCPU converts the RawBuffer to a numpy array with shape (size,). many backends are 0 copy here
  def toCPU(self) -> np.ndarray: raise NotImplementedError("must be implemented")

# RawMallocBuffer is the simplest concrete version of this (in tinygrad/ops.py). it's used for the CLANG and LLVM backends
# it's just malloc(size * dtype.itemsize)
from tinygrad.ops import RawMallocBuffer

# Runtime is what actually runs the kernels
class Runtime(ABC):
  # `name` is the name of the function, and `prg` is the code
  # the constructor compiles the code
  def __init__(self, name:str, prg:str): pass
  # call runs the code on the bufs. NOTE: the output is always bufs[0], but this is just a convention
  def __call__(self, global_size:Optional[List[int]], local_size:Optional[List[int]], *bufs:List[RawBuffer]): pass

# ClangProgram is the simplest version (in tinygrad/runtime/ops_clang.py)
# __init__ calls clang, and __call__ calls the function in the *.so outputted by clang
# in CLANG, global_size and local_size are ignored
from tinygrad.runtime.ops_clang import ClangProgram

# a concrete example looks like this, that adds two size 1 RawBuffer
from tinygrad.helpers import dtypes

# first we create two numpy buffers containing 2 and 3
# then we copy the numpy in to RawMallocBuffers
# last, we create an empty output buffer
numpy_a, numpy_b = np.array([2], dtype=np.float32), np.array([3], dtype=np.float32)
input_a, input_b = RawMallocBuffer.fromCPU(numpy_a), RawMallocBuffer.fromCPU(numpy_b)
output = RawMallocBuffer(1, dtypes.float32)

# compile the program, run it, and 2+3 does indeed equal 5
program = ClangProgram("add", "void add(float *a, float *b, float *c) { *a = *b + *c; }")
program(None, None, output, input_a, input_b)  # NOTE: the None are for global_size and local_size. clean this up?
np.testing.assert_allclose(output.toCPU(), numpy_a+numpy_b)

# but we are nowhere near done!
# we need the LazyOp ASTs to actually be turned into code
# the current class looks roughly like this, but this will change and we will update the docs
# this stuff is in the terrible 528 lines of (tinygrad/codegen/*, code 2/10 aka turd quality)
class ASTKernel:  # (from tinygrad/codegen/ast.py)
  # create the kernel with the AST
  # NOTE: the AST contains the CompiledBuffers themselves as the root nodes. this will change
  def __init__(self, ast:LazyOp): pass
  def codegen(self) -> ASTRunner: pass

# we return a class that runs code on CompiledBuffers
class ASTRunner:  # (from tinygrad/ops.py)
  def __init__(self, name, prg, global_size:Optional[List[int]], local_size:Optional[List[int]]): pass
  def build(self, runtime:Runtime): pass
  def exec(self, bufs:List[CompiledBuffer]): pass

# that hides a lot of complexity that will be refactored, but that's the basic idea of code generation

# last, but not least (in fact one of the nicest things in tinygrad). the ShapeTracker
class ShapeTracker: pass

# TODO: finish this, the coffee shop is closing
