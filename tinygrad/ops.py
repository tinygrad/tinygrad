import os
from enum import Enum
from typing import Union, Type, NamedTuple, Tuple, Any, List
import functools, operator

DEBUG = int(os.getenv("DEBUG", "0"))

# these are the llops your accelerator must implement, along with toCpu
UnaryOps = Enum("UnaryOps", ["NOOP", "NEG", "RELU", "EXP", "LOG", "SIGN", "RECIPROCAL"])
BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "CMPEQ"])
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "EXPAND", "FLIP", "STRIDED", "PAD", "SHRINK"])
ProcessingOps = Enum("ProcessingOps", ["CONV"])
LoadOps = Enum("LoadOps", ["FROMCPU", "CONTIGUOUS"])

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[ProcessingOps], Type[LoadOps]]

class LazyOp(NamedTuple):
  op: Op
  # Any == Union[LazyOp, LazyBuffer, DeviceBuffer]
  src: Tuple[Any, ...]  # type: ignore
  arg: Any = None
  # TODO: add dest to support multiple outputs

# Any == Union[LazyBuffer, DeviceBuffer]
def get_lazybuffers(op:LazyOp) -> List[Any]: return functools.reduce(operator.add, [get_lazybuffers(x) if isinstance(x, LazyOp) else [x] for x in op.src], [])
def get_lazyops(op:LazyOp) -> List[LazyOp]: return functools.reduce(operator.add, [get_lazyops(x) for x in op.src if isinstance(x, LazyOp)], [op])