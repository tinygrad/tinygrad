import os
from enum import Enum
from typing import Tuple, NamedTuple, Union, Any, Type

DEBUG = int(os.getenv("DEBUG", "0"))

# these are the llops your accelerator must implement, along with toCpu
LoadOps = Enum("LoadOps", ["FROMCPU"])  # TODO: add zeros and randomness as load sources

# these 4 form the core ops of tinygrad
UnaryOps = Enum("UnaryOps", ["NOOP", "NEG", "RELU", "EXP", "LOG", "SIGN", "RECIPROCAL"])  # f(x)
BinaryOps = Enum("BinaryOps", ["ADD", "SUB", "MUL", "DIV", "POW", "CMPEQ"])  # f(x,y)
ReduceOps = Enum("ReduceOps", ["SUM", "MAX"])
MovementOps = Enum("MovementOps", ["RESHAPE", "PERMUTE", "EXPAND", "FLIP", "STRIDED", "PAD", "SHRINK"])  # these don't change data, only move it

# ProcessingOps don't have to be implemented if you can fuse Binary + Reduce
ProcessingOps = Enum("ProcessingOps", ["CONV"])

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, ProcessingOps, LoadOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[ProcessingOps], Type[LoadOps]]

class LazyOp(NamedTuple):
  op: Op
  src: Tuple[Any, ...]  # type: ignore
  arg: Any = None
  # TODO: add dest to support multiple outputs
