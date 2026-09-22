import ctypes
from dataclasses import dataclass
from typing import Any

@dataclass(eq=False)
class BoundCFunctionRef:
  pointer: int
  name: str
  owner: Any
  restype: Any = ctypes.c_int
  lib: str = 'libc'

  @classmethod
  def bind(cls, callback: Any, name: str) -> 'BoundCFunctionRef':
    pointer = ctypes.cast(callback, ctypes.c_void_p).value
    if pointer is None: raise ValueError(f'{name}: callback has no address')
    return cls(pointer, name, callback)

  def address(self) -> int:
    return self.pointer

  @property
  def __name__(self): return self.name
