import os

from tinygrad.tensor import Tensor                                    # noqa: F401
from tinygrad.engine.jit import TinyJit                               # noqa: F401
from tinygrad.ops import UOp
Variable = UOp.variable
from tinygrad.dtype import dtypes                                     # noqa: F401
from tinygrad.helpers import GlobalCounters, fetch, Context, getenv   # noqa: F401
from tinygrad.device import Device                                    # noqa: F401

if int(os.getenv("TYPED", "0")):
  from typeguard import typechecked, install_import_hook
  install_import_hook(packages=['tinygrad'])
else:
  def typechecked(obj):
    return obj