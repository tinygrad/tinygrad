"""Compat shim — implementation lives in rdna3.py (Gabriel/x86 layout)."""
from tinygrad.renderer.isa.rdna3 import *  # noqa: F403
from tinygrad.renderer.isa import rdna3 as _rdna3
from tinygrad.renderer.isa.rdna3 import AMDRenderer as RDNA3Renderer, AMDOps as RDNA3Ops

# star-import skips underscore names; tests/codegen still poke amd_lib._*
def __getattr__(name: str):
  return getattr(_rdna3, name)

__all__ = ["AMDRenderer", "AMDOps", "RDNA3Renderer", "RDNA3Ops"]
