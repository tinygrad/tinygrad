"""Compat shim — implementation lives in rdna3.py (Gabriel/x86 layout)."""
from tinygrad.renderer.isa import rdna3 as _rdna3
from tinygrad.renderer.isa.rdna3 import AMDOps, AMDRenderer, AMDOps as RDNA3Ops, AMDRenderer as RDNA3Renderer

# tests/codegen still poke amd_lib._* private helpers
def __getattr__(name: str):
  return getattr(_rdna3, name)

__all__ = ["AMDRenderer", "AMDOps", "RDNA3Renderer", "RDNA3Ops"]
