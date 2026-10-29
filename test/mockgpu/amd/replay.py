#!/usr/bin/env python3
import functools, sys
from dataclasses import replace
from test.external.process_replay.process_replay import main
from test.mockgpu.amd.emu import lift_asm
from tinygrad.codegen import to_program
from tinygrad.helpers import VERSION, Target
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import ClangRenderer, HIPRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.uop.ops import UOp, Ops

def get_source(p:UOp) -> str: return next(x.arg for x in p.src if x.op is Ops.SOURCE)

def replay_asm(ast:UOp, code:bytes, arch:str, renderer:Renderer, dispatch:tuple, name:str,
               backends:tuple[str, ...]) -> tuple[str, str, tuple]:
  compare = lift_asm(code, arch, renderer, dispatch, name)
  good_sources, compare_sources = [], []
  for backend in backends:
    target = Target("AMD", "HIP", {"rdna3":"gfx1100", "rdna4":"gfx1200", "cdna":"gfx950"}[arch]) if backend == "amd" else \
      replace(renderer.target, device="CPU", renderer=backend.upper())
    ren = {"clang":ClangRenderer, "llvm":CPULLVMRenderer, "amd":HIPRenderer}[backend](target)
    good_sources.append(get_source(to_program(ast, ren)))
    compare_sources.append(get_source(to_program(compare, ren)))
  return "\n".join(compare_sources), "\n".join(good_sources), (name, backends, dispatch)

if __name__ == "__main__":
  backends = tuple((sys.argv[1] if len(sys.argv) > 1 else "clang").split(","))
  if any(b not in ("clang", "llvm", "amd") for b in backends): raise SystemExit("renderers must be comma-separated clang, llvm, amd")
  main({"lift_asm":functools.partial(replay_asm, backends=backends)}, f"emu_process_replay_{VERSION}")
