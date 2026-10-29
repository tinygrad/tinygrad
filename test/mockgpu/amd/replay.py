#!/usr/bin/env python3
from test.external.process_replay.process_replay import main
from test.mockgpu.amd.emu import compile_asm
from tinygrad.helpers import VERSION
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import UOp, Ops

def get_source(p:UOp) -> str: return next(x.arg for x in p.src if x.op is Ops.SOURCE)

def replay_asm(p:UOp, code:bytes, arch:str, renderer:Renderer, dispatch:tuple, name:str) -> tuple[str, str, tuple]:
  return get_source(compile_asm(code, arch, renderer, dispatch, name)), get_source(p), (name, arch, dispatch)

if __name__ == "__main__": main({"compile_asm":replay_asm}, f"emu_process_replay_{VERSION}")
