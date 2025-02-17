import pickle, sys
from tinygrad.engine.jit import TinyJit
from tinygrad.engine.realize import CompiledRunner
from tinygrad.renderer import ProgramSpec

if __name__ == "__main__":
  with open(sys.argv[1], "rb") as f:
    fxn: TinyJit = pickle.load(f)
    print(f"{f.tell()/1e6:.2f}M loaded")
  print(type(fxn))
  for ei in fxn.captured.jit_cache:
    if isinstance(ei.prg, CompiledRunner):
      p: ProgramSpec = ei.prg.p
      print(p.name)
      #print(p.ast)

