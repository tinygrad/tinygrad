import functools, pathlib
from tinygrad import UOp, Context
from tinygrad.uop.uir import parse_uir

@functools.cache
def load(name:str) -> UOp|None:
  if not (p := pathlib.Path(__file__).resolve().parent / f"{name}.uir").is_file():
    try:
      from slopkernels import generate_all
      generate_all()
    except ImportError: return None
  with Context(SPEC=0): return parse_uir(p.read_text(encoding="utf-8")) if p.is_file() else None
