import functools, pathlib
from tinygrad import UOp
from tinygrad.uop.uir import parse_uir

_DIR = pathlib.Path(__file__).resolve().parent

@functools.cache
def load(name:str) -> UOp|None:
  if not (p := _DIR / f"{name}.uir").is_file():
    try:
      from slopkernels import generate_all
      generate_all()
    except ImportError: return None
  return parse_uir(p.read_text(encoding="utf-8")) if p.is_file() else None
