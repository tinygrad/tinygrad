# type: ignore
import tempfile, platform
from pathlib import Path

OSX = platform.system() == "Darwin"
WINDOWS = platform.system() == "Windows"

def temp(x:str) -> str: return (Path(tempfile.gettempdir()) / x).as_posix()

def get_child(parent, key):
  obj = parent
  for k in key.split('.'):
    if k.isnumeric():
      obj = obj[int(k)]
    elif isinstance(obj, dict):
      obj = obj[k]
    else:
      obj = getattr(obj, k)
  return obj
