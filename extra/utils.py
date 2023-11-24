# type: ignore
import pickle, hashlib, zipfile, io, requests, struct, tempfile, platform, concurrent.futures
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Union

from tinygrad.helpers import prod, getenv, DEBUG, dtypes, get_child
from tinygrad.helpers import GlobalCounters
from tinygrad.tensor import Tensor
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import Device
from tinygrad.shape.view import strides_for_shape
OSX = platform.system() == "Darwin"
WINDOWS = platform.system() == "Windows"

def temp(x:str) -> str: return (Path(tempfile.gettempdir()) / x).as_posix()

def fetch(url):
  if url.startswith("/") or url.startswith("."):
    with open(url, "rb") as f:
      return f.read()
  fp = temp(hashlib.md5(url.encode('utf-8')).hexdigest())
  download_file(url, fp, skip_if_exists=not getenv("NOCACHE"))
  with open(fp, "rb") as f:
    return f.read()

def fetch_as_file(url):
  if url.startswith("/") or url.startswith("."):
    with open(url, "rb") as f:
      return f.read()
  fp = temp(hashlib.md5(url.encode('utf-8')).hexdigest())
  download_file(url, fp, skip_if_exists=not getenv("NOCACHE"))
  return fp

def download_file(url, fp, skip_if_exists=True):
  if skip_if_exists and Path(fp).is_file() and Path(fp).stat().st_size > 0:
    return
  r = requests.get(url, stream=True, timeout=10)
  assert r.status_code == 200
  progress_bar = tqdm(total=int(r.headers.get('content-length', 0)), unit='B', unit_scale=True, desc=url)
  (path := Path(fp).parent).mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
    for chunk in r.iter_content(chunk_size=16384):
      progress_bar.update(f.write(chunk))
    f.close()
    Path(f.name).rename(fp)


