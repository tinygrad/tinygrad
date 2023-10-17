# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.helpers import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Variable
inf, nan = float('inf'), float('nan')

# kernel unpacker
from tinygrad.codegen.linearizer import Linearizer
def ast_str_to_lin(ast_str): return Linearizer(eval(ast_str))

# load worlds, a dataset of about 12k kernels
import gzip
from pathlib import Path
import random
from tinygrad.helpers import dedup
def load_worlds(filter_reduce=True, filter_noimage=True, filter_novariable=True):
  fn = Path(__file__).parent.parent / "datasets/sops.gz"
  ast_strs = dedup(gzip.open(fn).read().decode('utf-8').strip().split("\n"))
  if filter_reduce: ast_strs = [x for x in ast_strs if "ReduceOps" in x]
  if filter_noimage: ast_strs = [x for x in ast_strs if "dtypes.image" not in x]
  if filter_novariable: ast_strs = [x for x in ast_strs if "Variable" not in x]
  random.seed(1337)
  random.shuffle(ast_strs)
  return ast_strs

def assert_same_lin(l1, l2):
  assert l1.colored_shape() == l2.colored_shape()
  assert all(x==y for x,y in zip(l1.sts, l2.sts))

# get features
import math
from tinygrad.shape.symbolic import Node

MAX_DIMS = 16
def lin_to_feats(lin):
  assert lin.shape_len < MAX_DIMS, "too many dims"

  all_colors = ["blue", "cyan", "white", "green", "red", "magenta", "yellow"]
  lc = [all_colors.index(x) for x in lin.colors()]
  #my_sts = dedup([(x.shape == lin.full_shape, x.real_strides()) for x in lin.sts[1:]])

  # first, the full shape, including the colors
  ret = []
  for s,c in zip(lin.full_shape,lc):
    if isinstance(s, Node):
      ret.append(False)
      ret += [0]*7
    else:
      ret.append(True)
      ret.append(math.log2(s))
      ret.append(min(33, s))
      ret.append(s%2 == 0)
      ret.append(s%3 == 0)
      ret.append(s%4 == 0)
      ret.append(s%8 == 0)
      ret.append(s%16 == 0)
    cc = [0]*7
    cc[c] = 1
    ret += cc
  ret += [0] * (15*(MAX_DIMS-len(lin.full_shape)))
  ret = [float(x) for x in ret]

  assert len(ret) == 240, f"wrong len {len(ret)}"
  return ret