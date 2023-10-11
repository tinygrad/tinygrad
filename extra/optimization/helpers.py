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

# actions
from tinygrad.helpers import dtypes, prod
from tinygrad.codegen.optimizer import Opt, OptOps
actions = [Opt(op=OptOps.UPCAST, axis=1, amt=21), Opt(op=OptOps.UPCAST, axis=3, amt=27), Opt(op=OptOps.LOCAL, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=4, amt=10), Opt(op=OptOps.UPCAST, axis=5, amt=9), Opt(op=OptOps.UPCAST, axis=5, amt=27), Opt(op=OptOps.UPCAST, axis=2, amt=3), Opt(op=OptOps.LOCAL, axis=0, amt=3), Opt(op=OptOps.UPCAST, axis=3, amt=2), Opt(op=OptOps.UPCAST, axis=1, amt=5), Opt(op=OptOps.GROUPTOP, axis=0, amt=256), Opt(op=OptOps.UPCAST, axis=2, amt=12), Opt(op=OptOps.UPCAST, axis=1, amt=14), Opt(op=OptOps.UPCAST, axis=3, amt=11), Opt(op=OptOps.UPCAST, axis=1, amt=32), Opt(op=OptOps.UPCAST, axis=4, amt=3), Opt(op=OptOps.LOCAL, axis=2, amt=3), Opt(op=OptOps.UPCAST, axis=5, amt=2), Opt(op=OptOps.UPCAST, axis=6, amt=31), Opt(op=OptOps.UPCAST, axis=5, amt=11), Opt(op=OptOps.UPCAST, axis=0, amt=24), Opt(op=OptOps.LOCAL, axis=1, amt=16), Opt(op=OptOps.UPCAST, axis=2, amt=5), Opt(op=OptOps.UPCAST, axis=1, amt=7), Opt(op=OptOps.UPCAST, axis=3, amt=4), Opt(op=OptOps.UPCAST, axis=6, amt=3), Opt(op=OptOps.UPCAST, axis=1, amt=16), Opt(op=OptOps.UPCAST, axis=3, amt=13), Opt(op=OptOps.UPCAST, axis=1, amt=25), Opt(op=OptOps.UPCAST, axis=4, amt=5), Opt(op=OptOps.UPCAST, axis=6, amt=24), Opt(op=OptOps.UPCAST, axis=5, amt=4), Opt(op=OptOps.UPCAST, axis=5, amt=13), Opt(op=OptOps.UPCAST, axis=4, amt=26), Opt(op=OptOps.UPCAST, axis=3, amt=6), Opt(op=OptOps.UPCAST, axis=1, amt=9), Opt(op=OptOps.UPCAST, axis=3, amt=15), Opt(op=OptOps.UPCAST, axis=2, amt=28), Opt(op=OptOps.UPCAST, axis=0, amt=10), Opt(op=OptOps.UPCAST, axis=6, amt=26), Opt(op=OptOps.UPCAST, axis=5, amt=6), Opt(op=OptOps.UPCAST, axis=7, amt=3), Opt(op=OptOps.LOCAL, axis=3, amt=8), Opt(op=OptOps.UPCAST, axis=1, amt=2), Opt(op=OptOps.LOCAL, axis=4, amt=16), Opt(op=OptOps.UPCAST, axis=1, amt=11), Opt(op=OptOps.UPCAST, axis=3, amt=8), Opt(op=OptOps.GROUPTOP, axis=1, amt=16), Opt(op=OptOps.UPCAST, axis=4, amt=28), Opt(op=OptOps.UPCAST, axis=1, amt=20), Opt(op=OptOps.UPCAST, axis=2, amt=21), Opt(op=OptOps.UPCAST, axis=0, amt=3), Opt(op=OptOps.GROUP, axis=1, amt=8), Opt(op=OptOps.UPCAST, axis=2, amt=30), Opt(op=OptOps.UPCAST, axis=0, amt=12), Opt(op=OptOps.UPCAST, axis=5, amt=8), Opt(op=OptOps.LOCAL, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=4, amt=12), Opt(op=OptOps.LOCAL, axis=0, amt=2), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=1, amt=13), Opt(op=OptOps.UPCAST, axis=3, amt=10), Opt(op=OptOps.UPCAST, axis=4, amt=30), Opt(op=OptOps.GROUPTOP, axis=1, amt=256), Opt(op=OptOps.UPCAST, axis=2, amt=14), Opt(op=OptOps.UPCAST, axis=0, amt=5), Opt(op=OptOps.LOCAL, axis=0, amt=32), Opt(op=OptOps.UPCAST, axis=2, amt=32), Opt(op=OptOps.LOCAL, axis=3, amt=3), Opt(op=OptOps.LOCAL, axis=4, amt=2), Opt(op=OptOps.UPCAST, axis=4, amt=14), Opt(op=OptOps.UPCAST, axis=1, amt=6), Opt(op=OptOps.UPCAST, axis=3, amt=3), Opt(op=OptOps.UPCAST, axis=1, amt=15), Opt(op=OptOps.UPCAST, axis=4, amt=32), Opt(op=OptOps.UPCAST, axis=2, amt=7), Opt(op=OptOps.UPCAST, axis=6, amt=5), Opt(op=OptOps.LOCAL, axis=0, amt=16), Opt(op=OptOps.UPCAST, axis=2, amt=16), Opt(op=OptOps.UPCAST, axis=2, amt=25), Opt(op=OptOps.UPCAST, axis=0, amt=7), Opt(op=OptOps.UPCAST, axis=3, amt=24), Opt(op=OptOps.LOCAL, axis=1, amt=8), Opt(op=OptOps.UPCAST, axis=4, amt=7), Opt(op=OptOps.LOCAL, axis=2, amt=16), Opt(op=OptOps.UPCAST, axis=4, amt=16), Opt(op=OptOps.UPCAST, axis=1, amt=8), Opt(op=OptOps.UPCAST, axis=5, amt=24), Opt(op=OptOps.UPCAST, axis=2, amt=9), Opt(op=OptOps.UPCAST, axis=6, amt=7), Opt(op=OptOps.UPCAST, axis=6, amt=16), Opt(op=OptOps.UPCAST, axis=2, amt=27), Opt(op=OptOps.UPCAST, axis=0, amt=9), Opt(op=OptOps.UPCAST, axis=3, amt=26), Opt(op=OptOps.UPCAST, axis=4, amt=9), Opt(op=OptOps.LOCAL, axis=3, amt=16), Opt(op=OptOps.UPCAST, axis=4, amt=27), Opt(op=OptOps.UPCAST, axis=5, amt=26), Opt(op=OptOps.UPCAST, axis=2, amt=2), Opt(op=OptOps.UPCAST, axis=2, amt=11), Opt(op=OptOps.UPCAST, axis=2, amt=20), Opt(op=OptOps.UPCAST, axis=0, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=28), Opt(op=OptOps.UPCAST, axis=6, amt=27), Opt(op=OptOps.LOCAL, axis=1, amt=3), Opt(op=OptOps.UPCAST, axis=4, amt=2), Opt(op=OptOps.LOCAL, axis=2, amt=2), Opt(op=OptOps.UPCAST, axis=4, amt=11), Opt(op=OptOps.UPCAST, axis=7, amt=7), Opt(op=OptOps.UPCAST, axis=1, amt=3), Opt(op=OptOps.UPCAST, axis=4, amt=20), Opt(op=OptOps.GROUPTOP, axis=2, amt=16), Opt(op=OptOps.UPCAST, axis=2, amt=4), Opt(op=OptOps.LOCAL, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=6, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=12), Opt(op=OptOps.UPCAST, axis=1, amt=24), Opt(op=OptOps.UPCAST, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=2, amt=31), Opt(op=OptOps.LOCAL, axis=3, amt=2), Opt(op=OptOps.UPCAST, axis=4, amt=4), Opt(op=OptOps.LOCAL, axis=2, amt=4), Opt(op=OptOps.UPCAST, axis=5, amt=3), Opt(op=OptOps.UPCAST, axis=4, amt=13), Opt(op=OptOps.UPCAST, axis=2, amt=6), Opt(op=OptOps.GROUPTOP, axis=2, amt=256), Opt(op=OptOps.UPCAST, axis=3, amt=5), Opt(op=OptOps.UPCAST, axis=6, amt=4), Opt(op=OptOps.UPCAST, axis=2, amt=15), Opt(op=OptOps.UPCAST, axis=1, amt=17), Opt(op=OptOps.UPCAST, axis=3, amt=14), Opt(op=OptOps.UPCAST, axis=6, amt=13), Opt(op=OptOps.UPCAST, axis=2, amt=24), Opt(op=OptOps.UPCAST, axis=0, amt=6), Opt(op=OptOps.UPCAST, axis=3, amt=32), Opt(op=OptOps.LOCAL, axis=3, amt=4), Opt(op=OptOps.UPCAST, axis=4, amt=6), Opt(op=OptOps.LOCAL, axis=4, amt=3), Opt(op=OptOps.UPCAST, axis=5, amt=5), Opt(op=OptOps.UPCAST, axis=7, amt=2), Opt(op=OptOps.UPCAST, axis=4, amt=15), Opt(op=OptOps.UPCAST, axis=4, amt=24), Opt(op=OptOps.LOCAL, axis=0, amt=8), Opt(op=OptOps.UPCAST, axis=2, amt=8), Opt(op=OptOps.UPCAST, axis=3, amt=7), Opt(op=OptOps.UPCAST, axis=1, amt=10), Opt(op=OptOps.UPCAST, axis=6, amt=6), Opt(op=OptOps.UPCAST, axis=2, amt=17), Opt(op=OptOps.UPCAST, axis=3, amt=16), Opt(op=OptOps.GROUP, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=1, amt=28), Opt(op=OptOps.LOCAL, axis=2, amt=8), Opt(op=OptOps.UPCAST, axis=4, amt=8), Opt(op=OptOps.UPCAST, axis=5, amt=7), Opt(op=OptOps.UPCAST, axis=7, amt=4), Opt(op=OptOps.UPCAST, axis=5, amt=16), Opt(op=OptOps.GROUPTOP, axis=0, amt=16), Opt(op=OptOps.UPCAST, axis=2, amt=10), Opt(op=OptOps.UPCAST, axis=1, amt=12), Opt(op=OptOps.UPCAST, axis=3, amt=9)]
#actions = [x for x in actions if x.op not in [OptOps.GROUP, OptOps.GROUPTOP]]

from copy import deepcopy
def get_linearizer_actions(lin):
  acted_lins = {}
  for i,a in enumerate(actions):
    lin2 = deepcopy(lin)
    try:
      lin2.apply_opt(a)
      up, lcl = 1, 1
      for s,c in zip(lin2.full_shape, lin2.colors()):
        if c in {"magenta", "yellow"}: up *= s
        if c in {"cyan", "green", "white"}: lcl *= s
      if up > 256 or lcl > 256: continue
      acted_lins[i] = lin2
    except Exception:
      pass
  return acted_lins

# running

from collections import defaultdict
from tinygrad.ops import Device
from tinygrad.lazy import var_vals_from_ast
from tinygrad.shape.symbolic import sym_infer

device = Device[Device.DEFAULT]

def time_linearizer(lin, rawbufs, allow_test_size=True):
  lin = deepcopy(lin)  # TODO: remove the need for this
  var_vals = {k:k.min for k in var_vals_from_ast(lin.ast)}
  try:
    lin.linearize()
    prg = device.to_program(lin)
    real_global_size = prg.global_size[:]
    prg.global_size = [1,1,1]
    tm = prg(rawbufs, var_vals, force_wait=True)
  except Exception as e:
    print("FAILED")
    print(lin.ast)
    print(lin.applied_opts)
    raise e

  if allow_test_size:
    test_global_size = real_global_size[:]
    while prod(test_global_size) > 4096:
      for j in range(2,-1,-1):
        if test_global_size[j] > 1:
          test_global_size[j] //= 2
          break
    factor = prod(real_global_size) / prod(test_global_size)
    prg.global_size = test_global_size
  else:
    prg.global_size = real_global_size
    factor = 1

  tm = min([prg(rawbufs, var_vals, force_wait=True) for _ in range(2)])
  tm *= factor
  gflops = sym_infer(lin.info.flops, var_vals)*1e-9/tm
  return tm*1e6, gflops

def bufs_from_lin(lin):
  bufsts = defaultdict(list)
  for x in lin.membufs: bufsts[x.idx].append(x)
  rawbufs = [None]*len(bufsts)
  for k,x in bufsts.items(): rawbufs[k] = device.buffer(max(y.st.size() for y in x), x[0].dtype)
  assert all(x is not None for x in rawbufs)
  return rawbufs

# load worlds
import random
from tinygrad.helpers import dedup
def load_worlds():
  ast_strs = dedup(open("/tmp/sops").read().strip().split("\n"))
  ast_strs = [x for x in ast_strs if "ReduceOps" in x and "dtypes.image" not in x and "Variable" not in x]
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