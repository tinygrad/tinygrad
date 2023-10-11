import sys
import random
from collections import defaultdict
from tqdm import tqdm
import math
from tinygrad.helpers import dedup, ImageDType, getenv, ansilen, prod
from tinygrad.graph import print_tree
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.lazy import var_vals_from_ast
from tinygrad.shape.symbolic import sym_infer, Node
from tinygrad.ops import Device, Compiled, MemBuffer

# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.helpers import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Variable
inf, nan = float('inf'), float('nan')


def bufs_from_lin(lin):
  bufsts = defaultdict(list)
  for x in lin.membufs: bufsts[x.idx].append(x)
  buffer_count = len(bufsts)
  rawbufs = [None]*buffer_count
  for k,x in bufsts.items():
    rawbufs[k] = device.buffer(max(y.st.size() for y in x), x[0].dtype)
  assert all(x is not None for x in rawbufs)
  return rawbufs

dset_x, dset_y = [], []
def time_linearizer(lin, rawbufs, preopt):
  lin.linearize()

  # example var vals
  var_vals = {k:k.min for k in var_vals_from_ast(ast)}

  # time
  prg = device.to_program(lin)
  #i = 0
  #while prod(prg.global_size) > 1024*16:
  #  ogs *= prg.global_size[i]
  #  prg.global_size[i] = 1
  #  i += 1

  tm = prg(rawbufs, var_vals, force_wait=True)
  if tm < 0.1: tm = min([tm]+[prg(rawbufs, var_vals, force_wait=True) for _ in range(10)])
  atm.append(tm)

  # print
  #print_tree(ast)
  #for u in lin.uops: print(u)
  gflops = sym_infer(lin.info.flops, var_vals)*1e-9/tm
  agflops.append(gflops)

  dset_x.append(lin_to_feats(lin))
  dset_y.append((tm*1e6, gflops))
  #if tm*1e6 > 100:
  print(f"{len(lin.uops):4d} uops, {len(lin.applied_opts)} opts, {len(dedup([x.st for x in lin.membufs]))} bufs, {len(lin.full_shape)} dims, {str(lin.global_size):18s} {str(lin.local_size):12s} {tm*1e6:8.2f} us {gflops:7.2f} GFLOPS", preopt+' '*(37-ansilen(preopt)), "->", lin.colored_shape())

# optimizer
from tinygrad.codegen.optimizer import Opt, OptOps
opt_options = {Opt(op=OptOps.LOCAL, axis=0, amt=3), Opt(op=OptOps.UPCAST, axis=3, amt=4), Opt(op=OptOps.UPCAST, axis=2, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=7), Opt(op=OptOps.UPCAST, axis=2, amt=5), Opt(op=OptOps.LOCAL, axis=4, amt=3), Opt(op=OptOps.GROUP, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=4, amt=2), Opt(op=OptOps.UPCAST, axis=4, amt=5), Opt(op=OptOps.GROUPTOP, axis=0, amt=256), Opt(op=OptOps.LOCAL, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=2, amt=3), Opt(op=OptOps.UPCAST, axis=1, amt=6), Opt(op=OptOps.UPCAST, axis=0, amt=4), Opt(op=OptOps.LOCAL, axis=3, amt=4), Opt(op=OptOps.UPCAST, axis=1, amt=3), Opt(op=OptOps.GROUPTOP, axis=2, amt=256), Opt(op=OptOps.LOCAL, axis=1, amt=16), Opt(op=OptOps.UPCAST, axis=0, amt=7), Opt(op=OptOps.UPCAST, axis=5, amt=3), Opt(op=OptOps.LOCAL, axis=3, amt=16), Opt(op=OptOps.LOCAL, axis=0, amt=2), Opt(op=OptOps.GROUPTOP, axis=1, amt=16), Opt(op=OptOps.LOCAL, axis=0, amt=8), Opt(op=OptOps.UPCAST, axis=3, amt=6), Opt(op=OptOps.UPCAST, axis=2, amt=4), Opt(op=OptOps.LOCAL, axis=4, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=3), Opt(op=OptOps.UPCAST, axis=2, amt=7), Opt(op=OptOps.UPCAST, axis=4, amt=4), Opt(op=OptOps.GROUP, axis=2, amt=8), Opt(op=OptOps.LOCAL, axis=2, amt=2), Opt(op=OptOps.LOCAL, axis=0, amt=32), Opt(op=OptOps.UPCAST, axis=0, amt=3), Opt(op=OptOps.LOCAL, axis=1, amt=3), Opt(op=OptOps.UPCAST, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=0, amt=6), Opt(op=OptOps.LOCAL, axis=2, amt=8), Opt(op=OptOps.UPCAST, axis=1, amt=5), Opt(op=OptOps.LOCAL, axis=3, amt=3), Opt(op=OptOps.UPCAST, axis=6, amt=4), Opt(op=OptOps.GROUPTOP, axis=1, amt=256), Opt(op=OptOps.UPCAST, axis=2, amt=3), Opt(op=OptOps.LOCAL, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=3, amt=2), Opt(op=OptOps.LOCAL, axis=0, amt=16), Opt(op=OptOps.UPCAST, axis=2, amt=6), Opt(op=OptOps.UPCAST, axis=5, amt=4), Opt(op=OptOps.UPCAST, axis=4, amt=3), Opt(op=OptOps.UPCAST, axis=3, amt=5), Opt(op=OptOps.GROUP, axis=1, amt=8), Opt(op=OptOps.LOCAL, axis=4, amt=16), Opt(op=OptOps.GROUPTOP, axis=0, amt=16), Opt(op=OptOps.LOCAL, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=0, amt=5), Opt(op=OptOps.GROUPTOP, axis=2, amt=16), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=0, amt=2), Opt(op=OptOps.LOCAL, axis=2, amt=4), Opt(op=OptOps.LOCAL, axis=3, amt=2), Opt(op=OptOps.LOCAL, axis=1, amt=8), Opt(op=OptOps.UPCAST, axis=1, amt=7), Opt(op=OptOps.LOCAL, axis=3, amt=8), Opt(op=OptOps.LOCAL, axis=2, amt=16)}

if __name__ == "__main__":
  ast_strs = dedup(open(sys.argv[1]).read().strip().split("\n"))
  print(len(opt_options))

  # reduce kernels only, no ImageDType
  #ast_strs = [x for x in ast_strs if "dtypes.image" not in x]
  #ast_strs = [x for x in ast_strs if "ReduceOps" in x and "dtypes.image" not in x]
  ast_strs = [x for x in ast_strs if "ReduceOps" in x and "dtypes.image" not in x and "Variable" not in x]

  # the device we are optimizing for
  device: Compiled = Device[Device.DEFAULT]
  print(f"optimizing for {Device.DEFAULT}")

  # random first kernels
  random.seed(1337)
  random.shuffle(ast_strs)
  #ast_strs = ast_strs[1:2]
  #ast_strs = ast_strs[:1000]
  #ast_strs = ast_strs[:1000]
  #ast_strs = ast_strs[1000:1100]

  print(f"loaded {len(ast_strs)} kernels")
  #print(f"optimizing with {len(opt_options)} actions")

  atm = []
  agflops = []
  all_opts = set()
  for ast_str in tqdm(ast_strs):
    ast = eval(ast_str)

    # linearize
    lin = Linearizer(ast)
    preopt = lin.colored_shape()
    lin.hand_coded_optimizations()

    # create output/input buffers
    rawbufs = bufs_from_lin(lin)

    print("NEW KERNEL")
    ao = lin.applied_opts
    for i in range(0, len(ao)):
      try:
        lin = Linearizer(ast)
        for o in ao[:i]: lin.apply_opt(o)
        time_linearizer(lin, rawbufs, preopt)
      except Exception:
        print("FAILEDDDD")

    """
    print("baseline done")
    for o in opt_options:
      for o2 in opt_options:
        if o.op is OptOps.GROUP or o.op is OptOps.GROUPTOP: continue
        if o2.op is OptOps.GROUP or o2.op is OptOps.GROUPTOP: continue
        lin = Linearizer(ast)
        for oo in ao[0:3]: lin.apply_opt(oo)
        try:
          lin.apply_opt(o)
        except Exception:
          continue
        try:
          lin.apply_opt(o2)
        except Exception:
          continue
        try:
          time_linearizer(lin, rawbufs, preopt)
        except Exception:
          continue
    """

  print(f"all kernels ran in {sum(atm)*1e3:.2f} ms")
  print(all_opts)

  from tinygrad.tensor import Tensor

  tx = Tensor(dset_x)
  ty = Tensor(dset_y)

  print(tx.shape, tx.dtype)
  print(ty.shape, ty.dtype)

  tx.to("disk:/tmp/allopt_x").realize()
  ty.to("disk:/tmp/allopt_y").realize()

  if getenv("SHOW"):
    import matplotlib.pyplot as plt
    #plt.hist(agflops, bins=100)
    #plt.yscale('log')
    plt.scatter(atm, agflops)
    plt.xscale('log')
    plt.show()
