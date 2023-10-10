import sys
import random
from collections import defaultdict
from tqdm import tqdm
from tinygrad.helpers import dedup, ImageDType, getenv, ansilen
from tinygrad.graph import print_tree
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.lazy import var_vals_from_ast
from tinygrad.shape.symbolic import sym_infer
from tinygrad.ops import Device, Compiled, MemBuffer

# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.helpers import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Variable
inf, nan = float('inf'), float('nan')

# optimizer
from tinygrad.codegen.optimizer import Opt, OptOps
opt_options = {Opt(op=OptOps.LOCAL, axis=0, amt=3), Opt(op=OptOps.UPCAST, axis=3, amt=4), Opt(op=OptOps.UPCAST, axis=2, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=7), Opt(op=OptOps.UPCAST, axis=2, amt=5), Opt(op=OptOps.LOCAL, axis=4, amt=3), Opt(op=OptOps.GROUP, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=4, amt=2), Opt(op=OptOps.UPCAST, axis=4, amt=5), Opt(op=OptOps.GROUPTOP, axis=0, amt=256), Opt(op=OptOps.LOCAL, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=2, amt=3), Opt(op=OptOps.UPCAST, axis=1, amt=6), Opt(op=OptOps.UPCAST, axis=0, amt=4), Opt(op=OptOps.LOCAL, axis=3, amt=4), Opt(op=OptOps.UPCAST, axis=1, amt=3), Opt(op=OptOps.GROUPTOP, axis=2, amt=256), Opt(op=OptOps.LOCAL, axis=1, amt=16), Opt(op=OptOps.UPCAST, axis=0, amt=7), Opt(op=OptOps.UPCAST, axis=5, amt=3), Opt(op=OptOps.LOCAL, axis=3, amt=16), Opt(op=OptOps.LOCAL, axis=0, amt=2), Opt(op=OptOps.GROUPTOP, axis=1, amt=16), Opt(op=OptOps.LOCAL, axis=0, amt=8), Opt(op=OptOps.UPCAST, axis=3, amt=6), Opt(op=OptOps.UPCAST, axis=2, amt=4), Opt(op=OptOps.LOCAL, axis=4, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=3), Opt(op=OptOps.UPCAST, axis=2, amt=7), Opt(op=OptOps.UPCAST, axis=4, amt=4), Opt(op=OptOps.GROUP, axis=2, amt=8), Opt(op=OptOps.LOCAL, axis=2, amt=2), Opt(op=OptOps.LOCAL, axis=0, amt=32), Opt(op=OptOps.UPCAST, axis=0, amt=3), Opt(op=OptOps.LOCAL, axis=1, amt=3), Opt(op=OptOps.UPCAST, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=0, amt=6), Opt(op=OptOps.LOCAL, axis=2, amt=8), Opt(op=OptOps.UPCAST, axis=1, amt=5), Opt(op=OptOps.LOCAL, axis=3, amt=3), Opt(op=OptOps.UPCAST, axis=6, amt=4), Opt(op=OptOps.GROUPTOP, axis=1, amt=256), Opt(op=OptOps.UPCAST, axis=2, amt=3), Opt(op=OptOps.LOCAL, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=3, amt=2), Opt(op=OptOps.LOCAL, axis=0, amt=16), Opt(op=OptOps.UPCAST, axis=2, amt=6), Opt(op=OptOps.UPCAST, axis=5, amt=4), Opt(op=OptOps.UPCAST, axis=4, amt=3), Opt(op=OptOps.UPCAST, axis=3, amt=5), Opt(op=OptOps.GROUP, axis=1, amt=8), Opt(op=OptOps.LOCAL, axis=4, amt=16), Opt(op=OptOps.GROUPTOP, axis=0, amt=16), Opt(op=OptOps.LOCAL, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=0, amt=5), Opt(op=OptOps.GROUPTOP, axis=2, amt=16), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=0, amt=2), Opt(op=OptOps.LOCAL, axis=2, amt=4), Opt(op=OptOps.LOCAL, axis=3, amt=2), Opt(op=OptOps.LOCAL, axis=1, amt=8), Opt(op=OptOps.UPCAST, axis=1, amt=7), Opt(op=OptOps.LOCAL, axis=3, amt=8), Opt(op=OptOps.LOCAL, axis=2, amt=16)}

if __name__ == "__main__":
  ast_strs = dedup(open(sys.argv[1]).read().strip().split("\n"))
  print(len(opt_options))

  # reduce kernels only, no ImageDType
  #ast_strs = [x for x in ast_strs if "dtypes.image" not in x]
  ast_strs = [x for x in ast_strs if "ReduceOps" in x and "dtypes.image" not in x]

  # the device we are optimizing for
  device: Compiled = Device[Device.DEFAULT]
  print(f"optimizing for {Device.DEFAULT}")

  # random first kernels
  random.seed(1337)
  random.shuffle(ast_strs)
  ast_strs = ast_strs[1:2]

  print(f"loaded {len(ast_strs)} kernels")

  atm = []
  agflops = []
  all_opts = set()
  for ast_str in tqdm(ast_strs):
    ast = eval(ast_str)

    # linearize
    lin = Linearizer(ast)
    preopt = lin.colored_shape()
    lin.hand_coded_optimizations()
    postopt = lin.colored_shape()
    print(lin.applied_opts)
    """
    #if not lin.apply_tensor_cores(getenv("TC", 1)):
    lin.hand_coded_optimizations()
    postopt = lin.colored_shape()

    for x in lin.applied_opts: all_opts.add(x)

    print(len(all_opts))
    continue
    """

    """
    # linearize_alt
    lin2 = Linearizer(ast)
    lin2.hand_coded_optimizations_old()
    postopt_alt = lin2.colored_shape()

    print(preopt+' '*(37-ansilen(preopt)), "->", postopt)
    assert postopt == postopt_alt, f"{postopt} != {postopt_alt}"
    for s1,s2 in zip(lin.sts, lin2.sts): assert s1 == s2
    continue
    """

    lin.linearize()

    # create output/input buffers
    bufsts = defaultdict(list)
    for x in lin.bufs:
      if isinstance(x, MemBuffer):
        bufsts[x.idx].append(x)
    buffer_count = len(bufsts)
    rawbufs = [None]*buffer_count
    for k,x in bufsts.items():
      rawbufs[k] = device.buffer(max(y.st.size() for y in x), x[0].dtype)
    assert all(x is not None for x in rawbufs)

    # example var vals
    var_vals = {k:k.min for k in var_vals_from_ast(ast)}

    # time
    prg = device.to_program(lin)
    tm = min([prg(rawbufs, var_vals, force_wait=True) for _ in range(10)])
    atm.append(tm)

    # print
    #print_tree(ast)
    #for u in lin.uops: print(u)
    gflops = sym_infer(lin.info.flops, var_vals)*1e-9/tm
    agflops.append(gflops)
    if tm*1e6 > 100:
      print(f"{len(lin.uops):4d} uops, {str(lin.global_size):18s} {str(lin.local_size):12s} {tm*1e6:8.2f} us {gflops:7.2f} GFLOPS", preopt+' '*(37-ansilen(preopt)), "->", postopt)

  print(f"all kernels ran in {sum(atm)*1e3:.2f} ms")
  print(all_opts)

  if getenv("SHOW"):
    import matplotlib.pyplot as plt
    #plt.hist(agflops, bins=100)
    #plt.yscale('log')
    plt.scatter(atm, agflops)
    plt.xscale('log')
    plt.show()
