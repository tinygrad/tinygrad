import numpy as np
from tinygrad.helpers import dedup, ansilen
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.tensor import Tensor

# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.helpers import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Variable
inf, nan = float('inf'), float('nan')

from extra.optimization.rl import PolicyNet, lin_to_feats, get_linearizer_actions, bufs_from_lin, time_linearizer

if __name__ == "__main__":
  # load worlds
  ast_strs = dedup(open("/tmp/sops").read().strip().split("\n"))
  ast_strs = [x for x in ast_strs if "ReduceOps" in x and "dtypes.image" not in x and "Variable" not in x]
  ast_strs = ast_strs[:7]

  # load net
  net = PolicyNet()
  load_state_dict(net, safe_load("/tmp/speednet.safetensors"))

  for ast_str in ast_strs:
    print("EPISODE")
    ast = eval(ast_str)

    # hand coded opt
    linhc = Linearizer(ast)
    rawbufs = bufs_from_lin(linhc)
    linhc.hand_coded_optimizations()
    tm1, gf1 = time_linearizer(linhc, rawbufs)
    print(linhc.colored_shape(), f"{tm1:7.2f}")

    # linearize
    lin = Linearizer(ast)
    tm1, gf1 = time_linearizer(lin, rawbufs)
    print(lin.colored_shape(), f"{tm1:7.2f}")

    while 1:
      # enumerate all the actions as linearizers and run the model
      acted_lins = get_linearizer_actions(lin)
      lfeats = [lin_to_feats(lin)] + [lin_to_feats(x) for x in acted_lins.values()]
      preds = net(Tensor(lfeats)).exp().numpy()

      tact = np.argmin(preds[:, 0])
      if tact == 0: break
      act = list(acted_lins.keys())[(tact-1)]
      lin2 = acted_lins[act]

      # take action
      tm2, gf2 = time_linearizer(lin2, rawbufs)
      print(lin2.colored_shape(), f"{tm2:7.2f}")

      # loop
      lin = lin2

