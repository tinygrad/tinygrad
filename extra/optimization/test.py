import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm
from tinygrad.helpers import dedup, ansilen
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.tensor import Tensor

from extra.optimization.rl import PolicyNet, lin_to_feats
from extra.optimization.helpers import ast_str_to_lin, get_linearizer_actions, time_linearizer, bufs_from_lin, actions

if __name__ == "__main__":
  # load worlds
  ast_strs = dedup(open("/tmp/sops").read().strip().split("\n"))
  ast_strs = [x for x in ast_strs if "ReduceOps" in x and "dtypes.image" not in x and "Variable" not in x]
  #ast_strs = [x for x in ast_strs if "Variable" not in x]

  # slow example
  #ast_strs = ast_strs[:1]

  #ast_strs = ast_strs[3:4]
  ast_strs = ast_strs[8:]

  # fast example
  #ast_strs = ast_strs[5:6]

  # load net
  #net = PolicyNet()
  #load_state_dict(net, safe_load("/tmp/speednet.safetensors"))

  for ast_str in ast_strs:
    print("\nEPISODE")
    lin = ast_str_to_lin(ast_str)

    linhc = deepcopy(lin)
    linhc.hand_coded_optimizations()
    assert all(x in actions for x in linhc.applied_opts), "all actions not available"

    rawbufs = bufs_from_lin(lin)
    tm1, gf1 = time_linearizer(linhc, rawbufs)
    print(f"{tm1:10.2f}", linhc.colored_shape(), f"with {len(linhc.applied_opts)} actions from {len(actions)} action space")

    step = 0
    while 1:
      tm, gflops = time_linearizer(lin, rawbufs)
      print(f"{tm:10.2f}", lin.colored_shape())
      acted_lins = get_linearizer_actions(lin)
      if len(acted_lins) == 0: break

      best_tm, best_lin = tm, lin
      for l in tqdm(list(acted_lins.values())):
        tm, gflops = time_linearizer(l, rawbufs)
        #print(f"{tm:10.2f}", l.colored_shape())
        if tm < best_tm:
          best_tm, best_lin = tm, l
      if lin == best_lin: break
      lin = best_lin



      #act = random.choice(list(acted_lins.keys()))
      #lin = acted_lins[act]

      step += 1


    continue
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

  #print(list(acts))
    """
    for o in linhc.applied_opts: lin.apply_opt(o)
    print(lin.colored_shape(), linhc.colored_shape())
    assert lin.colored_shape() == linhc.colored_shape()
    assert all(x==y for x,y in zip(lin.sts, linhc.sts))
    continue
    """