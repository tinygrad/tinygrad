import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm
from tinygrad.helpers import dedup, ansilen
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.tensor import Tensor

#from extra.optimization.pretrain import PolicyNet
from extra.optimization.helpers import load_worlds, ast_str_to_lin, get_linearizer_actions, time_linearizer, bufs_from_lin, actions, lin_to_feats

if __name__ == "__main__":
  # load worlds
  ast_strs = load_worlds()

  for ast_str in ast_strs:
    print("\nEPISODE")
    lin = ast_str_to_lin(ast_str)

    linhc = deepcopy(lin)
    linhc.hand_coded_optimizations()
    if not all(x in actions for x in linhc.applied_opts):
      print("skipping", linhc.colored_shape())
      continue

    rawbufs = bufs_from_lin(lin)
    tm1, gf1 = time_linearizer(linhc, rawbufs)
    print(f"{tm1:10.2f}", linhc.colored_shape(), f"with {len(linhc.applied_opts)} actions from {len(actions)} action space")

    while 1:
      tm, gflops = time_linearizer(lin, rawbufs)
      print(f"{tm:10.2f}", lin.colored_shape())
      acted_lins = get_linearizer_actions(lin)
      if len(acted_lins) == 0: break

      best_tm, best_lin = tm, lin
      for l in list(acted_lins.values()):
        tm, gflops = time_linearizer(l, rawbufs)
        #print(f"{tm:10.2f}", l.colored_shape())
        if tm < best_tm:
          best_tm, best_lin = tm, l
      if lin == best_lin: break
      lin = best_lin
