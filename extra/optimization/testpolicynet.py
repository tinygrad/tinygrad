import numpy as np
np.set_printoptions(suppress=True)
from copy import deepcopy
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad.codegen.search import bufs_from_lin, time_linearizer, actions
from extra.optimization.helpers import load_worlds, ast_str_to_lin, lin_to_feats
from extra.optimization.pretrain import PolicyNet

if __name__ == "__main__":
  net = PolicyNet()
  load_state_dict(net, safe_load("/tmp/policynet.safetensors"))

  ast_strs = load_worlds()

  for ep_num,ast_str in enumerate(ast_strs):
    print("\nEPISODE", ep_num)
    lin = ast_str_to_lin(ast_str)
    rawbufs = bufs_from_lin(lin)

    linhc = deepcopy(lin)
    linhc.hand_coded_optimizations()
    tm, gflops = time_linearizer(linhc, rawbufs)
    print(f"{tm:10.2f}", linhc.colored_shape())

    while 1:
      probs = net(Tensor([lin_to_feats(lin)]))
      dist = probs.exp().numpy()
      act = dist.argmax()
      if act == 0: break
      try:
        lin.apply_opt(actions[act-1])
      except Exception:
        print("FAILED")
        break
      tm, gflops = time_linearizer(lin, rawbufs)
      print(f"{tm:10.2f}", lin.colored_shape())