import numpy as np
import math
np.set_printoptions(suppress=True)
from copy import deepcopy
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad.codegen.search import bufs_from_lin, time_linearizer, actions, get_linearizer_actions
from extra.optimization.helpers import load_worlds, ast_str_to_lin, lin_to_feats
from extra.optimization.pretrain_policynet import PolicyNet
from extra.optimization.pretrain_valuenet import ValueNet

VALUE = getenv("VALUE")

if __name__ == "__main__":
  if VALUE:
    net = ValueNet()
    load_state_dict(net, safe_load("/tmp/valuenet.safetensors"))
  else:
    net = PolicyNet()
    load_state_dict(net, safe_load("/tmp/policynet.safetensors"))

  ast_strs = load_worlds()

  for ep_num,ast_str in enumerate(ast_strs):
    print("\nEPISODE", ep_num)
    lin = ast_str_to_lin(ast_str)
    rawbufs = bufs_from_lin(lin)

    """
    linhc = deepcopy(lin)
    linhc.hand_coded_optimizations()
    tm = time_linearizer(linhc, rawbufs)
    print(f"{tm*1e6:10.2f}", linhc.colored_shape())
    """

    pred_time = float('nan')
    while 1:
      if VALUE:
        acts,feats = [], []
        for k,v in get_linearizer_actions(lin).items():
          acts.append(k)
          feats.append(lin_to_feats(v))
        preds = net(Tensor(feats))
        pred_time = math.exp(preds.numpy().min())
        act = acts[preds.numpy().argmin()]
      else:
        probs = net(Tensor([lin_to_feats(lin)]))
        dist = probs.exp().numpy()
        act = dist.argmax()
      if act == 0: break
      try:
        lin.apply_opt(actions[act-1])
      except Exception:
        print("FAILED")
        break
      tm = time_linearizer(lin, rawbufs)
      print(f"{tm*1e6:10.2f} {pred_time:10.2f}", lin.colored_shape())