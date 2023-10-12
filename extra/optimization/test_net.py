import numpy as np
import math
import random
np.set_printoptions(suppress=True)
from copy import deepcopy
from tinygrad.helpers import getenv, colored
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

  ast_strs = load_worlds(filter_reduce=True)

  # real randomness
  random.seed()
  random.shuffle(ast_strs)

  wins = 0
  for ep_num,ast_str in enumerate(ast_strs):
    print("\nEPISODE", ep_num, f"win {wins*100/max(1,ep_num):.2f}%")
    lin = ast_str_to_lin(ast_str)
    rawbufs = bufs_from_lin(lin)

    linhc = deepcopy(lin)
    linhc.hand_coded_optimizations()
    tmhc = time_linearizer(linhc, rawbufs)
    print(f"{tmhc*1e6:10.2f}     HC    ", linhc.colored_shape())

    pred_time = float('nan')
    tm = float('inf')
    while 1:
      if VALUE:
        acts,feats = [], []
        for k,v in get_linearizer_actions(lin).items():
          acts.append(k)
          feats.append(lin_to_feats(v))
        preds = net(Tensor(feats))
        tms, sigmas = preds[:, 0], preds[:, 2].exp()
        gflops = preds[:, 1]
        # 1 stddev of overest
        #preds = tms + sigmas
        ngflops = gflops.exp().numpy()
        ntms = tms.exp().numpy()
        gain = ((ngflops/ngflops[0]) + (ntms[0]/ntms))/2
        #print(gain)
        #print(gflops.numpy().argmax(), tms.numpy().argmin())
        #act = acts[tms.numpy().argmin()]
        #act = acts[gflops.numpy().argmax()]
        act = acts[gain.argmax()]

        pred_time = math.exp(tms.numpy().min())
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
      print(f"{tm*1e6:10.2f} {pred_time*1e6:10.2f}", lin.colored_shape())

    print(f"{colored('BEAT', 'green') if tm < tmhc else colored('lost', 'red')} hand coded {tmhc/tm:5.2f}x")
    wins += int(tm < tmhc)