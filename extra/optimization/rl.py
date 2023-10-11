import random
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.ops import Device
from tinygrad.helpers import dedup, ansilen
from tinygrad.lazy import var_vals_from_ast
from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Adam
from tinygrad.shape.symbolic import sym_infer
from tinygrad.codegen.optimizer import Opt, OptOps
from tinygrad.codegen.linearizer import Linearizer
actions = [Opt(op=OptOps.LOCAL, axis=0, amt=3), Opt(op=OptOps.UPCAST, axis=3, amt=4), Opt(op=OptOps.UPCAST, axis=2, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=7), Opt(op=OptOps.UPCAST, axis=2, amt=5), Opt(op=OptOps.LOCAL, axis=4, amt=3), Opt(op=OptOps.GROUP, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=4, amt=2), Opt(op=OptOps.UPCAST, axis=4, amt=5), Opt(op=OptOps.GROUPTOP, axis=0, amt=256), Opt(op=OptOps.LOCAL, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=2, amt=3), Opt(op=OptOps.UPCAST, axis=1, amt=6), Opt(op=OptOps.UPCAST, axis=0, amt=4), Opt(op=OptOps.LOCAL, axis=3, amt=4), Opt(op=OptOps.UPCAST, axis=1, amt=3), Opt(op=OptOps.GROUPTOP, axis=2, amt=256), Opt(op=OptOps.LOCAL, axis=1, amt=16), Opt(op=OptOps.UPCAST, axis=0, amt=7), Opt(op=OptOps.UPCAST, axis=5, amt=3), Opt(op=OptOps.LOCAL, axis=3, amt=16), Opt(op=OptOps.LOCAL, axis=0, amt=2), Opt(op=OptOps.GROUPTOP, axis=1, amt=16), Opt(op=OptOps.LOCAL, axis=0, amt=8), Opt(op=OptOps.UPCAST, axis=3, amt=6), Opt(op=OptOps.UPCAST, axis=2, amt=4), Opt(op=OptOps.LOCAL, axis=4, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=3), Opt(op=OptOps.UPCAST, axis=2, amt=7), Opt(op=OptOps.UPCAST, axis=4, amt=4), Opt(op=OptOps.GROUP, axis=2, amt=8), Opt(op=OptOps.LOCAL, axis=2, amt=2), Opt(op=OptOps.LOCAL, axis=0, amt=32), Opt(op=OptOps.UPCAST, axis=0, amt=3), Opt(op=OptOps.LOCAL, axis=1, amt=3), Opt(op=OptOps.UPCAST, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=0, amt=6), Opt(op=OptOps.LOCAL, axis=2, amt=8), Opt(op=OptOps.UPCAST, axis=1, amt=5), Opt(op=OptOps.LOCAL, axis=3, amt=3), Opt(op=OptOps.UPCAST, axis=6, amt=4), Opt(op=OptOps.GROUPTOP, axis=1, amt=256), Opt(op=OptOps.UPCAST, axis=2, amt=3), Opt(op=OptOps.LOCAL, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=3, amt=2), Opt(op=OptOps.LOCAL, axis=0, amt=16), Opt(op=OptOps.UPCAST, axis=2, amt=6), Opt(op=OptOps.UPCAST, axis=5, amt=4), Opt(op=OptOps.UPCAST, axis=4, amt=3), Opt(op=OptOps.UPCAST, axis=3, amt=5), Opt(op=OptOps.GROUP, axis=1, amt=8), Opt(op=OptOps.LOCAL, axis=4, amt=16), Opt(op=OptOps.GROUPTOP, axis=0, amt=16), Opt(op=OptOps.LOCAL, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=0, amt=5), Opt(op=OptOps.GROUPTOP, axis=2, amt=16), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=0, amt=2), Opt(op=OptOps.LOCAL, axis=2, amt=4), Opt(op=OptOps.LOCAL, axis=3, amt=2), Opt(op=OptOps.LOCAL, axis=1, amt=8), Opt(op=OptOps.UPCAST, axis=1, amt=7), Opt(op=OptOps.LOCAL, axis=3, amt=8), Opt(op=OptOps.LOCAL, axis=2, amt=16)]
actions = [x for x in actions if x.op not in [OptOps.GROUP, OptOps.GROUPTOP]]

device = Device[Device.DEFAULT]

# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.helpers import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Variable
inf, nan = float('inf'), float('nan')

from extra.optimization.go import lin_to_feats

INNER = 256
class PolicyNet:
  def __init__(self):
    self.l1 = Linear(240,INNER)
    self.l2 = Linear(INNER,INNER)
    self.l3 = Linear(INNER,2)
    #self.l3 = Linear(INNER,len(actions))
  def __call__(self, x):
    x = self.l1(x).relu()
    x = self.l2(x).relu()
    return self.l3(x) #.log_softmax()

def bufs_from_lin(lin):
  bufsts = defaultdict(list)
  for x in lin.membufs: bufsts[x.idx].append(x)
  rawbufs = [None]*len(bufsts)
  for k,x in bufsts.items(): rawbufs[k] = device.buffer(max(y.st.size() for y in x), x[0].dtype)
  assert all(x is not None for x in rawbufs)
  return rawbufs

def time_linearizer(lin, rawbufs):
  lin = deepcopy(lin)  # TODO: remove the need for this
  var_vals = {k:k.min for k in var_vals_from_ast(lin.ast)}
  try:
    lin.linearize()
    prg = device.to_program(lin)
    tm = prg(rawbufs, var_vals, force_wait=True)
  except Exception as e:
    print("FAILED")
    print(lin.ast)
    print(lin.applied_opts)
    raise e
  if tm < 0.1: tm = min([tm]+[prg(rawbufs, var_vals, force_wait=True) for _ in range(10)])
  gflops = sym_infer(lin.info.flops, var_vals)*1e-9/tm
  return tm*1e6, gflops

if __name__ == "__main__":
  # load worlds
  ast_strs = dedup(open("/tmp/sops").read().strip().split("\n"))
  ast_strs = [x for x in ast_strs if "ReduceOps" in x and "dtypes.image" not in x and "Variable" not in x]

  #ast_strs = ast_strs[:100]

  # deterministic
  random.seed(1337)

  net = PolicyNet()
  optim = Adam(get_parameters(net))

  feats, acts, rews, gts = [], [], [], []

  old_worlds = []
  for i in range(20000):
    Tensor.no_grad, Tensor.training = True, False
    if random.random() < 0.7 or len(old_worlds) < 3:
      ast_str = random.choice(ast_strs)
      ast = eval(ast_str)
      lin = Linearizer(ast)
    else:
      lin = random.choice(old_worlds)

    # get description
    cs = lin.colored_shape()
    cs += ' '*(50-ansilen(cs))

    # enumerate all the actions as linearizers
    acted_lins = {}
    for i,a in enumerate(actions):
      lin2 = deepcopy(lin)
      try:
        lin2.apply_opt(a)
        acted_lins[i] = lin2
      except Exception:
        pass

    # run the model on all possible
    lfeats = [lin_to_feats(lin)] + [lin_to_feats(x) for x in acted_lins.values()]
    preds = net(Tensor(lfeats)).exp().numpy()

    tact = np.argmin(preds[:, 0])
    # stop if slow or no action
    if tact == 0 or preds[0, 0] > 5000: continue
    act = list(acted_lins.keys())[(tact-1)]
    lin2 = acted_lins[act]

    # take action
    rawbufs = bufs_from_lin(lin)
    tm1, gf1 = time_linearizer(lin, rawbufs)
    tm2, gf2 = time_linearizer(lin2, rawbufs)
    old_worlds.append(lin2)

    feats.append(lfeats[0])
    feats.append(lfeats[tact])
    gts.append((tm1, gf1))
    gts.append((tm2, gf2))

    print(f"{cs} actions: {len(acted_lins):3d} chose: {act:3d} tm: {tm1:7.2f} -> {tm2:7.2f} predgain: {(preds[0, 0]-preds[tact, 0])/preds[0, 0]:5.2f}x realgain: {(tm1-tm2)/tm1:5.2f}x")

    # training batch
    if len(feats) == 8:
      Tensor.no_grad, Tensor.training = False, True
      f = net(Tensor(feats))
      loss = (f - Tensor(gts).log()).square().mean()
      optim.zero_grad()
      loss.backward()
      optim.step()
      feats, acts, rews, gts = [], [], [], []
      print(f"trained {loss.numpy()}")

    """
    mask = np.zeros((len(acts), len(actions)), dtype=np.float32)
    mask[np.arange(len(acts)), acts] = 1
    logp = net(Tensor(feats)) * Tensor(mask)
    loss = -(logp * Tensor(rews).reshape(-1,1)).mean()
    """


    """
    continue

    # model first
    feat = lin_to_feats(lin)
    pred_tm, pred_gflops = net(Tensor(feat)).exp().numpy()
    if pred_tm > 100:   # 100 us
      #print("skipping slow")
      continue



    continue

    didnt_act = True
    for i in range(100):
      act = random.randint(0, len(actions)-1)
      lin2 = copy(lin)
      try:
        lin2.apply_opt(actions[act])
        didnt_act = False
        break
      except Exception:
        pass
    if didnt_act:
      print("DIDN'T ACT")
      continue

    rawbufs = bufs_from_lin(lin)
    tm1, gf1 = time_linearizer(lin, rawbufs)
    #print(act, f"pred_time: {pred_tm:7.2f}  time: {tm1:7.2f}", f"pred_gflops: {pred_gflops:7.2f}  gflops: {gf1:7.2f}")

    #act = np.random.choice(len(act), p=act)
    try:
      lin2 = copy(lin)
      lin2.apply_opt(actions[act])
      #rews.append(1)
      #print(cs, "okay", act, lin2.colored_shape())
      tm2, gf2 = time_linearizer(lin2, rawbufs)
      rews.append((tm1-tm2)/tm2)
      print(cs, "okay", actions[act], lin2.colored_shape(), f"{tm1:7.2f} -> {tm2:7.2f}, rew {rews[-1]}")

      feats.append(feat)
      gts.append((tm1, gf1))

      feats.append(lin_to_feats(lin2))
      gts.append((tm2, gf2))

      acts.append(act)
      old_worlds.append(lin2)
    except Exception:
      print(cs, f"INVALID ACTION {act} {actions[act]}")
      #rews.append(-3)
    """