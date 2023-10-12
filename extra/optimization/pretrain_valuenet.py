from tinygrad.codegen.linearizer import Linearizer
from tqdm import tqdm, trange
import math
import random
from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad.lazy import vars_from_ast
from tinygrad.shape.symbolic import sym_infer

# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.helpers import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Variable
inf, nan = float('inf'), float('nan')
from tinygrad.codegen.optimizer import Opt, OptOps

from extra.optimization.helpers import lin_to_feats, MAX_DIMS

def log_likelihood(x:Tensor, mu:Tensor, log_sigma:Tensor):
  #print(x.shape, mu.shape, log_sigma.shape)
  #return (x-mu).abs() * (-log_sigma).exp() + log_sigma
  return (x-mu).square() * (-2*log_sigma).exp() / 2 + log_sigma

# NOTE: this is not real value of the state, it's just a prediction of the runtime
INNER = 512
class ValueNet:
  def __init__(self):
    self.l1 = Linear(1021,1024)
    self.l2 = Linear(1024,INNER)
    self.l3 = Linear(INNER,INNER)
    self.l4 = Linear(INNER,4)
  def __call__(self, x):
    x = self.l1(x).relu()
    x = self.l2(x).relu()
    x = self.l3(x).relu() #.dropout(0.1)
    return self.l4(x)

if __name__ == "__main__":
  net = ValueNet()
  optim = Adam(get_parameters(net), lr=3e-4)

  TEST_SIZE = 256
  GEN = False

  if GEN:
    dset = open("/tmp/logtm").read().strip().split("\n")
    random.seed(1337)
    random.shuffle(dset)
    #dset = dset[:2000]

    X,Y = [], []
    for i,x in enumerate(tqdm(dset)):
      ast, opts, tms = eval(x)
      lin = Linearizer(ast)
      for o in opts: lin.apply_opt(o)
      if lin.shape_len >= MAX_DIMS: continue
      if min(tms) == float('inf'): continue
      X.append(lin_to_feats(lin, use_sts=True))
      gflops = sym_infer(lin.info.flops, {k:k.min for k in vars_from_ast(lin.ast)})*1e-9/min(tms)
      Y.append([math.log(min(tms)), math.log(gflops+0.01)])
    print(f"got {len(X)} samples")
    safe_save({"X": Tensor(X), "Y": Tensor(Y)}, "/tmp/dset_cache")
  else:
    dd = safe_load("/tmp/dset_cache")
    X, Y = dd["X"].numpy(), dd["Y"].numpy()

  X_test,Y_test = Tensor(X[-TEST_SIZE:]), Tensor(Y[-TEST_SIZE:])
  X,Y = X[:-TEST_SIZE], Y[:-TEST_SIZE]

  def get_minibatch(X,Y,bs):
    xs, ys = [], []
    for _ in range(bs):
      sel = random.randint(0, len(X)-1)
      xs.append(X[sel])
      ys.append(Y[sel])
    return Tensor(xs), Tensor(ys)

  @TinyJit
  def train(x,y):
    optim.lr *= 0.999
    out = net(x)
    loss = log_likelihood(y, out[:,0:2], out[:,2:4]).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.realize()

  Tensor.no_grad, Tensor.training = False, True
  losses = []
  test_losses = []
  test_loss = float('inf')
  for i in (t:=trange(2000)):
    x,y = get_minibatch(X,Y,bs=512)
    loss = train(x,y)

    t.set_description(f"loss {loss.numpy():7.2f}, test mse loss {test_loss:7.2f}")
    losses.append(loss.numpy().item())
    test_losses.append(test_loss)
    if i % 10: test_loss = (net(X_test)[:,0:2]-Y_test).square().mean().numpy().item()

  safe_save(get_state_dict(net), "/tmp/valuenet.safetensors")

  import matplotlib.pyplot as plt
  plt.plot(losses[200:])
  plt.plot(test_losses[200:])
  plt.show()
