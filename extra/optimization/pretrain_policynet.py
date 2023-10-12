import os
from copy import deepcopy
from tinygrad.nn import Linear
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad.codegen.search import actions
from extra.optimization.helpers import load_worlds, ast_str_to_lin, lin_to_feats, assert_same_lin

INNER = 32
class PolicyNet:
  def __init__(self):
    self.l1 = Linear(240,INNER)
    self.l2 = Linear(INNER,INNER)
    self.l3 = Linear(INNER,1+len(actions))
  def __call__(self, x):
    x = self.l1(x).relu()
    x = self.l2(x).relu()
    return self.l3(x).log_softmax()

if __name__ == "__main__":
  ast_strs = load_worlds(False, False, filter_novariable=True)

  net = PolicyNet()
  if os.path.isfile("/tmp/policynet.safetensors"): load_state_dict(net, safe_load("/tmp/policynet.safetensors"))
  optim = Adam(get_parameters(net))

  X,Y = [], []
  steps = 0

  for ast_str in ast_strs:
    lin = ast_str_to_lin(ast_str)
    linhc = deepcopy(lin)
    linhc.hand_coded_optimizations()
    print(lin.colored_shape(50), "->", linhc.colored_shape())

    lin2 = deepcopy(lin)
    for o in linhc.applied_opts:
      X.append(lin_to_feats(lin2))
      Y.append(actions.index(o)+1)
      lin2.apply_opt(o)
    X.append(lin_to_feats(lin2))
    Y.append(0)
    assert_same_lin(linhc, lin2)

    BS = 64
    if len(X) >= BS:
      Tensor.no_grad, Tensor.training = False, True
      x,y = Tensor(X[:BS]), Tensor(Y[:BS])
      out = net(x)
      loss = out.sparse_categorical_crossentropy(y)
      optim.zero_grad()
      loss.backward()
      optim.step()
      cat = out.argmax(axis=-1)
      accuracy = (cat == y).mean()
      print(loss.numpy(), accuracy.numpy())

      X = X[BS:]
      Y = Y[BS:]

      if steps%10 == 0:
        safe_save(get_state_dict(net), "/tmp/policynet.safetensors")
        print("saved model")
      steps += 1
