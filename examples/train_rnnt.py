from datasets.librispeech import iterate
from tinygrad.helpers import LazyNumpyArray
from tinygrad.lazy import LazyBuffer
from tinygrad.tensor import Tensor, Function
from tinygrad.nn.optim import LAMB, get_parameters
import numpy as np

from models.rnnt import RNNT

def rnnt_loss_forward(x, y, blank=28):
  T, U, _ = x.shape

  alphas = np.zeros((T, U))
  alphas[1,0] = 1

  for t in range(1, T):
    alphas[t, 0] = alphas[t - 1, 0] + x[t - 1, 0, blank]

  for u in range(1, U):
    alphas[0, u] = alphas[0, u - 1] + x[0, u - 1, y[u - 1]]

  for t in range(1, T):
    for u in range(1, U):
      no_emit = alphas[t - 1, u] + x[t - 1, u, blank]
      emit = alphas[t, u - 1] + x[t, u - 1, y[u - 1]]
      alphas[t, u] = np.logaddexp(emit, no_emit)

  log_likelihood = alphas[T - 1, U - 1] + x[T - 1, U - 1, blank]
  return alphas, -log_likelihood

def rnnt_loss_backward(x, y, blank=28):
  T, U, _ = x.shape

  betas = np.zeros((T, U))
  betas[T - 1, U - 1] = x[T - 1, U - 1, blank]

  for t in reversed(range(T - 1)):
    betas[t, U - 1] = betas[t + 1, U - 1] + x[t, U - 1, blank]

  for u in reversed(range(U - 1)):
    betas[T - 1, u] = betas[T - 1, u + 1] + x[T - 1, u, y[u]]

  for t in reversed(range(T - 1)):
    for u in reversed(range(U - 1)):
      no_emit = betas[t + 1, u] + x[t, u, blank]
      emit = betas[t, u + 1] + x[t, u, y[u]]
      betas[t, u] = np.logaddexp(emit, no_emit)

  return betas

def rnnt_loss_grad(x, alphas, betas, y, blank=28):
  T, U, _ = x.shape

  grads = np.full(x.shape, -np.inf)
  log_likelihood = betas[0, 0]

  grads[T - 1, U - 1, blank] = alphas[T - 1, U - 1]
  grads[:T - 1, :, blank] = alphas[:T - 1, : ] + betas[1:, :]

  for u, l in enumerate(y):
    grads[:, u, l] = alphas[:, u] + betas[:, u + 1]

  grads = -np.exp(grads + x - log_likelihood)

  return grads

def rnnt_loss(x, y, blank=28):
  alphas, log_likelihood = rnnt_loss_forward(x, y, blank)
  betas = rnnt_loss_backward(x, y, blank)
  grads = rnnt_loss_grad(x, alphas, betas, y, blank)
  return log_likelihood, grads

def rnnt_loss_batch(x, x_lens, y, y_lens, blank=28):
  grads = np.zeros_like(x)
  losses = []
  for b in range(x.shape[0]):
    t = int(x_lens[b])
    u = int(y_lens[b]) + 1
    loss, grad = rnnt_loss(x[b, :t, :u, :], y[b, :u - 1], blank)
    losses.append(loss)
    grads[b, :t, :u, :] = grad
  return np.array(losses, dtype=np.float32), grads

class RNNTLoss(Function):
  def forward(self, x, x_lens, y, y_lens):
    self.x, self.x_lens, self.y, self.y_lens = x, x_lens, y, y_lens

    x_np = x.toCPU()
    x_lens_np = x_lens.toCPU().astype(np.int32)
    y_np = y.toCPU().astype(np.int32)
    y_lens_np = y_lens.toCPU().astype(np.int32)

    loss, grads = rnnt_loss_batch(x_np, x_lens_np, y_np, y_lens_np)
    self.grads = grads.astype(np.float32)

    return LazyBuffer.fromCPU(LazyNumpyArray(loss, loss.shape, loss.dtype), x.device)

  def backward(self, grad_output):
    return LazyBuffer.fromCPU(LazyNumpyArray(self.grads, self.grads.shape, self.grads.dtype), grad_output.device), None, None, None

if __name__ == "__main__":
  Tensor.training = True
  np.set_printoptions(linewidth=200)

  mdl = RNNT()
  mdl.load_from_pretrained()
  optim = LAMB(get_parameters(mdl), lr=4e-3, wd=1e-3)

  LABELS = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

  optim.zero_grad()
  for epoch in range(100):
    for X, Y, y_raw in iterate(val=False, bs=1):
      x, y = Tensor(X[0]), Tensor(Y)
      out = mdl(x, y)

      tt = mdl.decode(Tensor(X[0]), Tensor(X[1]))
      for n, t in enumerate(tt):
        tnp = np.array(t)
        print(["".join([LABELS[int(tnp[i])] for i in range(tnp.shape[0])])])
        print(y_raw[n])

      # print(out.shape)
      print("forward done")
      loss = RNNTLoss.apply(out.log_softmax(), Tensor([10, 10, 10, 10]), y, Tensor([10, 10, 10, 10])).mean()
      print("loss done")
      loss.backward()
      print("backward done")
      optim.step()
      print("step done")
      optim.zero_grad()
      print("zero grad done")

      print("loss %.4f" % loss.numpy()[0])