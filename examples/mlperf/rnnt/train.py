# %%
from tinygrad.features.graph import print_tree

from tinygrad import TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor, Function
from tinygrad.shape.shapetracker import ShapeTracker

from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.device import Device
from tinygrad.helpers import getenv

from data import load_data, iterate
from model import RNNT

from matplotlib import pyplot as plt
import numpy as np
import time


GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS"))]

def logsumexp(a:Tensor, b:Tensor):
    mx = Tensor.maximum(a,b).maximum(-1e10)
    s = (a-mx).exp() + (b-mx).exp()
    return s.log() + mx

inf = float('inf')

def shear(d:Tensor,value = 0):
    B,X,Y,C = d.shape
    d = d.pad(((0,0),(0,Y),(0,0),(0,0)),value=value)
    d = d.transpose(1,2).reshape((B,-1,C))
    d = d[:,:(X+Y-1)*Y,:].realize()
    return d.reshape((B,Y,X+Y-1,C)).transpose(1,2)

def unshear(x:Tensor):
    B,X,Y = x.shape
    x = x.reshape((B,-1,))
    x = x.pad(((0,0),(0,X),))
    x = x.reshape((B,X,Y+1))
    return x.shrink(((0,B),(0,X),(0,Y+1-X)))

class Loss:
  def __init__(self,d:Tensor, labels:Tensor):
    Tensor.no_grad = True

    self.d = d
    self.B,self.X,self.Y,self.C = d.shape

    self.labels = labels.pad(((0,0),(0,1)))
    self.lattice = shear(d, 0.)
    self.X = self.X+self.Y-1
    assert self.lattice.shape == (self.B,self.X,self.Y,self.C), f"{self.lattice.shape}"

    self.skip = shear(d[:,:,:,-1:],1.)[:,:,:,0].log()

    self.p = self.lattice[
      Tensor.arange(self.B).reshape((-1,1,1)),
      Tensor.arange(self.X).reshape((1,-1,1)),
      Tensor.arange(self.Y).reshape((1,1,-1)),
      self.labels.reshape((self.B,1,-1))].log()

    assert self.p.shape == (self.B, self.X, self.Y)
    self.a = [Tensor.zeros(self.B).reshape(-1,1).pad(((0,0),(0,self.Y-1),),-inf).realize()]

    for x in range(0,self.X-1):
      self.a.append(logsumexp(
        (self.a[-1] + self.skip[:,x,:]).realize(),
        (
          self.a[-1][:,:-1].pad(((0,0),(1,0),),-inf).realize() + self.p[:,x,:-1].pad(((0,0),(1,0),),-inf)
        ).realize()
      ))
    self.value = -self.a[-1].max(1).sum()

  def backward(self):
    Tensor.no_grad = True
    self.b: list[None | Tensor] = [None] * (self.X-1) + [Tensor.ones(self.B,self.Y)]
    for x in range(self.X-2,-1,-1):
      self.b[x] = (
        logsumexp(
          self.b[x+1] + self.skip[:,x,:],
          self.b[x+1][:,1:].pad(((0,0),(0,1),),-inf).realize() + self.p[:,x,:].realize()
        )).realize()

    self.skg, self.p_grad = None, None

    for a,b in zip(self.a[:-1], self.b[1:]):
      sg = (a + b).reshape(self.B, 1,-1)
      self.skg = sg if self.skg is None else self.skg.cat(sg,dim=1).realize()
      pg = a.unsqueeze(1) + b[:,1:].pad(((0,0),(0,1),),-inf).unsqueeze(1)
      self.p_grad = pg if self.p_grad is None else self.p_grad.cat(pg,dim=1).realize()
    
    self.skg = (unshear(self.skg.transpose(1,2)) - self.b[0][:,0].unsqueeze(1).unsqueeze(1)).transpose(1,2).exp().realize()
    self.p_grad = (unshear(self.p_grad.transpose(1,2))).transpose(1,2).realize() - self.b[0][:,0].unsqueeze(1).unsqueeze(1)

    self.p_grad = self.p_grad.exp().unsqueeze(-1).mul(Tensor.eye(self.C-1)[self.labels].unsqueeze(1))
    grad = - self.p_grad.cat(self.skg.unsqueeze(-1), dim=-1).pad(((0,0),(0,1),(0,0),(0,0)))

    Tensor.no_grad = False

    # inject loss gradient into autograd
    loss = (self.d - self.d.detach() + grad).square().sum() / 2
    backward(loss)


def backward(self) -> Tensor:
  assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

  # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
  # this is "implicit gradient creation"
  self.grad = Tensor(1.0, device=self.device, requires_grad=False)

  for t0 in reversed(self.deepwalk()):
    if t0.grad is None: raise RuntimeError("tensor has no grad")
    grads = t0._ctx.backward(t0.grad.lazydata)
    grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
      for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
    for t, g in zip(t0._ctx.parents, grads):
      if g is not None and t.requires_grad:
        assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
        t.grad = g if t.grad is None else (t.grad + g).realize()
    del t0._ctx
  return self


def mask(d,X_lens, Y_lens, maxX, maxY, vocab=28):

    d = d.pad(((0,0),(0,1),(0,0),(0,0))) # padding after X for masking
    xrange = Tensor.arange(maxX+1)
    mask = (xrange.unsqueeze(-1) < X_lens).T

    d = d * mask.unsqueeze(-1).unsqueeze(-1)
    mask = Tensor.arange

    yrange = Tensor.arange(maxY + 1)
    mask = (yrange.unsqueeze(-1).unsqueeze(-1)) <= Y_lens
    mask = mask.transpose(0,2)
    d = d * mask.unsqueeze(-1)

    line = (yrange.unsqueeze(0) == Y_lens.unsqueeze(-1))
    line = line.unsqueeze(1) * (xrange.unsqueeze(-1) >= X_lens.unsqueeze(1).unsqueeze(-1))
    line = line.unsqueeze(3).pad(((0,0),(0,0),(0,0),(vocab,0)))

    d = d + line
    return d

rnnt = RNNT()
opt = Adam(get_parameters(rnnt))
if (GPUS):
    for x in get_parameters(rnnt):
        x.to_(GPUS)


def fb_pass(X:Tensor,labels, X_lens, Y_lens, maxX, maxY):
    opt.zero_grad()
    L = forward(X, labels, X_lens, Y_lens, maxX, maxY)
    L.backward()
    for p in opt.params: p.grad.realize()
    opt.step()
    return (L.value/ X_lens.sum()).realize()

def forward(X:Tensor, labels, X_lens, Y_lens, maxx, maxy):
    X = rnnt.encoder.__call__(X) # LSTM expects (N,B,D)
    X_lens = (X_lens+1)/2
    Y = rnnt.prediction.__call__(labels,1)
    Y = Y.pad(((0,0),(1,0),(0,0)))
    d = rnnt.joint(X,Y).softmax(-1).realize()
    md = mask(d,X_lens, Y_lens, maxx/2, maxy)
    L = Loss(md, labels)
    return L

forward_jit = TinyJit(forward)

def test():
    iter = iterate(test_set,BS, maxx,maxy)
    Tensor.no_grad = True
    L = 0
    for sample in iter:
        X_lens = sample[2]
        l = forward_jit(*sample,maxx,maxy).value.numpy()/(X_lens.sum())
        L += l.numpy().item()

    Tensor.no_grad = False
    return L / int(len(test_set) / BS)

def timestring(s:int): return f"{int(s//3600)}:{int(s//60%60)}:{s%60:.4}"


BS = 4
maxsecs = 5

if __name__ == "__main__":

    ci, maxx, maxy = load_data(maxsecs)
    train_set = ci[:-(2*BS)]
    test_set = ci[-(2*BS):]
    print(f"eval: {test()}")
    interrupt = False
    step = TinyJit(fb_pass)
    for e in range(20,40):
        st = time.time()
        for i,sample in enumerate(iterate(train_set, BS, maxx,maxy)):
            try:
                L = fb_pass(*sample, maxx, maxy).numpy().item()
                print( end = f"\r {i}/{int(len(train_set)/BS)} L:{L:.4}", flush = True)
            except KeyboardInterrupt:
                rnnt.save(f"rnnt_e_{e+1}_i_{i+1}")
                interrupt = True
                break
        if interrupt: break
        print (f"\nepoch {e+1} finished in {timestring(time.time() - st )} val:{test()}")
        rnnt.save(f"rnnt_e_{e+1}")


# %%
