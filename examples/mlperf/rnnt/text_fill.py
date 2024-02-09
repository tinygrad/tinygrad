#%%
from tinygrad import Tensor
from tinygrad.tensor import Function
from model import LSTM
from tinygrad.nn import Embedding, Linear
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.jit import TinyJit
from utils import imshow

from utils import imshow
from data import load_data, iterate

import numpy as np
import matplotlib .pyplot as plt

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

class TransducerLoss(Function):

  def forward(self, d:Tensor, labels:Tensor):
    self.B,self.X,self.Y,self.C = d.shape

    self.labels = Tensor(labels).pad(((0,0),(0,1)))
    self.lattice = shear(Tensor(d), 0.)
    self.X = self.X+self.Y-1
    assert self.lattice.shape == (self.B,self.X,self.Y,self.C), f"{self.lattice.shape}"

    self.skip = shear(Tensor(d)[:,:,:,-1:],1.)[:,:,:,0].log()

    self.p = self.lattice[
      Tensor(np.arange(self.B).reshape((-1,1,1))),
      Tensor(np.arange(self.X).reshape((1,-1,1))),
      Tensor(np.arange(self.Y).reshape((1,1,-1))),
      self.labels.reshape((self.B,1,-1))].log()

    assert self.p.shape == (self.B, self.X, self.Y)
    self.a = [Tensor([0]*self.B).reshape(-1,1).pad(((0,0),(0,self.Y-1),),-inf).realize()]

    for x in range(0,self.X-1):
      self.a.append(logsumexp(
        (self.a[-1] + self.skip[:,x,:]).realize(),
        (
          self.a[-1][:,:-1].pad(((0,0),(1,0),),-inf).realize() + self.p[:,x,:-1].pad(((0,0),(1,0),),-inf)
        ).realize()
      ))

    return (-self.a[-1].max(1).sum()).lazydata
    
  def backward(self, g):

    self.b = [None] * (self.X-1) + [Tensor.ones(self.B,self.Y)]
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
    grad = self.p_grad.cat(self.skg.unsqueeze(-1), dim=-1).pad(((0,0),(0,1),(0,0),(0,0)))

    assert not (grad.numpy() == 0).all()

    return (-grad).lazydata, None

ci, maxx, maxy = load_data(5)
BS = 16
dim = 1024
VOCAB = 28

i2c = [" ","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","'"]
c2i = {c:i for i,c in enumerate(i2c)}

def text_encode(y:np.ndarray,maxlen=None):
  if maxlen is None: maxlen = max(len(t) for t in y)
  ylens = np.array([len(t) for t in y])
  y = [np.array([c2i[c] for c in t]) for t in y]
  y = np.array([np.pad(t, (0, maxlen - len(t)), "constant") for t in y])
  return y,ylens

def text_decode(y:np.ndarray):
  return ["".join([i2c[c] for c in t]) for t in y]

def iter(ci, BS):
  for i in range(0, len(ci), BS):

    files = ci[i:i+BS]
    if (len(files)< BS): return

    y = np.array([v["transcript"] for v in files])
    # y = np.array(["".join(filter(lambda c: c in "aeiou",v["transcript"])) for v in files])
    y,ylens = text_encode(y, maxy)
    x = np.array([v["transcript"] for v in files])
    # x = np.array(["".join(filter(lambda c: c not in "aeiou",v["transcript"])) for v in files])
    x,xlens = text_encode(x, maxy)

    yield map(lambda x:Tensor(x).realize(), [x,y,xlens,ylens]) 

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

def analysis():
  global A, B, AB, skip, p
  def stack(X):
    R = X[0].unsqueeze(1)
    for x in X[1:]:
        R = R.cat(x.unsqueeze(1), dim=1).realize()
    return unshear(R.transpose(1,2)).transpose(1,2)

  A = stack(ctx.a).realize()
  B = stack(ctx.b).realize()
  AB = A[0] + B[0]
  AB = AB - AB.max()
  skip = unshear(ctx.skip.transpose(1,2)).transpose(1,2).realize()
  p = unshear(ctx.p.transpose(1,2)).transpose(1,2).realize()

class Model:
  def __init__(self,dropout= 0):
    self.dropout = dropout
    self.xemb = Embedding(VOCAB, dim)
    self.yemb = Embedding(VOCAB, dim)
    self.encoder = LSTM(dim, dim, 2, dropout)
    self.decoder = LSTM(dim, dim, 1, dropout)
    self.joint = Linear(dim, VOCAB+1)
  
  def join(self,x:Tensor,y:Tensor,xlens:Tensor,ylens:Tensor,maxx,maxy):
    global px, py, pxy
    x = self.xemb(x)
    y = self.yemb(y)
    px,_ = self.encoder(x.T) # lstm expects (S,B,D)
    py,_ = self.decoder(y.T)
    py = py.pad(((1,0),(0,0),(0,0)))
    pxy = px.T.unsqueeze(2) + py.T.unsqueeze(1)
    pxy = self.joint(pxy).softmax(-1)
    pxy = mask(pxy, xlens, ylens, maxx, maxy)
    return pxy

def init_model():
  global model, opt
  Tensor.manual_seed()
  model = Model()
  for p in get_parameters(model):p.realize()
  opt = Adam(get_parameters(model), lr=0.001)

init_model()

def forward(x,y,xlens,ylens,maxx,maxy):
  p = model.join(x,y,xlens,ylens,maxx,maxy)
  L = TransducerLoss.apply(p, y)
  return L.realize()

def clip_grad_norm(params:list[Tensor]):
  grad_norm = 0
  for p in params: p.grad.numpy()
  for p in params: p.grad.numpy()
  for p in params: grad_norm += p.grad.square().sum()

  print(f"grad_norm: {grad_norm.numpy()}")
  grad_norm = grad_norm.sqrt().maximum(1)
  print(f"grad_norm: {grad_norm.numpy()}")
  for p in params:
    p.grad = (p.grad / grad_norm).realize()

def step(x,y,xlens, ylens, maxx,maxy):
  L = forward(x,y,xlens,ylens,maxx,maxy)
  opt.zero_grad()
  L.backward()
  # for p in opt.params: p.grad.realize()
  clip_grad_norm(opt.params)
  opt.step()
  return L.realize()

def check():
  Tensor.training = False
  global x,y,xlens,ylens,L,ctx
  x,y,xlens,ylens = next(iter(ci, BS))
  L = forward(x,y,xlens,ylens,maxy,maxy)
  ctx = L._ctx
  L.backward()
  analysis()
#%%
def loop():
  Tensor.training = True
  jit_step = TinyJit(step)
  for i, (x,y,xlens,ylens) in enumerate(iter(ci, BS)):
    try:
      L = jit_step(x,y,xlens,ylens,maxy,maxy)
      print(end = f"\r{i} {L.numpy() / xlens.sum().numpy() :.6}" + " "*10)
    except KeyboardInterrupt: break

#%%
def run():
  loop()
  check()
  imshow(AB.exp())

#%%
L = step(x,y,xlens,ylens,maxy,maxy)
L.numpy()

#%%
# greedy decode
def greedy_decode(logits):
  maxlen = 20
  outputs = []
  encoded = model.xemb(logits)
  encoded, _ = model.encoder.__call__(encoded.T)
  px = encoded.T
  u = 0
  hc = None
  py = Tensor([[0]])
  while len (outputs) < maxlen:
    py, hc = model.decoder(model.yemb(py), hc)
    pxy = py + px[:,u]

    py = model.joint(pxy)
    py = py.argmax(-1).realize()

    outputs.append(py)

  return outputs


x = Tensor(text_encode(["h ws n dr strghts"],)[0])
x = model.xemb(x)
x,_ = model.encoder(x)

imshow(x[0,:,:100])

#%%
# run 2X2
for i in range(4):
  init_model()
  it = iter(ci, BS)
  for i in range(2):
    x,y,xlens,ylens = next(it)

    L = step(x,y,xlens,ylens,maxy,maxy)
    print(L.numpy())
  print()
# %%
