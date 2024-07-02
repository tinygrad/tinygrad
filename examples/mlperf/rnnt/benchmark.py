# %%
from tinygrad import Tensor, TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.nn import Embedding, Linear

import time

class LSTMCell:
  def __init__(self, input_size, hidden_size):
    k = hidden_size ** -0.5
    self.w_ih = Tensor.randn(input_size, hidden_size * 4,).realize()
    self.b_ih = Tensor.randn(hidden_size * 4).realize() * k
    self.w_hh = Tensor.randn(hidden_size, hidden_size * 4).realize() * k
    self.b_hh = Tensor.randn(hidden_size * 4).realize() * k

  def __call__(self, x):
    h = Tensor.zeros(x.shape[1], self.w_hh.shape[0])
    c = Tensor.zeros(x.shape[1], self.w_hh.shape[0])
    res = []
    for t in range(x.shape[0]):

      gates = x[t].linear(self.w_ih, self.b_ih) + h.linear(self.w_hh, self.b_hh)
      i, f, g, o = gates.chunk(4, 1)
      i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
      h = (o * c.tanh()).realize()
      c = (f * c) + (i * g).realize()

      res.append(Tensor.stack([h,c]))
      h = res[-1][0]
      c = res[-1][1]
    
    ret = res[0].unsqueeze(0)
    for e in res[1:]: ret = ret.cat(e.unsqueeze(0) , dim=0).realize()
    return ret[:,0].realize()

class LSTM:
  def __init__(self,input_size, hidden_size, layers,_):
    self.cells = [LSTMCell(input_size, hidden_size) if i == 0 else LSTMCell(hidden_size,hidden_size) for i in range(layers)]
  
  def __call__(self,x:Tensor):
    # expect x of shape (T,B,D)
    for cell in self.cells: x = cell(x)
    return x.realize()

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
    loss = (self.d - self.d.detach() + grad).square().sum() / 2
    # backward(loss)
    loss.backward()

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

STACKFACTOR = 2
def stacktime (x):
  x = x.pad(((0, (-x.shape[0]) % STACKFACTOR), (0, 0), (0, 0)))
  x = x.reshape(x.shape[0] // STACKFACTOR, x.shape[1], x.shape[2] * STACKFACTOR)
  return x

class RNNT:
  def __init__(self, input_features=240, vocab_size=29, enc_hidden_size=1024, pred_hidden_size=320, joint_hidden_size=512, pre_enc_layers=2, post_enc_layers=3, pred_layers=2, dropout=0.32):

    self.Encoder = [
      LSTM(input_features, enc_hidden_size, pre_enc_layers, dropout),
      stacktime,
      LSTM(STACKFACTOR * enc_hidden_size, enc_hidden_size, post_enc_layers, dropout)]
    self.Prediction = [
      Embedding(vocab_size - 1, pred_hidden_size),
      LSTM(pred_hidden_size, pred_hidden_size, pred_layers, dropout)]
    self.Joint = [
       Linear(pred_hidden_size + enc_hidden_size, joint_hidden_size),
       Linear(joint_hidden_size, vocab_size)]

  def encoder(self, x:Tensor): return x.sequential(self.Encoder).transpose(0, 1)

  def prediction(self, x):
    emb = self.Prediction[0](x)
    return self.Prediction[1](emb.transpose(0, 1))

  def joint(self,x,y):
    (_, T, H), (B, U, H2) = x.shape, y.shape
    x = x.unsqueeze(2).expand(B, T, U, H)
    y = y.unsqueeze(1).expand(B, T, U, H2)

    inp = x.cat(y, dim=3)
    t = self.Joint[0](inp).relu()
    t = t.dropout(0.32)
    return self.Joint[1](t)

def fb_pass(X:Tensor,labels, xlens, ylens, maxX, maxY):
    opt.zero_grad()
    L = forward(X, labels, xlens, ylens, maxX, maxY)
    L.backward()
    for p in opt.params: 
      if (p.grad.numpy() == 0).all(): print("zero grad")
    opt.step()
    return (L.value/ xlens.sum()).realize()

def forward(X:Tensor, labels:Tensor, xlens, ylens, maxx, maxy):
    X = rnnt.encoder.__call__(X) # LSTM expects (N,B,D)
    xlens = (xlens+1)/2
    Y = rnnt.prediction(labels)
    Y = Y.pad(((0,0),(1,0),(0,0)))
    d = rnnt.joint(X,Y).softmax(-1).realize()
    md = mask(d,xlens, ylens, maxx/2, maxy)
    L = Loss(md, labels.T)
    return L

def timestring(s:int): return f"{int(s//3600)}:{int(s//60%60)}:{s%60:.4}"

BS = 4
Tensor.manual_seed()
rnnt = RNNT(dropout=0)
for p in get_parameters(rnnt):p.realize()
opt = Adam(get_parameters(rnnt))
if __name__ == "__main__":
    Tensor.manual_seed()
    maxx, maxy = 100, 100
    step = TinyJit(fb_pass)
    st = time.time()
    for i in range(78):
        x = Tensor.randn(100, BS, 240)
        y = Tensor.randint(100, BS, high=28)

        xlens = Tensor.randint(BS, low=50, high=100)
        ylens = Tensor.randint(BS, low=50, high=100)
        try:
            L = step(x,y,xlens,ylens,maxx,maxy).item()
            print(f" {i}/{int(100)} {timestring(time.time()-st)} L:{L:.4}", flush = True)
        except KeyboardInterrupt:
            break
