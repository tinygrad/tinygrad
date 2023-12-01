# %%
from tinygrad.graph import print_tree
from tinygrad.helpers import dtypes
from tinygrad.jit import TinyJit
from tinygrad.nn import Linear, Embedding
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.ops  import ConstBuffer, UnaryOps, LoadOps
from tinygrad.tensor import Tensor, Function
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker

from matplotlib import pyplot as plt
import numpy as np

# %%
X,Y,B,C = 20,22,5,6
labels = Tensor(np.random.randint(0,C,size=(B,Y)))

# %%
def imshow(x):
    if isinstance(x,Tensor): x = x.numpy()
    while len(x.shape) > 2: x = x[:,:,0]
    plt.imshow(x[:,:])
    plt.show()

# %%
class Model:
    def __init__(self, C:int,hdim = 10):
        self.C = C
        self.hdim = hdim
        self.input_emb = Embedding(C,hdim)   
        self.out_emb = Embedding(C,hdim)
        self.lin = Linear(hdim*2, C+1)

    def distribution(self, labels):
        B,N = labels.shape
        X = self.input_emb( labels)
        Y = self.out_emb (labels.pad(((0,0),(1,0))))

        d = Tensor.cat(X.unsqueeze(2).expand((-1,-1,N+1,-1)), Y.unsqueeze(1).expand((-1,N,-1,-1)),dim=-1)
        d = self.lin(d)
        return d

# %%
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
        self.lattice = shear(Tensor(d))
        self.X = self.X+self.Y-1
        assert self.lattice.shape == (self.B,self.X,self.Y,self.C), f"{self.lattice.shape}"

        self.skip = self.lattice[:,:,:,-1].log()
        self.p = self.lattice[
            Tensor(np.arange(self.B).reshape((-1,1,1))),
            Tensor(np.arange(self.X).reshape((1,-1,1))),
            Tensor(np.arange(self.Y).reshape((1,1,-1))),
            self.labels.reshape((self.B,1,-1))].log()

        assert self.p.shape == (self.B, self.X, self.Y)
        self.a = [Tensor([0]*self.B).reshape(-1,1).pad(((0,0),(0,self.Y-1),),-inf).realize()]

        for x in range(0,self.X-1):
            self.a.append(logsumexp((self.a[-1] + self.skip[:,x,:]).realize(), (self.a[-1][:,:-1].pad(((0,0),(1,0),),-inf).realize() + self.p[:,x,:-1].pad(((0,0),(1,0),),-inf)).realize()))

        return (-self.a[-1][:,-1] - self.skip[:,-1,-1]).sum().lazydata
    
    def backward(self, g):
        self.b = [None] * (self.X-1) + [Tensor([0]*self.B).reshape(-1,1).pad(((0,0),(self.Y-1,0),),-inf).realize()]
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

        self.skg = (unshear(Tensor.cat(self.skg,(self.a[-1] + self.b[-1]).reshape(self.B, 1,-1),dim=1).realize().transpose(1,2)) - self.b[0][:,0].unsqueeze(1).unsqueeze(1)).exp().realize()
        self.p_grad = (unshear(self.p_grad.pad(((0,0),(0,1),(0,0))).transpose(1,2)) +Tensor([1]*(self.Y-1) + [-inf]).unsqueeze(-1) - self.b[0][:,0].realize().unsqueeze(1).unsqueeze(1)).exp().realize()
        self.p_grad = self.p_grad.unsqueeze(-1).mul(Tensor.eye(self.C-1)[self.labels].unsqueeze(2))

        return (-Tensor.cat(self.p_grad,self.skg.unsqueeze(-1), dim=-1)).transpose(1,2).realize().lazydata,None

# %%
def setup():
    global model,opt
    model = Model(C)
    opt = Adam(get_parameters(model))
setup()

# %%
def merge(x):
    m = None
    for e in x:
        e = e.unsqueeze(1)
        m = e if m is None else Tensor.cat(m,e,dim=1)
    return unshear(m.transpose(1,2)).transpose(1,2)

# %%
def analyse():
    d = model.distribution(labels)
    imshow(d[0])
    opt.zero_grad()
    l = TransducerLoss.apply(d.softmax(-1),labels)
    ctx = l._ctx
    l.backward()
    imshow(d.grad[0])
    a = merge(ctx.a)
    b = merge(ctx.b)
    imshow((a[0]+b[0]-a[0].max()).exp())

# %%
@TinyJit
def step(model,labels):
    d = model.distribution(labels)
    L = TransducerLoss.apply(d.softmax(-1),labels)
    opt.zero_grad()
    L.backward()
    opt.step()
    return L.realize(), d.realize()

# %%
for i in range(10):
    for i in range(10):
        labels = Tensor(np.random.randint(0,C,size=(B,Y)))
        l,d = step(model,labels)
    print(l.numpy())

# %%
analyse()

# %% [markdown]
# ## LSTM

# %%
from extra.models.rnnt import RNNT, LSTM
from tinygrad.nn import Linear, Embedding
from temp.lstm import LSTM
from tinygrad.helpers import Timing

# %%
emb = Embedding(C,10)
H = 100

class RNN:
    def __init__(self,dim:int):
        self.dim = dim
        self.layers = [
            LSTM(dim,H,1,0),
            LSTM(H,H,10,0),
            LSTM(H,dim,1,0)]

    def __call__(self,x:Tensor):
        for l in self.layers:
            x,_ = l(x,None)
        return x

rnn = RNN(10)
opt = Adam(get_parameters(rnn))

# %%
x = emb(labels)
# x = Tensor.uniform(20,4,10)
y = x[:,1:].contiguous().realize()
x = x[:,:-1].contiguous().realize()

# %%
@TinyJit
def rnn_step(rnn,x,y):
    p = rnn(x.T)
    # loss = Tensor.sparse_categorical_crossentropy(p.T,y)
    loss = (p.T-y).square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.realize(),p.realize()

# %%
with Timing("all: "):
    for i in range (500):
        l,p = rnn_step(rnn,x,y)
        if (i+1)%10==0:
            print(f"\r {i}",l.numpy(),end="")
    print()

# %% [markdown]
# ## RNNT

# %%
from examples.mlperf.rnnt.model import RNNT
from tinygrad import Tensor

# %%
rnnt = RNNT()

# %%
Tensor.randint(5,15,low=0,high=29).max().numpy()

# %%
X = Tensor.uniform(20,5,240)
X_lens = Tensor.ones(5)*20
Y = Tensor.randint(5,15,high=29)

# %%

def dis(data_item):
    enc, xlens = rnnt.encoder(X,X_lens)
    preds,ylens = rnnt.prediction.__call__(Y,None,1)
    d = rnnt.joint(enc,preds).softmax(-1)
    return d, xlens,ylens,labels

# %% [markdown]
# ## real data

# %%
from data import iterate

# %%
i2c = list("abcdefghijklmnopqrstuvwxyz' ")+["<pad>"]
c2i = dict(map(reversed,enumerate(i2c)))
C = len(i2c) # the last index stands for either the skip or pad. thesee are different.
def text_encode(s):
    if type(s[0]) == str: s = [s]
    lens = list(map(len,s))
    maxlen = max(lens)
    encs = [list(map(c2i.__getitem__,e)) + (maxlen-l) * [c2i['<pad>']] for e,l in zip(s,lens)]
    encs = np.array(encs)
    return encs,lens

# %%
X,Y = next(iterate(4))
X,X_lens = X
Y,Y_lens = text_encode(Y)
X,X_lens = rnnt.encoder.__call__(Tensor(X),X_lens) # LSTM expects (N,B,D)
Y,_ = rnnt.prediction(Tensor(Y),None,1)
Y = Y.pad(((0,0),(1,0),(0,0)))
d = rnnt.joint(X,Y).softmax(-1)

maxx = max(X_lens)
ar = Tensor.arange(maxx)
mask = (ar.unsqueeze(1) < Tensor(X_lens)).T
d = d * mask.unsqueeze(-1).unsqueeze(-1)
skip_mask = Tensor.eye(128)[Tensor(Y_lens)].unsqueeze(1).mul(1-mask.unsqueeze(-1))
skip_mask = skip_mask.unsqueeze(-1).mul(Tensor([0]*(C-1)+[1]).reshape((1,1,1,C)))
d = d + skip_mask
X,Y,d

# %%
imshow(d[0,:,:,-1])


