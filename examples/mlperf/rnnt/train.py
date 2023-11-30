# %%
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.lazy import LazyBuffer
from tinygrad.ops  import ConstBuffer
from tinygrad.ops import UnaryOps, LoadOps
from tinygrad.helpers import dtypes
from tinygrad.graph import print_tree
from tinygrad.tensor import Tensor, Function
from tinygrad.jit import TinyJit
from tinygrad.nn.optim import Adam
from matplotlib import pyplot as plt
from tinygrad.nn import Linear, Embedding
from tinygrad.nn.state import get_parameters
import numpy as np

# %%
X = 20
Y = 22
B = 5
C = 6
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
    d = d.reshape((B,Y,X+Y-1,C)).transpose(1,2)
    return d

def unshear(x:Tensor):
    B,X,Y = x.shape
    x = x.reshape((B,-1,))
    x = x.pad(((0,0),(0,X),))
    x = x.reshape((B,X,Y+1))
    x = x.shrink(((0,B),(0,X),(0,Y+1-X)))
    return x

class Loss(Function):

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
            self.labels.reshape((self.B,1,-1))
        ].log()

        assert self.p.shape == (self.B, self.X, self.Y)
        self.a = [Tensor([0]*self.B).reshape(-1,1).pad(((0,0),(0,self.Y-1),),-inf).realize()]

        for x in range(0,self.X-1):
            self.a.append(
                logsumexp(
                (self.a[-1] + self.skip[:,x,:]).realize(),
                (self.a[-1][:,:-1].pad(((0,0),(1,0),),-inf).realize() + self.p[:,x,:-1].pad(((0,0),(1,0),),-inf)).realize()
            ))
        loss = -(self.a[-1][:,-1] + self.skip[:,-1,-1]).sum()
        return loss.lazydata
    
    def backward(self, g):
        self.b = [None] * (self.X-1) + [Tensor([0]*self.B).reshape(-1,1).pad(((0,0),(self.Y-1,0),),-inf).realize()]
        for x in range(self.X-2,-1,-1):
            self.b[x] = (
                logsumexp(
                self.b[x+1] + self.skip[:,x,:],
                self.b[x+1][:,1:].pad(((0,0),(0,1),),-inf).realize() + self.p[:,x,:].realize()
             )).realize()

        self.skip_grad = None
        self.p_grad = None
        for a,b in zip(self.a[:-1], self.b[1:]):

            sg = (a + b).reshape(self.B, 1,-1)
            self.skip_grad = sg if self.skip_grad is None else self.skip_grad.cat(sg,dim=1).realize()
            pg = a.unsqueeze(1) + b[:,1:].pad(((0,0),(0,1),),-inf).unsqueeze(1)
            self.p_grad = pg if self.p_grad is None else self.p_grad.cat(pg,dim=1).realize()
        

        self.skip_grad = Tensor.cat(self.skip_grad,(self.a[-1] + self.b[-1]).reshape(self.B, 1,-1),dim=1).realize()
        self.skip_grad = self.skip_grad.transpose(1,2)

        
        # print(self.skip_grad[0].numpy())
        # plt.show()

        self.skip_grad = unshear(self.skip_grad) 
        self.skip_grad = self.skip_grad - self.b[0][:,0].unsqueeze(1).unsqueeze(1)
        self.skip_grad = Tensor.exp(self.skip_grad).realize()

        self.p_grad = unshear(self.p_grad.pad(((0,0),(0,1),(0,0))).transpose(1,2))
        self.p_grad = self.p_grad +Tensor([1]*(self.Y-1) + [-inf]).unsqueeze(-1)

        self.p_grad -= self.b[0][:,0].realize().unsqueeze(1).unsqueeze(1)
        self.p_grad =  Tensor.exp(self.p_grad).realize()

        idx = Tensor.eye(self.C-1)[self.labels].unsqueeze(2)
        self.p_grad = self.p_grad.unsqueeze(-1).mul(idx)

        return (-Tensor.cat(self.p_grad,self.skip_grad.unsqueeze(-1), dim=-1)).transpose(1,2).realize().lazydata,None

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
    l = Loss.apply(d.softmax(-1),labels)
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
    L = Loss.apply(d.softmax(-1),labels)
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
