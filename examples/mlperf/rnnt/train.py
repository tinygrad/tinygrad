# %%
from tinygrad.graph import print_tree

from tinygrad.jit import TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor, Function
from tinygrad.shape.shapetracker import ShapeTracker

from tinygrad import Tensor
from tinygrad.nn.state import get_parameters
from tinygrad.device import Device

from matplotlib import pyplot as plt

from data import load_data, iterate
from model import RNNT
import numpy as np

from tinygrad.helpers import getenv


GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS"))]


def imshow(x):
    if isinstance(x,Tensor): x = x.numpy()
    while len(x.shape) > 2: x = x[:,:,0]
    plt.imshow(x[:,:])
    plt.show()

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
    
        # return (-self.a[-1][:,-1] - self.skip[:,-1,-1]).sum().lazydata
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

        self.skg = (unshear(Tensor.cat(self.skg,(self.a[-1] + self.b[-1]).reshape(self.B, 1,-1),dim=1).realize().transpose(1,2)) - self.b[0][:,0].unsqueeze(1).unsqueeze(1)).exp().realize()
        self.p_grad = (unshear(self.p_grad.pad(((0,0),(0,1),(0,0))).transpose(1,2)) +Tensor([1]*(self.Y-1) + [-inf]).unsqueeze(-1) - self.b[0][:,0].realize().unsqueeze(1).unsqueeze(1)).exp().realize()
        self.p_grad = self.p_grad.unsqueeze(-1).mul(Tensor.eye(self.C-1)[self.labels].unsqueeze(2))

        return (-Tensor.cat(self.p_grad,self.skg.unsqueeze(-1), dim=-1)).transpose(1,2).realize().lazydata,None

# %%
# rnnt = RNNT()
# opt = Adam(get_parameters(rnnt))
# X = Tensor.uniform(20,5,240)
# X_lens = Tensor.ones(5)*20
# Y = Tensor.randint(5,15,high=29)

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

def mask(d,X_lens, Y_lens, maxX, maxY):
    d = d.pad(((0,0),(0,1),(0,0),(0,0)))
    xrange = Tensor.arange(maxX+1)
    mask = (xrange.unsqueeze(-1) < X_lens).T

    d = d * mask.unsqueeze(-1).unsqueeze(-1)
    mask = Tensor.arange

    yrange = Tensor.arange(maxY + 1)
    mask = (yrange.unsqueeze(-1).unsqueeze(-1)) < Y_lens
    mask = mask.transpose(0,2)
    d = d * mask.unsqueeze(-1)

    line = (yrange.unsqueeze(0) == Y_lens.unsqueeze(-1)-1)
    line = line.unsqueeze(1) * (xrange.unsqueeze(-1) >= X_lens.unsqueeze(1).unsqueeze(-1))
    line = line.unsqueeze(3).pad(((0,0),(0,0),(0,0),(28,0)))

    d = d + line
    return d

rnnt = RNNT()
opt = Adam(get_parameters(rnnt))
if (GPUS):
    for x in get_parameters(rnnt):
        x.to_(GPUS)

def step(iter, maxX, maxY):
    X,labels = next(iter)
    X,X_lens = X
    labels,Y_lens = text_encode(labels)
    X_lens = Tensor(X_lens)
    Y_lens = Tensor(Y_lens)
    # print(X_lens.numpy(),Y_lens.numpy())
    X = Tensor(X).pad(((0,maxX-X.shape[0]),(0,0),(0,0))).contiguous().realize()
    labels = Tensor(labels).pad(((0,0),(0,maxY - labels.shape[1]))).contiguous().realize()
    return fb_pass(X,labels,X_lens, Y_lens)/(X_lens.sum() + Y_lens.sum(), maxX, maxY)

@TinyJit
def fb_pass(X:Tensor,labels, X_lens, Y_lens, maxX, maxY):
    opt.zero_grad()
    X = rnnt.encoder.__call__(X) # LSTM expects (N,B,D)
    X_lens = (X_lens+1)/2

    Y,_ = rnnt.prediction(labels,None,1)
    Y = Y.pad(((0,0),(1,0),(0,0)))

    d = rnnt.joint(X,Y).softmax(-1).realize()
    md = mask(d,X_lens, Y_lens, maxX/2, maxY)

    L = TransducerLoss.apply(md, labels)
    L.backward()
    opt.step()
    return L.realize()

@TinyJit 
def f_pass(X:Tensor, labels, X_lens, Y_lens):
    X = rnnt.encoder.__call__(X) # LSTM expects (N,B,D)
    X_lens = (X_lens+1)/2
    Y,_ = rnnt.prediction(labels,None,1)
    Y = Y.pad(((0,0),(1,0),(0,0)))
    d = rnnt.joint(X,Y).softmax(-1).realize()
    md = mask(d,X_lens, Y_lens, maxX/2, maxY)
    L = TransducerLoss.apply(md, labels)
    return L.realize()


def test():
    iter = iterate(test_set,B)
    Tensor.no_grad = True
    L = 0
    for X, labels in iter:

        X,X_lens = X
        labels,Y_lens = text_encode(labels)
        X_lens = Tensor(X_lens)
        Y_lens = Tensor(Y_lens)

        X = Tensor(X).pad(((0,maxX-X.shape[0]),(0,0),(0,0))).contiguous().realize()
        labels = Tensor(labels).pad(((0,0),(0,maxY - labels.shape[1]))).contiguous().realize()
        l = f_pass(X,labels,X_lens, Y_lens)/(X_lens.sum() + Y_lens.sum())
        L += l.numpy().item()
        
    Tensor.no_grad = False
    return L / int(len(test_set) / B)

def timestring(s:int): return f"{int(s//3600)}:{int(s%3600//60)}:{int(s%60)}"


B = 1
if __name__ == "__main__":

    ci, maxX, maxY = load_data(15)
    train_set = ci[:-(2*B)]
    test_set = ci[-(2*B):]

    # for e in range(10):
    #     iter = iterate(train_set, B)
    #     L = step(iter,maxX, maxY).numpy().item()
    #     print(end = f"\r{L}")


