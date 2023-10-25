#%%

from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam, LAMB
from tinygrad.nn import Embedding, Linear
from tinygrad.helpers import dtypes
from tinygrad.ops import Device
from tinygrad.jit import TinyJit
from tinygrad.mlops import Function
from tinygrad.lazy import LazyBuffer
from tinygrad.graph import print_tree

import itertools
import pathlib
import json
import numpy as np
from math import prod
import matplotlib.pyplot as plt
import librosa
import soundfile

# try:
#   import lovely_grad
#   lovely_grad.monkey_patch()
# except:pass


"""
The dataset has to be downloaded manually from https://www.openslr.org/12/ and put in `extra/datasets/librispeech`.
For mlperf validation the dev-clean dataset is used.

Then all the flacs have to be converted to wav using something like:
```fish
for file in $(find * | grep flac); do ffmpeg -i $file -ar 16k "$(dirname $file)/$(basename $file .flac).wav"; done
```

Then this [file](https://github.com/mlcommons/inference/blob/master/speech_recognition/rnnt/dev-clean-wav.json) has to also be put in `extra/datasets/librispeech`.
"""

BASEDIR = pathlib.Path("../../../extra/datasets/librispeech")
with open(BASEDIR / "dev-clean-wav.json") as f:
  ci = json.load(f)
FILTER_BANK = np.expand_dims(librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000), 0)
WINDOW = librosa.filters.get_window("hann", 320)

def feature_extract(x, x_lens):
  x_lens = np.int32(x_lens / 160 / 3) + 1
  x = np.concatenate((np.expand_dims(x[:, 0], 1), x[:, 1:] - 0.97 * x[:, :-1]), axis=1)
  x = librosa.stft(x, n_fft=512, window=WINDOW, hop_length=160, win_length=320, center=True, pad_mode="reflect")
  x = np.stack((x.real, x.imag), axis=-1)

  x = (x**2).sum(-1)
  x = np.matmul(FILTER_BANK, x)
  x = np.log(x + 1e-20)

  seq = [x]
  for i in range(1, 3):
    tmp = np.zeros_like(x)
    tmp[:, :, :-i] = x[:, :, i:]
    seq.append(tmp)
  features = np.concatenate(seq, axis=1)[:, :, ::3]

  # normalize
  features_mean = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
  features_std = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
  for i in range(features.shape[0]):
    features_mean[i, :] = features[i, :, :x_lens[i]].mean(axis=1)
    features_std[i, :] = features[i, :, :x_lens[i]].std(axis=1, ddof=1)
  features_std += 1e-5
  features = (features - np.expand_dims(features_mean, 2)) / np.expand_dims(features_std, 2)
  return features.transpose(2, 0, 1), x_lens.astype(np.float32)
def load_wav(file):
  sample = soundfile.read(file)[0].astype(np.float32)
  return sample, sample.shape[0]
#%%
characters = [*" 'abcdefghijklmnopqrstuvwxyz","<skip>"]
c2i= dict([(c,i) for i,c in enumerate(characters)])
charn = len(characters)

def text_encode(text:list[str])->(Tensor,np.ndarray):
    if isinstance(text,str):text = [text]
    seqs = [np.array([np.array(c2i[char]) for char in seq]) for seq in text]
    seq_lens = [len(s) for s in seqs]
    seqs = list(map (lambda s: np.pad(s,(0,max(seq_lens)-len(s)),"constant"),seqs))
    return Tensor(seqs,dtype=dtypes.int16, requires_grad=False), Tensor(np.array(seq_lens,dtype=np.int16),requires_grad=False)

def text_decode(toks:Tensor):
    ret = []
    for seq in toks:
        ret.append("".join([characters[int(tok)] for tok in seq ]))
    return ret

from typing import Generator

def iterate(bs=1, start=0)->Generator[tuple[Tensor, Tensor, Tensor, Tensor],None,None]:
  for i in range(start, len(ci), bs):
    samples, sample_lens = zip(*[load_wav(BASEDIR / v["files"][0]["fname"]) for v in ci[i : i + bs]])
    samples = list(samples)
    X,X_lens = list(zip(*[feature_extract(np.array(samples[i:i+1]),np.array(sample_lens[i:i+1])) for i in range(bs)]))
    max_len = max(X_lens)
    X = [np.pad(X[j],((0,int(max_len[0]-X_lens[j][0])),(0,0),(0,0)),'constant') for j in range (len(X))]
    yield (
      Tensor(np.concatenate(X,axis=1),requires_grad=False),
      Tensor(np.array(X_lens),requires_grad=False),
      *text_encode([v['transcript'] for v in ci[i:i+bs]]))

#%%
class RNNT:
  def __init__(self, input_features=240, vocab_size=29, enc_hidden_size=1024, pred_hidden_size=320, joint_hidden_size=512, pre_enc_layers=2, post_enc_layers=3, pred_layers=2, stack_time_factor=2, dropout=0.32):
    self.encoder = Encoder(input_features, enc_hidden_size, pre_enc_layers, post_enc_layers, stack_time_factor, dropout)
    self.prediction = Prediction(vocab_size, pred_hidden_size, pred_layers, dropout)
    self.joint = Joint(vocab_size, pred_hidden_size, enc_hidden_size, joint_hidden_size, dropout)
    self.params:list[Tensor] = [*self.encoder.params, *self.prediction.params, *self.joint.params]

  def decode(self, x, x_lens, max_output = 1e9):
    logits, logit_lens = self.encoder(x, x_lens)
    outputs = []
    for b in range(logits.shape[0]):
      inseq = logits[b, :, :].unsqueeze(1)
      logit_len = logit_lens[b]
      seq = self._greedy_decode(inseq, int(np.ceil(logit_len.numpy()).item()),max_output)
      outputs.append(seq)
    return outputs

  def _greedy_decode(self, logits, logit_len,max_output):
    hc = Tensor.zeros(self.prediction.rnn.layers, 2, self.prediction.hidden_size, requires_grad=False)
    labels = []
    label = Tensor.zeros(1, 1, requires_grad=False)
    mask = Tensor.zeros(1, requires_grad=False)
    for time_idx in range(logit_len):
      logit = logits[time_idx, :, :].unsqueeze(0)
      not_blank = True
      added = 0
      while not_blank and added < 30:
        if len(labels) > 0:
          mask = (mask + 1).clip(0, 1)
          label = Tensor([[labels[-1] if labels[-1] <= 28 else labels[-1] - 1]], requires_grad=False) + 1 - 1
        jhc = self._pred_joint(Tensor(logit.numpy()), label, hc, mask)
        k = jhc[0, 0, :29].argmax(axis=0).numpy()
        not_blank = k != 28
        if not_blank:
          labels.append(k)
          hc = jhc[:, :, 29:] + 1 - 1
        added += 1
    return labels

  # @TinyJit
  def _pred_joint(self, logit, label, hc, mask):
    g, hc = self.prediction(label, hc, mask)
    j = self.joint(logit, g)[0]
    j = j.pad(((0, 1), (0, 1), (0, 0)))
    out = j.cat(hc, dim=2)
    return out.realize()
  
class LSTMCell:
  def __init__(self, input_size, hidden_size, dropout):
    self.input_size,self.hidden_size = input_size,hidden_size
    self.dropout = dropout

    self.weights_ih = Tensor.uniform(hidden_size * 4, input_size)
    self.bias_ih = Tensor.uniform(hidden_size * 4)
    self.weights_hh = Tensor.uniform(hidden_size * 4, hidden_size)
    self.bias_hh = Tensor.uniform(hidden_size * 4)

  def __call__(self, x:Tensor, hc:Tensor):

    assert (BS, self.input_size) == x.shape, f"{self.input_size} {x.shape}"
    assert (2,BS,self.hidden_size) == hc.shape
    last_h,last_c = hc
    gates = x.linear(self.weights_ih.T, self.bias_ih) + last_h.linear(self.weights_hh.T, self.bias_hh)

    i, f, g, o = gates.chunk(4, 1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()

    c = ((f * last_c) + (i * g)).unsqueeze(0)
    h = (o * c.tanh()).dropout(self.dropout)

    return Tensor.cat(h, c).realize()

T = BS = 0
class LSTM:
  def __init__(self, input_size, hidden_size, layers, dropout):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.layers = layers

    self.cells = [LSTMCell(input_size if i == 0 else hidden_size, hidden_size, dropout if i != layers - 1 else 0) for i in range(layers)]

    self.params = list(itertools.chain(*[[cell.bias_hh,cell.bias_ih,cell.weights_hh,cell.weights_ih] for cell in self.cells]))

  def do_step(self, x, hc):
    assert (self.layers,2,BS,self.hidden_size) == hc.shape
    h = x
    new_hc = []
    for i, cell in enumerate(self.cells):
      res = cell(h, hc[i])
      assert (2,BS,self.hidden_size) == res.shape, f"{(2,BS,self.hidden_size)} {res.shape}"
      h = res[0]
      new_hc.append(res)
    return Tensor.stack(new_hc)

  def __call__(self, x, hc):
    @TinyJit
    def _do_step(x_, hc_):
      return self.do_step(x_, hc_)

    global BS,T
    T,BS,IS = x.shape
    assert IS == self.input_size
    
    if hc is None:
      hc = Tensor.zeros(self.layers, 2, BS, self.hidden_size, requires_grad=False)

    output = None
    for t in range(T):
      hc = _do_step(x[t] + 1 - 1, hc) # TODO: why do we need to do this?
      assert (self.layers,2,BS,self.hidden_size) == hc.shape
      if output is None:
        output = hc[-1][:1]
      else:
        output = output.cat(hc[-1][:1], dim=0).realize()

    return output, hc

class Joint:
  def __init__(self, vocab_size, pred_hidden_size, enc_hidden_size, joint_hidden_size, dropout):
    self.dropout = dropout
    self.l1 = Linear(pred_hidden_size + enc_hidden_size, joint_hidden_size)
    self.l2 = Linear(joint_hidden_size, vocab_size)
    self.params = [self.l1.bias, self.l1.weight, self.l2.bias, self.l2.weight]

  def __call__(self, f, g):
    (_, T, H), (B, U, H2) = f.shape, g.shape
    f = f.unsqueeze(2).expand(B, T, U, H)
    g = g.unsqueeze(1).expand(B, T, U, H2)

    inp = f.cat(g, dim=3)
    t = self.l1(inp).relu()
    t = t.dropout(self.dropout)
    return self.l2(t)

class Encoder:
  def __init__(self, input_size, hidden_size, pre_layers, post_layers, stack_time_factor, dropout):
    self.pre_rnn = LSTM(input_size, hidden_size, pre_layers, dropout)
    self.stack_time = StackTime(stack_time_factor)
    self.post_rnn = LSTM(stack_time_factor * hidden_size, hidden_size, post_layers, dropout)
    self.params = [*self.pre_rnn.params, *self.post_rnn.params]

  def __call__(self, x:Tensor, x_lens):

    Timer.start("enc.pre")
    x, _ = self.pre_rnn.__call__(x, None)
    Timer.start("enc.stack")
    x, x_lens = self.stack_time.__call__(x, x_lens)
    Timer.start("enc.post")
    x, _ = self.post_rnn.__call__(x, None)
    return x.transpose(0, 1), x_lens
  
class StackTime:
  def __init__(self, factor):
    self.factor = factor

  def __call__(self, x:Tensor, x_lens:Tensor):
    x = x.pad(((0, x.shape[0] % self.factor), (0, 0), (0, 0))).permute((1,0,2))
    x = x.reshape( x.shape[0], x.shape[1] // self.factor, x.shape[2] * self.factor).permute((1,0,2))
    return x, x_lens / self.factor if x_lens is not None else None

class Prediction:
  def __init__(self, vocab_size, hidden_size, layers, dropout):
    self.hidden_size = hidden_size

    self.emb = Embedding(vocab_size - 1, hidden_size)
    self.rnn = LSTM(hidden_size, hidden_size, layers, dropout)

    self.params = [self.emb.weight,*self.rnn.params]

  def __call__(self, x, hc, m):
    emb = self.emb(x) * m
    x_, hc = self.rnn.__call__(emb.transpose(0, 1), hc)
    return x_.transpose(0, 1), hc

def autocompare(x,x2):
  if type(x) == Tensor: 
    x = x.numpy()
    x2 = x2.numpy()
  shape = tuple(min(a,b) for a,b in zip (x.shape,x2.shape))
  def forceshape(x:np.ndarray):
    x = x.reshape ((1,*x.shape))
    for i in range(len(shape)):
      x = x[:,:shape[i]]
      x = x.reshape ((-1,*x.shape[2:]))
    return x
  x2= forceshape(x2).reshape(shape)
  x = forceshape(x).reshape(shape)
  shape = tuple(s for s in shape if s > 1)
  x2= x2.reshape(shape)
  x= x.reshape(shape)

  if np.allclose(x,x2):
    return shape,True
  else:
    err = np.abs(x-x2).max()
    print(err)
    return False
#%%
# @TinyJit
def encode(rnnt:RNNT,X:Tensor,X_lens:np.ndarray,Y:Tensor,Y_lens:np.ndarray):
  bs = X.shape[1]
  Timer.start("enc")
  enc, enc_lens  = rnnt.encoder.__call__(X,X_lens)
  Timer.start("pred")
  preds,hc = rnnt.prediction.__call__(Tensor.zeros((bs,1)).cat(Y,dim=1),None,1)
  Timer.start("joint")
  distribution = rnnt.joint.__call__(preds, enc).softmax(3).realize()
  Timer.end("joint")
  return enc,enc_lens,distribution

#%%

def timestring(s):
  s = round(s)
  m,s = s//60,s%60
  h,m = m//60,m%60
  return f"{str(h).rjust(3)}:{str(m).rjust(2,'0')}:{str(s).rjust(2,'0')}"
import time

def logsumexp(a:np.ndarray,b:np.ndarray):
  mx = np.where(a>b,a,b)
  return np.log(np.exp(a-mx)+np.exp(b-mx)) + mx

class LogLoss(Function):

  def forward(self,distribution:LazyBuffer,X_lens:Tensor,Y:LazyBuffer, Y_lens:Tensor):
    bs=Y.shape[0]
    self.device = distribution.device
    distribution:np.ndarray = distribution.realized.toCPU().reshape(distribution.shape)
    self.labels=Y.realized.toCPU().reshape(Y.shape)
    X_lens = X_lens.realized.toCPU()
    Y_lens = Y_lens.realized.toCPU()

    self.logdists, self.logalphas, self.Ts, self.Us = [],[],[],[]
    Loss = []

    for bi in range(bs):
      T, U = round(X_lens[bi] + .2), int(Y_lens[bi])+1
      self.Ts.append(T)
      self.Us.append(U)
      
      logdist = np.log(distribution[bi,:U,:T])
      self.logdists.append(logdist)
      global logalpha
      logalpha = np.full((T,U),-np.inf,np.float32)
      logalpha [0,0] = 0
      self.logalphas.append(logalpha)

      for i in range(1,T+U-1):
        offset = max(0,i-T+1)
        u=np.arange(offset,min(i+1,U))
        t=i-u
        _t,_u = t[np.where(t>0)] , u[np.where(t>0)]
        logalpha[_t,_u] = logsumexp(logalpha[_t,_u], logalpha[_t-1,_u] + logdist[_u,_t-1,-1])
        _t,_u = t[np.where(u>0)], u[np.where(u>0)]
        logalpha[_t,_u] = logsumexp(logalpha[_t,_u], logalpha[_t,_u-1] + logdist[_u-1,_t,self.labels[bi,_u-1]])
      Loss .append( -logalpha[-1,-1] - logdist[-1,-1,-1])

    return LazyBuffer.fromCPU(np.sum(Loss))

  def backward(self,grad):

    bn = len(self.logalphas)
    logdgrads=[]

    for bi in range(bn):

      # global alpha,beta
      logalpha,logdist = self.logalphas[bi],self.logdists[bi]
      logalpha:np.ndarray
      T,U = self.Ts[bi], self.Us[bi]
      global logbeta
      logbeta = np.full((T,U), -np.inf,np.float32)
      logbeta [-1,-1] = 0
      global logab
      logab = logalpha.copy()

      for i in range(T+U-2,-1,-1):

        offset= max(0,i-T+1)
        global t,u
        u=np.arange(offset,min(i+1,U))
        t=i-u

        logab [t,u] += logbeta [t,u]
        _t,_u = t[np.where(t>0)] , u[np.where(t>0)]
        logbeta[_t-1,_u] = logsumexp(logbeta[_t-1,_u], logbeta[_t,_u] + logdist[_u,_t-1,-1])
        _t,_u = t[np.where(u>0)], u[np.where(u>0)]
        logbeta[_t,_u-1] = logsumexp(logbeta[_t,_u-1], logbeta[_t,_u] + logdist[_u-1,_t,self.labels[bi,_u-1]])

      t,u =np.arange(T-1), np.arange(U-1)
      global logdgrad
      # logdgrad = np.ful()
      logdgrad = np.full(self.logdists[bi].shape, -np.inf,np.float32)

      logdgrad[:,t,-1]                = logalpha.transpose()[:,t] - logalpha[-1,-1] + logbeta.transpose()[:,t+1]
      logdgrad[u,:,self.labels[bi,u]] = logalpha.transpose()[u,:] - logalpha[-1,-1] + logbeta.transpose()[u+1,:]
      
      logdgrad[-1,-1,-1] = 0

      logdgrad = np.pad(logdgrad, [[0,max(self.Us)-logdgrad.shape[0]],[0,max(self.Ts)-logdgrad.shape[1]],[0,0]],"constant",constant_values=-np.inf)
      logdgrads.append(logdgrad)

    assert not np.isnan(logdgrads).any()
    return LazyBuffer.fromCPU(- np.exp(logdgrads)), None, None, None, None



#%%
rnnt= RNNT()
opt = LAMB(rnnt.params)
opt.zero_grad()



#%%
# opt.step()

#%%
def save(name:str = "rnnt"):
  with open(name+'.npy', 'wb') as f:
    for par in rnnt.params:
      np.save(f, par.numpy())

# save()

def load(name:str ="rnnt"):
  with open(name+'.npy', 'rb') as f:
    for par in rnnt.params:
      #if par.requires_grad:
      try:
        par.numpy()[:] = np.load(f)
      except:
        print('Could not load parameter')

class Timer:

  data:dict[str,any] = {}
  running_items:set[str]= set()

  def start(item:str):

    for other in list(Timer.running_items):
      if (not item.startswith(other)):
        Timer.end(other)

    Timer.running_items.add(item)
    if not item in Timer.data: Timer.data[item] = {"time":0.}
    Timer.data[item]["start"] = time.time()

  def end(item):
    Timer.running_items.remove(item)
    Timer.data[item]['time'] += time.time() - Timer.data[item]['start']
    del Timer.data[item]['start']

    
  def reset(): 
    for v in Timer.data.values: v['time'] = 0

  def table(): 
    for k in Timer.data: print (str(k).ljust(20)+ f": {Timer.data[k]['time']:.5}")

#%%
if __name__ == "__main__":

  try:
    rnnt = RNNT()
    batch_size = 8
    start_time = time.time()
    it = enumerate(iterate(batch_size))
    Timer.start("")
    for i,(X,X_lens,Y,Y_lens) in it:


      last_time = time.time()
      enc,enclens,d = encode(rnnt,X,X_lens,Y,Y_lens)
      c = i * batch_size

      delta = time.time() - last_time
      Timer.start("forward")
      loss = LogLoss.apply(d,enclens.realize(), Y,Y_lens.realize())

      Timer.end("forward")

      print (f"{timestring(time.time()-start_time)} Btime: {(time.time()-last_time):.5} ")
      if i >= 10:
        Timer.end("")
        Timer.table()
        break

  except KeyboardInterrupt:pass




# %%
