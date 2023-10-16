
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam, LAMB
from tinygrad.nn import Embedding, Linear
from tinygrad.helpers import dtypes
from tinygrad.ops import Device
from tinygrad.jit import TinyJit
from tinygrad.mlops import Function
from tinygrad.lazy import LazyBuffer
from extra.utils import print_tree

import itertools
import pathlib
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile

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
def iterate(bs=1, start=0):
  print(f"there are {len(ci)} samples in the dataset")
  for i in range(start, len(ci), bs):
    samples, sample_lens = zip(*[load_wav(BASEDIR / v["files"][0]["fname"]) for v in ci[i : i + bs]])
    samples = list(samples)
    X,X_lens = list(zip(*[feature_extract(np.array(samples[i:i+1]),np.array(sample_lens[i:i+1])) for i in range(bs)]))
    max_len = max(X_lens)
    X = [np.pad(X[j],((0,int(max_len[0]-X_lens[j][0])),(0,0),(0,0)),'constant') for j in range (len(X))]
    yield (
      np.concatenate(X,axis=1),
      (np.array(X_lens)),
      *text_encode([v['transcript'] for v in ci[i:i+bs]]))

characters = [*" 'abcdefghijklmnopqrstuvwxyz","<skip>"]
c2i= dict([(c,i) for i,c in enumerate(characters)])
charn = len(characters)

def text_encode(text:list[str]):
    if isinstance(text,str):text = [text]
    seqs = [np.array([np.array(c2i[char]) for char in seq]) for seq in text]
    seq_lens = [len(s) for s in seqs]
    seqs = list(map (lambda s: np.pad(s,(0,max(seq_lens)-len(s)),"constant"),seqs))
    return Tensor(seqs,dtype=dtypes.int16), seq_lens

def text_decode(toks:Tensor):
    ret = []
    for seq in toks:
        ret.append("".join([characters[int(tok)] for tok in seq ]))
    return ret

class RNNT:
  def __init__(self, input_features=240, vocab_size=29, enc_hidden_size=1024, pred_hidden_size=320, joint_hidden_size=512, pre_enc_layers=2, post_enc_layers=3, pred_layers=2, stack_time_factor=2, dropout=0.32):
    self.encoder = Encoder(input_features, enc_hidden_size, pre_enc_layers, post_enc_layers, stack_time_factor, dropout)
    self.prediction = Prediction(vocab_size, pred_hidden_size, pred_layers, dropout)
    self.joint = Joint(vocab_size, pred_hidden_size, enc_hidden_size, joint_hidden_size, dropout)
    self.params = [*self.encoder.params, *self.prediction.params, *self.joint.params]

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
    x, _ = self.pre_rnn(x, None)
    x, x_lens = self.stack_time(x, x_lens)
    x, _ = self.post_rnn(x, None)
    return x.transpose(0, 1), x_lens
  
class StackTime:
  def __init__(self, factor):
    self.factor = factor

  def __call__(self, x:Tensor, x_lens):
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
    x_, hc = self.rnn(emb.transpose(0, 1), hc)
    return x_.transpose(0, 1), hc

rnnt = RNNT()
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

def encode(X,X_lens,Y,Y_lens):

  enc, enc_lens  = rnnt.encoder(Tensor(X),Tensor(X_lens))
  bs = X.shape[1]
  preds,hc = rnnt.prediction(Tensor.zeros((bs,1)).cat(Y,dim=1),None,1)
  distribution = rnnt.joint.__call__(preds, enc).softmax(3).realize()
  return enc,enc_lens,distribution

def timestring(s):
  s = round(s)
  m,s = s//60,s%60
  h,m = m//60,m%60
  return f"{str(h).rjust(3)}:{str(m).rjust(2,'0')}:{str(s).rjust(2,'0')}"
import time
progress_start = time.time()
def progressbar(i = None):
  global progress_start
  if i == None : 
    progress_start = time.time()
    return
  dur = time.time() - progress_start
  sampled = (i+1) * batch_size
  print (f"{sampled}/{len(ci)} done. {timestring(dur)} projected total: {timestring(dur*len(ci)/sampled)}".ljust(50),end = "",flush=True)
  
class RNNTBatchLoss(Function):

  def forward(self,distribution:LazyBuffer,X_lens:LazyBuffer,Y:LazyBuffer, Y_lens:LazyBuffer):
    bs=Y.shape[0]
    self.device = distribution.device
    distribution = distribution.toCPU()
    self.labels=Y.toCPU()
    self.dists, self.alphas, self.Ts, self.Us, self.dists = [],[],[],[],[]
    Loss = []

    for bi in range(bs):
      T, U = round(X_lens.toCPU()[bi][0] + .2), int(Y_lens.toCPU()[bi])+1
      self.Ts.append(T)
      self.Us.append(U)

      _, du,dt, _ = distribution.shape
      dist = distribution[bi,:U,:T]
      self.dists.append(dist)

      alpha = np.zeros((T,U),dtype=np.float32)
      alpha [0,0] = 1
      self.alphas.append(alpha)
      alpha_norm_log = np.zeros((),np.float32)
      for i in range(1,T+U-1):
        offset = max(0,i-T+1)
        u=np.arange(offset,min(i+1,U))
        t=i-u
        _t,_u = t[np.where(t>0)] , u[np.where(t>0)]
        alpha[_t,_u] += alpha[_t-1,_u] * dist[_u,_t-1,-1]
        _t,_u = t[np.where(u>0)], u[np.where(u>0)]
        alpha[_t,_u] += alpha[_t,_u-1] * dist[_u-1,_t,self.labels[bi,_u-1]]
        
        alpha_norm = alpha[t,u].sum()
        alpha [t,u] /= alpha_norm
        alpha_norm_log += np.log(alpha_norm,dtype=np.float32)
      Loss .append(-alpha_norm_log - np.log(dist[-1,-1,-1]))

    return LazyBuffer.fromCPU(np.sum(Loss))

  def backward(self,grad):

    bn = len(self.alphas)
    dgrads=[]

    for bi in range(bn):

      global alpha,beta
      alpha,dist = self.alphas[bi],self.dists[bi]
      T,U = self.Ts[bi], self.Us[bi]
      beta = np.zeros((T,U))
      beta [-1,-1] = 1
      global ab
      ab = alpha # we store a * b normalized across diagonal dont use alpha after this.

      for i in range(T+U-2,-1,-1):

        offset= max(0,i-T+1)
        global t,u
        u=np.arange(offset,min(i+1,U))
        t=i-u

        beta[t,u] /= beta[t,u].sum()
        ab [t,u] *= beta [t,u]
        ab [t,u] /= ab[t,u].sum() + 1e-40 # numerical stability 
        _t,_u = t[np.where(t>0)] , u[np.where(t>0)]
        beta[_t-1,_u] += beta[_t,_u] * dist[_u,_t-1,-1]
        _t,_u = t[np.where(u>0)], u[np.where(u>0)]
        beta[_t,_u-1] += (beta[_t,_u] * dist[_u-1,_t,self.labels[bi,_u-1]])
      
      beta += 1e-40
      dgrad = np.zeros_like(dist)
      t = np.arange(T-1)
      u = np.arange(U-1)
      dgrad[:-1,:-1,-1] = ab[:-1,:-1].T / (dist[:-1,:-1,-1] + (beta[:-1,1:]/beta[1:,:-1]).T * dist[u,:-1,self.labels[bi,u]] )
      dgrad[-1,:,-1] = ab[:,-1].T / (dist[-1,:,-1]  )

      dgrad[u,:-1,self.labels[bi,u]] = ab[:-1,:-1].T / (dist[:-1,:-1,-1]* (beta[1:,:-1]/beta[:-1,1:]).T + dist[u,:-1,self.labels[bi,u]])
      dgrad[u,-1,self.labels[bi,u]] = ab[-1,:-1].T / ( dist[u,-1,self.labels[bi,u]])

      if (np.isnan(dgrad).any()):
        raise "Nan value encountered"

      dgrad = np.pad(dgrad, [[0,max(self.Us)-dgrad.shape[0]],[0,max(self.Ts)-dgrad.shape[1]],[0,0]],"constant")
      dgrads.append(dgrad)

    return LazyBuffer.fromCPU(- np.array(dgrads)), None

rnnt= RNNT()
opt = LAMB(rnnt.params)
opt.zero_grad()
batch_size=2
it = enumerate(iterate(batch_size))
progressbar()

for i, (X,X_lens,Y,Y_lens ) in it:
  opt.zero_grad()
  global enc2, enc2lens, dis2 
  enc2,enc2lens,dis2 = encode(X,X_lens,Y,Y_lens)
  loss = RNNTBatchLoss.apply(dis2,enc2lens, Y,Tensor(Y_lens,requires_grad=False))
  loss.backward()
  loss_normal = loss.numpy() / (sum(X_lens) + sum(Y_lens))
  opt.step()

  progressbar(i)
  print (f" loss: {loss.numpy():.5} normalized: {loss_normal:.5}",flush=True)
  if i %20==0:
    print ("beta")
    plt.imshow(beta)
    plt.show()
    print ("ab")
    plt.imshow(ab)
    plt.show()
