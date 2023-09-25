#%%
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam, LAMB
from tinygrad.nn import Embedding, Linear
from tinygrad.helpers import dtypes
from tinygrad.ops import Device
from tinygrad.jit import TinyJit
from tinygrad.mlops import Function
from tinygrad.lazy import LazyBuffer

import itertools
import pathlib
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile

#%% data extract
BASEDIR = pathlib.Path("../../../extra/datasets/librispeech")
with open(BASEDIR / "dev-clean-wav.json") as f:
  ci = json.load(f)
FILTER_BANK = np.expand_dims(librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000), 0)
WINDOW = librosa.filters.get_window("hann", 320)

def feature_extract(x, x_lens):
  x_lens = np.ceil((x_lens / 160) / 3).astype(np.int32)

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
    # pad to same length
    max_len = max(sample_lens)
    for j in range(len(samples)):
      samples[j] = np.pad(samples[j], (0, max_len - sample_lens[j]), "constant")
    samples, sample_lens = np.array(samples), np.array(sample_lens)

    yield feature_extract(samples, sample_lens), np.array([v["transcript"] for v in ci[i : i + bs]])

characters = [*" 'abcdefghijklmnopqrstuvwxyz","<skip>"]
c2i= dict([(c,i) for i,c in enumerate(characters)])
charn = len(characters)

def text_encode(text:list[str]):
    if isinstance(text,str):text = [text]
    return [[np.array(c2i[char]) for char in seq] for seq in text]

def text_decode(toks:Tensor):
    ret = []
    for seq in toks:
        ret.append("".join([characters[int(tok)] for tok in seq ]))
    return ret

# %% model
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
    self.dropout = dropout

    self.weights_ih = Tensor.uniform(hidden_size * 4, input_size)
    self.bias_ih = Tensor.uniform(hidden_size * 4)
    self.weights_hh = Tensor.uniform(hidden_size * 4, hidden_size)
    self.bias_hh = Tensor.uniform(hidden_size * 4)

  def __call__(self, x:Tensor, hc:Tensor):
    gates = x.linear(self.weights_ih.T, self.bias_ih) + hc[:x.shape[0]].linear(self.weights_hh.T, self.bias_hh)

    i, f, g, o = gates.chunk(4, 1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()

    c = (f * hc[x.shape[0]:]) + (i * g)
    h = (o * c.tanh()).dropout(self.dropout)

    return Tensor.cat(h, c).realize()

class LSTM:
  def __init__(self, input_size, hidden_size, layers, dropout):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.layers = layers

    self.cells = [LSTMCell(input_size if i == 0 else hidden_size, hidden_size, dropout if i != layers - 1 else 0) for i in range(layers)]

    self.params = list(itertools.chain(*[[cell.bias_hh,cell.bias_ih,cell.weights_hh,cell.weights_ih] for cell in self.cells]))

  def do_step(self, x, hc):
    new_hc = [x]
    for i, cell in enumerate(self.cells):
      new_hc.append(cell(new_hc[i][:x.shape[0]], hc[i]))
    return Tensor.stack(new_hc[1:]).realize()

  def __call__(self, x, hc):
    @TinyJit
    def _do_step(x_, hc_):
      return self.do_step(x_, hc_)

    if hc is None:
      hc = Tensor.zeros(self.layers, 2 * x.shape[1], self.hidden_size, requires_grad=False)

    output = None
    for t in range(x.shape[0]):
      hc = _do_step(x[t] + 1 - 1, hc) # TODO: why do we need to do this?
      if output is None:
        output = hc[-1:, :x.shape[1]]
      else:
        output = output.cat(hc[-1:, :x.shape[1]], dim=0).realize()

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

  def __call__(self, x, x_lens):
    x = x.pad(((0, (-x.shape[0]) % self.factor), (0, 0), (0, 0)))
    x = x.reshape(x.shape[0] // self.factor, x.shape[1], x.shape[2] * self.factor)
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

 #%%
def check32(arg):assert arg.dtype == np.float32 or arg.dtype == dtypes.float32 , f'{arg.dtype}'
#%% Loss
class RNNTLoss(Function):
  def forward(self,distribution:LazyBuffer,labels:LazyBuffer):
    self.device = distribution.device
    self.distribution = distribution.toCPU()
    assert isinstance(self.distribution,np.ndarray)
    self.labels=labels.toCPU()

    self.T,self.U = distribution.shape[2],distribution.shape[1]
    assert len(self.labels) == self.U-1, f"len labels {len(self.labels)} doesnt match U-1 {self.U-1}"

    self.alpha = np.zeros((self.T,self.U),dtype=np.float32)
    self.alpha [0,0] = 1

    alpha_norm_log = np.zeros((),np.float32)

    for i in range(1,self.T+self.U-1):

      offset= max(0,i-self.T+1)
      u=np.arange(offset,min(i+1,self.U))
      t=i-u

      _t,_u = t[np.where(t>0)] , u[np.where(t>0)]

      self.alpha[_t,_u] += self.alpha[_t-1,_u] * self.distribution[0,_u,_t-1,-1]

      _t,_u = t[np.where(u>0)], u[np.where(u>0)]
      self.alpha[_t,_u] += self.alpha[_t,_u-1] * self.distribution[0,_u-1,_t,self.labels[_u-1]]

      alpha_norm = self.alpha[t,u].sum()
      self.alpha [t,u] /= alpha_norm
      alpha_norm_log += np.log(alpha_norm,dtype=np.float32)
    
    Loss= -alpha_norm_log - np.log(self.distribution[0,-1,-1,-1])
    return LazyBuffer.fromCPU(Loss)

  def backward(self,grad):
    beta = np.zeros((self.T,self.U))
    beta [-1,-1] = 1

    ab = self.alpha
    
    for i in range(self.T+self.U-2,-1,-1):

      offset= max(0,i-self.T+1)
      u=np.arange(offset,min(i+1,self.U))
      t=i-u

      beta[t,u] /= beta[t,u].sum()

      ab [t,u] *= beta [t,u]
      ab [t,u] /= ab[t,u].sum()

      _t,_u = t[np.where(t>0)] , u[np.where(t>0)]
      beta[_t-1,_u] += beta[_t,_u] * self.distribution[0,_u,_t-1,-1]

      _t,_u = t[np.where(u>0)], u[np.where(u>0)]
      beta[_t,_u-1] += beta[_t,_u] * self.distribution[0,_u-1,_t,self.labels[_u-1]]

    dgrad = np.zeros_like(self.distribution)
    t = np.arange(self.T-1)
    u = np.arange(self.U-1)
    dgrad[0,:-1,:-1,-1] = ab[:-1,:-1].T / (self.distribution[0,:-1,:-1,-1] + (beta[:-1,1:]/beta[1:,:-1]).T * self.distribution[0,u,:-1,self.labels[u]] )
    # u=U
    dgrad[0,-1,:,-1] = ab[:,-1].T / (self.distribution[0,-1,:,-1]  )

    dgrad[0,u,:-1,self.labels[u]] = ab[:-1,:-1].T / (self.distribution[0,:-1,:-1,-1]* (beta[1:,:-1]/beta[:-1,1:]).T + self.distribution[0,u,:-1,self.labels[u]])
    dgrad[0,u,-1,self.labels[u]] = ab[-1,:-1].T / ( self.distribution[0,u,-1,self.labels[u]])

    return LazyBuffer.fromCPU(-dgrad), None


Loss = RNNTLoss.apply(distribution_tensor, labels)
Loss.numpy()
Loss.backward()

#%% encode
def encode(X,labels):
  enc, enc_lens  = rnnt.encoder(Tensor(X[0]),Tensor(X[1]))
  preds = None
  hc = Tensor.zeros(rnnt.prediction.rnn.layers, 2, rnnt.prediction.hidden_size)
  for x in [0] + list(labels.numpy()):
    pred,hc = rnnt.prediction.__call__(Tensor([[x]]),hc,1)
    preds = pred if preds is None else preds.cat(pred,dim=1).realize()

  distribution_tensor = rnnt.joint.__call__(preds, enc).softmax(3).realize()
  distribution = distribution_tensor.numpy()
  return enc,distribution,distribution_tensor


opt = LAMB(rnnt.params)
def train_step(X,Y):
  opt.zero_grad()
  labels = Tensor(text_encode(Y[0])[0],dtype=dtypes.int16)
  enc,distribution,distribution_tensor = encode(X[0],labels)

  loss = RNNTLoss.apply(distribution_tensor,labels)
  ll = loss.numpy()
  print (f'loss: {ll:.5} latice shape: {distribution.shape[1:-1]} normalized: {ll/sum(distribution.shape[:-1])}')

  loss.backward()
  opt.step()
  return ll
# %%
rnnt= RNNT()
opt = LAMB(rnnt.params)

# %%
hist = []
def epoch ():
  for i,(X,Y) in enumerate(iterate()):
    print(end=(f'{i}: ').rjust(5))
    Loss = train_step(X,Y)
    hist.append(Loss)

epoch()
# %%
