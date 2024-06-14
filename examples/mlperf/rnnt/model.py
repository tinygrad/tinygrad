from tinygrad import Tensor
from tinygrad.nn import Linear, Embedding


class LSTMCell:
  def __init__(self, input_size, hidden_size):
    k = hidden_size ** -0.5
    self.w_ih = Tensor.randn(input_size, hidden_size * 4).realize() * k
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

      res.append(Tensor.stack(h,c))
      h = res[-1][0]
      c = res[-1][1]
    
    ret = res[0].unsqueeze(0)
    for e in res[1:]: ret = ret.cat(e.unsqueeze(0) , dim=0).realize()
    return ret[:,0].realize()

class LSTM:
  def __init__(self,input_size, hidden_size, layers,_):
    self.cells = [LSTMCell(input_size, hidden_size) if i == 0 else LSTMCell(hidden_size,hidden_size) for i in range(layers)]
  
  def __call__(self,x:Tensor):
    for cell in self.cells: x = cell(x)
    return x.realize()

class StackTime:
  def __init__(self, factor):
    self.factor = factor

  def __call__(self, x):
    x = x.pad(((0, (-x.shape[0]) % self.factor), (0, 0), (0, 0)))
    x = x.reshape(x.shape[0] // self.factor, x.shape[1], x.shape[2] * self.factor)
    return x


STACKFACTOR = 2
class Encoder:
  def __init__(self, input_size, hidden_size, pre_layers, post_layers, dropout):
    self.pre_rnn = LSTM(input_size, hidden_size, pre_layers, dropout)
    self.stack_time = StackTime(STACKFACTOR)
    self.post_rnn = LSTM(STACKFACTOR * hidden_size, hidden_size, post_layers, dropout)

  def __call__(self, x):
    x = self.pre_rnn(x)
    x = self.stack_time(x)
    x = self.post_rnn(x)
    return x.transpose(0, 1)

class Prediction:
  def __init__(self, vocab_size, hidden_size, layers, dropout):
    self.hidden_size = hidden_size

    self.emb = Embedding(vocab_size - 1, hidden_size)
    self.rnn = LSTM(hidden_size, hidden_size, layers, dropout)

  def __call__(self, x, m):
    emb = self.emb(x) * m
    x_ = self.rnn(emb.transpose(0, 1))
    return x_.transpose(0, 1)

class Joint:
  def __init__(self, vocab_size, pred_hidden_size, enc_hidden_size, joint_hidden_size, dropout):
    self.dropout = dropout

    self.l1 = Linear(pred_hidden_size + enc_hidden_size, joint_hidden_size)
    self.l2 = Linear(joint_hidden_size, vocab_size)

  def __call__(self, f, g):
    (_, T, H), (B, U, H2) = f.shape, g.shape
    f = f.unsqueeze(2).expand(B, T, U, H)
    g = g.unsqueeze(1).expand(B, T, U, H2)

    inp = f.cat(g, dim=3)
    t = self.l1(inp).relu()
    t = t.dropout(self.dropout)
    return self.l2(t)

from tinygrad.nn.state import get_state_dict, load_state_dict, safe_save, safe_load

class RNNT:
  def __init__(self, input_features=240, vocab_size=29, enc_hidden_size=1024, pred_hidden_size=320, joint_hidden_size=512, pre_enc_layers=2, post_enc_layers=3, pred_layers=2, dropout=0.32):
    self.encoder = Encoder(input_features, enc_hidden_size, pre_enc_layers, post_enc_layers, dropout)
    self.prediction = Prediction(vocab_size, pred_hidden_size, pred_layers, dropout)
    self.joint = Joint(vocab_size, pred_hidden_size, enc_hidden_size, joint_hidden_size, dropout)

  def __call__(self, x, y):
    f = self.encoder(x)
    g = self.prediction(y, Tensor.ones(1, requires_grad=False))
    out = self.joint(f, g)
    return out.realize()

  def decode(self, x):
    logits, logit_lens = self.encoder(x)
    outputs = []
    for b in range(logits.shape[0]):
      inseq = logits[b, :, :].unsqueeze(1)
      logit_len = logit_lens[b]
      seq = self._greedy_decode(inseq, int(np.ceil(logit_len.numpy()).item()))
      outputs.append(seq)
    return outputs

  def _greedy_decode(self, logits, logit_len):
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

  def _pred_joint(self, logit, label, hc, mask):
    g, hc = self.prediction(label, hc, mask)
    j = self.joint(logit, g)[0]
    j = j.pad(((0, 1), (0, 1), (0, 0)))
    out = j.cat(hc, dim=2)
    return out.realize()

  def save(self,fname): safe_save(get_state_dict(self),fname)

  def load(self,fname:str): load_state_dict(self, safe_load(fname))

