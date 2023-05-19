from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
import numpy as np
from extra.utils import download_file
from pathlib import Path


class RNNT:
  def __init__(self, input_features=240, vocab_size=29, enc_hidden_size=1024, pred_hidden_size=320, joint_hidden_size=512, pre_enc_layers=2, post_enc_layers=3, pred_layers=2, stack_time_factor=2, dropout=0.32):
    self.encoder = Encoder(input_features, enc_hidden_size, pre_enc_layers, post_enc_layers, stack_time_factor, dropout)
    self.prediction = Prediction(vocab_size, pred_hidden_size, pred_layers, dropout)
    self.joint = Joint(vocab_size, pred_hidden_size, enc_hidden_size, joint_hidden_size, dropout)

  def __call__(self, x, x_lens, y, h=None, c=None):
    f, _ = self.encoder(x, x_lens)
    g, _, _ = self.prediction(y, h, c)
    out = self.joint(f, g)
    return out

  def decode(self, x, x_lens):
    logits, logit_lens = self.encoder(x, x_lens)
    outputs = []
    for b in range(logits.shape[0]):
      inseq = logits[b, :, :].unsqueeze(1)
      logit_len = logit_lens[b]
      seq = self._greedy_decode(inseq, logit_len)
      outputs.append(seq)
    return outputs

  def _greedy_decode(self, logits, logit_lens):
    h = c = None
    labels = []
    for time_idx in range(int(logit_lens.numpy().item())):
      logit = logits[time_idx, :, :].unsqueeze(0)
      not_blank = True
      added = 0
      while not_blank and added < 30:
        label = Tensor([[labels[-1] if labels[-1] <= 28 else labels[-1] - 1]]) if len(labels) > 0 else None
        g, h_, c_ = self.prediction(label, h, c)
        j = self.joint(logit, g)[:, 0, 0, :][0, :].numpy()

        k = np.argmax(j, axis=0)
        not_blank = k != 28
        if not_blank:
          labels.append(k)
          h, c = h_, c_
        added += 1
    return labels

  def load_from_pretrained(self):
    fn = Path(__file__).parent.parent / "weights/rnnt.pt"
    download_file("https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1", fn)

    import torch
    state_dict = torch.load(open(fn, "rb"), map_location="cpu")["state_dict"]

    # encoder
    for i in range(2):
      self.encoder.pre_rnn.cells[i].weights_ih.assign(state_dict[f"encoder.pre_rnn.lstm.weight_ih_l{i}"].numpy())
      self.encoder.pre_rnn.cells[i].weights_hh.assign(state_dict[f"encoder.pre_rnn.lstm.weight_hh_l{i}"].numpy())
      self.encoder.pre_rnn.cells[i].bias_ih.assign(state_dict[f"encoder.pre_rnn.lstm.bias_ih_l{i}"].numpy())
      self.encoder.pre_rnn.cells[i].bias_hh.assign(state_dict[f"encoder.pre_rnn.lstm.bias_hh_l{i}"].numpy())
    for i in range(3):
      self.encoder.post_rnn.cells[i].weights_ih.assign(state_dict[f"encoder.post_rnn.lstm.weight_ih_l{i}"].numpy())
      self.encoder.post_rnn.cells[i].weights_hh.assign(state_dict[f"encoder.post_rnn.lstm.weight_hh_l{i}"].numpy())
      self.encoder.post_rnn.cells[i].bias_ih.assign(state_dict[f"encoder.post_rnn.lstm.bias_ih_l{i}"].numpy())
      self.encoder.post_rnn.cells[i].bias_hh.assign(state_dict[f"encoder.post_rnn.lstm.bias_hh_l{i}"].numpy())

    # prediction
    self.prediction.emb.weight.assign(state_dict["prediction.embed.weight"].numpy())
    for i in range(2):
      self.prediction.rnn.cells[i].weights_ih.assign(state_dict[f"prediction.dec_rnn.lstm.weight_ih_l{i}"].numpy())
      self.prediction.rnn.cells[i].weights_hh.assign(state_dict[f"prediction.dec_rnn.lstm.weight_hh_l{i}"].numpy())
      self.prediction.rnn.cells[i].bias_ih.assign(state_dict[f"prediction.dec_rnn.lstm.bias_ih_l{i}"].numpy())
      self.prediction.rnn.cells[i].bias_hh.assign(state_dict[f"prediction.dec_rnn.lstm.bias_hh_l{i}"].numpy())

    # joint
    self.joint.l1.weight.assign(state_dict["joint_net.0.weight"].numpy())
    self.joint.l1.bias.assign(state_dict["joint_net.0.bias"].numpy())
    self.joint.l2.weight.assign(state_dict["joint_net.3.weight"].numpy())
    self.joint.l2.bias.assign(state_dict["joint_net.3.bias"].numpy())


class LSTMCell:
  def __init__(self, input_size, hidden_size):
    self.hidden_size = hidden_size

    self.weights_ih = Tensor.uniform(hidden_size * 4, input_size)
    self.bias_ih = Tensor.uniform(hidden_size * 4)
    self.weights_hh = Tensor.uniform(hidden_size * 4, hidden_size)
    self.bias_hh = Tensor.uniform(hidden_size * 4)

  def __call__(self, x, h, c):
    t = (x @ self.weights_ih.T) + self.bias_ih + (h @ self.weights_hh.T) + self.bias_hh

    i, f, g, o = t.chunk(4, 1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()

    c = (f * c) + (i * g)
    h = o * c.tanh()

    return h.realize(), c.realize()


class LSTM:
  def __init__(self, input_size, hidden_size, layers, dropout):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.layers = layers
    self.dropout = dropout

    self.cells = [LSTMCell(input_size, hidden_size) if i == 0 else LSTMCell(hidden_size, hidden_size) for i in range(layers)]

  def __call__(self, x, h, c):
    if h is None or c is None:
      h = [Tensor.zeros(x.shape[1], self.hidden_size) for _ in range(self.layers)]
      c = [Tensor.zeros(x.shape[1], self.hidden_size) for _ in range(self.layers)]
    else:
      h, c = h.copy(), c.copy()

    output = None
    for t in range(x.shape[0]):
      input = x[t]
      for i, cell in enumerate(self.cells):
        h[i], c[i] = cell(input, h[i], c[i])
        # dropout
        if i != self.layers - 1:
          h[i] = h[i].dropout(self.dropout)
        input = h[i]
      output = h[-1].unsqueeze(0) if output is None else Tensor.cat(output, h[-1].unsqueeze(0), dim=0).realize()

    return output, h, c


class StackTime:
  def __init__(self, factor):
    self.factor = factor

  def __call__(self, x, x_lens):
    r = x.transpose(0, 1)
    if r.shape[1] % self.factor != 0:
      r = r.cat(Tensor.zeros(r.shape[0], (-r.shape[1]) % self.factor, r.shape[2]), dim=1)
    r = r.reshape(r.shape[0], r.shape[1] // self.factor, r.shape[2] * self.factor)
    x_lens = Tensor(np.ceil((x_lens / self.factor).numpy()))
    return r.transpose(0, 1), x_lens


class Encoder:
    def __init__(self, input_size, hidden_size, pre_layers, post_layers, stack_time_factor, dropout):
      self.pre_rnn = LSTM(input_size, hidden_size, pre_layers, dropout)
      self.stack_time = StackTime(stack_time_factor)
      self.post_rnn = LSTM(stack_time_factor * hidden_size, hidden_size, post_layers, dropout)

    def __call__(self, x, x_lens):
      x, _, _ = self.pre_rnn(x, None, None)
      x, x_lens = self.stack_time(x, x_lens)
      x, _, _ = self.post_rnn(x, None, None)
      return x.transpose(0, 1), x_lens


class Embedding:
  def __init__(self, vocab_size: int, embed_size: int):
    self.vocab_size = vocab_size
    self.weight = Tensor.scaled_uniform(vocab_size, embed_size)

  def __call__(self, idx: Tensor) -> Tensor:
    idxnp = idx.numpy().astype(np.int32)
    onehot = np.zeros((idx.shape[0], idx.shape[1], self.vocab_size), dtype=np.float32)
    for i in range(idx.shape[0]):
      onehot[i, np.arange(idx.shape[1]), idxnp[i]] = 1
    return Tensor(onehot, requires_grad=False) @ self.weight


class Prediction:
  def __init__(self, vocab_size, hidden_size, layers, dropout):
    self.hidden_size = hidden_size

    self.emb = Embedding(vocab_size - 1, hidden_size)
    self.rnn = LSTM(hidden_size, hidden_size, layers, dropout)

  def __call__(self, x, h, c):
    x = self.emb(x) if x is not None else (Tensor.zeros(1, 1, self.hidden_size) if h is None else Tensor.zeros(h[0].shape[1], 1, self.hidden_size))
    x, h, c = self.rnn(x.transpose(0, 1), h, c)
    return x.transpose(0, 1), h, c


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
