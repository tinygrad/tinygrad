import numpy as np
from tqdm import trange
from transformers import BertTokenizer

from models.transformer_bert import Bert
from tinygrad.tensor import Tensor
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.nn.optim import Adam, get_parameters


class BertDataset:
  loss_mask = -100
  mask_ratio = 0.3
  def __init__(self, data, block_size, tokenizer):
    self.data = data
    self.block_size = block_size
    self.cls_id = tokenizer.cls_token_id
    self.mask_id = tokenizer.mask_token_id
    self.vocab_size = tokenizer.vocab_size

  def __getitem__(self, idx):
    x = self.data[idx: idx+self.block_size].astype(np.int32)
    y = self.data[idx: idx+self.block_size].astype(np.int32)
    x, y = self.mask_sample(x, y)
    return x, y

  def mask_sample(self, x, y):
    # replace first token with a [CLS] token, it's ok bc we sample from a contiguous blob of text anyway
    x[0] = self.cls_id
    for i in range(1, len(x)):
      if np.random.random() < self.mask_ratio:
        if (p := np.random.random()) < 0.8:
          mask_token = self.mask_id
        elif p < 0.9:
          random_token = np.random.randint(self.vocab_size)
          mask_token = random_token
        else:
          mask_token = x[i]
        x[i] = mask_token
      else:
        # ignore unmasked for loss calculation
        y[i] = self.loss_mask
    return x, y

  def get_batch(self, batch_size):
    ix = np.random.randint(len(self.data) - block_size, size=(batch_size,))
    batch = []
    for i in ix:
      batch.append(self[i])
    input_ids = np.array([c[0] for c in batch])
    targets = np.array([c[1] for c in batch])
    return input_ids, targets
  
if __name__ == "__main__":
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  tokenizer.vocab_size
  vocab_size = tokenizer.vocab_size
  block_size = 128
  layers = 12
  num_heads = 12
  embed_dim = 768
  ff_dim = 4 * embed_dim
  bert = Bert(vocab_size,
        block_size,
        layers,
        embed_dim,
        num_heads,
        ff_dim)

  dataset_path = 'train.bin'
  train_data = np.memmap(dataset_path, dtype=np.uint16, mode='r')
  bert_dataset = BertDataset(train_data, block_size, tokenizer)

  Tensor.training = True
  noloss = False
  BS = 32
  model = bert
  steps = 1_000_000 * 8
  optim = Adam(get_parameters(bert), lr=1e-4)

  def sparse_categorical_crossentropy(out, Y, ignore_index):
    num_classes = out.shape[-1]
    YY = Y.flatten().astype(np.int32)
    y = np.zeros((YY.shape[0], num_classes), np.float32)
    loss_mask = YY != ignore_index
    # correct loss for NLL, torch NLL loss returns one per row
    y[loss_mask, YY[loss_mask]] = -1.0
    y = y.reshape(list(Y.shape)+[num_classes])
    y = Tensor(y)
    return out.mul(y).sum() / sum(loss_mask)
    
  losses, accuracies = [], []
  for i in (t := trange(steps, disable=getenv('CI', False))):
    x, y = bert_dataset.get_batch(BS)
    x = Tensor(x, requires_grad=False)

    # network
    logprobs = model.forward(x) if hasattr(model, 'forward') else model(x)
    loss = sparse_categorical_crossentropy(logprobs, y, ignore_index=bert_dataset.loss_mask)
    optim.zero_grad()
    loss.backward()
    if noloss: del loss
    optim.step()

    # printing
    if not noloss and i%20==0:
      cat = np.argmax(logprobs.cpu().numpy(), axis=-1)
      loss = loss.detach().cpu().numpy()
      losses.append(loss)
      t.set_description("loss %.2f" % (loss))
