#!/usr/bin/env python
import os
import argparse
import urllib.request
import numpy as np
import tinygrad.nn as nn
from tinygrad.nn import optim
from tinygrad.state import get_parameters
from tinygrad.tensor import Tensor, dtypes
import torch # remove if faster multimonial selection is found
from torch import multinomial

# tiny shakespear input text
input_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

Tensor.manual_seed(31337)

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 50
n_embd = 32
n_head = 4
n_layer = 3
dropout = 0.2

def get_input_text(url, file_path, clean):
  """ fetch and return tiny Shakespear input text """
  urllib.request.urlretrieve(url, file_path)
  with open(file_path, 'r') as f:
    text = f.read()
  if clean:
    os.remove(file_path)
  return text

def cross_entropy(out, Y):
  """ negative Log Loss function """
  num_classes = out.shape[-1]
  YY = Y.flatten().astype(np.int32)
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y)
  return out.mul(y).mean()

def estimate_loss():
  out = {}
  Tensor.training = False
  for split in ['train', 'val']:
    losses = np.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      _, loss = model(X, Y)
      losses[k] = loss.numpy()
    out[split] = losses.mean()
  Tensor.training = True
  return out

# data loading
def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == "train" else val_data
  ix = Tensor.uniform(batch_size, low=0, high=(data.shape[0] - block_size)).cast(dtype=dtypes.int32)
  x = Tensor.stack([data[i:i+block_size] for i in ix.numpy()])
  y = Tensor.stack([data[i+1:i+block_size+1] for i in ix.numpy()])
  return x, y


class Head():
  """ one head of self-attention """

  def __init__(self, head_size):
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False) # usually bias term is added on q
    self.value = nn.Linear(n_embd, head_size, bias=False) # usually bias term is added on v

  def __call__(self, x):
    B,T,C = x.shape
    k = self.key(x)   # (B,T,C)
    q = self.query(x) # (B,T,C)
    # compute attention scores ("affinities")
    wei = q @ Tensor.transpose(k, -2, -1) * C**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
    # equal to a lower triangular matrix (tril) masked_fill in pytorch
    mask = Tensor(np.triu(np.ones((T,T), dtype=np.float32) * -np.inf, k=1))
    wei = wei + mask
    wei = wei.softmax(-1)
    wei = wei.dropout(dropout)
    # perform the weighted aggregation of the values
    v = self.value(x) # (B,T,hs)
    out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
    return out


class MultiHeadAttention():

  """ multiple heads of self-attention in parallel """
  def __init__(self, num_heads, head_size):
    self.heads = [Head(head_size) for _ in range(num_heads)]
    self.proj = nn.Linear(n_embd, n_embd)

  def __call__(self, x):
    out = self.heads[0](x).cat(*[h(x) for h in self.heads[1:]], dim=-1)
    out = self.proj(out).dropout(dropout)
    return out


class FeedForward():
  """ a simple linear layer followed by a non-linearity """

  def __init__(self, n_embd):
    self.net = [
      nn.Linear(n_embd, 4 * n_embd),
      Tensor.relu,
      nn.Linear(4 * n_embd, n_embd)
    ]

  def __call__(self, x):
    return x.sequential(self.net).dropout(dropout)


class Block():
  """ transformer block: communication followed by computation """
  def __init__(self, n_embd, n_head):
    # n_embd: embedding dimension, n_head: the number of heads we'd like
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def __call__(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


class GPTLanguageModel():
  """ a decoder only transformer """

  def __init__(self, vocab_size):
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def __call__(self, idx, targets=None):
    B, T = idx.shape

    # idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.position_embedding_table(Tensor.arange(T, dtype=dtypes.int8).reshape(1,T)) # (T,C)
    x = tok_emb + pos_emb # (B,T,C)
    for block in self.blocks: x = block(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # (B,T,vocab_size)
    
    if targets is None:
      loss = None
    else:
      # log softmax to get predictions for cross_entropy loss calculation
      predictions = logits.log_softmax(-1)
      loss = cross_entropy(predictions, targets.numpy())
    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # get the predictions
      logits, _ = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probabilities
      probs = logits.softmax(-1) # (B, C)
      # sample from the distribution
      idx_next = multinomial(torch.tensor(probs.numpy()), num_samples=1).numpy() # (B, 1)
      # multinomial sampling using numpy  **slow**
      #idx_next = np.array([np.random.choice(len(p), size=1, p=p) for p in probs.numpy()])
      idx_next = Tensor(idx_next)
      # append sampled index to the running sequence
      idx = Tensor.cat(idx, idx_next, dim=1) # (B, T+1)
      print(decode(idx_next.numpy()[0]), end='')
    return idx


if __name__ == "__main__":
  """
  Generative Transformer implementation Pre-trained on tiny Shakespeare data set.
  This is almost a direct copy of the video how-to by Andrej Karpathy implemented in tinygrad.
  Reference of the pytorch implementation and video.
  
      YouTube : https://youtu.be/kCc8FmEb1nY
      Github  : https://github.com/karpathy/ng-video-lecture/tree/master
  """

  parser = argparse.ArgumentParser(description="""Tiny Shakespeare GPT""")
  parser.add_argument('--path', default=os.path.join(os.path.sep, 'tmp', 'input.txt'), 
                      help='Location to save the input text, defaults to /tmp/input.txt')
  parser.add_argument('--clean', action="store_true",
                      help='Delete the input text file after run, defaults to False.')
  args = parser.parse_args()

  text = get_input_text(input_url, args.path, args.clean)

  # here are all the unique characters that occur in this text
  chars = sorted(list(set(text)))
  vocab_size = len(chars)
  # create a mapping from characters to integers
  stoi = { ch:i for i,ch in enumerate(chars) }
  itos = { i:ch for i,ch in enumerate(chars) }

  # encoder: take a string, output a list of integers
  encode = lambda s: [stoi[c] for c in s]  # noqa: E731
  # decoder: take a list of integers, output a string
  decode = lambda l: ''.join([itos[i] for i in l])  # noqa: E731

  # train and test splits
  data = Tensor(encode(text), dtype=dtypes.int64, requires_grad=False)
  n = int(0.9*data.shape[0])
  train_data = data[:n]
  val_data = data[n:]
      
  model = GPTLanguageModel(vocab_size)
  
  parameters = get_parameters(model)
  optimizer = optim.AdamW(parameters, lr=1e-3)

  Tensor.training = True
  for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
      losses = estimate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()        
    optimizer.step()
  
  print("final loss: {0}".format(loss.numpy()))

  Tensor.training = False
  context = Tensor.zeros((1, 1), dtypes.int64)
  print("-- hark! --")
  model.generate(context, max_new_tokens=500).numpy()[0]
