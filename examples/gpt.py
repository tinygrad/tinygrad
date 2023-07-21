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
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 32
n_head = 6
n_layer = 6
dropout = 0.2

def tempestLoop():
  pass

def get_input_text(url, file_path, clean):
  """Fetch and return tiny Shakespear input text"""
  urllib.request.urlretrieve(url, file_path)
  with open(file_path, 'r') as f:
    text = f.read()
  if clean:
    os.remove(file_path)
  return text

def cross_entropy(out, Y):
  """Negative Log Loss function"""
  num_classes = out.shape[-1]
  YY = Y.flatten().astype(np.int32)
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y)
  return out.mul(y).mean()

# data loading
def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == "train" else val_data
  ix = Tensor.uniform(batch_size, low=0, high=(data.shape[0] - block_size)).cast(dtype=dtypes.int32)
  x = Tensor.stack([data[i:i+block_size] for i in ix.numpy()])
  y = Tensor.stack([data[i+1:i+block_size+1] for i in ix.numpy()])
  return x, y


class BigramLanguageModel():
  def __init__(self, vocab_size):
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  #def __call__(self, idx, targets=None):
  def __call__(self, idx, targets=None):
    B, T = idx.shape

    # idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.position_embedding_table(Tensor.arange(T, dtype=dtypes.int8).reshape(1,T)) # (T,C)
    x = tok_emb + pos_emb # (B,T,C)
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
      # get the predictions
      logits, _ = self(idx)
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

  encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
  decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

  # train and test splits
  data = Tensor(encode(text), dtype=dtypes.int64, requires_grad=False)
  n = int(0.9*data.shape[0])
  train_data = data[:n]
  val_data = data[n:]
      
  m = BigramLanguageModel(vocab_size)
  
  parameters = get_parameters(m)
  optimizer = optim.AdamW(parameters, lr=1e-3)

  Tensor.training = True
  for epoch in range(1000):
    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    _, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()        
    optimizer.step()
    if (epoch % 50 == 0):
      print("epoch: {0} loss: {1}".format(epoch, loss.numpy()))
  
  print("final loss: {0}".format(loss.numpy()))

  Tensor.training = False
  print(decode(m.generate(Tensor.zeros((1, 1), dtypes.int64), max_new_tokens=500).numpy()[0]))