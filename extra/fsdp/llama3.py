# CUDA=1 LRU=0 SHARD=4 RING=2 PYTHONPATH='.' python extra/fsdp/llama3.py
# CUDA=1 LRU=0 SHARD=4 RING=2 PYTHONPATH='.' setsid python -u extra/fsdp/llama3.py > tmp/llama_fsdp_log.txt 2>&1 &
import argparse
import time
import pathlib
import os
from tinygrad import Tensor, nn, Device, TinyJit
import tinygrad.nn as nn
from tinygrad.helpers import prod, trange, fetch
import math
from extra.models.llama import Transformer
from examples.llama3 import Tokenizer
from typing import List, Callable, Union
from tinygrad.multi import MultiLazyBuffer
Tensor.manual_seed(2)

def get_size(tensors: List[Tensor],
             getter: Callable[[Tensor], int]=lambda t: t.nbytes(),
             units: List[str]=["bytes", "KB", "MB", "GB"],
             divisor: int=1000):
  size = sum([getter(t) if isinstance(t, Tensor) else t.size for t in tensors])
  size, unit = size_unit(size, units, divisor)
  return size, unit

def size_unit(size: Union[float, int],
              units: List[str]=["bytes", "KB", "MB", "GB"],
              divisor: int=1000):
  for i, unit in enumerate(units):
    if size < divisor or i == len(units) - 1: break
    size /= divisor
  return size, unit

SHARD = int(os.environ.get("SHARD", 1))
GPUS = [f"{Device.DEFAULT}:{i+1}" for i in range(SHARD)]

print(f"Allocating data on {Device.DEFAULT}, Training on {GPUS}")
B = 16
dim = 4096
n_layers = 16
n_heads = 32
n_kv_heads = 8
hidden_dim = 14336
epoch = 2000
eval_interval = 50
eval_interval_mod = eval_interval - 1

# small for testing
# dim = 64
# n_layers = 2
# n_heads = 4
# n_kv_heads = 4 
# hidden_dim = 64

T = 16
vocab_size = 128256
max_context = 8192
rope_theta=50000
lr = 1e-4
weight_decay=0
generate_tokens = 20
assert dim % n_heads == 0 and dim % SHARD == 0
head_dim = dim // n_heads
shard_dim = dim // SHARD
norm_eps = 1e-5
TEMPERATURE = 0.95
TOP_K = 0
TOP_P = 0.0
ALPHA_F = 0.0
ALPHA_P = 0.0

def shard_model(model):
  seen = set()
  for k, p in nn.state.get_state_dict(model).items():
    if isinstance(p.lazydata, MultiLazyBuffer):
      continue
    if p in seen: continue
    seen.add(p)
    axis = 0
    if k == 'tok_embeddings.weight':
      axis = 1
    elif k == 'freqs_cis':
      axis = None
    elif p.shape[0] == 1:
      axis = None
    elif prod(p.shape) <= 1:
      axis = None
    p.shard_(GPUS, axis)
  # for k, p in nn.state.get_state_dict(opt).items():
  #   if p in seen: continue
  #   seen.add(p)
  #   axis = None if prod(p.shape) <= 1 else 0
  #   if p.shape[0] == 1:
  #     axis = None
  #   p.shard_(GPUS, axis)

tokenizer = Tokenizer(fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="llama3-8b-sfr").as_posix())
def tokenize_data():
  with open(fetch("https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt", "tiny_shakespeare.txt")) as f:
    text = f.read()
    length = len(text)
    split = math.floor(length * 0.8)
    train = tokenizer.encode(text[:split])
    val = tokenizer.encode(text[split:])
    return Tensor(train), Tensor(val)

def get_batch(tokens, batch):
  i = 0
  while True:
    x = tokens[i:i+batch*T].view(batch, T)
    y = tokens[i+1:i+batch*T+1].view(batch, T)
    if len(GPUS) > 1:
      x.shard_(GPUS)
      y.shard_(GPUS)
    yield x, y
    i += batch*T
    if i + batch*T + 1 >= len(tokens):
      i = 0 # in prod we'd want to randomize the start point a bit

train, val = tokenize_data()
train_data = iter(get_batch(train, B))
x_test, y_test = next(iter(get_batch(val, 32)))

model = Transformer(
  dim=dim,
  hidden_dim=hidden_dim,
  n_heads=n_heads,
  n_layers=n_layers,
  norm_eps=norm_eps,
  vocab_size=vocab_size,
  n_kv_heads=n_kv_heads,
  max_context=max_context,
  jit=True
)

@Tensor.train()
def train():
  opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=lr, weight_decay=weight_decay)
  if len(GPUS) > 1:
    shard_model(model)
    shard_model(opt)
  model_size, model_size_unit = get_size(nn.state.get_parameters(model))
  opt_size, opt_size_unit = get_size(nn.state.get_parameters(opt))
  print(f"Model {model_size:.2f} {model_size_unit} Optimizer: {opt_size:.2f} {opt_size_unit}")
  model_elem, model_elem_unit = get_size(
    nn.state.get_parameters(model),
    lambda t: t.numel(),
    units=["", "K", "M", "B"]
  )
  optim_elem, optim_elem_unit = get_size(
    nn.state.get_parameters(opt),
    lambda t: t.numel(),
    units=["", "K", "M", "B"]
  )
  print(f"Model params: {model_elem:.2f} {model_elem_unit} Optimizer params: {optim_elem:.2f} {optim_elem_unit}")

  def forward_pass(x: Tensor, y: Tensor):
    logits = model(x, 0, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
    loss = logits.sparse_categorical_crossentropy(y)
    return loss

  @TinyJit
  def train_step(x: Tensor, y: Tensor):
    opt.zero_grad()
    loss = forward_pass(x, y)
    loss.backward()
    loss.realize(*opt.schedule_step())
    return loss

  @TinyJit
  @Tensor.test()
  def test_step() -> Tensor:
    loss = forward_pass(x_test, y_test)
    return loss

  test_loss = float('nan')
  start_time = time.time()
  for i in range(epoch):
    x, y = next(train_data)
    loss = train_step(x.contiguous(), y.contiguous())
    if i % eval_interval == eval_interval_mod:
      test_loss = test_step().item()
      time_elapsed, unit = size_unit(time.time() - start_time, ["s", "min", "hr"], 60)
      print(f"Epoch {i+1:4} loss: {loss.item():6.2f} test_loss: {test_loss:6.2f} elapsed time: {time_elapsed: 2.1f} {unit}")
  opt.zero_grad()

def generate():
  for p in nn.state.get_parameters(model):
    p.requires_grad = False
  tokens = Tensor([tokenizer.encode("<|begin_of_text|>", allow_special=True)])
  if len(GPUS) > 1:
    shard_model(model)
    tokens.shard_(GPUS, axis=None)
  for start_pos in (t:= trange(generate_tokens)):
    idx_next = model(tokens, start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).unsqueeze(0)
    if len(GPUS) > 1:
      idx_next.shard_(GPUS, axis=None)
    tokens = tokens.cat(idx_next, dim=1)
    t.set_description(f"Inferenced tokens {start_pos+1}")
  text = tokenizer.decode(tokens.tolist()[0])
  print("Inference: ", text)
  return text

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="""
Default mode is train the model, save the weights into WEIGHT_DIR and do inference with trained weights in memory
Optionally you can skip saving the weights, specify a different weight directory, or do inference only
with the weights on disk.
""")
  parser.add_argument("--weights_dir", type=str, default="weights")
  parser.add_argument("--weights_name", type=str, default="llama3_fsdp")
  parser.add_argument("--no_save_weights", action="store_true")
  parser.add_argument("--infer_only", action="store_true", help="Load the saved weights in WEIGHT_DIR and do inference only")
  args = parser.parse_args()
  weights_dir = args.weights_dir
  weights_name = args.weights_name
  weights_path = pathlib.Path(weights_dir).joinpath(f"{weights_name}.safetensors").as_posix()
  no_save_weights = args.no_save_weights
  infer_only = args.infer_only
  if not os.path.exists(weights_dir):
    print("creating directory to save weights:", weights_dir)
    os.mkdir(weights_dir)
  if infer_only:
    print("Loading weights from", weights_path)
    state_dict = nn.state.safe_load(weights_path)
    nn.state.load_state_dict(model, state_dict)
    generate()
  else:
    train()
    if not no_save_weights:
      nn.state.safe_save(nn.state.get_state_dict(model), weights_path)
      print(f"Saved weights to {weights_path}")
    generate()
  