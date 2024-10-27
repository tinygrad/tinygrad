import os
os.environ["TRACEMETA"] = "0"
from tinygrad import Tensor, nn, Device, TinyJit
from tinygrad.helpers import prod, trange, size_unit
import math
from extra.models.llama import Transformer
from extra.fsdp.utils import get_size
from examples.llama3 import Tokenizer
Tensor.manual_seed(2)

SHARD = int(os.environ.get("SHARD", 1))
GPUS = [f"{Device.DEFAULT}:{i}" for i in range(SHARD)]
GPU_NAME = Device.DEFAULT
if len(GPUS) > 1:
  Device.DEFAULT = "CLANG"
B = 4
T = 16
vocab_size = 128256
dim = 64
n_layers = 1
n_heads = 32
n_kv_heads = 8
max_context = 8192
rope_theta=50000
hidden_dim = 32
epoch = 2
lr = 1e-4
weight_decay=0
generate_tokens = 5
assert dim % n_heads == 0 and dim % SHARD == 0
head_dim = dim // n_heads
shard_dim = dim // SHARD
norm_eps = 1e-5
TEMPERATURE = 0.95
TOP_K = 0
TOP_P = 0.0
ALPHA_F = 0.0
ALPHA_P = 0.0



def shard_model(model, opt):
  seen = set()
  for k, p in nn.state.get_state_dict(model).items():
    if p in seen: continue
    seen.add(p)
    axis = 0
    if k == 'tok_embeddings.weight':
      axis = 1
    elif k == 'freqs_cis':
      axis = None
    elif p.shape[0] == 1:
      axis = None
    p.shard_(GPUS, axis)
  for k, p in nn.state.get_state_dict(opt).items():
    if p in seen: continue
    seen.add(p)
    axis = None if prod(p.shape) <= 1 else 0
    if p.shape[0] == 1:
      axis = None
    p.shard_(GPUS, axis)

tokenizer = Tokenizer("tmp/tokenizer.model")
def tokenize_data():
  with open("tmp/tiny_shakespeare.txt") as f:
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
  jit=False
)
opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=lr, weight_decay=weight_decay)
if len(GPUS) > 1:
  shard_model(model, opt)
model_size, model_size_unit = get_size(nn.state.get_parameters(model))
opt_size, opt_size_unit = get_size(nn.state.get_parameters(opt))
print(f"Model {model_size:.4f} {model_size_unit} Opt: {opt_size:.4f} {opt_size_unit}")

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

@Tensor.train()
def train():
  Device.DEFAULT = GPU_NAME
  test_loss = float('nan')
  for i in (t:= trange(epoch)):
    x, y = next(train_data)
    loss = train_step(x.contiguous(), y.contiguous())
    if i % 10 == 9: test_loss = test_step().item()
    t.set_description(f"loss: {loss.item():6.2f} test_loss: {test_loss:5.2f}")

def generate():
  opt.zero_grad()
  for p in nn.state.get_parameters(model):
    p.requires_grad = False
  tokens = Tensor([tokenizer.encode("<|begin_of_text|>", allow_special=True)])
  if len(GPUS) > 1:
    tokens.shard_(GPUS, axis=None)
  for start_pos in range(generate_tokens):
    idx_next = model(tokens, start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).unsqueeze(0)
    if len(GPUS) > 1:
      idx_next.shard_(GPUS, axis=None)
    tokens = tokens.cat(idx_next, dim=1)
  return tokenizer.decode(tokens.tolist()[0])

train()

mem_usage = []
for device in GPUS:
  device = Device[device].allocator
  highest, unit = size_unit(device.mem_high)
  mem_usage.append(f"{device.name}: {highest:.2f} {unit}")
