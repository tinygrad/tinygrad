import json, math, time
from pathlib import Path
import numpy as np
from tinygrad.helpers import GlobalCounters, getenv
from tinygrad.jit import TinyJit
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor, dtypes
from extra.lr_scheduler import OneCycleLR
from extra.models.bert import Bert
from extra.datasets.wikipedia import iterate
from extra import dist

if __name__ == "__main__":
  if getenv("DIST"):
    dist.preinit()

if getenv('HALF', 0):
  Tensor.default_type = dtypes.float16
  np_dtype = np.float16
else:
  Tensor.default_type = dtypes.float32
  np_dtype = np.float32

BS, EVAL_BS, STEPS, WARMUP_STEPS, EPOCH, MAX_LR  = getenv("BS", 32), getenv('EVAL_BS', 8), getenv("STEPS", 100000), getenv("WARMUP_STEPS", 10000), getenv("EPOCHS", 30), getenv('MAX_LR', 2.0)
EVAL_FREQ = math.floor(min(0.05*(230.23 * BS + 3000000), 25000))

def get_model_and_config(path:str):
  with open(path, 'r') as f:
    config = json.load(f)
  model = Bert(config["hidden_size"],
               config["intermediate_size"], 
               config["max_position_embeddings"], 
               config["num_attention_heads"], 
               config["num_hidden_layers"], 
               config["type_vocab_size"], 
               config["vocab_size"], 
               config["attention_probs_dropout_prob"], 
               config["hidden_dropout_prob"]
  )
  return model, config

def one_hot(arr: Tensor, num_classes=3): #TODO: REMOVE GRAPH BREAK
  res = np.eye(num_classes)[np.array(arr.numpy(), dtype=np.int32).reshape(-1)]
  arr = res.reshape(list(arr.shape) + [num_classes])
  return Tensor(arr.astype(np_dtype))

def pool_output(config:dict, output:Tensor):
  pooled_output = output[:, 0]
  return Tensor.tanh(pooled_output.linear(Tensor.empty(*(config["hidden_size"], config["hidden_size"])))) # Weight init?

def gather_indexes(sequence_tensor:Tensor, positions:Tensor):
  assert len(sequence_tensor.shape) == 3, f"Expected tensor to have rank 3, but got {len(sequence_tensor.shape)}"
  sequence_shape = list(sequence_tensor.shape)
  batch_size, seq_length, width = sequence_shape[0], sequence_shape[1], sequence_shape[2]

  flat_offsets = Tensor.arange(0, batch_size).reshape([1, -1]) * seq_length
  flat_positions = (positions + flat_offsets.reshape(-1, 1)).reshape([-1])
  flat_sequence_tensor = sequence_tensor.reshape([batch_size * seq_length, width])
  return flat_sequence_tensor[flat_positions]

def get_masked_lm_output(bert_config:dict, input_tensor:Tensor, output_weights:Tensor, positions:Tensor, label_ids: np.ndarray, label_weights:Tensor): 
  input_tensor = gather_indexes(input_tensor, positions)
  input_tensor = Tensor.gelu(input_tensor.linear(Tensor.empty(*(1024, input_tensor.shape[1])))) # TODO get_activation, hidden_size and init weight range from config file
  input_tensor = Tensor.layernorm(input_tensor)
  output_bias = Tensor.zeros((bert_config["vocab_size"],))
  logits = input_tensor.matmul(output_weights.transpose()).add(output_bias)
  log_probs = logits.log_softmax()

  label_weights = label_weights.reshape((-1))
  one_hot_labels = one_hot(label_ids.reshape(-1), num_classes=bert_config["vocab_size"])
  per_example_loss = -1 * (log_probs * one_hot_labels).sum(axis=-1)
  numerator = (label_weights * per_example_loss).sum()
  denominator = label_weights.sum() + 1e-5
  loss = numerator / denominator
  return loss, per_example_loss, log_probs

def get_next_sentence_output(bert_config:dict, input_tensor:Tensor, labels: Tensor):
  logits = input_tensor.matmul(Tensor.empty(*(2, bert_config["hidden_size"])).transpose()).add(Tensor.zeros(2)) # Weight init?
  log_probs = logits.log_softmax(axis=-1)
  labels = labels.reshape([-1])
  one_hot_labels = one_hot(labels, num_classes=2)
  per_example_loss = -1 * ((log_probs * one_hot_labels).sum(axis=-1))
  loss = per_example_loss.mean()
  return loss, per_example_loss, log_probs

@TinyJit
def train_step_jitted(model, config:dict, embedding_table, optimizer, lr_scheduler, input_ids, input_mask, masked_lm_ids, masked_lm_weights, segment_ids, masked_lm_positions, next_sentence_labels):
  output = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids) # check input_mask == attention_mask?
  pooled_output = pool_output(config, output)

  masked_lm_loss, _, _  = get_masked_lm_output(config, output, embedding_table, masked_lm_positions, masked_lm_ids, masked_lm_weights)
  next_sentence_loss, _, _ = get_next_sentence_output(config, pooled_output, next_sentence_labels)
  loss = masked_lm_loss + next_sentence_loss

  if not getenv('DISABLE_BACKWARD', 0):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
  return loss.realize()

def pretrain():
  model, config = get_model_and_config(Path(__file__).parent.parents[2] / "extra" / "datasets" / "wiki" / "bert_config.json")
  embedding_table = model.embeddings.word_embeddings.weight
  optimizer = optim.LAMB(get_parameters(model), 1 / WARMUP_STEPS, eps=1e-6, wd=0.01, adam=True) # TODO: Exclude LayerNorm, and bias from weight decay
  lr_scheduler = OneCycleLR(optimizer, getenv('INIT_LR', 2.0), MAX_LR * WARMUP_STEPS, MAX_LR * 1e12, STEPS, WARMUP_STEPS / STEPS)

  for _ in range(EPOCH):
    i = 0
    while i <= STEPS:
      st = time.monotonic()
      for X, Y in iterate(bs=BS, val=False):
        GlobalCounters.reset()
        loss = train_step_jitted(model, config, embedding_table, optimizer, lr_scheduler, Tensor(X["input_ids"]), Tensor(X["input_mask"]), Tensor(X["masked_lm_ids"]), Tensor(X["masked_lm_weights"]), Tensor(X["segment_ids"]), Tensor(X["masked_lm_positions"]), Tensor(X["next_sentence_labels"]))
        et = time.monotonic()

        loss_cpu = loss.numpy()
        cl = time.monotonic()

        print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {optimizer.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
        st = cl
        i += 1