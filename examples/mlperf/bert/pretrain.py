import json, math, time, os
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
  model = Bert(
    config["hidden_size"],
    config["intermediate_size"], 
    config["max_position_embeddings"], 
    config["num_attention_heads"], 
    config["num_hidden_layers"], 
    config["type_vocab_size"], 
    config["vocab_size"], 
    config["attention_probs_dropout_prob"], 
    config["hidden_dropout_prob"]
  )
  os.environ['VOCAB_SIZE'] = str(config["vocab_size"])

  embedding_table = model.embeddings.word_embeddings.weight
  s_weights = Tensor.uniform(*(2, config["hidden_size"]), low=-0.1, high=0.1) #TODO: change init range
  s_bias = Tensor.zeros(2)
  m_weights = Tensor.uniform(*(config["hidden_size"], config["hidden_size"]), low=-0.1, high=0.1) #TODO: change init range
  m_bias = Tensor.zeros((config["vocab_size"],))
  p_weights = Tensor.uniform(*(config["hidden_size"], config["hidden_size"]), low=-0.1, high=0.1) #TODO: change init range
  return model, embedding_table, s_weights, s_bias, m_weights, m_bias, p_weights 

def one_hot(arr:Tensor, num_classes=3):
  res = Tensor.eye(num_classes)[arr.reshape(-1)]
  return res.reshape(list(arr.shape) + [num_classes])

def pool_output(output:Tensor, weights:Tensor):
  pooled_output = output[:, 0]
  return Tensor.tanh(pooled_output.linear(weights))

def gather_indexes(sequence_tensor:Tensor, positions:Tensor):
  assert len(sequence_tensor.shape) == 3, f"Expected tensor to have rank 3, but got {len(sequence_tensor.shape)}"
  sequence_shape = list(sequence_tensor.shape)
  batch_size, seq_length, width = sequence_shape[0], sequence_shape[1], sequence_shape[2]

  flat_offsets = Tensor.arange(0, batch_size, requires_grad=False).reshape([1, -1]) * seq_length
  flat_positions = (positions + flat_offsets.reshape(-1, 1)).reshape([-1])
  flat_sequence_tensor = sequence_tensor.reshape([batch_size * seq_length, width])
  return flat_sequence_tensor[flat_positions]

def get_masked_lm_output(input_tensor:Tensor, output_weights:Tensor, transform_weights:Tensor, transform_bias:Tensor, positions:Tensor, label_ids:Tensor): 
  input_tensor = gather_indexes(input_tensor, positions)
  input_tensor = Tensor.gelu(input_tensor.matmul(transform_weights))
  input_tensor = Tensor.layernorm(input_tensor)
  logits = input_tensor.matmul(output_weights.transpose()).add(transform_bias)
  return logits.sparse_categorical_crossentropy(label_ids.flatten())

def get_next_sentence_output(input_tensor:Tensor, labels: Tensor, weights:Tensor, bias:Tensor):
  logits = input_tensor.matmul(weights.transpose()).add(bias)
  print('logits: ', logits.softmax().numpy())
  print('labels: ', labels.numpy())
  return logits.softmax().binary_crossentropy(labels)

def pretrain():
  model, embedding_table, s_weights, s_bias, m_weights, m_bias, p_weights = get_model_and_config(Path(__file__).parent.parents[2] / "extra" / "datasets" / "wiki" / "bert_config.json")
  optimizer = optim.LAMB(get_parameters(model), 1 / WARMUP_STEPS, eps=1e-6, wd=0.01, adam=True) # TODO: Exclude LayerNorm, and bias from weight decay
  lr_scheduler = OneCycleLR(optimizer, MAX_LR, MAX_LR * WARMUP_STEPS, MAX_LR * 1e12, STEPS, WARMUP_STEPS / STEPS)

  @TinyJit
  def train_step_jitted(model, embedding_table, optimizer, lr_scheduler, input_ids, input_mask, segment_ids, masked_lm_ids, masked_lm_positions, next_sentence_labels):
    output = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
    pooled_output = pool_output(output, p_weights)

    masked_lm_loss = get_masked_lm_output(output, embedding_table, m_weights, m_bias, masked_lm_positions, masked_lm_ids)
    next_sentence_loss = get_next_sentence_output(pooled_output, next_sentence_labels, s_weights, s_bias)
    print('next_sentence_loss: ', next_sentence_loss.numpy())
    loss = masked_lm_loss + next_sentence_loss

    if not getenv('DISABLE_BACKWARD', 0):
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      lr_scheduler.step()
    return loss.realize()
  
  for _ in range(EPOCH):
    i = 0
    while i <= STEPS:
      st = time.monotonic()
      for X, _ in iterate(bs=BS, val=False):
        GlobalCounters.reset()
        input_ids, input_mask, segment_ids, masked_lm_ids, masked_lm_positions, next_sentence_labels = Tensor(X["input_ids"]), Tensor(X["input_mask"]), Tensor(X["segment_ids"]), Tensor(X["masked_lm_ids"]), Tensor(X["masked_lm_positions"]), Tensor(X["next_sentence_labels"])
        loss = train_step_jitted(model, embedding_table, optimizer, lr_scheduler, input_ids, input_mask, segment_ids, masked_lm_ids, masked_lm_positions, next_sentence_labels)
        et = time.monotonic()

        loss_cpu = loss.numpy()
        cl = time.monotonic()

        print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {optimizer.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
        st = cl
        i += 1