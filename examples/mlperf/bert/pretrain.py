import json, math
import numpy as np
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor, dtypes
from extra.models.bert import Bert
from examples.mlperf.metrics import one_hot
from extra.dist import dist

if __name__ == "__main__":
  if getenv("DIST"):
    dist.preinit()

if getenv('HALF', 0):
  Tensor.default_type = dtypes.float16
  np_dtype = np.float16
else:
  Tensor.default_type = dtypes.float32
  np_dtype = np.float32

BS, EVAL_BS, STEPS, WARMUP_STEPS, EPOCH,  = getenv("BS", 32), getenv('EVAL_BS', 8), getenv("STEPS", 100000), getenv("WARMUP_STEPS", 10000), getenv("EPOCHS", 30)
EVAL_FREQ = math.floor(0.05*(230.23 * BS + 3000000), 25000)

def get_model_config(path:str):
  with open(path, 'r') as f:
    config = json.load(f)
  return config

# TODO: add output weights back to dataloader
def get_masked_lm_output(bert_config, input_tensor, output_weights:Tensor, positions, label_ids:Tensor, label_weights): # untested 
  input_tensor = gather_indexes(input_tensor, positions)
  input_tensor = Tensor.gelu(input_tensor.linear(Tensor.empty(*(input_tensor.shape[-1], 1024)))) # TODO get_activation, hidden_size and init weight range from config file
  input_tensor = Tensor.layernorm(input_tensor)

  output_bias = Tensor.zeros((bert_config.vocab_size,))
  logits = input_tensor.matmul(output_weights.transpose()).add(output_bias)
  log_probs = logits.softmax()

  label_ids, label_weights =  label_ids.reshape((-1)), label_weights.reshape((-1))
  one_hot_labels = one_hot(label_ids, num_classes=bert_config.vocab_size)
  per_example_loss = -(log_probs * one_hot_labels).sum(axis=-1)
  numerator = (label_weights * per_example_loss).sum()
  denominator = label_weights.sum() + 1e-5
  loss = numerator / denominator
  return (loss, per_example_loss, log_probs)

def gather_indexes(sequence_tensor:Tensor, positions:Tensor):
  assert len(sequence_tensor.shape) == 3, "Expected tensor to have rank %d, but got %d" % (3, len(sequence_tensor.shape))
  sequence_shape = list(sequence_tensor.shape)
  batch_size, seq_length, width = sequence_shape[0], sequence_shape[1], sequence_shape[2]

  flat_offsets = Tensor.arange(0, batch_size, dtypes=Tensor.default_type).reshape([1, -1]) * seq_length
  flat_positions = (positions + flat_offsets).reshape([-1])
  flat_sequence_tensor = sequence_tensor.reshape([batch_size * seq_length, width])
  return flat_sequence_tensor.gather(flat_positions, dim=0)

def pretrain():
  ...