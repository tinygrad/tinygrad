import json, math, time
from pathlib import Path
import numpy as np

from tinygrad.helpers import getenv
from tinygrad.features.jit import TinyJit
from tinygrad.ops import GlobalCounters
from tinygrad.nn import Linear, optim
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor, dtypes
from extra.lr_scheduler import OneCycleLR
from extra.models.bert import Bert
from extra.datasets.wikipedia import iterate

if getenv('HALF', 0):
  Tensor.default_type = dtypes.float16
  np_dtype = np.float16
else:
  Tensor.default_type = dtypes.float32
  np_dtype = np.float32

class BertMLperf:
  def __init__(self, hidden_size, intermediate_size, max_position_embeddings, num_attention_heads, num_hidden_layers, type_vocab_size, vocab_size, attention_probs_dropout_prob, hidden_dropout_prob) -> None:
    self.model = Bert(
      hidden_size,
      intermediate_size, 
      max_position_embeddings, 
      num_attention_heads, 
      num_hidden_layers, 
      type_vocab_size, 
      vocab_size, 
      attention_probs_dropout_prob, 
      hidden_dropout_prob
    )
    # for clsf:
    self.fc = Linear(hidden_size, hidden_size)
    self.activation1 = Tensor.tanh
    self.classifier = Linear(hidden_size, 2)

    # for lm:
    self.linear = Linear(hidden_size, hidden_size)
    self.activation2 = Tensor.gelu
    self.norm = Tensor.layernorm

    self.decoder = Linear(hidden_size, vocab_size, bias=False)
    self.decoder.weight = self.model.embeddings.word_embeddings.weight
    self.decoder_bias = Tensor.zeros(vocab_size)
  
  def __call__(self, input_ids:Tensor, segment_ids:Tensor, attention_mask:Tensor, masked_positions:Tensor):
    output = self.model(input_ids, attention_mask, segment_ids)
    clsf_logits = self.classifier(self.activation1(self.fc(output[:, 0])))

    masked_positions = masked_positions[:, :, None].expand(-1, -1, output.shape[-1])
    h_masked = Tensor.gather(output, masked_positions, 1)
    h_masked = self.norm(self.activation2(self.linear(h_masked)))
    lm_logits = self.decoder(h_masked) + self.decoder_bias

    return lm_logits, clsf_logits

BS, EVAL_BS, STEPS, MAX_EVAL_STEPS, WARMUP_STEPS, EPOCH, MAX_LR  = getenv("BS", 32), getenv('EVAL_BS', 8), getenv("STEPS", 100000), getenv("MAX_EVAL_STEPS", 100), getenv("WARMUP_STEPS", 10000), getenv("EPOCHS", 30), getenv('MAX_LR', 2.0)
EVAL_FREQ = math.floor(min(0.05*(230.23 * BS + 3000000), 25000))

def get_model(config_path:str):
  with open(config_path, 'r') as f:
    config = json.load(f)
  return BertMLperf(
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

# ************ Actual training ************

def pretrain():
  model = get_model(Path(__file__).parent.parents[2] / "extra" / "datasets" / "wiki" / "bert_config.json")
  optimizer = optim.LAMB(get_parameters(model), 1 / WARMUP_STEPS, eps=1e-6, wd=0.01, adam=True) # TODO: Keep in FP32?, Exclude LayerNorm, and bias from weight decay
  lr_scheduler = OneCycleLR(optimizer, MAX_LR, MAX_LR * WARMUP_STEPS, MAX_LR * 1e12, STEPS, WARMUP_STEPS / STEPS)

  @TinyJit
  def train_step_jitted(input_ids:Tensor, segment_ids:Tensor, attention_mask:Tensor, masked_positions:Tensor, masked_lm_ids:Tensor, next_sentence_labels:Tensor):
    lm_logits, clsf_logits = model(input_ids, segment_ids, attention_mask, masked_positions)
    lm_loss = lm_logits.sparse_categorical_crossentropy(masked_lm_ids)
    clsf_loss = clsf_logits.binary_crossentropy_logits(next_sentence_labels)
    loss = lm_loss + clsf_loss

    if not getenv('DISABLE_BACKWARD', 0):
      optimizer.zero_grad()
      loss.backward()

      optimizer.step()
      lr_scheduler.step()
    return loss.realize()
  
  @TinyJit
  def eval_step_jitted(input_ids:Tensor, segment_ids:Tensor, attention_mask:Tensor, masked_positions:Tensor, masked_lm_ids:Tensor):
    lm_logits, _ = model(input_ids, segment_ids, attention_mask, masked_positions)
    predictions = lm_logits.log_softmax().argmax(-1)
    return (predictions == masked_lm_ids).float().mean()
  
  train_batcher = iterate(bs=BS, val=False)
  eval_batcher = iterate(bs=EVAL_BS, val=True)

  epoch = 0
  while epoch < EPOCH:
    step = 0
    while step < STEPS:
      if step % 10 == 0 and step > 0:
        X, Y = next(eval_batcher)
        Tensor.train = False
        acc = eval_step_jitted(Tensor(X["input_ids"]).realize(), Tensor(X["segment_ids"]).realize(), Tensor(X["input_mask"]).realize(), Tensor(X["masked_lm_positions"]).realize(), Tensor(Y["masked_lm_ids"]).realize())
        Tensor.train = True
        print(f"{step:3d} {acc.numpy()*100:.2f}% MLM Acc")
      st = time.monotonic()
      X, Y = next(train_batcher) 
      
      GlobalCounters.reset()

      loss = train_step_jitted(Tensor(X["input_ids"]).realize(), Tensor(X["segment_ids"]).realize(), Tensor(X["input_mask"]).realize(), Tensor(X["masked_lm_positions"]).realize(), Tensor(Y["masked_lm_ids"]).realize(), Tensor(Y["next_sentence_labels"]).realize())
      et = time.monotonic()
      loss_cpu = loss.numpy()
      cl = time.monotonic()

      print(f"{step:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {optimizer.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
      st = cl
      step += 1
    epoch += 1

if __name__ == "__main__":
  with Tensor.train(): pretrain()
