import json, math, time
from pathlib import Path
import numpy as np

from tinygrad.helpers import getenv
from tinygrad.device import Device
from tinygrad.features.jit import TinyJit
from tinygrad.ops import GlobalCounters
from tinygrad.nn import Linear, optim
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save
from tinygrad.tensor import Tensor, dtypes
from extra.lr_scheduler import OneCycleLR
from extra.models.bert import Bert
from extra.datasets.wikipedia import iterate

BS, EVAL_BS, STEPS, MAX_EVAL_STEPS, WARMUP_STEPS, EPOCH, MAX_LR = getenv("BS", 32), getenv('EVAL_BS', 8), getenv("STEPS", 100000), getenv("MAX_EVAL_STEPS", 100), getenv("WARMUP_STEPS", 10000), getenv("EPOCHS", 30), getenv('MAX_LR', 2.0)
EVAL_STEP_FREQ = int(math.floor(0.05 * (230.23 * BS + 3000000) / 25000) * 25000 / BS)
GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv('GPUS', 1))]
for x in GPUS: Device[x]

if getenv('WANDB', 0): 
  import wandb
  wandb.init(project="MLPerf-BERT", config={
    "max_lr": MAX_LR,
    "batch_size": BS,
    "steps": STEPS,
    "max_eval_steps": MAX_EVAL_STEPS,
    "warmup_steps": WARMUP_STEPS,
    "epochs": EPOCH,
    "eval_freq": EVAL_STEP_FREQ
})

if getenv('HALF', 0):
  dtypes.default_float = dtypes.float16
  np_dtype = np.float16
else:
  dtypes.default_float = dtypes.float32
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

def save_model(model: BertMLperf, path:str = "/tmp/model"): safe_save(get_state_dict(model), path)
def load_model(model:BertMLperf, path:str = "/tmp/model"): get_state_dict(model, safe_load(path))

# ************ Actual training ************
def pretrain():
  model = get_model(Path(__file__).parent.parents[2] / "extra" / "datasets" / "wiki" / "bert_config.json")

  if len(GPUS) > 1:
    for x in get_parameters(model):
      x.to_(GPUS)

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

  epoch = 1
  wallclock_start = time.monotonic()
  accuracy_achieved = False
  while epoch <= EPOCH:
    step = 0
    while step < STEPS:
      if step % EVAL_STEP_FREQ == 0 and step > 0 and not getenv('DISABLE_EVAL', 0):
        train_step_jitted.reset()
        Tensor.train = False
        accu = 0.0
        for _ in range(MAX_EVAL_STEPS):
          X, Y = next(eval_batcher)
          accu += eval_step_jitted(Tensor(X["input_ids"]), Tensor(X["segment_ids"]), Tensor(X["input_mask"]), Tensor(X["masked_lm_positions"]), Tensor(Y["masked_lm_ids"])).numpy()
        Tensor.train = True
        print(f"{step:3d} {(acc := (accu/MAX_EVAL_STEPS))*100:.2f}% MLM Acc")
        wandb.log({"MLM Accuracy": acc*100}) if getenv('WANDB', 0) else None
        if acc >= 0.72:
          wallclock_end = time.monotonic()
          hours, minutes = divmod((wallclock_end - wallclock_start) / 3600, 1)
          print(f"MLM accuracy achieved in {int(hours)} hours and {int(minutes * 60)} minutes.")
          save_model(model, getenv('SAVE_PATH', getenv("SAVE_PATH", "/raid/tmp/")) + "final.safetensors")
          accuracy_achieved = True
          break
        eval_step_jitted.reset()
      
      if accuracy_achieved: break

      st = time.monotonic()
      X, Y = next(train_batcher) 
      input_ids, segment_ids, input_mask, masked_lm_positions, masked_lm_ids, next_sentence_labels = Tensor(X["input_ids"], dtype=dtypes.default_float), Tensor(X["segment_ids"], dtype=dtypes.default_float), Tensor(X["input_mask"], dtype=dtypes.default_float), Tensor(X["masked_lm_positions"], dtype=dtypes.default_float), Tensor(Y["masked_lm_ids"], dtype=dtypes.default_float), Tensor(Y["next_sentence_labels"], dtype=dtypes.default_float)
      if len(GPUS) > 1:
        input_ids.shard_(GPUS, axis=0)
        segment_ids.shard_(GPUS, axis=0)
        input_mask.shard_(GPUS, axis=0)
        masked_lm_positions.shard_(GPUS, axis=0)
        masked_lm_ids.shard_(GPUS, axis=0)
        next_sentence_labels.shard_(GPUS, axis=0)

      mem = time.monotonic()
      GlobalCounters.reset()

      loss = train_step_jitted(input_ids, segment_ids, input_mask, masked_lm_positions, masked_lm_ids, next_sentence_labels)
      et = time.monotonic()
      loss_cpu = loss.numpy()
      cl = time.monotonic()
      device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"

      print(f"{step:3d} {(cl-st)*1000.0:7.2f} ms run, {(mem-st)*1000.0:7.2f} ms fetch, {(et-mem)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms {device_str}, {loss_cpu:7.2f} loss, {(lr := optimizer.lr.numpy()[0]):.6f} LR, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
      wandb.log({"Loss": loss_cpu, "LR": lr}) if getenv('WANDB', 0) else None
      st = cl
      step += 1
    epoch += 1

if __name__ == "__main__":
  with Tensor.train(): pretrain()
