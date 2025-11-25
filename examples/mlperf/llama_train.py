import math
from pathlib import Path
from tinygrad import Device, nn, Tensor, TinyJit
from tinygrad.helpers import getenv, profile_marker
from extra.models.llama import Transformer
from examples.llama3 import MODEL_PARAMS
from examples.mlperf.lr_schedulers import CosineAnnealingLRWithWarmup

config = {}
BASEDIR            = config["BASEDIR"]                = Path(getenv("BASEDIR", "/raid/datasets/c4/"))
BS                 = config["BS"]                     = getenv("BS", 16)
grad_acc           = config["GRADIENT_ACC_STEPS"]     = getenv("GRADIENT_ACC_STEPS", 1)
GBS                = config["GLOBAL_BATCH_SIZE"]      = BS * grad_acc
SEED               = config["SEED"]                   = getenv("SEED", 5760)
SEQLEN             = config["SEQLEN"]                 = getenv("SEQLEN", 8192)
TRAIN_ON_VAL       = config["TRAIN_ON_VAL"]           = getenv("TRAIN_ON_VAL", 0)
SMALL              = config["SMALL"]                  = getenv("SMALL", 0)
SAMPLES            = config["SAMPLES"]                = getenv("SAMPLES", 5_760 if TRAIN_ON_VAL else 1_200_000 * 1152)
EVAL_FREQ          = config["EVAL_FREQ"]              = getenv("EVAL_FREQ", 46080)
EVAL_BS            = config["EVAL_BS"]                = getenv("EVAL_BS", 16)
EVAL_TARGET        = config["EVAL_TARGET"]            = getenv("EVAL_TARGET", 5.6)

opt_adamw_beta_1 = 0.9
opt_adamw_beta_2 = 0.95
opt_adamw_epsilon = 1e-5
opt_adamw_weight_decay = 0.1

opt_gradient_clip_norm = 1.0
opt_learning_rate_warmup_steps = getenv("WARMUP_STEPS", math.ceil(8000 * 1152 / GBS))
opt_learning_rate_decay_steps = getenv("MAX_STEPS", math.ceil(1_200_000 * 1152 / GBS)) - opt_learning_rate_warmup_steps
opt_base_learning_rate = getenv("LR", 8e-5 * GBS / 1152)  # NOTE: cannot change for benchmark
opt_end_learning_rate = getenv("END_LR", 8e-7)

# TODO: confirm weights are in bf16
# vocab_size from the mixtral tokenizer
params = MODEL_PARAMS[getenv("LLAMA3_SIZE", "8B")]["args"]
params = params | {"vocab_size": 32000} if not SMALL else params
if (llama_layers:=getenv("LLAMA_LAYERS")) != 0: params['n_layers'] = llama_layers

if __name__ == "__main__":
  profile_marker("create model")

  model = Transformer(**params, max_context=SEQLEN, jit=False, disable_kv_cache=True)

  # shard the model, either data parallel (DP) or model parallel (MP)

  if (DP := getenv("DP", 1)) > 1:
    device = tuple(f"{Device.DEFAULT}:{i}" for i in range(DP))
    for v in nn.state.get_parameters(model):
      v.shard_(device, axis=None)

  if (MP := getenv("MP", 1)) > 1:
    device = tuple(f"{Device.DEFAULT}:{i}" for i in range(MP))
    for k,v in nn.state.get_state_dict(model).items():
      if 'scale' in k: v.shard_(device, axis=None)  # from quantized
      elif '.attention.wq' in k: v.shard_(device, axis=0)
      elif '.attention.wk' in k: v.shard_(device, axis=0)
      elif '.attention.wv' in k: v.shard_(device, axis=0)
      elif '.attention.wo' in k: v.shard_(device, axis=1)
      elif '.feed_forward.w1.' in k: v.shard_(device, axis=0)
      elif '.feed_forward.w2.' in k: v.shard_(device, axis=1)
      elif '.feed_forward.w3.' in k: v.shard_(device, axis=0)
      elif 'tok_embeddings.weight' in k: v.shard_(device, axis=0)
      elif 'output.weight' in k: v.shard_(device, axis=0)
      else:
        # attention_norm, ffn_norm, norm
        v.shard_(device, axis=None)
      # prevents memory spike on device 0
      v.realize()

  profile_marker("create optim")
  optim = nn.optim.AdamW(nn.state.get_parameters(model), lr=0.0,
                         b1=opt_adamw_beta_1, b2=opt_adamw_beta_2, eps=opt_adamw_epsilon, weight_decay=opt_adamw_weight_decay, fused=True)
  scheduler = CosineAnnealingLRWithWarmup(optim, opt_base_learning_rate, opt_end_learning_rate,
                                          opt_learning_rate_warmup_steps, opt_learning_rate_decay_steps)

  profile_marker("init params")
  optim.lr.realize(*[p.replace(p.contiguous()) for p in optim.params])

  # TODO: make this work with multigpu
  cat_params = Tensor.cat(*[t.flatten() for t in optim.params], dim=0)
  cat_grads = Tensor.zeros_like(cat_params)

  @profile_marker("microbatch")
  @TinyJit
  @Tensor.train()
  def microbatch(batch:Tensor):
    logits:Tensor = model(batch[:, :-1], start_pos=0, temperature=math.nan)
    loss = logits.sparse_categorical_crossentropy(batch[:, 1:]).backward()
    return loss.realize(cat_grads)



