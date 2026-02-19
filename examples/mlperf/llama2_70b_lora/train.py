#!/usr/bin/env python3
# Llama2 70B LoRA training for MLPerf
import time, json, re
from pathlib import Path
from collections import Counter
from tinygrad import Device, GlobalCounters, Tensor, TinyJit
from tinygrad.helpers import getenv, diskcache_clear, Context
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load, safe_save
from tinygrad.nn.optim import AdamW
from extra.models.llama import Transformer, convert_from_huggingface, fix_bf16
from examples.mlperf.helpers import get_training_state
from examples.mlperf.llama2_70b_lora.lora import apply_lora, get_lora_params
from examples.mlperf.llama2_70b_lora.dataset import load_data, batch_iter, get_tokenizer

try:
  import mlperf_logging.mllog as mllog_mod
  import mlperf_logging.mllog.constants as mlc
  MLPERF = True
except ImportError:
  mllog_mod, MLPERF = None, False

#ROUGE scoring (simple implementation for MLPerf - avoids external deps)
#ref: https://aclanthology.org/W04-1013.pdf
def tokenize_text(t): return re.findall(r'\b\w+\b', t.lower())
def get_ngrams(toks, n): return Counter(' '.join(toks[i:i+n]) for i in range(len(toks)-n+1))
def rouge_n(p_toks, r_toks, n):
  p_ng, r_ng = get_ngrams(p_toks, n), get_ngrams(r_toks, n)
  if not r_ng: return {"p": 0.0, "r": 0.0, "f": 0.0}
  overlap = sum((p_ng & r_ng).values())
  prec, rec = overlap/max(sum(p_ng.values()),1), overlap/sum(r_ng.values())
  return {"p": prec, "r": rec, "f": (2*prec*rec)/max(prec+rec, 1e-8)}
def rouge_l(p_toks, r_toks):
  def lcs_len(x, y):
    if len(x)==0 or len(y)==0: return 0
    if len(y) > len(x): x, y = y, x
    prev = [0]*(len(y)+1)
    for x_tok in x:
      curr = [0]
      for j,y_tok in enumerate(y, start=1):
        curr.append(prev[j-1]+1 if x_tok==y_tok else max(prev[j], curr[j-1]))
      prev = curr
    return prev[-1]
  if not r_toks: return {"p": 0.0, "r": 0.0, "f": 0.0}
  lcs = lcs_len(p_toks, r_toks)
  prec, rec = lcs/max(len(p_toks),1), lcs/len(r_toks)
  return {"p": prec, "r": rec, "f": (2*prec*rec)/max(prec+rec, 1e-8)}
def compute_rouge(preds, refs):
  r1, r2, rl = [], [], []
  for p,r in zip(preds, refs):
    pt, rt = tokenize_text(p), tokenize_text(r)
    r1.append(rouge_n(pt,rt,1)), r2.append(rouge_n(pt,rt,2)), rl.append(rouge_l(pt,rt))
  avg = lambda scores: {k: sum(s[k] for s in scores)/len(scores) for k in ["p","r","f"]} if scores else {k:0.0 for k in ["p","r","f"]}
  return {"rouge-1": avg(r1), "rouge-2": avg(r2), "rouge-l": avg(rl)}

@TinyJit
def train_step(inp, labels, model, opt):
  opt.zero_grad()
  logits = model.forward(inp, start_pos=0, temperature=float('nan'), top_k=0, top_p=0.0, alpha_f=0.0, alpha_p=0.0)
  sl, slabels = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
  sl_flat, slabels_flat = sl.reshape(-1, sl.shape[-1]), slabels.reshape(-1)
  valid = (slabels_flat != -100).cast(sl_flat.dtype)
  safe_labels = (slabels_flat * valid.cast(slabels_flat.dtype)).cast('int32')
  token_nll = -sl_flat.log_softmax(axis=-1).gather(1, safe_labels.unsqueeze(1)).squeeze(1)
  loss = (token_nll * valid).sum() / valid.sum().maximum(1)
  loss.backward()
  opt.step()
  Tensor.realize(loss)
  return loss.detach()

@Tensor.train(mode=False)
def evaluate(model, data, tok, bs, maxlen, max_eval):
  tot_loss, nb, preds, refs = 0.0, 0, [], []
  print(f"eval on {max_eval} batches...")
  for i,batch in enumerate(batch_iter(data, tok, bs, maxlen, shuffle=False)):
    if i >= max_eval: break
    with Tensor.no_grad():
      inp, labels = batch['input_ids'], batch['labels']
      logits = model.forward(inp, start_pos=0, temperature=float('nan'), top_k=0, top_p=0.0, alpha_f=0.0, alpha_p=0.0)
      sl, slabels = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
      sl_flat, slabels_flat = sl.reshape(-1, sl.shape[-1]), slabels.reshape(-1)
      if int((slabels_flat != -100).sum().item()) > 0:
        loss = sl_flat.sparse_categorical_crossentropy(slabels_flat, ignore_index=-100)
        tot_loss += loss.item()
        nb += 1
      pred_ids = logits.argmax(axis=-1).numpy()
      pred_text = tok.decode(pred_ids[0].tolist())
      ref_ids = labels[0][labels[0]!=-100].numpy().tolist()
      ref_text = tok.decode(ref_ids)
      preds.append(pred_text)
      refs.append(ref_text)
  return tot_loss/max(nb,1), compute_rouge(preds, refs)

def save_ckpt(model, opt, path):
  ckpt = get_training_state(model, opt, None)
  cpu_ckpt = {k: v.detach().to("CPU").realize().cast(v.dtype.base).contiguous() for k,v in ckpt.items()}
  Tensor.realize(*cpu_ckpt.values())
  safe_save(cpu_ckpt, path)
  print(f"saved to {path}")

def train():
  # config from env
  GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1)))
  SEED, BS, LR = getenv("SEED", 42), getenv("BS", len(GPUS)), getenv("LR", 1e-4)
  MAXLEN, TARGET = getenv("MAXLEN", 8192), getenv("TARGET_ROUGE", 0.270)
  EVAL_STEPS, CKPT_STEPS = getenv("EVAL_STEPS", 500), getenv("CKPT_STEPS", 500)
  MAX_EVAL, EPOCHS = getenv("MAX_EVAL", 100), getenv("EPOCHS", 3)
  DATADIR = Path(getenv("DATADIR", "./dataset/govreport"))
  MODELDIR = Path(getenv("MODELDIR", "./models/llama-2-70b"))
  CKPTDIR = Path(getenv("CKPTDIR", "./checkpoints"))
  LORA_R, LORA_ALPHA = getenv("LORA_R", 16), getenv("LORA_ALPHA", 32.0)
  MAX_STEPS = getenv("MAX_STEPS", 0)  # 0 = unlimited
  DEFAULT_CFG = (8192, 28672, 64, 8, 80)

  print(f"training on {GPUS}, bs={BS}, lr={LR}")
  for d in GPUS: Device[d]
  Tensor.manual_seed(SEED)
  Tensor.training = True

  #mlperf logging
  mllog = None
  if getenv("LOGMLPERF") and MLPERF:
    mllog_mod.config(filename=f"result_llama2_lora_{SEED}.txt")
    mllog_mod.config(root_dir=Path(__file__).parents[3].as_posix())
    mllog = mllog_mod.get_mllogger()
  if mllog and getenv("INITMLPERF"):
    mllog.event(key=mlc.SUBMISSION_ORG, value="tinycorp")
    mllog.event(key=mlc.SUBMISSION_PLATFORM, value=getenv("SUBMISSION_PLATFORM", "tinybox"))
    mllog.event(key=mlc.SUBMISSION_DIVISION, value=mlc.CLOSED)
    mllog.event(key=mlc.SUBMISSION_STATUS, value=mlc.ONPREM)
    mllog.event(key="submission_benchmark", value="llama2_70b_lora")
    diskcache_clear()
    mllog.event(key=mlc.CACHE_CLEAR, value=True)
    mllog.start(key=mlc.INIT_START)

  # load model (dims configurable via env for smoke testing)
  DIM, HIDDEN_DIM = getenv("DIM", 8192), getenv("HIDDEN_DIM", 28672)
  N_HEADS, N_KV_HEADS, N_LAYERS = getenv("N_HEADS", 64), getenv("N_KV_HEADS", 8), getenv("N_LAYERS", 80)
  cfg = (DIM, HIDDEN_DIM, N_HEADS, N_KV_HEADS, N_LAYERS)
  smoke_mode = MAX_STEPS > 0 or cfg != DEFAULT_CFG
  print(f"loading llama2 (dim={DIM}, layers={N_LAYERS})...")
  model = Transformer(dim=DIM, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, n_layers=N_LAYERS,
                      norm_eps=1e-5, vocab_size=32000, max_context=MAXLEN, jit=False, disable_kv_cache=True)
  if MODELDIR.exists():
    if MODELDIR.is_dir():
      st_single, st_index = MODELDIR/"model.safetensors", MODELDIR/"model.safetensors.index.json"
      if st_single.exists(): weights = safe_load(st_single)
      elif st_index.exists():
        idx = json.load(open(st_index))
        wmap = idx.get("weight_map", {})
        shards = sorted({MODELDIR/f for f in wmap.values()})
        shard_st = {str(sf.name): safe_load(sf) for sf in shards}
        weights = {n: shard_st[wmap[n]][n] for n in wmap.keys()}
      else: weights = None
    else: weights = safe_load(MODELDIR)
    if weights:
      try:
        weights = fix_bf16(weights)
        if any('model.layers' in k for k in weights.keys()): weights = convert_from_huggingface(weights, N_LAYERS, N_HEADS, N_KV_HEADS)
        load_state_dict(model, weights)
        print(f"loaded weights from {MODELDIR}")
      except Exception as e:
        if not smoke_mode: raise
        print(f"warn: failed to load weights in smoke mode ({type(e).__name__}: {e}), using random weights")
  else: print(f"warn: {MODELDIR} not found, using random weights")

  # freeze base model
  for p in get_parameters(model): p.requires_grad_(False)

  print("applying lora...")
  lora_target = [x.strip() for x in getenv("LORA_TARGET", "wq,wv,wk,wo,w1,w2,w3").split(",") if x.strip()]
  apply_lora(model, LORA_R, LORA_ALPHA, target=lora_target)
  lora_params = get_lora_params(model)
  for p in lora_params: p.requires_grad_(True)
  print(f"lora target: {lora_target}, params: {len(lora_params)}")

  if len(GPUS) > 1:
    params = get_parameters(model)
    for p in params: p.to_(GPUS)
    with Context(BEAM=0): Tensor.realize(*params)

  opt = AdamW(lora_params, lr=LR, weight_decay=getenv("WD", 0.01))
  CKPTDIR.mkdir(parents=True, exist_ok=True)

  tok = get_tokenizer(MODELDIR)
  train_data, val_data = load_data(DATADIR, "train"), load_data(DATADIR, "validation")

  if mllog: mllog.end(key=mlc.INIT_END)
  if mllog and getenv("RUNMLPERF"):
    mllog.start(key=mlc.RUN_START)
    mllog.event(key=mlc.SEED, value=SEED)

  print("training...")
  best_rouge, achieved, gstep = 0.0, False, 0
  stop_training = False
  for epoch in range(EPOCHS):
    print(f"\nepoch {epoch+1}/{EPOCHS}")
    for batch in batch_iter(train_data, tok, BS, MAXLEN, shuffle=True):
      gstep += 1
      do_log = gstep%10 == 0 or MAX_STEPS
      if do_log:
        GlobalCounters.reset()
        t1 = time.perf_counter()
      inp, labels = batch['input_ids'], batch['labels']
      if len(GPUS)>1: inp.shard_(GPUS, axis=0), labels.shard_(GPUS, axis=0)
      loss = train_step(inp, labels, model, opt)
      if do_log:
        loss_v = loss.item()
        t2 = time.perf_counter()
        gf = GlobalCounters.global_ops * 1e-9 / (t2-t1)
        print(f"step {gstep}: {gf:9.2f} GFLOPS, loss: {loss_v:.5f}")
      if MAX_STEPS and gstep >= MAX_STEPS:
        print(f"reached MAX_STEPS={MAX_STEPS}, stopping.")
        stop_training = True
        break
      if gstep % EVAL_STEPS == 0:
        print(f"\neval @ {gstep}...")
        eloss, rouge = evaluate(model, val_data, tok, BS, MAXLEN, MAX_EVAL)
        rf = rouge.get('rouge-l',{}).get('f',0.0)
        print(f"eval - loss: {eloss:.4f}, rouge-l f1: {rf:.4f}")
        if mllog: mllog.event(key="eval_rouge_l_f1", value=rf, metadata={"step": gstep})
        if rf >= TARGET and not achieved:
          print(f"target rouge-l {TARGET} achieved! ({rf:.4f})")
          achieved, best_rouge = True, rf
          if mllog: mllog.end(key=mlc.RUN_STOP, metadata={"status": "success", "step": gstep})
          save_ckpt(model, opt, CKPTDIR/f"final_{gstep}.safetensors")
          return
        if rf > best_rouge:
          best_rouge = rf
          save_ckpt(model, opt, CKPTDIR/f"best_{gstep}.safetensors")
      if gstep % CKPT_STEPS == 0:
        save_ckpt(model, opt, CKPTDIR/f"ckpt_{gstep}.safetensors")
    if stop_training: break
  print(f"\ntraining done! best rouge-l: {best_rouge:.4f}, target achieved: {achieved}")
  if mllog and not achieved: mllog.end(key=mlc.RUN_STOP, metadata={"status": "aborted", "step": gstep})

if __name__ == "__main__":
  import multiprocessing
  multiprocessing.set_start_method('spawn')
  train()
