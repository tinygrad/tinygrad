import functools
import os
import time
from tqdm import tqdm
import multiprocessing

from tinygrad import Device, GlobalCounters, Tensor, TinyJit, dtypes
from tinygrad.helpers import getenv, BEAM, WINO, Context
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save
from tinygrad.nn.optim import LARS, SGD, OptimizerGroup

from extra.lr_scheduler import LRSchedulerGroup
from examples.mlperf.helpers import get_training_state, load_training_state

def train_resnet():
  from extra.models import resnet
  from examples.mlperf.dataloader import batch_load_resnet
  from extra.datasets.imagenet import get_train_files, get_val_files
  from examples.mlperf.lr_schedulers import PolynomialDecayWithWarmup
  from examples.mlperf.initializers import Conv2dHeNormal, Linear
  from examples.hlb_cifar10 import UnsyncedBatchNorm

  config = {}
  seed = config["seed"] = getenv("SEED", 42)
  Tensor.manual_seed(seed)  # seed for weight initialization

  GPUS = config["GPUS"] = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  print(f"Training on {GPUS}")
  for x in GPUS: Device[x]

  TRAIN_BEAM = getenv("TRAIN_BEAM", BEAM.value)
  EVAL_BEAM = getenv("EVAL_BEAM", BEAM.value)

  # ** model definition and initializers **
  num_classes = 1000
  resnet.Conv2d = Conv2dHeNormal
  resnet.Linear = Linear
  if not getenv("SYNCBN"): resnet.BatchNorm = functools.partial(UnsyncedBatchNorm, num_devices=len(GPUS))
  model = resnet.ResNet50(num_classes)

  # shard weights and initialize in order
  for k, x in get_state_dict(model).items():
    if not getenv("SYNCBN") and ("running_mean" in k or "running_var" in k):
      x.realize().shard_(GPUS, axis=0)
    else:
      x.realize().to_(GPUS)
  parameters = get_parameters(model)

  # ** hyperparameters **
  epochs            = config["epochs"]            = getenv("EPOCHS", 37)
  BS                = config["BS"]                = getenv("BS", 104 * len(GPUS))  # fp32 GPUS<=6 7900xtx can fit BS=112
  EVAL_BS           = config["EVAL_BS"]           = getenv("EVAL_BS", BS)
  base_lr           = config["base_lr"]           = getenv("LR", 7.4 * (BS/1632))
  lr_warmup_epochs  = config["lr_warmup_epochs"]  = getenv("WARMUP_EPOCHS", 2)
  decay             = config["decay"]             = getenv("DECAY", 5e-5)

  loss_scaler       = config["LOSS_SCALER"]       = getenv("LOSS_SCALER", 128.0 if dtypes.default_float == dtypes.float16 else 1.0)

  target, achieved  = getenv("TARGET", 0.759), False
  eval_start_epoch  = getenv("EVAL_START_EPOCH", 0)
  eval_epochs       = getenv("EVAL_EPOCHS", 1)

  steps_in_train_epoch  = config["steps_in_train_epoch"]  = (len(get_train_files()) // BS)
  steps_in_val_epoch    = config["steps_in_val_epoch"]    = (len(get_val_files()) // EVAL_BS)

  config["DEFAULT_FLOAT"] = dtypes.default_float.name
  config["BEAM"]          = BEAM.value
  config["TRAIN_BEAM"]    = TRAIN_BEAM
  config["EVAL_BEAM"]     = EVAL_BEAM
  config["WINO"]          = WINO.value
  config["SYNCBN"]        = getenv("SYNCBN")

  # ** Optimizer **
  skip_list = [v for k, v in get_state_dict(model).items() if "bn" in k or "bias" in k or "downsample.1" in k]
  parameters = [x for x in parameters if x not in set(skip_list)]
  optimizer = LARS(parameters, base_lr, momentum=.9, weight_decay=decay)
  optimizer_skip = SGD(skip_list, base_lr, momentum=.9, weight_decay=0.0, classic=True)
  optimizer_group = OptimizerGroup(optimizer, optimizer_skip)

  # ** LR scheduler **
  scheduler = PolynomialDecayWithWarmup(optimizer, initial_lr=base_lr, end_lr=1e-4,
                                        train_steps=epochs * steps_in_train_epoch,
                                        warmup=lr_warmup_epochs * steps_in_train_epoch)
  scheduler_skip = PolynomialDecayWithWarmup(optimizer_skip, initial_lr=base_lr, end_lr=1e-4,
                                             train_steps=epochs * steps_in_train_epoch,
                                             warmup=lr_warmup_epochs * steps_in_train_epoch)
  scheduler_group = LRSchedulerGroup(scheduler, scheduler_skip)
  print(f"training with batch size {BS} for {epochs} epochs")

  # ** resume from checkpointing **
  start_epoch = 0
  if ckpt:=getenv("RESUME", ""):
    load_training_state(model, optimizer_group, scheduler_group, safe_load(ckpt))
    start_epoch = int(scheduler.epoch_counter.numpy().item() / steps_in_train_epoch)
    print(f"resuming from {ckpt} at epoch {start_epoch}")

  # ** init wandb **
  WANDB = getenv("WANDB")
  if WANDB:
    import wandb
    wandb_args = {"id": wandb_id, "resume": "must"} if (wandb_id := getenv("WANDB_RESUME", "")) else {}
    wandb.init(config=config, **wandb_args)

  BENCHMARK = getenv("BENCHMARK")

  # ** jitted steps **
  input_mean = Tensor([123.68, 116.78, 103.94], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  # mlperf reference resnet does not divide by input_std for some reason
  # input_std = Tensor([0.229, 0.224, 0.225], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  def normalize(x): return (x.permute([0, 3, 1, 2]) - input_mean).cast(dtypes.default_float)
  @TinyJit
  def train_step(X, Y):
    with Context(BEAM=TRAIN_BEAM):
      optimizer_group.zero_grad()
      X = normalize(X)
      out = model.forward(X)
      loss = out.cast(dtypes.float32).sparse_categorical_crossentropy(Y, label_smoothing=0.1)
      top_1 = (out.argmax(-1) == Y).sum()
      (loss * loss_scaler).backward()
      for t in optimizer_group.params: t.grad = t.grad.contiguous() / loss_scaler
      optimizer_group.step()
      scheduler_group.step()
      return loss.realize(), top_1.realize()

  @TinyJit
  def eval_step(X, Y):
    with Context(BEAM=EVAL_BEAM):
      X = normalize(X)
      out = model.forward(X)
      loss = out.cast(dtypes.float32).sparse_categorical_crossentropy(Y, label_smoothing=0.1)
      top_1 = (out.argmax(-1) == Y).sum()
      return loss.realize(), top_1.realize()

  def data_get(it):
    x, y, cookie = next(it)
    return x.shard(GPUS, axis=0).realize(), Tensor(y, requires_grad=False).shard(GPUS, axis=0), cookie

  # ** epoch loop **
  step_times = []
  for e in range(start_epoch, epochs):
    # ** train loop **
    Tensor.training = True
    batch_loader = batch_load_resnet(batch_size=BS, val=False, shuffle=True, seed=seed*epochs + e)
    it = iter(tqdm(batch_loader, total=steps_in_train_epoch, desc=f"epoch {e}", disable=BENCHMARK))
    i, proc = 0, data_get(it)
    st = time.perf_counter()
    while proc is not None:
      GlobalCounters.reset()
      (loss, top_1_acc), proc = train_step(proc[0], proc[1]), proc[2]

      pt = time.perf_counter()

      try:
        next_proc = data_get(it)
      except StopIteration:
        next_proc = None

      dt = time.perf_counter()

      device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
      loss, top_1_acc = loss.numpy().item(), top_1_acc.numpy().item() / BS

      cl = time.perf_counter()
      if BENCHMARK:
        step_times.append(cl - st)

      tqdm.write(
        f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms fetch data, "
        f"{(cl - dt) * 1000.0:7.2f} ms {device_str}, {loss:5.2f} loss, {top_1_acc:3.2f} acc, {optimizer.lr.numpy()[0]:.6f} LR, "
        f"{GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
      if WANDB:
        wandb.log({"lr": optimizer.lr.numpy(), "train/loss": loss, "train/top_1_acc": top_1_acc, "train/step_time": cl - st,
                   "train/python_time": pt - st, "train/data_time": dt - pt, "train/cl_time": cl - dt,
                   "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st), "epoch": e + (i + 1) / steps_in_train_epoch})

      st = cl
      proc, next_proc = next_proc, None  # return old cookie
      i += 1

      if i == BENCHMARK:
        median_step_time = sorted(step_times)[(BENCHMARK + 1) // 2]  # in seconds
        estimated_total_minutes = int(median_step_time * steps_in_train_epoch * epochs / 60)
        print(f"Estimated training time: {estimated_total_minutes // 60}h{estimated_total_minutes % 60}m")
        # if we are doing beam search, run the first eval too
        if (TRAIN_BEAM or EVAL_BEAM) and e == start_epoch: break
        return

    # ** eval loop **
    if (e + 1 - eval_start_epoch) % eval_epochs == 0 and steps_in_val_epoch > 0:
      train_step.reset()  # free the train step memory :(
      eval_loss = []
      eval_times = []
      eval_top_1_acc = []
      Tensor.training = False

      it = iter(tqdm(batch_load_resnet(batch_size=EVAL_BS, val=True, shuffle=False), total=steps_in_val_epoch))
      proc = data_get(it)
      while proc is not None:
        GlobalCounters.reset()
        st = time.time()

        (loss, top_1_acc), proc = eval_step(proc[0], proc[1]), proc[2]  # drop inputs, keep cookie

        try:
          next_proc = data_get(it)
        except StopIteration:
          next_proc = None

        loss, top_1_acc = loss.numpy().item(), top_1_acc.numpy().item() / EVAL_BS
        eval_loss.append(loss)
        eval_top_1_acc.append(top_1_acc)
        proc, next_proc = next_proc, None  # return old cookie

        et = time.time()
        eval_times.append(et - st)

      eval_step.reset()
      total_loss = sum(eval_loss) / len(eval_loss)
      total_top_1 = sum(eval_top_1_acc) / len(eval_top_1_acc)
      total_fw_time = sum(eval_times) / len(eval_times)
      tqdm.write(f"eval loss: {total_loss:.2f}, eval time: {total_fw_time:.2f}, eval top 1 acc: {total_top_1:.3f}")
      if WANDB:
        wandb.log({"eval/loss": total_loss, "eval/top_1_acc": total_top_1, "eval/forward_time": total_fw_time, "epoch": e + 1})

      # save model if achieved target
      if not achieved and total_top_1 >= target:
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        fn = f"./ckpts/resnet50.safe"
        safe_save(get_state_dict(model), fn)
        print(f" *** Model saved to {fn} ***")
        achieved = True

      # checkpoint every time we eval
      if getenv("CKPT"):
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        if WANDB and wandb.run is not None:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_{wandb.run.id}_e{e}.safe"
        else:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_e{e}.safe"
        print(f"saving ckpt to {fn}")
        safe_save(get_training_state(model, optimizer_group, scheduler_group), fn)

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  # TODO: Unet3d
  pass

def train_rnnt():
  traint0 = time.time()
  import numpy as np
  import sentencepiece
  import gc
  from extra.datasets import librispeech2
  from extra.models.rnnt2 import RNNT
  import math

  seed = 0
  print(f"{seed=}")

  n_classes = 1024
  in_feats = 256
  enc_n_hid = 1024
  enc_pre_rnn_layers = 2
  enc_post_rnn_layers = 3
  enc_dropout = 0.1
  pred_dropout = 0.3
  joint_dropout = 0.3

  pred_n_hid = 512
  pred_rnn_layers = 2
  joint_n_hid = 512
  batch_size = 32

  forget_gate_bias = 1.0
  weights_init_scale = 0.45

  maxT = 642
  maxU = 126

  opt_eps=1e-9
  wd=1e-3
  ema=0.994
  opt_b1 = 0.9
  opt_b2 = 0.9985
  lr=0.0062
  min_lr=1e-5
  lr_exp_gamma=0.915
  max_global_norm=1.0

  epochs = 60
  warmup_epochs=1
  hold_epochs=11
  eval_epochs =1

  samples_per_step = 512

  gpus = [f"cuda:{i}" for i in range(2)]
  # gpus = [f"cuda:{i}" for i in range(1)]
  librispeech2.download_and_process_alldata(remove_notfinal_files=True)
  train_loader = librispeech2.DataLoader(batch_size=batch_size,gpus=gpus,eval=False,shuffle=True,seed=seed)

  Tensor.manual_seed(seed)
  dtype=dtypes.float32

  model = RNNT(gpus,
  nclasses = n_classes,
  enc_input_size = in_feats,
  enc_hid_size = enc_n_hid,
  enc1_layers = enc_pre_rnn_layers,
  enc2_layers = enc_post_rnn_layers,
  pred_layers = pred_rnn_layers,
  batch_size = batch_size,
  pred_input_size = pred_n_hid,
  pred_hid_size = pred_n_hid,
  lin_outputsize = joint_n_hid,
  enc_dropout = enc_dropout,
  pred_dropout = pred_dropout,
  joint_dropout = joint_dropout,
  forget_gate_bias = forget_gate_bias,
  weights_init_scale = weights_init_scale,
  maxT=maxT,maxU=maxU,
  opt_eps=opt_eps,
  wd=wd,
  ema=ema,
  opt_b1 = opt_b1,
  opt_b2 = opt_b2,
  lr=lr,
  min_lr=min_lr,
  max_global_norm=max_global_norm,
  lr_exp_gamma=lr_exp_gamma,
  grad_accumulation_factor=1/int(samples_per_step/(batch_size*len(gpus))),
  warmup_epochs=warmup_epochs,
  hold_epochs=hold_epochs,
  beam=16, debug=0,dtype=dtype)

  assert samples_per_step % (len(gpus)*batch_size) == 0

  acc_steps_per_epoch = math.floor(len(train_loader.active_data)/(samples_per_step))
  batch_steps_per_step = int(samples_per_step/(len(gpus)*batch_size))
  model.opt_steps_per_epoch = acc_steps_per_epoch
  print(f"all files {len(train_loader.filedata)}, within maxduration {len(train_loader.active_data)}, minibatch count {len(train_loader.iter)}, files per epoch {samples_per_step*acc_steps_per_epoch}")
  print(f"{epochs=} {hold_epochs=} {warmup_epochs=} {acc_steps_per_epoch=} {batch_steps_per_step=} {samples_per_step=} {batch_size=}")

  def warmup():
    for lrtens in model.lrtens:
      lrtens.assign(Tensor.zeros_like(lrtens).contiguous()).realize()
    for len_a, len_t in ((8,8),(maxT,maxU-1)):
      audio = [Tensor.ones(len_a,batch_size,in_feats, device=gpu, dtype=dtype).pad(((0,model.maxT-len_a),None,None)).contiguous().realize() for gpu in gpus]
      audio_len = [Tensor.full(shape=(batch_size,), fill_value=len_a, device=gpu, dtype=dtypes.int32).contiguous().realize() for gpu in gpus]
      txt = [Tensor.ones(batch_size,len_t, device=gpu, dtype=dtypes.int32).pad((None,(0,model.maxU-1-len_t))).contiguous().realize() for gpu in gpus]
      txt_len = [Tensor.full(batch_size,fill_value=len_t, device=gpu).contiguous().realize() for gpu in gpus]
      loss = model.forward(audio,audio_len,txt,txt_len)
      model.backward()
      model.step()
      model.zero_grads()
  warmup()

  def levenshtein(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]
  def evaluate():
    model.del_joint_bufs()

    batch_size = 120

    model_eval = RNNT(gpus,
      nclasses = n_classes,
      enc_input_size = in_feats,
      enc_hid_size = enc_n_hid,
      enc1_layers = enc_pre_rnn_layers,
      enc2_layers = enc_post_rnn_layers,
      pred_layers = pred_rnn_layers,
      batch_size = batch_size,
      pred_input_size = pred_n_hid,
      pred_hid_size = pred_n_hid,
      lin_outputsize = joint_n_hid,
      enc_dropout = 0,
      pred_dropout = 0,
      joint_dropout = 0,
      maxT=1092,maxU=189,
      beam=2, debug=0,dtype=dtype,eval=True)
    model_eval.copy_parameters(model_eval.parameters,model.ema_parameters)
    # model_eval.load_parameters(f"{librispeech2.basedir}/weights_wer12.66.dat")

    sent = sentencepiece.SentencePieceProcessor(model_file="/datasets/LibriSpeech/librispeech1023.model")
    import torch

    loader = librispeech2.DataLoader(batch_size,gpus=gpus,eval=True,shuffle=False,seed=seed)
    words_tot, scores_tot = 0,0
    maxn = math.ceil(len(loader.iter)/len(gpus))
    t2 = time.time()
    for n in range(maxn):
      t1 = time.time()
      if n == maxn-1:
        n2 = len(loader.iter) % len(gpus)
        if n2 == 0: n2 = len(gpus)
        data = [next(loader) for _ in range(n2)]
        for _ in range(n2,len(gpus)):
          data += [[torch.zeros_like(el) for el in data[0]]]
      else:
        data = [next(loader) for _ in range(len(gpus))]

      audio, audio_len, txt, txt_len=list(zip(*data))
      topad = batch_size-audio_len[0].shape[0]
      audio = [Tensor(audio.cpu().numpy(), device=gpu).pad((None,(0,topad),None)).contiguous().realize() for audio,gpu in zip(audio,gpus)]
      audio_len = [Tensor(audio_len.cpu().numpy(), device=gpu).pad(((0,topad),)).contiguous().realize() for audio_len,gpu in zip(audio_len,gpus)]
      txt = [Tensor(txt.cpu().numpy(), device=gpu).pad(((0,topad),None)).contiguous().realize() for txt,gpu in zip(txt,gpus)]
      txt_len = [Tensor(txt_len.cpu().numpy(), device=gpu).pad(((0,topad),)).contiguous().realize() for txt_len,gpu in zip(txt_len,gpus)]

      token_pred = model_eval.evaluate_batch(audio,audio_len)

      pred = sent.decode(token_pred)

      alltxt = np.concatenate([el.numpy() for el in txt])
      alltxt_len = np.concatenate([el.numpy() for el in txt_len])
      res = sent.decode([el[:len].tolist() for el,len in zip(alltxt, alltxt_len)])

      words = 0
      scores = 0
      for p,r in zip(pred,res):
        words += len(r.split())
        scores += levenshtein(p.split(), r.split())
      words_tot += words
      scores_tot += scores
      wer = scores/words
      print(f"{(t:=time.time())-traint0:>5.1f}s {t-t1:.2f}s {n:>3}/{math.ceil(len(loader.iter)/len(gpus))} wer {wer*100:>6.3f}% words {words:>4} scores {scores:>3} samples {batch_size*len(gpus)}, speed {10**3*(t-t1)/(batch_size*len(gpus)):>3.0f} ms/s")

    wer = scores_tot/words_tot
    print(f"{(t:=time.time())-t2:>5.1f}s total wer {wer*100:.3f}% words {words_tot} scores {scores_tot}")

    del model_eval, loader, data, audio, audio_len, txt, txt_len
    gc.collect()

    for gpu in gpus: Device[gpu].allocator.free_cache()
    model.remake_joint_bufs()

    return wer

  # evaluate()

  lossacc = 0
  for epoch in range(epochs):
    epochtime = time.time()
    for n1 in range(acc_steps_per_epoch):
      t2 = time.time()
      for n2 in range(batch_steps_per_step):
        t1 = time.time()
        data = [next(train_loader) for _ in range(len(gpus))]
        audio, audio_len, txt, txt_len=list(zip(*data))
        audio = [Tensor(audio.cpu().numpy(), device=gpu).realize() for audio,gpu in zip(audio,gpus)]
        audio_len = [Tensor(audio_len.cpu().numpy(), device=gpu).realize() for audio_len,gpu in zip(audio_len,gpus)]
        txt = [Tensor(txt.cpu().numpy(), device=gpu).realize() for txt,gpu in zip(txt,gpus)]
        txt_len = [Tensor(txt_len.cpu().numpy(), device=gpu).realize() for txt_len,gpu in zip(txt_len,gpus)]

        losses = model.forward(audio,audio_len,txt,txt_len)
        loss = np.concatenate([el.numpy() for el in losses]).sum().item()
        lossacc += loss
        model.backward()
        model.synchronize()

        # print(f"{(t:=time.time())-traint0:.3f}s epoch {epoch+1} {(n1*(batch_steps_per_step)+n2+1)}/{math.floor(len(train_loader.active_data)/(samples_per_step))*batch_steps_per_step} {t-t1:.3f}s, loss {loss:.4e}, {10**3*(t-t1)/(len(gpus)*batch_size):.2f} ms/s, {len(gpus)*batch_size/(t-t1):.1f} s/sec")

      model.step(epoch,n1)
      model.zero_grads()
      model.apply_ema(*model.ema_parameters,*model.parameters)
      print(f"{(t:=time.time())-traint0:.3f}s epoch {epoch+1} stepped {n1+1}/{acc_steps_per_epoch}, {t-t2:.3f}s, loss {lossacc:.3f}, {10**3*(t-t2)/(len(gpus)*batch_size*batch_steps_per_step):.2f} ms/s, lr {model.lrlast:.5f}")
      # print()

      lossacc = 0
    for data in train_loader:
      pass
    train_loader.iter.reset()
    model.save_parameters(filename=f"{librispeech2.basedir}/LibriSpeech/parameters_ema.dat", ema=True)
    model.save_parameters(filename=f"{librispeech2.basedir}/LibriSpeech/parameters_opt.dat", ema=False,opt_parameters=True)
    print(f"{(t:=time.time())-traint0:.3f}s epoch time {time.time()-epochtime:.3f}s")
    if (epoch+1) % eval_epochs == 0:
      evaluate()

def train_bert():
  # TODO: BERT
  pass

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  train_rnnt()
  # multiprocessing.set_start_method('spawn')
  # with Tensor.train():
  #   for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
  #     nm = f"train_{m}"
  #     if nm in globals():
  #       print(f"training {m}")
  #       globals()[nm]()
