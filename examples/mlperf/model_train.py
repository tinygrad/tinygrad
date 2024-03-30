import functools, os, time, math
from pathlib import Path
from tqdm import tqdm

from tinygrad import Device, GlobalCounters, Tensor, TinyJit, dtypes
from tinygrad.helpers import getenv, BEAM, WINO
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save
from tinygrad.nn.optim import LAMB, LARS, SGD, OptimizerGroup

from extra.lr_scheduler import LRSchedulerGroup, OneCycleLR
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
  epochs            = config["epochs"]            = getenv("EPOCHS", 41)
  BS                = config["BS"]                = getenv("BS", 104 * len(GPUS))  # fp32 GPUS<=6 7900xtx can fit BS=112
  EVAL_BS           = config["EVAL_BS"]           = getenv("EVAL_BS", BS)
  base_lr           = config["base_lr"]           = getenv("LR", 8.5 * (BS/2048))
  lr_warmup_epochs  = config["lr_warmup_epochs"]  = getenv("WARMUP_EPOCHS", 5)
  decay             = config["decay"]             = getenv("DECAY", 2e-4)

  loss_scaler       = config["LOSS_SCALER"]       = getenv("LOSS_SCALER", 128.0 if dtypes.default_float == dtypes.float16 else 1.0)

  target, achieved  = getenv("TARGET", 0.759), False
  eval_start_epoch  = getenv("EVAL_START_EPOCH", 0)
  eval_epochs       = getenv("EVAL_EPOCHS", 1)

  steps_in_train_epoch  = config["steps_in_train_epoch"]  = (len(get_train_files()) // BS)
  steps_in_val_epoch    = config["steps_in_val_epoch"]    = (len(get_val_files()) // EVAL_BS)

  config["DEFAULT_FLOAT"] = dtypes.default_float.name
  config["BEAM"]    = BEAM.value
  config["WINO"]    = WINO.value
  config["SYNCBN"]  = getenv("SYNCBN")

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
    it = iter(tqdm(batch_load_resnet(batch_size=BS, val=False, shuffle=True, seed=seed*epochs + e),
                   total=steps_in_train_epoch, desc=f"epoch {e}", disable=BENCHMARK))
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
        estimated_total_hours = median_step_time * steps_in_train_epoch * epochs / 60 / 60
        print(f"Estimated training time: {estimated_total_hours:.0f}h{(estimated_total_hours - int(estimated_total_hours)) * 60:.0f}m")
        return

    # ** eval loop **
    if (e + 1 - eval_start_epoch) % eval_epochs == 0:
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
  # TODO: RNN-T
  pass

def train_bert():
  from examples.mlperf.dataloader import batch_load_bert
  from examples.mlperf.helpers import get_mlperf_bert_model
  from examples.mlperf.lr_schedulers import PolynomialDecayWithWarmup

  config = {}

  GPUS = config["GPUS"] = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  print(f"Training on {GPUS}")
  for x in GPUS: Device[x]
  seed = config["seed"] = getenv("SEED", 12345)

  # ** hyperparameters **
  epochs             = config["epochs"]                 = getenv("EPOCHS", 1)
  BS                 = config["GLOBAL_BATCH_SIZE"]      = getenv("GLOBAL_BATCH_SIZE", 32 * len(GPUS))
  EVAL_BS            = config["EVAL_BS"]                = getenv("EVAL_BS", 8)
  max_lr             = config["OPT_BASE_LEARNING_RATE"] = getenv("OPT_BASE_LEARNING_RATE", 1e-4)

  train_steps        = config["TRAIN_STEPS"]            = getenv("TRAIN_STEPS", 100000)
  warmup_steps       = config["NUM_WARMUP_STEPS"]       = getenv("NUM_WARMUP_STEPS", 10000)
  start_warmup_steps = config["START_WARMUP_STEPS"]     = getenv("START_WARMUP_STEPS", 0)
  max_eval_steps     = config["MAX_EVAL_STEPS"]         = getenv("MAX_EVAL_STEPS", 100)
  eval_step_freq     = config["EVAL_STEP_FREQ"]         = (math.floor(0.05 * (230.23 * BS + 3000000) / 25000) * 25000) / BS
  save_ckpt_freq     = config["SAVE_CKPT_FREQ"]         = getenv("SAVE_CKPT_FREQ", 1000)

  decay              = config["decay"]                  = getenv("DECAY", 0.01)
  poly_power         = config["poly_power"]             = getenv("POLY_POWER", 1.0)

  target, achieved                                      = getenv("TARGET", 0.72), False

  half              = config["FP16"]                    = getenv("FP16", 0)
  config["BEAM"]    = BEAM.value

  Tensor.manual_seed(seed)  # seed for weight initialization

  if half: dtypes.default_float = dtypes.float16

  config_path = getenv("BERT_CONFIG_PATH", Path(__file__).parent.parents[1] / "extra" / "datasets" / "wiki" / "bert_config.json")
  model = get_mlperf_bert_model(config_path)

  # shard weights and initialize in order
  for _, x in get_state_dict(model).items():
    x.realize().to_(GPUS)
  parameters = get_parameters(model)

  assert 800 % EVAL_BS == 0, "Evaluation batch size must divide 800 without remainder"

  # ** Log hparams **
  for key, value in config.items():
      print(f'HParam: "{key}": {value}')

  # ** Optimizer **
  skip_list = [v for k, v in get_state_dict(model).items() if "bias" in k or "LayerNorm" in k]
  parameters = [x for x in parameters if x not in set(skip_list)]
  optimizer = LAMB(parameters, 1 / warmup_steps, eps=1e-6, wd=decay, adam=False) # TODO: Double check optimizer settings
  optimizer_skip = LAMB(skip_list, 1 / warmup_steps, eps=1e-6, wd=0.0, adam=False)
  optimizer_group = OptimizerGroup(optimizer, optimizer_skip)

  # ** LR scheduler **
  scheduler = PolynomialDecayWithWarmup(optimizer, max_lr, 0, train_steps, warmup_steps, power=poly_power)
  print(f"Training with batch size {BS} for {epochs} epochs with each {train_steps} steps")

  # ** resume from checkpointing **
  start_step = 0
  if ckpt:=getenv("RESUME", ""):
    load_training_state(model, optimizer_group, scheduler, safe_load(ckpt))
    start_step = scheduler.epoch_counter.numpy().item() # We will likely only do one epoch
    print(f"resuming from {ckpt} at step {start_step}")

  # ** init wandb **
  WANDB = getenv("WANDB")
  if WANDB:
    import wandb
    wandb_args = {"id": wandb_id, "resume": "must"} if (wandb_id := getenv("WANDB_RESUME", "")) else {}
    wandb.init(config=config, **wandb_args, project="MLPerf-BERT")

  BENCHMARK = getenv("BENCHMARK")

  @TinyJit
  def train_step(input_ids:Tensor, segment_ids:Tensor, attention_mask:Tensor, masked_positions:Tensor, masked_lm_ids:Tensor, masked_lm_weights:Tensor, next_sentence_labels:Tensor):
    lm_logits, clsf_logits = model(input_ids, segment_ids, attention_mask, masked_positions)
    lm_loss = lm_logits.sparse_categorical_crossentropy(masked_lm_ids, ignore_index=masked_lm_weights)
    clsf_loss = clsf_logits.binary_crossentropy_logits(next_sentence_labels)
    loss = lm_loss + clsf_loss

    if not getenv('DISABLE_BACKWARD', 0):
      optimizer_group.zero_grad()
      loss.backward()

      optimizer_group.step()
      scheduler.step()
    return loss.realize()
  @TinyJit
  def eval_step(input_ids:Tensor, segment_ids:Tensor, attention_mask:Tensor, masked_positions:Tensor, masked_lm_ids:Tensor, masked_lm_weights:Tensor, next_sentence_labels:Tensor):
    lm_logits, clsf_logits = model(input_ids, segment_ids, attention_mask, masked_positions)

    clsf_predictions = clsf_logits.log_softmax().argmax(-1)
    clsf_accuracy = (clsf_predictions == next_sentence_labels).float().mean()

    mlm_predictions = lm_logits.log_softmax().argmax(-1)
    mask = (masked_lm_weights == 1.0)
    mlm_accuracy = (((mlm_predictions == masked_lm_ids) == mask).float() / mask.float().sum()).mean()

    lm_loss = lm_logits.sparse_categorical_crossentropy(masked_lm_ids, ignore_index=masked_lm_weights)
    clsf_loss = clsf_logits.binary_crossentropy_logits(next_sentence_labels)
    return {
      "masked_lm_accuracy": mlm_accuracy.realize(), 
      "masked_lm_loss": lm_loss.realize(), 
      "next_sentence_accuracy": clsf_accuracy.realize(), 
      "next_sentence_loss": clsf_loss.realize()
      }

  def data_get(it):
    data: dict[str, Tensor] = next(it)
    for key in data.keys(): data[key].shard_(GPUS, axis=0)
    return data
  
  eval_it = iter(batch_load_bert(EVAL_BS, val=True))

  # ** epoch loop **
  step_times = []
  for e in range(1, epochs + 1):
    # ** train loop **
    Tensor.training = True
    train_it = iter(tqdm(batch_load_bert(BS, val=False), total=train_steps))
    i, data = 0, data_get(train_it)
    while data is not None and i < train_steps:
      st = time.perf_counter()
      GlobalCounters.reset()
      loss = train_step(data["input_ids"], data["segment_ids"], data["input_mask"], data["masked_lm_positions"], \
                        data["masked_lm_ids"], data["masked_lm_weights"], data["next_sentence_labels"])

      pt = time.perf_counter()

      try:
        next_data = data_get(train_it)
      except StopIteration:
        next_data = None

      dt = time.perf_counter()

      device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
      loss = loss.numpy().item()

      cl = time.perf_counter()
      if BENCHMARK:
        step_times.append(cl - st)

      tqdm.write(
        f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms fetch data, "
        f"{(cl - dt) * 1000.0:7.2f} ms {device_str}, {loss:5.2f} loss, {optimizer.lr.numpy()[0]:.6f} LR, "
        f"{GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
      if WANDB:
        wandb.log({"lr": optimizer.lr.numpy(), "train/loss": loss, "train/step_time": cl - st,
                   "train/python_time": pt - st, "train/data_time": dt - pt, "train/cl_time": cl - dt,
                   "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st)})

      data, next_data = next_data, None
      i += 1

      if i == BENCHMARK:
        median_step_time = sorted(step_times)[(BENCHMARK + 1) // 2]  # in seconds
        estimated_total_hours = median_step_time * train_steps * epochs / 60 / 60
        print(f"Estimated training time: {estimated_total_hours:.0f}h{(estimated_total_hours - int(estimated_total_hours)) * 60:.0f}m")
        return

      # ** eval loop **
      if i % eval_step_freq == 0:
        train_step.reset()  # free the train step memory :(
        eval_loss = []
        eval_accuracy = []
        eval_times = []
        Tensor.training = False

        for _ in tqdm(range((max_eval_steps * 8) // EVAL_BS), desc="Evaluating", total=(max_eval_steps * 8) // EVAL_BS):
          data = data_get(eval_it)
          GlobalCounters.reset()
          st = time.time()

          eval_result: dict[str, Tensor] = eval_step(data["input_ids"], data["segment_ids"], data["input_mask"], data["masked_lm_positions"], \
                                                    data["masked_lm_ids"], data["masked_lm_weights"], data["next_sentence_labels"])

          lm_loss, clsf_loss  = eval_result["masked_lm_loss"].numpy().item(), eval_result["next_sentence_loss"].numpy().item()
          mlm_accuracy, clsf_accuracy = eval_result["masked_lm_accuracy"].numpy().item(), eval_result["next_sentence_accuracy"].numpy().item()
          eval_loss.append([lm_loss, clsf_loss])
          eval_accuracy.append([mlm_accuracy, clsf_accuracy])

          et = time.time()
          eval_times.append(et - st)

        eval_step.reset()
        total_lm_loss = sum(pair[0] for pair in eval_loss) / len(eval_loss)
        total_clsf_loss = sum(pair[1] for pair in eval_loss) / len(eval_loss)
        total_lm_accuracy = sum(pair[0] for pair in eval_accuracy) / len(eval_accuracy)
        total_clsf_accuracy = sum(pair[1] for pair in eval_accuracy) / len(eval_accuracy)
        total_fw_time = sum(eval_times) / len(eval_times)
        tqdm.write("\nEVAL Results:\n\n")
        tqdm.write(f"eval lm loss: {total_lm_loss:.2f}, eval clsf loss: {total_clsf_loss:.2f}, eval lm accuracy: {total_lm_accuracy:.2f}, \
                    eval clsf accuracy: {total_clsf_accuracy:.2f}, eval time: {total_fw_time:.2f}")
        if WANDB:
          wandb.log({"eval/lm_loss": total_lm_loss, "eval/clsf_loss": total_clsf_loss, "eval/lm_accuracy": total_lm_accuracy, \
                      "eval/clsf_accuracy": total_clsf_accuracy, "eval/forward_time": total_fw_time})

        # save model if achieved target
        if not achieved and total_lm_accuracy >= target:
          if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
          fn = f"./ckpts/bert-large.safe"
          safe_save(get_state_dict(model), fn)
          print(f" *** Model saved to {fn} ***")
          print(f"Reference Convergence point reached after {(e - 1) * train_steps * BS + i * BS} datasamples.")
          achieved = True

      if getenv("CKPT") and i % save_ckpt_freq == 0:
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        if WANDB and wandb.run is not None:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_{wandb.run.id}_e{e}.safe"
        else:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_e{e}.safe"
        print(f"saving ckpt to {fn}")
        safe_save(get_training_state(model, optimizer_group, scheduler), fn)

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()
