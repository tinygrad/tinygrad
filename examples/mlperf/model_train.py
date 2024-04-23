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
        print(f"epoch global_ops: {steps_in_train_epoch * GlobalCounters.global_ops:_}, "
              f"epoch global_mem: {steps_in_train_epoch * GlobalCounters.global_mem:_}")
        # if we are doing beam search, run the first eval too
        if (TRAIN_BEAM or EVAL_BEAM) and e == start_epoch: break
        return

    # ** eval loop **
    if (e + 1 - eval_start_epoch) % eval_epochs == 0 and steps_in_val_epoch > 0:
      if getenv("RESET_STEP", 1): train_step.reset()  # free the train step memory :(
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

      if getenv("RESET_STEP", 1): eval_step.reset()
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
  from examples.mlperf.losses import dice_ce_loss
  from examples.mlperf.metrics import dice_score
  from extra.models.unet3d import UNet3D
  from extra.datasets.kits19 import iterate, get_train_files, get_val_files, sliding_window_inference, preprocess, BASEDIR
  from tinygrad import Device, Tensor
  from tinygrad.nn.optim import SGD
  from tinygrad import GlobalCounters
  from math import ceil

  import numpy as np
  import random
  import time

  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  for x in GPUS: Device[x]
  print(f"Training on {GPUS}")

  TARGET_METRIC = 0.908
  NUM_EPOCHS = getenv("NUM_EPOCHS", 4000)
  BS = getenv("BS", 1 * len(GPUS))
  LR = getenv("LR", 0.8 * (BS / 2))
  LR_WARMUP_EPOCHS = getenv("LR_WARMUP_EPOCHS", 200)
  LR_WARMUP_INIT_LR = getenv("LR_WARMUP_INIT_LR", 0.0001)
  WANDB = getenv("WANDB")
  PROJ_NAME = getenv("PROJ_NAME", "tinygrad_unet3d_mlperf")
  SEED = getenv("SEED")
  TRAIN_DATASET_SIZE = len(get_train_files())
  VAL_DATASET_SIZE = len(get_val_files())
  SAMPLES_PER_EPOCH = TRAIN_DATASET_SIZE // BS
  START_EVAL_AT = getenv("START_EVAL_AT", ceil(1000 * TRAIN_DATASET_SIZE / (SAMPLES_PER_EPOCH * BS)))
  EVALUATE_EVERY = getenv("EVALUATE_EVERY", ceil(20 * TRAIN_DATASET_SIZE / (SAMPLES_PER_EPOCH * BS)))
  PREPROCESSED_DIR = BASEDIR / ".." / "preprocessed"
  TRAIN_BEAM = getenv("TRAIN_BEAM", BEAM.value)
  EVAL_BEAM = getenv("EVAL_BEAM", BEAM.value)
  BENCHMARK = getenv("BENCHMARK")

  config = {
    "num_epochs": NUM_EPOCHS,
    "batch_size": BS,
    "learning_rate": LR,
    "learning_rate_warmup_epochs": LR_WARMUP_EPOCHS,
    "learning_rate_warmup_init": LR_WARMUP_INIT_LR,
    "start_eval_at": START_EVAL_AT,
    "evaluate_every": EVALUATE_EVERY,
    "train_beam": TRAIN_BEAM,
    "eval_beam": EVAL_BEAM,
    "wino": WINO.value,
    "gpus": GPUS,
    "default_float": dtypes.default_float.name
  }

  if WANDB:
    try:
      import wandb
    except ImportError:
      raise "Need to install wandb to use it"

  if SEED:
    config["seed"] = SEED

    Tensor.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

  model = UNet3D()
  params = get_parameters(model)

  for p in params: p.realize().to_(GPUS)

  optim = SGD(params, lr=LR, momentum=0.9, nesterov=True)

  def preprocess_dataset(filenames, is_val=False):
    for fn in tqdm(filenames, desc=f"preprocess {'val' if is_val else 'train'}"):
      case = os.path.basename(fn)
      image_preproc_path, label_preproc_path = PREPROCESSED_DIR / f"{case}_x.npy", PREPROCESSED_DIR / f"{case}_y.npy"
      image, label = preprocess(fn)
      image, label = image.astype(np.float32), label.astype(np.uint8)
      np.save(image_preproc_path, image, allow_pickle=False)
      np.save(label_preproc_path, label, allow_pickle=False)
      tqdm.write(f"Saved preprocessed data {case}")

  def lr_warm_up(optim, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    optim.lr.assign(Tensor([init_lr + (lr - init_lr) * scale], device=GPUS))

  @TinyJit
  def train_step(model, x, y):
    with Context(BEAM=TRAIN_BEAM):
      optim.zero_grad()

      y_hat = model(x)
      loss = dice_ce_loss(y_hat, y)

      loss.backward()
      optim.step()
      return loss.realize()
  
  def eval_step(model, x, y):
    with Context(BEAM=EVAL_BEAM):
      y_hat, y = sliding_window_inference(model, x, y, gpus=GPUS)
      y_hat, y = Tensor(y_hat), Tensor(y, requires_grad=False)
      loss = dice_ce_loss(y_hat, y)
      score = dice_score(y_hat, y)
      return loss.realize(), score.realize()
  
  if WANDB: wandb.init(config=config, project=PROJ_NAME)

  step_times, start_epoch = [], 0
  is_successful, diverged = False, False
  start_eval_at, evaluate_every = START_EVAL_AT, EVALUATE_EVERY
  next_eval_at = start_eval_at
  print(f"Eval starts at epoch {start_eval_at} and every {evaluate_every} epochs afterwards")

  if not PREPROCESSED_DIR.exists():
    PREPROCESSED_DIR.mkdir()
    preprocess_dataset(get_train_files())
    preprocess_dataset(get_val_files(), is_val=True)

  for epoch in range(1, NUM_EPOCHS + 1):
    Tensor.training = True

    if epoch <= LR_WARMUP_EPOCHS and LR_WARMUP_EPOCHS > 0:
      lr_warm_up(optim, LR_WARMUP_INIT_LR, LR, epoch, LR_WARMUP_EPOCHS)

    st = time.perf_counter()

    for i, (x, y) in enumerate(tqdm(iterate(val=False, shuffle=True, bs=BS), total=SAMPLES_PER_EPOCH, desc=f"epoch {epoch}", disable=BENCHMARK), start=1):
      GlobalCounters.reset()

      x, y = Tensor(x).realize().shard(GPUS, axis=0), Tensor(y, requires_grad=False).shard(GPUS, axis=0)

      loss = train_step(model, x, y)
      pt = time.perf_counter()

      loss = loss.numpy().item()
      cl = time.perf_counter()

      if BENCHMARK:
        step_times.append(cl - st)

      tqdm.write(
        f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, "
        f"{loss:5.3f} loss, {optim.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used / 1e9:.2f} GB used,"
        f"{GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS"
      )

      if WANDB:
        wandb.log({"lr": optim.lr.numpy(), "train/loss": loss, "train/step_time": cl - st, "train/python_time": pt - st,
                   "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st), "epoch": epoch + (i + 1) / SAMPLES_PER_EPOCH})

      st = cl

      if i == BENCHMARK:
        median_step_time = sorted(step_times)[(BENCHMARK + 1) // 2]  # in seconds
        estimated_total_minutes = int(median_step_time * SAMPLES_PER_EPOCH * NUM_EPOCHS / 60)
        print(f"Estimated training time: {estimated_total_minutes // 60}h{estimated_total_minutes % 60}m")
        # if we are doing beam search, run the first eval too
        if (TRAIN_BEAM or EVAL_BEAM) and epoch == start_epoch: break
        return

    if epoch == next_eval_at:
      Tensor.training = False

      next_eval_at += evaluate_every
      eval_loss = []
      scores = []

      for x, y in tqdm(iterate(), total=VAL_DATASET_SIZE):
        eval_loss_value, score = eval_step(model, x, y)
        eval_loss.append(eval_loss_value)
        scores.append(score)

      scores = Tensor.mean(Tensor.stack(scores, dim=0), axis=0).numpy()
      eval_loss = Tensor.mean(Tensor.stack(eval_loss, dim=0), axis=0).numpy()

      l1_dice, l2_dice = scores[0][-2], scores[0][-1]
      mean_dice = (l2_dice + l1_dice) / 2

      tqdm.write(f"{l1_dice} L1 dice, {l2_dice} L2 dice, {mean_dice:.3f} mean_dice, {eval_loss:5.2f} eval_loss")

      if WANDB:
        wandb.log({"eval/loss": eval_loss, "eval/mean_dice": mean_dice, "epoch": epoch})

      if mean_dice >= TARGET_METRIC:
        is_successful = True

        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        fn = f"./ckpts/unet3d.safe"
        safe_save(get_state_dict(model), fn)
        print(f" *** Model saved to {fn} ***")
      elif mean_dice < 1e-6:
        print("Model diverging. Aborting.")
        diverged = True

    if is_successful or diverged:
      break

def train_rnnt():
  # TODO: RNN-T
  pass

def train_bert():
  # TODO: BERT
  pass

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  multiprocessing.set_start_method('spawn')
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()
