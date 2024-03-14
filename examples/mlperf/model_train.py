import functools
import os
import time
from tqdm import tqdm

from tinygrad import Device, GlobalCounters, Tensor, TinyJit, dtypes
from tinygrad.helpers import getenv, BEAM, WINO
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save

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
  epochs            = config["epochs"]            = getenv("EPOCHS", 45)
  BS                = config["BS"]                = getenv("BS", 104 * len(GPUS))  # fp32 GPUS<=6 7900xtx can fit BS=112
  EVAL_BS           = config["EVAL_BS"]           = getenv("EVAL_BS", BS)
  base_lr           = config["base_lr"]           = getenv("LR", 8.4 * (BS/2048))
  lr_warmup_epochs  = config["lr_warmup_epochs"]  = getenv("WARMUP_EPOCHS", 5)
  decay             = config["decay"]             = getenv("DECAY", 2e-4)

  target, achieved  = getenv("TARGET", 0.759), False
  eval_start_epoch  = getenv("EVAL_START_EPOCH", 0)
  eval_epochs       = getenv("EVAL_EPOCHS", 1)

  steps_in_train_epoch  = config["steps_in_train_epoch"]  = (len(get_train_files()) // BS)
  steps_in_val_epoch    = config["steps_in_val_epoch"]    = (len(get_val_files()) // EVAL_BS)

  config["BEAM"]    = BEAM.value
  config["WINO"]    = WINO.value
  config["SYNCBN"]  = getenv("SYNCBN")

  # ** Optimizer **
  from examples.mlperf.optimizers import LARS
  skip_list = {v for k, v in get_state_dict(model).items() if "bn" in k or "bias" in k or "downsample.1" in k}
  optimizer = LARS(parameters, base_lr, momentum=.9, weight_decay=decay, skip_list=skip_list)

  # ** LR scheduler **
  scheduler = PolynomialDecayWithWarmup(optimizer, initial_lr=base_lr, end_lr=1e-4,
                                        train_steps=epochs * steps_in_train_epoch,
                                        warmup=lr_warmup_epochs * steps_in_train_epoch)
  print(f"training with batch size {BS} for {epochs} epochs")

  # ** resume from checkpointing **
  start_epoch = 0
  if ckpt:=getenv("RESUME", ""):
    load_training_state(model, optimizer, scheduler, safe_load(ckpt))
    start_epoch = int(scheduler.epoch_counter.numpy().item() / steps_in_train_epoch)
    print(f"resuming from {ckpt} at epoch {start_epoch}")

  # ** init wandb **
  WANDB = getenv("WANDB")
  if WANDB:
    import wandb
    wandb_args = {"id": wandb_id, "resume": "must"} if (wandb_id := getenv("WANDB_RESUME", "")) else {}
    wandb.init(config=config, **wandb_args)

  # ** jitted steps **
  input_mean = Tensor([123.68, 116.78, 103.94], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  # mlperf reference resnet does not divide by input_std for some reason
  # input_std = Tensor([0.229, 0.224, 0.225], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  def normalize(x): return x.permute([0, 3, 1, 2]) - input_mean
  @TinyJit
  def train_step(X, Y):
    optimizer.zero_grad()
    X = normalize(X)
    out = model.forward(X)
    loss = out.sparse_categorical_crossentropy(Y, label_smoothing=0.1)
    top_1 = (out.argmax(-1) == Y).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.realize(), top_1.realize()
  @TinyJit
  def eval_step(X, Y):
    X = normalize(X)
    out = model.forward(X)
    loss = out.sparse_categorical_crossentropy(Y, label_smoothing=0.1)
    top_1 = (out.argmax(-1) == Y).sum()
    return loss.realize(), top_1.realize()
  def data_get(it):
    x, y, cookie = next(it)
    return x.shard(GPUS, axis=0).realize(), Tensor(y, requires_grad=False).shard(GPUS, axis=0), cookie

  # ** epoch loop **
  for e in range(start_epoch, epochs):
    # ** train loop **
    Tensor.training = True
    it = iter(tqdm(batch_load_resnet(batch_size=BS, val=False, shuffle=True, seed=seed*epochs + e),
                   total=steps_in_train_epoch, desc=f"epoch {e}", disable=getenv("BENCHMARK")))
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

      if i == getenv("BENCHMARK"): return

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
        safe_save(get_training_state(model, optimizer, scheduler), fn)

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  from examples.mlperf.losses import dice_ce_loss
  from examples.mlperf.metrics import dice_score
  from extra.models.unet3d import UNet3D
  from extra.datasets.kits19 import iterate, get_train_files, get_val_files, sliding_window_inference
  from tinygrad import dtypes, Device, TinyJit, Tensor
  from tinygrad.nn.optim import SGD
  from tinygrad.nn.state import load_state_dict
  from tinygrad.ops import GlobalCounters

  import numpy as np
  import random

  TARGET_METRIC = 0.908
  NUM_EPOCHS = getenv("NUM_EPOCHS", 4000)
  BS = getenv("BS", 1)
  LR = getenv("LR", 0.8)
  MOMENTUM = getenv("MOMENTUM", 0.9)
  LR_WARMUP_EPOCHS = getenv("LR_WARMUP_EPOCHS", 200)
  LR_WARMUP_INIT_LR = getenv("LR_WARMUP_INIT_LR", 0.0001)
  EVAL_AT = getenv("EVAL_AT", 20)
  CHECKPOINT_EVERY = getenv("CHECKPOINT_EVERY", 10)
  CHECKPOINT_FN = getenv("CHECKPOINT_FN")
  WANDB = getenv("WANDB")
  PROJ_NAME = getenv("PROJ_NAME", "tinygrad_unet3d_mlperf")
  SIZE = (64, 64, 64) if getenv("SMALL") else (128, 128, 128)
  SEED = getenv("SEED")

  if getenv("HALF"): dtypes.default_float = dtypes.half

  if SEED:
    assert 1 <= SEED <= 9, "seed must be between 1-9"
    Tensor.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

  if WANDB:
    try:
      import wandb
    except ImportError:
      raise "Need to install wandb to use it"

  if ((num_gpus := getenv("GPUS")) > 1):
    print(f"Using {num_gpus} GPUS for training")
    GPUS = tuple([Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(num_gpus)])
    assert BS % len(GPUS) == 0, f"{BS=} is not a multiple of {len(GPUS)=}"
    for x in GPUS: Device[x]
  else:
    GPUS = tuple()

  model = UNet3D()

  if CHECKPOINT_FN:
    state_dict = safe_load(CHECKPOINT_FN)
    load_state_dict(model, state_dict)
    print(f"Loaded checkpoint {CHECKPOINT_FN} into model")

  params = get_parameters(model)

  if len(GPUS) > 1:
    for p in params:
      p.to_(GPUS)

  optim = SGD(params, lr=LR, momentum=MOMENTUM, nesterov=True)

  def _lr_warm_up(optim, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    optim.lr.assign(Tensor([init_lr + (lr - init_lr) * scale], device=GPUS if len(GPUS) > 1 else None))

  @TinyJit
  def _train_step(model, x, y):
    optim.zero_grad()

    y_hat = model(x)
    loss = dice_ce_loss(y_hat, y)

    loss.backward()
    optim.step()
    return loss.realize()
  
  def _eval_step(model, x, y):
    y_hat, y = sliding_window_inference(model, x, y)
    score = dice_score(Tensor(y_hat), Tensor(y)).mean().item()
    return score

  if WANDB: wandb.init(project=PROJ_NAME)

  for epoch in range(1, NUM_EPOCHS + 1):
    if epoch <= LR_WARMUP_EPOCHS and LR_WARMUP_EPOCHS > 0:
      _lr_warm_up(optim, LR_WARMUP_INIT_LR, LR, epoch, LR_WARMUP_EPOCHS)

    for x, y in (t:=tqdm(iterate(val=False, shuffle=True, bs=BS, size=SIZE), desc=f"[Training][Epoch: {epoch}/{NUM_EPOCHS}]", total=len(get_train_files()) // BS)):
      x, y = Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)

      if len(GPUS) > 1:
        x.shard_(GPUS, axis=0)
        y.shard_(GPUS, axis=0)

      loss = _train_step(model, x, y)
      t.set_description(f"[Training][Epoch: {epoch}/{NUM_EPOCHS}][Loss: {loss.item():.3f}][RAM used: {GlobalCounters.mem_used/1e9:5.2f} GB]")
      if WANDB: wandb.log({"train_loss": loss.item()})

    if epoch % CHECKPOINT_EVERY == 0:
      state_dict = get_state_dict(model)
      safe_save(state_dict, f"unet3d_mlperf_epoch_{epoch}.safetensors")
      print(f"Saved checkpoint at epoch {epoch}")

    if epoch % EVAL_AT == 0:
      with Tensor.train(val=False):
        scores = 0

        for i, (x, y) in enumerate((t:=tqdm(iterate(), desc=f"[Validation][Epoch: {epoch}/{NUM_EPOCHS}]", total=len(get_val_files()))), start=1):
          scores += _eval_step(model , x, y) # NOTE: passing in model instead since it is jitted
          t.set_description(f"[Validation][Epoch: {epoch}/{NUM_EPOCHS}][Mean DICE: {scores / i:.3f}]")

        scores /= i

        if WANDB: wandb.log({"val_mean_dice": scores})

        if scores >= TARGET_METRIC:
          print(f"Target metric ({TARGET_METRIC:.3f}) has been reached with validation metric of {scores:.3f}")
          print("Training complete")
          break
        else:
          print(f"Target metric ({TARGET_METRIC:.3f}) has not been reached with validation metric of {scores:.3f}")

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
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()


