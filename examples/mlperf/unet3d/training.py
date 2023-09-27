import random
import time

import numpy as np
import torch
from tqdm import tqdm

from examples.mlperf.metrics import dice_ce_loss, get_dice_score, get_dice_score_np
from examples.mlperf.unet3d.inference import evaluate
from extra.datasets import kits19
from extra.datasets.kits19 import sliding_window_inference
from extra.lr_scheduler import MultiStepLR
from extra.training import lr_warmup
from tinygrad.helpers import dtypes, getenv
from tinygrad.jit import TinyJit

from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
from tinygrad.ops import Device
from tinygrad.tensor import Tensor

from examples.mlperf.unet3d.data_loader import get_data_loaders
from examples.mlperf.unet3d.flags import Flags
from models.unet3d import UNet3D

def train(flags, model:UNet3D, train_loader, val_loader, loss_fn):
  is_successful, diverged = False, False
  optimizer = optim.SGD(get_parameters(model), lr=flags.learning_rate, momentum=flags.momentum, weight_decay=flags.weight_decay)
  # scaler = GradScaler() # scalar is only needed when doing mixed precision. The default args have this disabled.
  if flags.lr_decay_epochs:
    scheduler = MultiStepLR(optimizer, milestones=flags.lr_decay_epochs, gamma=flags.lr_decay_factor)
  next_eval_at = flags.start_eval_at

  jit_model = TinyJit(lambda x: model(x).realize()) if getenv("JIT") else model # todo this might be nicer. Such that it can also be used for evaluate

  def training_step(model_output, label, lr):
    optimizer.zero_grad()
    print("No jit (yet)")
    optimizer.lr = lr
    loss_value = loss_fn(model_output, label)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    return loss_value.realize() # loss_value.realize is needed
  training_step_fn = TinyJit(training_step) if getenv("JIT") else training_step
  if getenv("OVERFIT"):
    loader = [(0, next(iter(train_loader)))]

  for epoch in range(1, flags.max_epochs + 1):
    Tensor.training = True
    Tensor.no_grad = False

    print('epoch', epoch)
    cumulative_loss = []
    if epoch <= flags.lr_warmup_epochs:
      lr_warmup(optimizer, flags.init_learning_rate, flags.learning_rate, epoch, flags.lr_warmup_epochs)
    loss_value = None
    if not getenv("OVERFIT"):
      loader = enumerate(tqdm(train_loader, disable=not flags.verbose))
      # print('len(loader)', len(loader))
      # loader = loader[:4]
    # 1 EPOCH with FILTERS="8 64 128 256 320" FP16=0 With 24 images currently takes 31 seconds. ~1.3 seconds per step. LOSS is actually going down!! To around 0.3 - 0.5
    # dice score is now also at 0.3-0.4 mean: 0.298
    start_time_epoch = time.time() # for 19 steps its currently ~5 seconds.
    for iteration, batch in loader:
      # print('optimizer.lr', optimizer.lr.numpy())
      image, label = batch

      dtype_img = dtypes.half if getenv("FP16") else dtypes.float
      image, label = Tensor(image.numpy(), dtype=dtype_img), Tensor(label.numpy(), dtype=dtype_img)

      output = jit_model(image)
      if getenv("MODEL_OUT", 0):
        output = jit_model(image)
        exit()
      del image
      loss_value = training_step_fn(output, label, optimizer.lr)
      del output, label
      # print('grad', loss_value.grad.numpy())
      cumulative_loss.append(loss_value)
      print('loss_value', loss_value.numpy())
      if flags.lr_decay_epochs:
        scheduler.step()

    if epoch == next_eval_at and getenv("EVAL", 1):
      next_eval_at += flags.evaluate_every
      Tensor.training = False
      Tensor.no_grad = True
      dtype_img = dtypes.half if getenv("FP16") else dtypes.float

      eval_model = lambda x : model(Tensor(x, dtype=dtype_img)).numpy() # todo maybe change
      eval_metrics = evaluate(flags, eval_model, val_loader, epoch=epoch)
      eval_metrics["train_loss"] = (sum(cumulative_loss) / len(cumulative_loss)).numpy().item()

      Tensor.training = True
      print('eval_metrics', [(k, f"{m:.7f}") for k,m in eval_metrics.items()])
      if eval_metrics["mean_dice"] >= flags.quality_threshold:
        print("SUCCESSFULL", eval_metrics["mean_dice"], ">", flags.quality_threshold)
        # is_successful = True
      elif eval_metrics["mean_dice"] < 1e-6:
        print("MODEL DIVERGED. ABORTING.", eval_metrics["mean_dice"], "<", 1e-6)
        # diverged = True

    if is_successful or diverged:
      break
    print('epoch time', time.time()-start_time_epoch)

if __name__ == "__main__":
  print('Device', Device.DEFAULT)
  # batch_size 2 is default: https://github.com/mlcommons/training/blob/00f04c57d589721aabce4618922780d29f73cf4e/image_segmentation/pytorch/runtime/arguments.py
  # this is the real starting script: https://github.com/mlcommons/training/blob/00f04c57d589721aabce4618922780d29f73cf4e/image_segmentation/pytorch/run_and_time.sh
  # todo check batch size?
  print(kits19.BASEDIR)
  flags = Flags(batch_size=getenv("BATCH", 1), verbose=True, data_dir=getenv("DATA_DIR", kits19.BASEDIR))#os.environ["KITS19_DATA_DIR"])
  # flags = Flags(batch_size=getenv("BATCH", 1), verbose=True, data_dir=getenv("DATA_DIR", '/home/gijs/code_projects/tinyrad/extra/datasets/kits19/data'))#os.environ["KITS19_DATA_DIR"])
  flags.num_workers = 0 # for debugging
  seed = flags.seed
  flags.evaluate_every = getenv("EVAL_STEPS", 100) # todo
  flags.start_eval_at = getenv("EVAL_START", 1) # todo
  if seed is not None:
    Tensor._seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
  model = UNet3D(debug_speed=getenv("SPEED", 2), filters=getenv("FILTERS", ()))
  if getenv("PRETRAINED"):
    model.load_from_pretrained()
  if getenv("FP16"):
    weights = get_state_dict(model)
    for k, v in weights.items():
      weights[k] = v.cpu().half()
    load_state_dict(model, weights)
  print("Model params: {:,.0f}".format(sum([p.numel() for p in get_parameters(model)])))

  train_loader, val_loader = get_data_loaders(flags, 1, 0) # todo change to tinygrad loader
  loss_fn = dice_ce_loss

  # score_fn = get_dice_score # these might work better and are much simpler
  if getenv("OVERFIT"):
    val_loader = train_loader
  train(flags, model, train_loader, val_loader, loss_fn)
# FP16=1 JIT=1 python training.py
# DATA_DIR=kits19/data_processed SPEED=1 FP16=1 JIT=1 python training.py
# HIP=1 WINO=1 DATA_DIR=kits19/data_processed SPEED=0 FP16=1 JIT=1 python training.py
# reference: https://github.com/mlcommons/training/blob/00f04c57d589721aabce4618922780d29f73cf4e/image_segmentation/pytorch/model/losses.py#L63
# todo eventually cleanup duplicate stuff. There is also things in extra/kits19