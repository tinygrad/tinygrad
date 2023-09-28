import random
import time
import logging

import numpy as np
import torch
from tqdm import tqdm

from examples.mlperf.metrics import dice_ce_loss, get_dice_score_np
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

def evaluate(flags, model, loader, score_fn=get_dice_score_np, epoch=0):
    s, i = 0, 0
    for i, batch in enumerate(tqdm(loader, disable=not flags.verbose)):
      image, label = batch
      image, label = image.numpy(), label.numpy()
      output, label = sliding_window_inference(model, image, label)
      label = label.squeeze(axis=1)
      score = score_fn(output, label).mean()
      s += score
      del output, label

    val_dice_score = s / (i+1)

    return {"epoch": epoch, "mean_dice": val_dice_score}

def train(flags, model:UNet3D, train_loader, val_loader, loss_fn):
  time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
  logging.basicConfig(filename=f'train_log_{time_str}.log', level=logging.INFO)
  logging.info('START')

  is_successful, diverged = False, False
  optimizer = optim.SGD(get_parameters(model), lr=flags.learning_rate, momentum=flags.momentum, weight_decay=flags.weight_decay)
  if flags.lr_decay_epochs:
    scheduler = MultiStepLR(optimizer, milestones=flags.lr_decay_epochs, gamma=flags.lr_decay_factor) # not used at the moment
  next_eval_at = flags.start_eval_at

  jit_model = TinyJit(lambda x: model(x).realize()) if getenv("JIT") else model

  def training_step(model_output, label, lr):
    print("No jit (yet)")
    optimizer.lr = lr
    loss_value = loss_fn(model_output, label)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    return loss_value.realize()
  training_step_fn = TinyJit(training_step) if getenv("JIT") else training_step

  for epoch in range(1, flags.max_epochs + 1):
    cumulative_loss = []
    if epoch <= flags.lr_warmup_epochs:
      lr_warmup(optimizer, flags.init_learning_rate, flags.learning_rate, epoch, flags.lr_warmup_epochs)
    start_time_epoch = time.time() # for 19 steps its currently ~5 seconds.
    for iteration, batch in enumerate(tqdm(train_loader, disable=not flags.verbose)):
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
      loss_value = loss_value.numpy()
      cumulative_loss.append(loss_value)

      if flags.lr_decay_epochs:
        scheduler.step()
    logging.info(f'loss_value epoch {epoch} {sum(cumulative_loss) / len(cumulative_loss)}')

    if epoch == next_eval_at and getenv("EVAL", 1):
      next_eval_at += flags.evaluate_every
      dtype_img = dtypes.half if getenv("FP16") else dtypes.float

      eval_model = lambda x : jit_model(Tensor(x, dtype=dtype_img)).numpy()
      eval_metrics = evaluate(flags, eval_model, val_loader, epoch=epoch)
      eval_metrics["train_loss"] = sum(cumulative_loss) / len(cumulative_loss)
      logging.info(eval_metrics)
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
  flags = Flags(batch_size=getenv("BATCH", 1), verbose=True, data_dir=getenv("DATA_DIR", kits19.BASEDIR))#os.environ["KITS19_DATA_DIR"])
  # flags = Flags(batch_size=getenv("BATCH", 1), verbose=True, data_dir=getenv("DATA_DIR", '/home/gijs/code_projects/tinyrad/extra/datasets/kits19/data'))#os.environ["KITS19_DATA_DIR"])
  flags.num_workers = 0 if getenv("SPEED", 2) > 0 else 1
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