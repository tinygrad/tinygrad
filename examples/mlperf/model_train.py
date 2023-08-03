from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import getenv, dtypes
from tqdm import tqdm
import numpy as np
import math
import random
import wandb
import time

def train_resnet():
  # TODO: Resnet50-v1.5
  from models.resnet import ResNet50
  from extra.datasets.imagenet import iterate, get_val_files
  from extra.lr_scheduler import CosineAnnealingLR
  from extra.training import sparse_categorical_crossentropy

  seed = getenv('SEED', 42)
  Tensor.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  wandb.init()

  num_classes = 1000
  model = ResNet50(num_classes)
  parameters = get_parameters(model)

  BS = 16
  lr = BS*1e-3
  epochs = 50
  optimizer = optim.SGD(parameters, lr, momentum=.875, weight_decay=1/2**15)
  scheduler = CosineAnnealingLR(optimizer, epochs)
  print(f"training with batch size {BS} for {epochs} epochs")

  @TinyJit
  def train_step(X, Y):
    optimizer.zero_grad()
    out = model.forward(X)
    loss = sparse_categorical_crossentropy(out, Y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.realize(), out

  Tensor.training = True
  steps_in_epoch = math.floor(len(get_val_files()) / BS)
  for epoch in range(epochs):
    for X, Y in (t := tqdm(iterate(bs=BS, val=True), total=steps_in_epoch)):
      X = Tensor(X, requires_grad=False)
      st = time.time()
      loss, out = train_step(X, Y)
      et = time.time()

      cat = np.argmax(out.cpu().numpy(), axis=-1)
      accuracy = (cat == Y).mean()
      t.set_description(f"loss: {loss.numpy().item():.3f} | acc: {accuracy:.2f}")

      wandb.log({"loss": loss.numpy().item(), 
                 "acc": accuracy,
                 "forward_time": et - st,}
      )

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
  # TODO: BERT
  pass

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  Tensor.training = True

  for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
    nm = f"train_{m}"
    if nm in globals():
      print(f"training {m}")
      globals()[nm]()


