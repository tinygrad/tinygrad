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

  def cross_entropy(out, Y, label_smoothing=0):
    num_classes = out.shape[-1]
    YY = Y.flatten().astype(np.int32)
    y = np.zeros((YY.shape[0], num_classes), np.float32)
    y[range(y.shape[0]),YY] = -1.0*num_classes
    y = y.reshape(list(Y.shape)+[num_classes])
    y = Tensor(y)
    return (1 - label_smoothing) * out.mul(y).mean() + (-1 * label_smoothing * out.mean())

  seed = getenv('SEED', 42)
  Tensor.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  wandb.init()

  num_classes = 1000
  model = ResNet50(num_classes)
  parameters = get_parameters(model)

  BS = 16
  lr = 0.256 * (BS / 256)  # Linearly scale from BS=256, lr=0.256
  epochs = 50
  optimizer = optim.SGD(parameters, lr, momentum=.875, weight_decay=1/2**15)
  scheduler = CosineAnnealingLR(optimizer, epochs)
  print(f"training with batch size {BS} for {epochs} epochs")

  def train_step(X, Y):
    optimizer.zero_grad()
    out = model.forward(X)
    loss = cross_entropy(out, Y, label_smoothing=0.1)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss, out
  
  def calculate_accuracy(out, Y, top_n):
    out_top_n = np.argpartition(out.cpu().numpy(), -top_n, axis=-1)[:, -top_n:]
    YY = np.expand_dims(Y, axis=1)
    YY = np.repeat(YY, top_n, axis=1)

    eq_elements = np.equal(out_top_n, YY)
    top_n_acc = np.count_nonzero(eq_elements) / eq_elements.size * top_n
    return top_n_acc

  Tensor.training = True
  steps_in_epoch = math.floor(len(get_val_files()) / BS)
  for epoch in range(epochs):
    for X, Y in (t := tqdm(iterate(bs=BS, val=True), total=steps_in_epoch)):
      X = Tensor(X, requires_grad=False)
      st = time.time()
      loss, out = train_step(X, Y)
      et = time.time()

      t.set_description(f"loss: {loss.numpy().item():.3f}")
      top_1_acc = calculate_accuracy(out, Y, 1)
      wandb.log({"loss": loss.numpy().item(), 
                 "top_1_acc": top_1_acc,
                 "lr": scheduler.get_lr().cpu().numpy().item(),
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


