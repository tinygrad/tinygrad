from tqdm import tqdm
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, dtypes
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
from tinygrad.ops import Device
from tinygrad.shape.symbolic import Node
from extra.lr_scheduler import MultiStepLR
from examples.mlperf.metrics import get_dice_score

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  device = Device.DEFAULT

  def cross_entropy_loss(x:Tensor, y:Tensor, reduction:str='mean', label_smoothing:float=0.0) -> Tensor:
    divisor = y.shape[1]
    assert not isinstance(divisor, Node), "sint not supported as divisor"
    y = (1 - label_smoothing)*y + label_smoothing / divisor
    if reduction=='none': return -x.log_softmax(axis=1).mul(y).sum(axis=1)
    if reduction=='sum': return -x.log_softmax(axis=1).mul(y).sum(axis=1).sum()
    return -x.log_softmax(axis=1).mul(y).sum(axis=1).mean()

  def dice_ce_loss(out, label):
    ce = cross_entropy_loss(out, label)
    dice_score = get_dice_score(out, label)
    dice = (1. - dice_score).mean()
    return (ce + dice) / 2

  train_loader = None
  rank, world_size = getenv("RANK"), getenv("WORLD_SIZE", 1)

  flags = {"epochs": 1, "ga_steps": 1, "warmup_step": 4, "batch_size": 2, "optimizer": "sgd", "init_lr": 1e-4, "lr": 0.1, "lr_decay_epochs": [], "weight_decay": 0.0, "momentum": 0.9, "verbose": False}

  def get_batch(generator, batch_size=32):
    bX, bY = [], []
    for _ in range(batch_size):
      try:
        X, Y = next(generator)
        bX.append(X)
        bY.append(X)
      except StopIteration:
        break
    return np.concatenate(bX, axis=0), np.concatenate(bY, axis=0)

  def get_optimizer(params, flags: dict):
    from tinygrad.nn.optim import Adam, SGD
    if flags["optimizer"] == "adam":
      optim = Adam(params, lr=flags["lr"], weight_decay=flags["weight_decay"])
    elif flags["optimizer"] == "sgd":
      optim = SGD(params, lr=flags["lr"], momentum=flags["momentum"], nesterov=True, weight_decay=flags["weight_decay"])
    elif flags["optimizer"] == "lamb":
      pass
    else:
      raise ValueError("Optimizer {} unknown.".format(flags["optimizer"]))
    return optim

  def lr_warmup(optimizer, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr + (lr - init_lr) * scale

  # model init
  from extra.models.unet3d import UNet3D
  mdl = UNet3D()
  if getenv("PRETRAINED"):
    mdl.load_from_pretrained()
  if getenv("FP16"):
    weights = get_state_dict(mdl)
    for k, v in weights.items():
      weights[k] = v.cpu().half()
    load_state_dict(mdl, weights)
  print("Model params: {:,.0f}".format(sum([p.numel() for p in get_parameters(mdl)])))

  from tinygrad.jit import TinyJit
  mdl_run = TinyJit(lambda x: mdl(x).realize())

  is_successful, diverged = False, False
  optimizer = get_optimizer(get_parameters(mdl), flags)
  if flags["lr_decay_epochs"]:
    scheduler = MultiStepLR(optimizer, milestones=flags["lr_decay_epochs"], gamma=flags["lr_decay_factor"])

  if getenv("TESTTRAIN", 0):
    train_loader = [(Tensor.rand(1,1,128,128,128), Tensor.rand(1,1,128,128,128)) for i in range(3)]
  else:
    train_loader = [(Tensor.rand(1,1,128,128,128), Tensor.rand(1,1,128,128,128)) for i in range(3)]

  for epoch in range(1, flags["epochs"] + 1):
    cumulative_loss = []

    loss_value = None
    optimizer.zero_grad()
    # for i, batch in enumerate(tqdm(train_loader, disable=(rank != 0) or not flags["verbose"])):
    for i, batch in enumerate(tqdm(train_loader)):
      im, label = batch
      print(im.shape, label.shape)

      im, label = im.to(device), label.to(device)
      out = mdl(im).numpy()

      loss_value = dice_ce_loss(out, label)
      loss_value /= flags["ga_steps"]
      print("loss", loss_value)

      loss_value.backward()
      optimizer.step()
      optimizer.zero_grad()

      cumulative_loss.append(loss_value)

    if flags["lr_decay_epochs"]:
      scheduler.step()

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
