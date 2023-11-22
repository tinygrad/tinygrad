from tqdm import tqdm
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.nn.state import get_parameters
from tinygrad.ops import Device
from extra.lr_scheduler import MultiStepLR

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  device = Device.DEFAULT

  train_loader = None
  rank, world_size = getenv("RANK"), getenv("WORLD_SIZE", 1)

  flags = {"epochs": 1, "ga_steps": 1, "warmup_step": 4, "batch_size": 2, "optimizer": "sgd", "init_lr": 1e-4, "lr": 0.1, "lr_decay_epochs": [], "weight_decay": 0.0, "momentum": 0.9, "verbose": False}

  def get_optimizer(params, flags: dict):
    from tinygrad.nn.optim import Adam, SGD
    if flags["optimizer"] == "adam":
      optim = Adam(params, lr=flags["lr"], weight_decay=flags["weight_decay"])
    elif flags["optimizer"] == "sgd":
      optim = SGD(params, lr=flags["lr"], momentum=flags["momentum"], nesterov=True, weight_decay=flags["weight_decay"])
    elif flags["optimizer"] == "lamb":
      pass
      # import apex
      # optim = apex.optimizers.FusedLAMB(params, lr=flags.learning_rate, betas=flags.lamb_betas, weight_decay=flags.weight_decay)
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

  optimizer = get_optimizer(get_parameters(mdl), flags)
  if flags["lr_decay_epochs"]:
    scheduler = MultiStepLR(optimizer, milestones=flags["lr_decay_epochs"], gamma=flags["lr_decay_factor"])
  # scaler = GradScaler()

  for epoch in range(1, flags["epochs"] + 1):
    cumulative_loss = []

    loss_value = None
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(train_loader, disable=(rank != 0) or not flags["verbose"])):
      im, label = batch
      im, label = im.to(device), label.to(device)
      # for callback in callbacks:
      #   callback.on_batch_start()

#       with autocast(enabled=flags["amp"]):
      out = mdl(im)
      loss_value = loss_fn(out, label)
      loss_value /= flags.ga_steps

      if flags["amp"]:
        scaler.scale(loss_value).backward()
      else:
        loss_value.backward()

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
