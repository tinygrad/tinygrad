from tqdm import tqdm
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, dtypes
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
from tinygrad.ops import Device
from tinygrad.shape.symbolic import Node
from extra.lr_scheduler import MultiStepLR
from extra.datasets.kits19 import get_batch, get_val_files

from examples.mlperf.metrics import get_dice_score
from examples.mlperf.conf import Conf

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  device = Device.DEFAULT
  conf = Conf()
  rank, world_size = getenv("RANK"), getenv("WORLD_SIZE", 1)
  is_successful, diverged = False, False

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

  def get_optimizer(params, conf: dict):
    from tinygrad.nn.optim import Adam, SGD, LAMB
    if conf.optimizer == "adam":
      optim = Adam(params, lr=conf.lr, weight_decay=conf.weight_decay)
    elif conf.optimizer == "sgd":
      optim = SGD(params, lr=conf.lr, momentum=conf.momentum, nesterov=True, weight_decay=conf.weight_decay)
    elif conf.optimizer == "lamb":
      optim = LAMB(params, lr=conf.lr, weight_decay=conf.weight_decay)
    else:
      raise ValueError("Optimizer {} unknown.".format(conf.optimizer))
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
  optim = get_optimizer(get_parameters(mdl), conf)
  if conf.lr_decay_epochs:
    scheduler = MultiStepLR(optim, milestones=conf.lr_decay_epochs, gamma=conf.lr_decay_factor)

  if getenv("MOCKTRAIN", 0):
    train_loader = [(Tensor.rand(1,1,128,128,128), Tensor.rand(1,1,128,128,128)) for i in range(3)]
  else:
    total_files = len(get_val_files())
    total_batches = (total_files + conf.batch_size - 1) // conf.batch_size
    train_loader = get_batch(conf.batch_size, conf.input_shape, conf.oversampling)

  @TinyJit
  def train_step(im, y):
    out = mdl_run(im).numpy()
    loss = dice_ce_loss(out, y)
    optim.zero_grad()
    loss.backward()
    # if noloss: del loss
    optim.step()
    # if noloss: return None
    return loss.realize()

  for epoch in range(1, conf.epochs + 1):
    cumulative_loss = []
    # if epoch <= conf.lr_warmup_epochs:
    #   lr_warmup(optim, conf.init_lr, conf.lr, epoch, conf.lr_warmup_epochs)

    for i, batch in enumerate(tqdm(train_loader, total=total_batches, disable=(rank != 0) or not conf.verbose)):
      im, label = batch
      print(im.shape, label.shape)

      dtype_im = dtypes.half if getenv("FP16") else dtypes.float
      im, label = Tensor(im, dtype=dtype_im), Tensor(label, dtype=dtype_im)

      loss_value = train_step(im, label)
      cumulative_loss.append(loss_value)

    if conf.lr_decay_epochs:
      scheduler.step()
    print(f'loss for epoch {epoch} {sum(cumulative_loss) / len(cumulative_loss)}')

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
