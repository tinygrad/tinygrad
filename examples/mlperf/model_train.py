import time
from tqdm import tqdm
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, dtypes
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
from tinygrad.ops import Device
from tinygrad.shape.symbolic import Node
from extra.lr_scheduler import MultiStepLR
from extra.datasets.kits19 import get_batch, get_val_files, sliding_window_inference

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
  next_eval_at = conf.start_eval_at

  def evaluate(conf, model, loader, score_fn, epoch=0):
    s, i = 0, 0
    for i, batch in enumerate(tqdm(loader, disable=not conf.verbose)):
      vol, label = batch
      vol, label = vol.numpy(), label.numpy()
      output, label = sliding_window_inference(model, vol, label)
      label = label.squeeze(axis=1)
      score = score_fn(output, label).mean()
      s += score
      del output, label

    val_dice_score = s / (i+1)

    return {"epoch": epoch, "mean_dice": val_dice_score}

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
    total_batches = 1
  else:
    def get_train_val_split(files): return files[:-int(len(files)*conf.val_split)], files[-int(len(files)*conf.val_split):]
    files = get_val_files()
    train_files, val_files = get_train_val_split(files)
    total_files = len(train_files)
    total_batches = (total_files + conf.batch_size - 1) // conf.batch_size
    train_loader = get_batch(train_files, conf.batch_size, conf.input_shape, conf.oversampling)
    val_loader = get_batch(val_files, 1, conf.val_input_shape, conf.oversampling)

  @TinyJit
  def train_step(out, y):
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
    start_time_epoch = time.time()

    for i, batch in enumerate(tqdm(train_loader, total=total_batches, disable=(rank != 0) or not conf.verbose)):
      im, label = batch

      dtype_im = dtypes.half if getenv("FP16") else dtypes.float
      im, label = Tensor(im, dtype=dtype_im), Tensor(label, dtype=dtype_im)

      out = mdl_run(im)
      del im

      loss_value = train_step(out, label)
      cumulative_loss.append(loss_value)

    if conf.lr_decay_epochs:
      scheduler.step()
    print(f'loss for epoch {epoch} {sum(cumulative_loss) / len(cumulative_loss)}')

    if epoch == next_eval_at:
      next_eval_at += conf.evaluate_every
      dtype_im = dtypes.half if getenv("FP16") else dtypes.float

      eval_model = lambda x : mdl_run(Tensor(x, dtype=dtype_im)).numpy()
      eval_metrics = evaluate(conf, eval_model, val_loader, epoch=epoch)
      eval_metrics["train_loss"] = sum(cumulative_loss) / len(cumulative_loss)
      print(eval_metrics)

      Tensor.training = True
      print('eval_metrics', [(k, f"{m:.7f}") for k,m in eval_metrics.items()])
      if eval_metrics["mean_dice"] >= conf.quality_threshold:
        print("success", eval_metrics["mean_dice"], ">", conf.quality_threshold)
        is_successful = True
      elif eval_metrics["mean_dice"] < 1e-6:
        print("model diverged. exit.", eval_metrics["mean_dice"], "<", 1e-6)
        diverged = True

    if is_successful or diverged:
      break
    print('epoch time', time.time()-start_time_epoch)

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
