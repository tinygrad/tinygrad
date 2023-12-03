import time
from tqdm import tqdm
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, dtypes
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
from tinygrad.shape.symbolic import Node
from extra.lr_scheduler import MultiStepLR
from extra.datasets.kits19 import get_batch, get_val_batch, get_val_files, sliding_window_inference
from extra import dist

from examples.mlperf.metrics import get_dice_score, get_dice_score_np
from examples.mlperf.conf import Conf

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  import pycuda.autoinit

  def train_single_unet3d(conf):
    is_successful, diverged = False, False
    next_eval_at = conf.start_eval_at

    def evaluate(conf, model, loader, score_fn=get_dice_score_np, epoch=0):
      s, i = 0, 0
      for i, batch in enumerate(tqdm(loader, disable=not conf.verbose)):
        vol, label = batch
        out, label = sliding_window_inference(model, vol, label)
        label = label.squeeze(axis=1)
        score = score_fn(out, label).mean()
        s += score
        del out, label

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

    from extra.models.unet3d import UNet3D
    mdl = UNet3D()
    if getenv("PRETRAINED"):
      mdl.load_from_pretrained()
    if getenv("FP16"):
      weights = get_state_dict(mdl)
      for k, v in weights.items():
        weights[k] = v.cpu().half()
      load_state_dict(mdl, weights)
    print("UNet3D params: {:,.0f}".format(sum([p.numel() for p in get_parameters(mdl)])))

    from tinygrad.jit import TinyJit
    mdl_run = TinyJit(lambda x: mdl(x).realize())

    is_successful, diverged = False, False
    optim = get_optimizer(get_parameters(mdl), conf)

    if conf.lr_decay_epochs:
      scheduler = MultiStepLR(optim, milestones=conf.lr_decay_epochs, gamma=conf.lr_decay_factor)

    if getenv("MOCKTRAIN", 0):
      train_loader = [(Tensor.rand((1,1,128,128,128), dtype=dtypes.half), Tensor.rand((1,128,128,128), dtype=dtypes.uint8)) for i in range(3)]
      total_batches = 1
    else:
      def get_train_val_split(files): return files[:-int(len(files)*conf.val_split)], files[-int(len(files)*conf.val_split):]
      files = get_val_files()
      train_files, val_files = get_train_val_split(files)
      total_files = len(train_files)
      total_batches = (total_files + conf.batch_size - 1) // conf.batch_size

    @TinyJit
    def train_step(out, y):
      with Tensor.train():
        optim.zero_grad()
        loss = dice_ce_loss(out, y)
        loss.backward()
        # if noloss: del loss
        optim.step()
        # if noloss: return None
        return loss.realize()

    for epoch in range(0, conf.epochs):
      cumulative_loss = []

      train_loader = get_batch(train_files, conf.batch_size, conf.input_shape, conf.oversampling)
      val_loader = get_val_batch(val_files)
      epoch_st = time.monotonic()
      for i, batch in enumerate(tqdm(train_loader, total=total_batches)):
        im, label = batch

        dtype_im = dtypes.half if getenv("FP16") else dtypes.float
        im, label = Tensor(im, dtype=dtype_im), Tensor(label, dtype=dtypes.uint8)

        out = mdl_run(im)
        del im

        loss_value = train_step(out, label)
        cumulative_loss.append(loss_value.detach())

      if conf.lr_decay_epochs:
        scheduler.step()

      if len(cumulative_loss):
        print(f'  (train) epoch {epoch} | loss: {(sum(cumulative_loss).numpy() / len(cumulative_loss)):.6f}')

      if epoch == next_eval_at:
        next_eval_at += conf.eval_every
        dtype_im = dtypes.half if getenv("FP16") else dtypes.float

        eval_metrics = evaluate(conf, mdl, val_loader, epoch=epoch)
        print("  (eval):", eval_metrics)

        Tensor.training = True
        if eval_metrics["mean_dice"] >= conf.quality_threshold:
          print("\nsuccess", eval_metrics["mean_dice"], ">", conf.quality_threshold)
          is_successful = True
        elif eval_metrics["mean_dice"] < 1e-6:
          print("\nmodel diverged. exit.", eval_metrics["mean_dice"], "<", 1e-6)
          diverged = True

      if is_successful or diverged:
        break
      print(f'  epoch time {time.monotonic()-epoch_st:.2f}s', )

  conf = Conf()
  if not getenv("DIST"):
    train_single_unet3d(conf)
  else:
    if getenv("CUDA"):
      pass
    else:
      from tinygrad.runtime.ops_gpu import CL
      devices = [f"gpu:{i}" for i in range(len(CL.devices))]
    world_size = len(devices)

    # ensure that the batch size is divisible by the number of devices
    assert conf.batch_size % world_size == 0, f"batch size {conf.batch_size} is not divisible by world size {world_size}"
    # init out-of-band communication
    dist.init_oob(world_size)
    # start the processes
    processes = []
    for rank, device in enumerate(devices):
      processes.append(dist.spawn(rank, device, fn=train_single_unet3d, args=(conf)))
    for p in processes: p.join()

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


