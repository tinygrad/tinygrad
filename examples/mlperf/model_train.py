import os
import time
from tqdm import tqdm
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, dtypes
from tinygrad.nn.state import get_parameters, safe_load, safe_save, get_state_dict, load_state_dict
from tinygrad.nn.optim import SGD
from extra.lr_scheduler import MultiStepLR
from extra.datasets.kits19 import get_batch, sliding_window_inference, get_data_split
from extra import dist

from examples.mlperf.metrics import dice_ce_loss, get_dice_score_np
from examples.mlperf.conf import Conf

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  """Before running this script, please get the data using `extra/datasets/kits19.py` script."""

  def train_single_unet3d(conf):
    is_successful, diverged = False, False
    next_eval_at = conf.start_eval_at

    def evaluate(conf, model, loader, score_fn=get_dice_score_np, epoch=0):
      s, i = 0, 0
      for i, batch in enumerate(tqdm(loader, disable=not conf.verbose)):
        vol, label = batch
        out, label = sliding_window_inference(model, vol, label)
        label = label.squeeze(axis=1)
        score = score_fn(out, label).mean(axis=0)
        s += score
        del out, label

      val_dice_score = s / (i+1)
      return {"epoch": epoch, "mean_dice": val_dice_score}

    def lr_warmup(optim, init_lr, lr, current_epoch, warmup_epochs):
      scale = current_epoch / warmup_epochs
      optim.lr.assign(Tensor([init_lr + (lr - init_lr) * scale])).realize()

    from extra.models.unet3d import UNet3D
    mdl = UNet3D()
    if getenv("PRETRAINED"):
      mdl.load_from_pretrained()
    if getenv("CONTINUE"):
      load_state_dict(mdl, safe_load(os.path.join(conf.save_ckpt_path, f"unet3d-ckpt-{conf.start_epoch}.safetensors")))
    if getenv("FP16"):
      weights = get_state_dict(mdl)
      for k, v in weights.items():
        weights[k] = v.cpu().half()
      load_state_dict(mdl, weights)
    print("UNet3D params: {:,.0f}".format(sum([p.numel() for p in get_parameters(mdl)])))

    from tinygrad.jit import TinyJit
    mdl_run = TinyJit(lambda x: mdl(x).realize()) if getenv("JIT") else mdl

    is_successful, diverged = False, False
    optim = SGD(get_parameters(mdl), lr=conf.lr, momentum=conf.momentum, nesterov=True, weight_decay=conf.weight_decay)

    if conf.lr_decay_epochs:
      scheduler = MultiStepLR(optim, milestones=conf.lr_decay_epochs, gamma=conf.lr_decay_factor)

    if getenv("MOCKTRAIN", 0):
      import numpy as np
      train_loader = [(np.random.rand(2,1,32,32,32), np.random.rand(2,1,32,32,32)) for i in range(3)]
      total_batches = 1
    else:
      train_x, train_y, val_x, val_y = get_data_split()
      total_files = len(train_x)
      total_batches = (total_files + conf.batch_size - 1) // conf.batch_size

    @TinyJit
    def train_step(out, y):
      with Tensor.train():
        optim.zero_grad()
        loss = dice_ce_loss(out, y)
        loss.backward()
        optim.step()
        return loss.realize()

    t0_total = time.monotonic()
    for epoch in range(conf.start_epoch, conf.epochs):
      cumulative_loss = []
      if epoch <= conf.lr_warmup_epochs and conf.lr_warmup_epochs > 0:
        lr_warmup(optim, conf.init_lr, conf.lr, epoch, conf.lr_warmup_epochs)

      if not getenv("MOCKTRAIN"):
        train_loader = get_batch(train_x, train_y, conf.batch_size, conf.input_shape, conf.oversampling)
        val_loader = get_batch(val_x, val_y, batch_size=1, shuffle=False, augment=False)

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

      print(f'  (train) epoch {epoch} | loss: {(sum(cumulative_loss).numpy() / len(cumulative_loss)):.6f}')
      if epoch == next_eval_at:
        next_eval_at += conf.eval_every
        dtype_im = dtypes.half if getenv("FP16") else dtypes.float

        eval_metrics = evaluate(conf, mdl, val_loader, epoch=epoch)
        print("  (eval):", eval_metrics)

        # safe_save(get_state_dict(mdl), os.path.join(conf.save_ckpt_path, f"unet3d-ckpt-{epoch}.safetensors"))
        Tensor.training = True
        if eval_metrics["mean_dice"] >= conf.quality_threshold:
          print("\nsuccess", eval_metrics["mean_dice"], ">", conf.quality_threshold, "runtime", time.monotonic()-t0_total)
          safe_save(get_state_dict(mdl), os.path.join(conf.save_ckpt_path, f"unet3d-ckpt-{conf.epochs}.safetensors"))
          is_successful = True
        elif eval_metrics["mean_dice"] < 1e-6:
          print("\nmodel diverged. exit.", eval_metrics["mean_dice"], "<", 1e-6)
          diverged = True

      if is_successful or diverged:
        break
      print(f'  epoch time {time.monotonic()-epoch_st:.2f}s')

    safe_save(get_state_dict(mdl), os.path.join(conf.save_ckpt_path, f"unet3d-ckpt-{conf.epochs}.safetensors"))
    print(f"  total runtime {time.monotonic()-t0_total:.1f} | epochs {conf.epochs}")

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
