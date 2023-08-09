from tqdm import tqdm
from tinygrad.helpers import getenv
from tinygrad.nn import optim
from tinygrad.state import get_parameters
from tinygrad.tensor import Tensor

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d(target=0.908, roi_shape=(128,128,128)):
  from examples.mlperf.metrics import dice_ce_loss, get_dice_score, one_hot
  from extra.datasets.kits19 import (get_train_files, get_val_files, iterate,
                                     sliding_window_inference)
  from extra.training import lr_warmup
  from models.unet3d import UNet3D
  Tensor.training = True
  in_channels, n_class, BS = 1, 3, 2
  mdl = UNet3D(in_channels, n_class)
  lr_warmup_epochs = 200
  init_lr, lr = 1e-4, 0.8
  max_epochs = 4000
  opt = optim.SGD(get_parameters(mdl), lr=init_lr)
  for epoch in range(max_epochs):
    if epoch <= lr_warmup_epochs and lr_warmup_epochs > 0:
      lr_warmup(opt, init_lr, lr, epoch, lr_warmup_epochs)
    for image, label in (t := tqdm(iterate(BS=BS, val=False, roi_shape=roi_shape), total=len(get_train_files())//BS)):
      opt.zero_grad()
      out = mdl(Tensor(image).half())
      loss = dice_ce_loss(out, label, n_class)
      loss.backward()
      opt.step()
      t.set_description(f"loss {loss.numpy().item()}")
    if (epoch + 1) % 20 == 0:
      Tensor.training = False
      s = 0
      for image, label in iterate(BS=BS, val=True):
        pred, label = sliding_window_inference(mdl, image, label, roi_shape)
        label = one_hot(label, n_class)
        label = Tensor(label, requires_grad=False)
        s += get_dice_score(pred, label).mean().numpy()
      val_dice_score = s / len(get_val_files())
      print(f"[Epoch {epoch}] Val dice score: {val_dice_score:.4f}. Target: {target}")
      Tensor.training = True
      if val_dice_score >= target:
        break

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


