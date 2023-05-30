import time
from tqdm import tqdm
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.nn import optim

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d(target=0.908):
  from models.unet3d import UNet3D
  from datasets.kits19 import iterate, get_train_files, get_val_files, sliding_window_inference
  from examples.mlperf.metrics import dice_ce_loss, dice_score
  mdl = UNet3D()
  for i in range(4000):
    opt = optim.SGD(optim.get_parameters(mdl), lr=0.8)
    for image, label in (t := tqdm(iterate(val=False), total=len(get_train_files()))):
      opt.zero_grad()
      out = mdl(Tensor(image))
      loss = dice_ce_loss(out, label)
      loss.backward()
      opt.step()
      t.set_description(f"loss {loss.numpy().item()}")
    if (i + 1) % 20 == 0:
      s = 0
      for image, label in iterate():
        mt = time.perf_counter()
        pred, label = sliding_window_inference(mdl, image, label)
        s += dice_score(pred, label).mean()
      if s / len(get_val_files()) >= target:
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
