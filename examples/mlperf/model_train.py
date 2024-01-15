from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  from examples.mlperf.losses import dice_ce_loss
  from extra.models.unet3d import UNet3D
  from extra.datasets.kits19 import iterate, get_train_files
  from tinygrad import dtypes
  from tinygrad import TinyJit
  import tinygrad.nn as nn
  from tqdm import tqdm

  epochs = getenv("NUM_EPOCHS", 4000)
  bs = getenv("BS", 2)
  lr = getenv("LR", 0.8)
  lr_warmup_epochs = getenv("LR_WARMUP_EPOCHS", 200)
  lr_warmup_init_lr = getenv("LR_WARMUP_INIT_LR", 0.0001)

  model = UNet3D()
  optim = nn.optim.SGD(nn.state.get_parameters(model), lr=1.0, momentum=0.9, nesterov=True)

  def _lr_warm_up(optim, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    optim.lr.assign(Tensor([init_lr + (lr - init_lr) * scale]))

  @TinyJit
  def _train_step(x, y):
    y_hat = model(x)
    loss = dice_ce_loss(y_hat, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.realize()

  for epoch in range(1, epochs + 1):
    if epoch <= lr_warmup_epochs and lr_warmup_epochs > 0:
      _lr_warm_up(optim, lr_warmup_init_lr, lr, epoch, lr_warmup_epochs)

    for x, y in (t:=tqdm(iterate(val=False, shuffle=True, bs=bs), desc=f"[Epoch {epoch}]", total=len(get_train_files()))):
      x, y = Tensor(x, dtype=dtypes.float32), Tensor(y, dtype=dtypes.uint8)
      loss = _train_step(x, y)
      t.set_description(f"[Epoch {epoch}][Loss: {loss.item():.3f}]")

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


