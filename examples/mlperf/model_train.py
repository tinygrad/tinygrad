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
  from tinygrad import dtypes, Device, TinyJit
  from tinygrad.nn.optim import SGD
  from tinygrad.nn.state import get_parameters
  from tqdm import tqdm

  NUM_EPOCHS = getenv("NUM_EPOCHS", 4000)
  BS = getenv("BS", 2)
  LR = getenv("LR", 0.8)
  MOMENTUM = getenv("MOMENTUM", 0.9)
  LR_WARMUP_EPOCHS = getenv("LR_WARMUP_EPOCHS", 200)
  LR_WARMUP_INIT_LR = getenv("LR_WARMUP_INIT_LR", 0.0001)

  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  assert BS % len(GPUS) == 0, f"{BS=} is not a multiple of {len(GPUS)=}"
  for x in GPUS: Device[x]

  model = UNet3D()
  if len(GPUS) > 1: model.shard_(GPUS)

  optim = SGD(get_parameters(model), lr=LR, momentum=MOMENTUM, nesterov=True)

  def _lr_warm_up(optim, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    optim.lr.assign(Tensor([init_lr + (lr - init_lr) * scale]))

  @TinyJit
  def _train_step(x, y):
    y_hat = model(x)
    loss = dice_ce_loss(y_hat, y, gpus=GPUS)

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.realize()

  for epoch in range(1, NUM_EPOCHS + 1):
    if epoch <= LR_WARMUP_EPOCHS and LR_WARMUP_EPOCHS > 0:
      _lr_warm_up(optim, LR_WARMUP_INIT_LR, LR, epoch, LR_WARMUP_EPOCHS)

    for x, y in (t:=tqdm(iterate(val=False, shuffle=True, bs=BS), desc=f"[Epoch {epoch}]", total=len(get_train_files()) // BS)):
      x, y = Tensor(x, dtype=dtypes.float32), Tensor(y, dtype=dtypes.uint8)
      if len(GPUS) > 1: x, y = x.shard(GPUS, axis=0), y.shard(GPUS, axis=0)

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


