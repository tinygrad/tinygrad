from tinygrad.tensor import Tensor,Device
from tinygrad.helpers import getenv
from tinygrad.nn import optim
from tqdm import tqdm

def train_resnet():
  # TODO: Resnet50-v1.5
  # there's no weight decay for sgd ?
  from models.resnet import ResNet50
  from datasets.imagenet import iterate
  from extra.lr_scheduler import CosineAnnealingLR
  from examples.mlperf.metrics import cross_entropy_loss
  from extra.training import sparse_categorical_crossentropy

  model = ResNet50()
  BS = 8
  lr = BS*1e-3
  optimizer = optim.SGD(optim.get_parameters(model), lr=lr, momentum = .875)
  args_epoch = 100
  scheduler = CosineAnnealingLR(optimizer, args_epoch)
  def warmup_factor(epoch, step):
    return min(1.0, (step+1)/warmup_period)

  for epoch in range(args_epoch):
    n,d = 0,0
    if epoch+1 % 10 == 0:
      # image - (BS,C,X,X), label - (BS,) -> Int
      for image, label in ( v:=tqdm(iterate(bs=BS, val=True))):
        out = model(Tensor(image))
        out = out.numpy()
        loss = cross_entropy_loss(out, label)
        out = out.argmax(axis=1)
        n += (out==label).sum()
        d += len(out)
        v.set_description(f"Validation Loss : {loss} ; {n}/{d}  {n*100./d:.2f}%")

    for image, label in (t :=tqdm(iterate(bs=BS,val=False))):
      image = Tensor(image)
      optimizer.zero_grad()
      out = model.forward(image)
      loss = sparse_categorical_crossentropy(out, label) # using sparse categorical : labels -> int
      loss.backward()
      optimizer.step()
      t.set_description(f"Training Loss : {loss.detach().cpu().numpy()} ; Learning Rate : {lr}")
    lr = scheduler.get_lr() if BS < 512 else scheduler.get_lr * warmup_factor(epoch, 8)

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  # TODO: Unet3d
  pass

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


