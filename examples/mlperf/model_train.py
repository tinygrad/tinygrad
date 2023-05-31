from tinygrad.tensor import Tensor,Device
from tinygrad.helpers import getenv
from tinygrad.nn import optim
from tqdm import tqdm

def train_resnet():
  # TODO: Resnet50-v1.5
  # cosine LR or OneCycleLR, use linear warmup for bigger batch sizes, weight decay, 250 epochs (best possible accuracy) 
  # use mixup regularization, write cross entropy loss
  # for data augmentation - Norm, random croped, horizontal flip
  # there's no weight decay for sgd ?
  from models.resnet import ResNet50
  from datasets.imagenet import iterate
  from extra.lr_scheduler import CosineAnnealingLR
  from examples.mlperf.metrics import cross_entropy_loss
  from tinygrad.jit import TinyJit
  from extra.helpers import cross_process
  from extra.training import sparse_categorical_crossentropy

  model = ResNet50()
  optimizer = optim.SGD(optim.get_parameters(model), lr=1e-4, momentum = .875, wd=1/2**15)
  scheduler = CosineAnnealingLR(optimizer, 250)
  args_epoch = 250

  for epoch in range(args_epoch):
    n,d = 0,0
    for image, label in ( v:=tqdm(iterate(bs=8, val=True))):
      out = model(Tensor(image))
      out = out.numpy()
      loss = cross_entropy_loss(out, label)
      out = out.argmax(axis=1)
      n += (out==label).sum()
      d += len(out)
      v.set_description(f"Validation Loss : {loss} ; {n}/{d}  {n*100./d:.2f}%")
      break

    Tensor.training= True
    for image, label in (t :=tqdm(iterate(bs=8,val=False))):
      image = Tensor(image, requires_grad=False)
      optimizer.zero_grad()
      out = model.forward(image)
      loss = sparse_categorical_crossentropy(out, label)
      loss.backward()
      optimizer.step()
      t.set_description(f"Training Loss : {loss.detach().cpu().numpy()}")

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


