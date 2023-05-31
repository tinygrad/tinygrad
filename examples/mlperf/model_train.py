from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.nn import optim
from tqdm import trange

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

  model = ResNet50()
  optimizer = optim.SGD(optim.get_parameters(model), lr=1e-4, momentum = .875)
  scheduler = CosineAnnealingLR(optimizer, 250)
  args_epoch = 250

  input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
  input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
  def input_fixup(x):
    x = x.permute([0,3,1,2]) / 255.0
    x -= input_mean
    x /= input_std
    return x

  mdlrun = TinyJit(lambda x: model(input_fixup(x)).realize())

  for epoch in (r := trange(args_epoch)):
    for image, label in iterate(bs=64,val=False):
      optimizer.zero_grad()
      out = mdlrun(Tensor(image))
      loss = cross_entropy_loss(out, label)
      loss.backwards()
      optimizer.step()
    r.set_description("some data")

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


