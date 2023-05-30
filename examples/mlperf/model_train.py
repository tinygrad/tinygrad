from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.nn import optim

def train_resnet():
  # TODO: Resnet50-v1.5
  # cosine LR or OneCycleLR, use linear warmup for bigger batch sizes, weight decay, 250 epochs (best possible accuracy) 
  # use mixup regularization
  # for data augmentation - Norm, random croped, horizontal flip
  from models.resnet import ResNet50
  model = ResNet50()
  optimizer = optim.SGD(optim.get_parameters(model), lr=1e-4, momentum = .875)

  pass

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


