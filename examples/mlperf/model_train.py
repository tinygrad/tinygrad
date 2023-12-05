from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv

def train_resnet():
  # TODO: Resnet50-v1.5
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
  from extra.datasets import iterate
  from extra.models.mask_rcnn import MaskRCNN
  from extra.models.resnet import ResNet

  batch_size = getenv("BS", default=2)
  # iterate()


  # backbone = ResNet(50)
  # mask_rcnn = MaskRCNN(backbone)

  # NOTE: mask_rcnn accepts a List[Tensor] as its input.
  # To load it, we can open a list of images and load it to a Tensor.

if __name__ == "__main__":
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()


