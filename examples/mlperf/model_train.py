from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, Context


def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  from models.resnet import ResNeXt50_32X4D
  from models.retinanet import RetinaNet
  from examples.mlperf.train_retinanet import RetinaNetTrainer
  backbone = ResNeXt50_32X4D(num_classes=None)
  retina = RetinaNet(backbone) #remember num_classes = 600 for openimages
  trainer = RetinaNetTrainer(retina)
  trainer.train()
  # reference torch implementation https://github.com/mlcommons/training/blob/master/object_detection/pytorch/maskrcnn_benchmark/modeling/rpn/retinanet


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


