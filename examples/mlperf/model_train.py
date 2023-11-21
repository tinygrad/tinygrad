from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, Context


def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet(n_epochs=1, target_mAP=0.34, walltime_h=24):
  from examples.mlperf.train_retinanet import RetinaNetTrainer
  trainer = RetinaNetTrainer(debug=False)
  for epoch in range(n_epochs):
    trainer.train_one_epoch()
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


