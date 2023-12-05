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
  from extra.datasets.coco import iterate, TRAIN_TRANSFORMS
  from extra.models.mask_rcnn import MaskRCNN
  from extra.models.resnet import ResNet
  from tinygrad.nn.optim import SGD
  from tinygrad.nn.state import get_parameters
  from tqdm import tqdm

  bs = getenv("BS", default=2)

  backbone = ResNet(50, num_classes=None, stride_in_1x1=True)
  backbone.load_from_pretrained()
  model = MaskRCNN(backbone)

  params = get_parameters(model)
  # TODO: need to implement LR scheduler
  optim = SGD(params, lr=0.02, momentum=0.9, weight_decay=0.0001)

  for imgs, tgts in tqdm(iterate(bs=bs, transforms=TRAIN_TRANSFORMS), desc="Training MASK-RCNN"):
    # features = model.backbone(imgs.tensors)
    # proposals, _ = model.rpn(imgs, features, tgts)
    pred = model(imgs, targets=tgts)


  # NOTE: mask_rcnn accepts a List[Tensor] as its input.
  # To load it, we can open a list of images and load it to a Tensor.

if __name__ == "__main__":
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()


