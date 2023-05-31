# load each model here, quick benchmark
from tinygrad.tensor import Tensor
from tinygrad.helpers import GlobalCounters, getenv
import numpy as np

def test_model(model, *inputs):
  GlobalCounters.reset()
  model(*inputs).numpy()
  # TODO: return event future to still get the time_sum_s without DEBUG=2
  print(f"{GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.time_sum_s*1000:.2f} ms")

def spec_resnet():
  # Resnet50-v1.5
  from models.resnet import ResNet50
  mdl = ResNet50()
  img = Tensor.randn(1, 3, 224, 224)
  test_model(mdl, img)

def spec_retinanet():
  # Retinanet with ResNet backbone
  from models.resnet import ResNet50
  from models.retinanet import RetinaNet
  mdl = RetinaNet(ResNet50(), num_classes=91, num_anchors=9)
  img = Tensor.randn(1, 3, 224, 224)
  test_model(mdl, img)

def spec_unet3d():
  # 3D UNET
  from models.unet3d import UNet3D
  mdl = UNet3D()
  mdl.load_from_pretrained()
  img = Tensor.randn(1, 1, 128, 128, 128)
  test_model(mdl, img)

def spec_rnnt():
  from models.rnnt import RNNT
  mdl = RNNT()
  mdl.load_from_pretrained()
  x = Tensor.randn(220, 1, 240)
  y = Tensor.randn(1, 220)
  test_model(mdl, x, y)

def spec_bert():
  from models.bert import BertForQuestionAnswering
  mdl = BertForQuestionAnswering()
  mdl.load_from_pretrained()
  x = Tensor.randn(1, 384)
  am = Tensor.randn(1, 384)
  tt = Tensor(np.random.randint(0, 2, (1, 384)).astype(np.float32))
  test_model(mdl, x, am, tt)

if __name__ == "__main__":
  # inference only for now
  Tensor.training = False
  Tensor.no_grad = True

  for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert").split(","):
    nm = f"spec_{m}"
    if nm in globals():
      print(f"testing {m}")
      globals()[nm]()


