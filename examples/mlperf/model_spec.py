# load each model here, quick benchmark
from tinygrad.tensor import Tensor
from tinygrad.helpers import GlobalCounters, getenv

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
  # TODO: Retinanet
  pass

def spec_unet3d():
  # 3D UNET
  from models.unet3d_v2 import UNet3D
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
  # TODO: BERT-large
  pass

if __name__ == "__main__":
  # inference only for now
  Tensor.training = False
  Tensor.no_grad = True

  for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert").split(","):
    nm = f"spec_{m}"
    if nm in globals():
      print(f"testing {m}")
      globals()[nm]()


