# load each model here, quick benchmark
from tinygrad.tensor import Tensor
from tinygrad.helpers import GlobalCounters

def test_model(model, *inputs):
  GlobalCounters.reset()
  model(*inputs).numpy()
  # TODO: return event future to still get the time_sum_s without DEBUG=2
  print(f"{GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.time_sum_s*1000:.2f} ms")

if __name__ == "__main__":
  # inference only for now
  Tensor.training = False
  Tensor.no_grad = True

  # Resnet50-v1.5
  from models.resnet import ResNet50
  mdl = ResNet50()
  img = Tensor.randn(1, 3, 224, 224)
  test_model(mdl, img)

  # Retinanet

  # 3D UNET
  from models.unet3d import UNet3D
  mdl = UNet3D()
  #mdl.load_from_pretrained()
  img = Tensor.randn(1, 1, 5, 224, 224)
  test_model(mdl, img)

  # RNNT

  # BERT-large
