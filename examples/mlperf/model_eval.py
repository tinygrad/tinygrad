import time
import numpy as np
from tinygrad.tensor import Tensor
from examples.mlperf import helpers

if __name__ == "__main__":
  # inference only
  Tensor.training = False
  Tensor.no_grad = True

  # Resnet50-v1.5
  from tinygrad.jit import TinyJit
  from models.resnet import ResNet50
  mdl = ResNet50()
  mdl.load_from_pretrained()

  input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
  input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
  def input_fixup(x):
    x = x.permute([0,3,1,2]) / 255.0
    x -= input_mean
    x /= input_std
    return x

  mdlrun = TinyJit(lambda x: mdl(input_fixup(x)).realize())

  # evaluation on the mlperf classes of the validation set from imagenet
  from datasets.imagenet import iterate
  from extra.helpers import cross_process

  n,d = 0,0
  st = time.perf_counter()
  for x,y in cross_process(iterate):
    dat = Tensor(x.astype(np.float32))
    mt = time.perf_counter()
    outs = mdlrun(dat)
    t = outs.numpy().argmax(axis=1)
    et = time.perf_counter()
    print(f"{(mt-st)*1000:.2f} ms loading data, {(et-mt)*1000:.2f} ms to run model")
    print(t)
    print(y)
    n += (t==y).sum()
    d += len(t)
    print(f"****** {n}/{d}  {n*100.0/d:.2f}%")
    st = time.perf_counter()

  # UNet3D
  from models.unet3d import UNet3D
  from datasets.kits19 import iterate
  mdl = UNet3D()
  mdl.load_from_pretrained()
  for x, y in iterate():
    image = x[np.newaxis, ...]
    result, norm_map, norm_patch = helpers.prepare_arrays(image)
    for i, j, k in helpers.get_slice(image):
      input_slice = Tensor(image[:, :, i:i+128, j:j+128, k:k+128])
      # TODO: if going_to_segfault(): dont()
      result[:, :, i:i+128, j:j+128, k:k+128] += mdl(input_slice).numpy() * norm_patch
      norm_map[:, :, i:i+128, j:j+128, k:k+128] += norm_patch
    final_result = helpers.finalize(result, norm_map)
