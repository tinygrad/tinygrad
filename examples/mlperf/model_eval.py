import time
from pathlib import Path
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
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
  from datasets.kits19 import iterate
  from pathlib import Path
  import torch
  fn = Path(__file__).parent.parent.parent / "weights/unet-3d.ckpt"
  mdl = torch.jit.load(fn, map_location=torch.device("cpu"))
  scores = []
  for X, Y, case in iterate():
    print(f"{case=}")
    result_file = Path(f"/tmp/{case}.npy")
    if getenv("LOAD_KITS") and result_file.is_file():
      pred = np.load(result_file)
    else:
      image = X[np.newaxis, ...]
      result, norm_map, norm_patch = helpers.prepare_arrays(image)
      for i, j, k in helpers.get_slice(image):
        out = mdl(torch.tensor(image[..., i:i+128, j:j+128, k:k+128])).detach().numpy()
        result[..., i:i+128, j:j+128, k:k+128] += out * norm_patch
        norm_map[..., i:i+128, j:j+128, k:k+128] += norm_patch
      result /= norm_map
      pred = np.argmax(result, axis=1).astype(np.uint8)
      if getenv("STORE_KITS"):
        np.save(result_file, pred)
    assert pred.shape == Y.shape
    scores.append(helpers.get_dice_score(pred, Y).mean())
  print(sum(scores) / len(scores))
