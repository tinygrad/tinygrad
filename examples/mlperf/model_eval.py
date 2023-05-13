import numpy as np
from tinygrad.tensor import Tensor

if __name__ == "__main__":
  # inference only
  Tensor.training = False
  Tensor.no_grad = True

  # Resnet50-v1.5
  from models.resnet import ResNet50
  mdl = ResNet50()
  mdl.load_from_pretrained()

  # evaluation on the mlperf classes of the validation set from imagenet
  from datasets.imagenet import iterate
  n,d = 0,0
  for x,y in iterate(32, True, shuffle=True):
    dat = Tensor(x.astype(np.float32))
    outs = mdl(dat)
    t = outs.numpy().argmax(axis=1)
    print(t)
    print(y)
    n += (t==y).sum()
    d += len(t)
    print(f"****** {n}/{d}  {n*100.0/d:.2f}%")


