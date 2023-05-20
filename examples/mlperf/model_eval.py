import time
import numpy as np
from tinygrad.tensor import Tensor

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

  # RNN-T
  from models.rnnt import RNNT
  mdl = RNNT()
  mdl.load_from_pretrained()

  from datasets.librispeech import iterate
  from examples.mlperf.metrics import word_error_rate

  LABELS = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

  wer = 0
  c = 0
  st = time.perf_counter()
  for X, Y in iterate():
    mt = time.perf_counter()
    tt = mdl.decode(Tensor(X[0]), Tensor([X[1]]))
    et = time.perf_counter()
    print(f"{(mt-st)*1000:.2f} ms loading data, {(et-mt)*1000:.2f} ms to run model")
    for n, t in enumerate(tt):
      tnp = np.array(t)
      wer += word_error_rate(["".join([LABELS[int(tnp[i])] for i in range(tnp.shape[0])])], [Y[n]])
    c += len(tt)
    print(f"WER: {wer/c}, {c} samples, raw wer: {wer}")
    st = time.perf_counter()
