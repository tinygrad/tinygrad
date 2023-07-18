from tqdm import tqdm

from tinygrad.helpers import getenv
from tinygrad.jit import TinyJit
from tinygrad.nn import optim
from tinygrad.state import get_parameters
from tinygrad.tensor import Tensor


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
  from examples.mlperf.metrics import neg_log_likelihood
  from extra.datasets.librispeech import iterate
  from extra.lr_scheduler import CosineAnnealingLR
  from models.rnnt import RNNT
  import numpy as np
  
  LABELS = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
            "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

  
  rnnt_model = RNNT()
  rnnt_model.load_from_pretrained()
  
  model_params = get_parameters(rnnt_model)
  sgd = optim.SGD(model_params, nesterov=True)
  sgd.zero_grad()
  lr_scheduler = CosineAnnealingLR(sgd, T_max=100, eta_min=1e-6)
  BS : int = 200
  EPHOCS : int = 50

  Tensor.training = True
  for epoch in (r := tqdm(range(EPHOCS))):
    for X,Y in (t := (iterate(bs=BS, mode="train"))):
      _ , pred_probs = rnnt_model.decode(X[0],X[1])
      for n,out in enumerate(pred_probs):
        out_ = np.array(out)
        target = np.array([LABELS.index(char.lower()) for char in Y[n]])
        loss = neg_log_likelihood(out_,target)
        sgd.zero_grad()
        loss.backward()
        sgd.step()
        
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


