from tqdm import tqdm, trange
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, dtypes, prod
from multiprocessing import Queue, Process
import multiprocessing.shared_memory as shared_memory

def train_resnet():
  from examples.mlperf.dataloader import batch_load_resnet
  iterator = batch_load_resnet(batch_size=64, val=False)

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
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()


