from tqdm import tqdm, trange
import time
from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.ops import Device
from tinygrad.helpers import getenv, dtypes, prod, GlobalCounters
from multiprocessing import Queue, Process
import multiprocessing.shared_memory as shared_memory
from tinygrad.nn.state import get_parameters, safe_save, get_state_dict
from tinygrad.nn.optim import Adam, SGD

def train_resnet():
  # https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
  from extra.datasets.imagenet import get_train_files
  from extra.models.resnet import ResNet50
  from examples.mlperf.dataloader import batch_load_resnet
  from extra.lr_scheduler import CosineAnnealingLR
  BS = getenv("BS", 64)
  EPOCHS = 50
  steps = len(get_train_files())//BS

  class ResnetTrainer:
    def __init__(self, device=None):
      self.mdl = ResNet50()
      for x in get_parameters(self.mdl) if device else []: x.to_(device)
      #self.mdl.load_from_pretrained()
      #self.opt = Adam(get_parameters(self.mdl))
      self.opt = SGD(get_parameters(self.mdl), lr=0.001*BS, momentum=0.875, weight_decay=1/32768)
      self.lr_schedule = CosineAnnealingLR(self.opt, EPOCHS*steps)
      self.input_mean = Tensor([0.485, 0.456, 0.406], device=device).reshape(1, -1, 1, 1)
      self.input_std = Tensor([0.229, 0.224, 0.225], device=device).reshape(1, -1, 1, 1)
    def __call__(self, x:Tensor, y:Tensor) -> Tensor:
      x = x.permute([0,3,1,2]).cast(dtypes.float32) / 255.0
      x -= self.input_mean
      x /= self.input_std
      with Tensor.train():
        self.opt.zero_grad()
        loss = self.mdl(x).sparse_categorical_crossentropy(y).backward()
        self.opt.step()
      self.lr_schedule.step()
      return loss.realize()

  trainer = ResnetTrainer()
  trainer_jit = TinyJit(trainer)

  for epoch in range(EPOCHS):
    safe_save(get_state_dict(trainer.mdl), f"/tmp/resnet_epoch_{epoch}.safetensors")
    iterator = batch_load_resnet(batch_size=BS, val=False)
    for x,y,c in (t:=tqdm(iterator, total=steps)):
      GlobalCounters.reset()
      st = time.perf_counter()
      loss = trainer_jit(x.to(Device.DEFAULT), Tensor(y, dtype=dtypes.int32)).item()
      et = time.perf_counter()-st
      t.set_description(f"loss: {loss:.2f} step: {et*1000:.2f} ms, {GlobalCounters.global_ops*1e-9/et:.2f} GFLOPS")

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


