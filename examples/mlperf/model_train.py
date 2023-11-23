from tqdm import tqdm, trange
import time
from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.ops import Device
from tinygrad.helpers import getenv, dtypes, prod, GlobalCounters, flatten
from multiprocessing import Queue, Process
import multiprocessing.shared_memory as shared_memory
from tinygrad.nn.state import get_parameters, safe_save, get_state_dict
from tinygrad.nn.optim import Adam, SGD
from extra.dist.collectives import allreduce
from extra.dist import init_oob
import wandb

def train_resnet():
  # https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
  from extra.datasets.imagenet import get_train_files
  from extra.models.resnet import ResNet50
  from examples.mlperf.dataloader import batch_load_resnet
  from extra.lr_scheduler import CosineAnnealingLR
  BS = getenv("BS", 64)
  EPOCHS = 50
  GPUS = getenv("GPUS", 1)
  steps = len(get_train_files())//(BS*GPUS)
  if getenv("WANDB"): wandb.init(project="tinygrad-resnet")

  class ResnetTrainer:
    def __init__(self, rank=-1):
      self.rank = rank
      device = None if rank == -1 or (rank == 0 and GPUS == 1) else f"gpu:{rank}"
      Tensor.manual_seed(1337)
      self.mdl = ResNet50()
      self.params = get_state_dict(self.mdl)
      for x in self.params.values() if device else []: x.to_(device)
      #self.mdl.load_from_pretrained()
      #self.opt = Adam(get_parameters(self.mdl))
      self.opt = SGD(self.params.values(), lr=0.001*BS*GPUS, momentum=0.875, weight_decay=1/32768)
      self.lr_schedule = CosineAnnealingLR(self.opt, EPOCHS*steps)
      self.input_mean = Tensor([0.485, 0.456, 0.406], device=device).reshape(1, -1, 1, 1)
      self.input_std = Tensor([0.229, 0.224, 0.225], device=device).reshape(1, -1, 1, 1)
    def __call__(self, x:Tensor, y:Tensor) -> Tensor:
      x = x.permute([0,3,1,2]).cast(dtypes.float32) / 255.0
      x -= self.input_mean
      x /= self.input_std
      self.opt.zero_grad()
      with Tensor.train():
        probs = self.mdl(x)
        loss = probs.sparse_categorical_crossentropy(y).backward()
        acc = (probs.argmax(-1) == y).mean()*100
        del probs, x, y
      return loss.realize(), acc.realize()
    def step(self):
      self.opt.step()
      self.lr_schedule.step()

  trainer = [ResnetTrainer(rank) for rank in range(GPUS)]
  @TinyJit
  def mega_train(*tensors):
    outs = [trainer[i](x,y) for i,(x,y) in enumerate(zip(tensors[::2], tensors[1::2]))]
    # allreduce
    if GPUS == 2:
      for k in trainer[0].params.keys():
        grads = [t.params[k].grad for t in trainer]
        if grads[0] is not None:
          grads[0] += grads[1].to(grads[0].device)
          grads[1] += grads[0].to(grads[1].device)
    elif GPUS > 2:
      raise NotImplementedError("write real allreduce")
    for t in trainer: t.step()
    return outs

  def data_get(iterator, rank=0):
    device = f"gpu:{rank}"
    x,y,c = next(iterator)
    return x.to(device).realize(), Tensor(y, dtype=dtypes.int32, device=device), c

  for epoch in range(EPOCHS):
    safe_save(get_state_dict(trainer[0].mdl), f"/tmp/resnet_epoch_{epoch}.safetensors")
    iterator = batch_load_resnet(batch_size=BS, val=False)
    proc = [data_get(iterator, rank) for rank in range(GPUS)]
    for _ in (t:=trange(steps)):
      GlobalCounters.reset()
      st = time.perf_counter()
      outs = mega_train(*flatten([(x,y) for x,y,_ in proc]))
      try: proc = [data_get(iterator, rank) for rank in range(GPUS)]
      except StopIteration: proc = None
      out_items = [(loss.item(), acc.item()) for loss, acc in outs]
      loss, acc = sum([x[0] for x in out_items])/len(out_items), sum([x[1] for x in out_items])/len(out_items)
      et = (time.perf_counter()-st)*1000
      t.set_description(f"loss: {loss:.2f} accuracy: {acc:.2f}% step: {et:.2f} ms, {GlobalCounters.global_ops*1e-9/et:.2f} GFLOPS")
      if getenv("WANDB"): wandb.log({"loss": loss, "accuracy": acc, "step_time_ms": et})

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


