from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import GlobalCounters, getenv, dtypes
from extra.datasets.imagenet import PreFetcher
from tqdm import tqdm
import numpy as np
import random
#import wandb
import time



def train_resnet():
  # TODO: Resnet50-v1.5
  from models.resnet import ResNet50
  from extra.datasets.imagenet import get_train_files, get_val_files
  from extra.datasets.dataloader import cross_process, iterate
  from extra.lr_scheduler import CosineAnnealingLR
  import torchvision.transforms.functional as F
  import torch
  import statistics

  def sparse_categorical_crossentropy(out, Y, label_smoothing=0):
    out = out.float()
    num_classes = out.shape[-1]
    y_counter = Tensor.arange(num_classes, requires_grad=False).unsqueeze(0).expand(Y.numel(), num_classes)
    y = (y_counter == Y.flatten().reshape(-1, 1)).where(-1.0 * num_classes, 0)
    y = y.reshape(*Y.shape, num_classes)
    return (1 - label_smoothing) * out.mul(y).mean() + (-1 * label_smoothing * out.mean())

  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]

  '''
  @TinyJit
  def normalize(X: Tensor):
    input_mean = Tensor(mean).reshape(-1,1,1)
    input_std = Tensor(std).reshape(-1,1,1)
    X = X.permute(0,3,1,2) / 255.0
    X -= input_mean
    X /= input_std
    return X
  '''

  @TinyJit
  def train_step(X, Y):
    input_mean = Tensor(mean).reshape(-1,1,1)
    input_std = Tensor(std).reshape(-1,1,1)
    X = X.permute(0,3,1,2) / 255.0
    X -= input_mean
    X /= input_std

    X = X.half()
    Y = Y.half()
    optimizer.zero_grad()
    out = model.forward(X)
    loss = sparse_categorical_crossentropy(out, Y, label_smoothing=0.1)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.realize(), out.realize()
  
  @TinyJit
  def eval_step(X, Y):
    out = model.forward(X)
    loss = sparse_categorical_crossentropy(out, Y, label_smoothing=0.1)
    return loss.realize(), out.realize()
  
  def calculate_accuracy(out, Y, top_n):
    out_top_n = np.argpartition(out.cpu().numpy(), -top_n, axis=-1)[:, -top_n:]
    YY = np.expand_dims(Y.numpy(), axis=1)
    YY = np.repeat(YY, top_n, axis=1)

    eq_elements = np.equal(out_top_n, YY)
    top_n_acc = np.count_nonzero(eq_elements) / eq_elements.size * top_n
    return top_n_acc

  seed = getenv('SEED', 42)
  Tensor.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  #wandb.init()

  num_classes = 1000
  fp16 = getenv("HALF", 0)
  if fp16 == 1:
    Tensor.default_type = dtypes.float16
  model = ResNet50(num_classes)
  parameters = get_parameters(model)

  BS = getenv("BS",16)
  WORKERS = getenv("WORKERS",16)
  lr = 0.256 * (BS / 256)  # Linearly scale from BS=256, lr=0.256
  epochs = 50
  optimizer = optim.SGD(parameters, lr, momentum=.875, weight_decay=1/2**15)
  scheduler = CosineAnnealingLR(optimizer, epochs)
  print(f"training with batch size {BS} for {epochs} epochs")

  steps_in_train_epoch = (len(get_train_files()) // BS) - 1
  steps_in_val_epoch = (len(get_val_files()) // BS) - 1
  epoch_avg_time = []
  for e in range(epochs):
    # train loop
    Tensor.training = True
    cl = time.monotonic() 
    for i, (X,Y,a,m) in enumerate(t:= tqdm(PreFetcher(iterate(bs=BS,val=False,shuffle=True,num_workers=WORKERS)), total=steps_in_train_epoch)):
      GlobalCounters.reset()

      st1 = time.monotonic()
      #X,Y = torch_normalize(X),Tensor(Y,requires_grad=False)
      X,Y = Tensor(X,requires_grad=False),Tensor(Y,requires_grad=False)

      st = time.monotonic()
      print(f'normalize {(st-st1)*1000:7.2f}')
      data_time = st-cl
      loss, out = train_step(X, Y)

      et = time.monotonic()
      if i % 1000 == 0: 
        loss_cpu = 0 #loss.numpy()

      cl = time.monotonic()
      train_time = (data_time+et-st)*steps_in_train_epoch*epochs/(60*60)
      val_time = (data_time+et-st)*steps_in_val_epoch*(epochs//4)/(60*60)
      print(f'{a*1000:7.2f} all img tm {m*1000:7.2f} multi tm')
      print(f"{(data_time+et-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {data_time*1000:7.2f} ms data {loss_cpu:7.2f} loss (every 1000)" + \
            f"{GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS {train_time+val_time:7.1f} hrs total {train_time:7.1f}hrs train {val_time:7.1f}hrs val")
      '''
      wandb.log({"lr": scheduler.get_lr().numpy().item(),
                 "train/data_time": data_time,
                 "train/python_time": et - st,
                 "train/step_time": cl - st,
                 "train/other_time": cl - et,
                 "train/loss": loss_cpu,
                 "train/GFLOPS": GlobalCounters.global_ops*1e-9/(cl-st),
      })
    '''
      epoch_avg_time.append((data_time+(et-st))*1000)

    epoch_avg = sum(epoch_avg_time)/len(epoch_avg_time)
    epoch_med = statistics.median(epoch_avg_time)
    val_time = epoch_avg*steps_in_val_epoch*(epochs//4)/(1000*60*60)
    train_time = epoch_avg*steps_in_train_epoch*epochs/(1000*60*60)
    print(f'EPOCH {e}: avg step time {epoch_avg:7.2f} ms {epoch_med:7.2f}ms median step time  {train_time+val_time:7.2f}hrs total')

    # 60ms mean 25ms median step for cross_process
    # 40ms mean 20ms median step  data, PreFetcher

    # before beam search, on METAL 1 step = ~1700ms compute
    # trying hand coded opt first, getting 2202.96 ms
    # trying beam search BEAM=10, getting 

    # CUDA
    # hand | ******* total 498.16 ms,   1062 GFLOPS
    # BEAM 6 | Tesla T4 ******* total 120.94 ms,   4376 GFLOPS

    # TODO: - does optimized for CUDA work on all CUDA machines
    # - BEAM 4 or 2.
    
    # "eval" loop. Evaluate every 4 epochs, starting with epoch 1

    # on runpod
    # idk why, data is 400ms... 

    if e % 4 == 1:
      eval_loss = []
      eval_times = []
      eval_top_1_acc = []
      eval_top_5_acc = []
      Tensor.training = False
      cl = time.monotonic()
      for i, (X, Y,a,m) in enumerate(t:=tqdm(PreFetcher(iterate(bs=BS,val=True, num_workers=WORKERS)), total=steps_in_val_epoch)):
        X, Y = normalize(Tensor(X, requires_grad=False)), Tensor(Y, requires_grad=False)
        loss, out = eval_step(X, Y)

        top_1_acc = calculate_accuracy(out, Y, 1)
        top_5_acc = calculate_accuracy(out, Y, 5)
        et = time.monotonic()
        eval_loss.append(loss.numpy().item())
        eval_times.append(et - st)
        eval_top_1_acc.append(top_1_acc)
        eval_top_5_acc.append(top_5_acc)
        '''
        wandb.log({"eval/loss": sum(eval_loss) / len(eval_loss),
                  "eval/forward_time": sum(eval_times) / len(eval_times),
                  "eval/top_1_acc": sum(eval_top_1_acc) / len(eval_top_1_acc),
                  "eval/top_5_acc": sum(eval_top_5_acc) / len(eval_top_5_acc),
                  "eval/avg_time": sum(epoch_avg_time) / len(epoch_avg_time)
        })
        '''
        val_time = (data_time+et-st)*steps_in_val_epoch*(epochs//4)/(60*60)
        print(f'{(et-cl)*1000:7.2f} ms run {val_time:7.2f}hrs total val')
        cl = time.monotonic()
      epoch_avg_time = []

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

import sqlite3

def recover_corrupted_db(corrupted_db, new_db):
    # Connect to the corrupted database
    conn_corrupted = sqlite3.connect(corrupted_db)
    
    # Get the dump of the corrupted database
    dumped_data = '\n'.join(conn_corrupted.iterdump())

    # Close the corrupted database connection
    conn_corrupted.close()

    # Create and/or write the dump to the new database
    conn_new = sqlite3.connect(new_db)
    conn_new.executescript(dumped_data)
    conn_new.close()

    print(f"Attempted to recover data from {corrupted_db} to {new_db}")



if __name__ == "__main__":
  import subprocess
  import sys
  modules_to_check = ['simplejpeg', 'wandb', 'pycuda', 'cloudpickle']
  def install(package):
      subprocess.check_call([sys.executable, "-m", "pip", "install", package])
  for module in modules_to_check:
      try:
          __import__(module)
          print(f"Module '{module}' is already installed.")
      except ImportError:
          print(f"Module '{module}' not found. Installing...")
          try:
              install(module)
              print(f"Module '{module}' has been installed.")
          except Exception as e:
              print(f"An error occurred while installing '{module}'.", e)
    #recover_corrupted_db("/users/minjunes/downloads/tinygrad_cache BEAM=6", "/tmp/tinygrad_cache")
    # NOTE: to run with resnet_dali, do export=resnet_dali
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()

