from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import GlobalCounters, getenv, dtypes
from examples.mlperf.helpers import PreFetcher
from tqdm import tqdm
import numpy as np
import random
import wandb
import time

# TODO: auto-down DALI
def train_resnet_dali():
  from models.resnet import ResNet50
  from extra.datasets.imagenet import BASEDIR
  from extra.datasets.imagenet_dali import create_dali_pipeline
  try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
  except ImportError:
    print("nvidia.dali package not found. Attempting to install...")
    import subprocess
    subprocess.call(["pip", "install", "--extra-index-url", "https://developer.download.nvidia.com/compute/redist", "--upgrade", "nvidia-dali-cuda120"])
    try:
        from nvidia.dali.pipeline import pipeline_def
        import nvidia.dali.types as types
        import nvidia.dali.fn as fn
    except ImportError:
        raise ImportError("Failed to import nvidia.dali modules even after installation attempt.")

  from extra.lr_scheduler import CosineAnnealingLR
  import os
  BS = getenv("BS",16)
  WORKERS = getenv("WORKERS", 4)
  traindir, valdir = os.path.join(BASEDIR, 'train'),os.path.join(BASEDIR,'val')
  val_size,crop_size = 256,224
  
  train_pipe = create_dali_pipeline(batch_size=BS,
                                        num_threads=WORKERS,
                                        device_id=0,
                                        seed=12,
                                        data_dir=traindir,
                                        crop=crop_size,
                                        size=val_size,
                                        dali_cpu=False,
                                        shard_id=0, # local_rank
                                        num_shards=0, # world size
                                        is_training=True)
  train_pipe.build()
  train_loader = DALIClassificationIterator(train_pipe, reader_name="Reader",
                                            last_batch_policy=LastBatchPolicy.PARTIAL,
                                            auto_reset=True)

  val_pipe = create_dali_pipeline(batch_size=BS,
                                        num_threads=WORKERS,
                                        device_id=0,
                                        seed=12,
                                        data_dir=valdir,
                                        crop=crop_size,
                                        size=val_size,
                                        dali_cpu=False,
                                        shard_id=0, # local_rank
                                        num_shards=0, # world size
                                        is_training=True)
  val_pipe.build()
  val_loader = DALIClassificationIterator(val_pipe, reader_name="Reader",
                                          last_batch_policy=LastBatchPolicy.PARTIAL,
                                          auto_reset=True)
  def sparse_categorical_crossentropy(out, Y, label_smoothing=0):
    out = out.float()
    num_classes = out.shape[-1]
    y_counter = Tensor.arange(num_classes, requires_grad=False).unsqueeze(0).expand(Y.numel(), num_classes)
    y = (y_counter == Y.flatten().reshape(-1, 1)).where(-1.0 * num_classes, 0)
    y = y.reshape(*Y.shape, num_classes)
    return (1 - label_smoothing) * out.mul(y).mean() + (-1 * label_smoothing * out.mean())

  @TinyJit
  def train_step(X, Y):
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

  wandb.init()

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

  epoch_avg_time = []
  for e in range(epochs):
    # train loop
    Tensor.training = True
    cl = time.monotonic() 
    for i,data in enumerate(train_loader):
      X,Y = data[0]["data"],data[0]["label"].squeeze(-1).long()
      GlobalCounters.reset()
      st = time.monotonic()
      data_time = st-cl
      X, Y = Tensor(X, requires_grad=False), Tensor(Y, requires_grad=False)
      loss, out = train_step(X, Y)
      et = time.monotonic()
      loss_cpu = loss.numpy()
      cl = time.monotonic()

      print(f"{(data_time+et-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {actual_data*1000.0:7.2f} ms fetch data, {data_time*1000:7.2f} ms prefetch data {loss_cpu:7.2f} loss, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
      wandb.log({"lr": scheduler.get_lr().numpy().item(),
                 "train/data_time": data_time,
                 "train/python_time": et - st,
                 "train/step_time": cl - st,
                 "train/other_time": cl - et,
                 "train/loss": loss_cpu,
                 "train/GFLOPS": GlobalCounters.global_ops*1e-9/(cl-st),
      })
      epoch_avg_time.append((data_time+(et-st))*1000)
    
    # "eval" loop. Evaluate every 4 epochs, starting with epoch 1
    if e % 4 == 1:
      eval_loss = []
      eval_times = []
      eval_top_1_acc = []
      eval_top_5_acc = []
      Tensor.training = False
      for i,data in enumerate(val_loader):
        X, Y = data[0]["data"], data[0]["label"].squeeze(-1).long()
        X, Y = Tensor(X, requires_grad=False), Tensor(Y, requires_grad=False)
        st = time.time()
        loss, out = eval_step(X, Y)
        et = time.time()

        top_1_acc = calculate_accuracy(out, Y, 1)
        top_5_acc = calculate_accuracy(out, Y, 5)
        eval_loss.append(loss.numpy().item())
        eval_times.append(et - st)
        eval_top_1_acc.append(top_1_acc)
        eval_top_5_acc.append(top_5_acc)
      wandb.log({"eval/loss": sum(eval_loss) / len(eval_loss),
                "eval/forward_time": sum(eval_times) / len(eval_times),
                "eval/top_1_acc": sum(eval_top_1_acc) / len(eval_top_1_acc),
                "eval/top_5_acc": sum(eval_top_5_acc) / len(eval_top_5_acc),
                "eval/avg_time": sum(epoch_avg_time) / len(epoch_avg_time)
      })
      epoch_avg_time = []

def train_resnet():
  # TODO: Resnet50-v1.5
  from models.resnet import ResNet50
  from extra.datasets.imagenet import iterate, get_train_files, get_val_files
  from extra.lr_scheduler import CosineAnnealingLR

  def sparse_categorical_crossentropy(out, Y, label_smoothing=0):
    out = out.float()
    num_classes = out.shape[-1]
    y_counter = Tensor.arange(num_classes, requires_grad=False).unsqueeze(0).expand(Y.numel(), num_classes)
    y = (y_counter == Y.flatten().reshape(-1, 1)).where(-1.0 * num_classes, 0)
    y = y.reshape(*Y.shape, num_classes)
    return (1 - label_smoothing) * out.mul(y).mean() + (-1 * label_smoothing * out.mean())
  
  @TinyJit
  def train_step(X, Y):
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

  wandb.init()

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
    for X, Y, actual_data in (t := tqdm(PreFetcher(iterate(bs=BS, val=False, num_workers=WORKERS)), total=steps_in_train_epoch)):
      GlobalCounters.reset()
      st = time.monotonic()
      data_time = st-cl
      X, Y = Tensor(X, requires_grad=False), Tensor(Y, requires_grad=False)
      loss, out = train_step(X, Y)
      et = time.monotonic()
      loss_cpu = loss.numpy()
      cl = time.monotonic()

      print(f"{(data_time+et-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {actual_data*1000.0:7.2f} ms fetch data, {data_time*1000:7.2f} ms prefetch data {loss_cpu:7.2f} loss, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS")
      wandb.log({"lr": scheduler.get_lr().numpy().item(),
                 "train/data_time": data_time,
                 "train/python_time": et - st,
                 "train/step_time": cl - st,
                 "train/other_time": cl - et,
                 "train/loss": loss_cpu,
                 "train/GFLOPS": GlobalCounters.global_ops*1e-9/(cl-st),
      })
      epoch_avg_time.append((data_time+(et-st))*1000)
    
    # "eval" loop. Evaluate every 4 epochs, starting with epoch 1
    if e % 4 == 1:
      eval_loss = []
      eval_times = []
      eval_top_1_acc = []
      eval_top_5_acc = []
      Tensor.training = False
      for X, Y in (t := tqdm(PreFetcher(iterate(bs=BS, val=True, num_workers=16)), total=steps_in_val_epoch)):
        X, Y = Tensor(X, requires_grad=False), Tensor(Y, requires_grad=False)
        st = time.time()
        loss, out = eval_step(X, Y)
        et = time.time()

        top_1_acc = calculate_accuracy(out, Y, 1)
        top_5_acc = calculate_accuracy(out, Y, 5)
        eval_loss.append(loss.numpy().item())
        eval_times.append(et - st)
        eval_top_1_acc.append(top_1_acc)
        eval_top_5_acc.append(top_5_acc)
      wandb.log({"eval/loss": sum(eval_loss) / len(eval_loss),
                "eval/forward_time": sum(eval_times) / len(eval_times),
                "eval/top_1_acc": sum(eval_top_1_acc) / len(eval_top_1_acc),
                "eval/top_5_acc": sum(eval_top_5_acc) / len(eval_top_5_acc),
                "eval/avg_time": sum(epoch_avg_time) / len(epoch_avg_time)
      })
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

if __name__ == "__main__":
  # NOTE: to run with resnet_dali, do export=resnet_dali
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn,resnet_dali").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()

