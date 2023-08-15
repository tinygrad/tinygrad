from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import getenv
from tqdm import tqdm
import numpy as np
import random
import wandb
import time

def train_resnet():
  # TODO: Resnet50-v1.5
  from models.resnet import ResNet50
  from extra.datasets.imagenet import iterate, get_train_files, get_val_files
  from extra.lr_scheduler import CosineAnnealingLR

  def sparse_categorical_crossentropy(out, Y, label_smoothing=0):
    num_classes = out.shape[-1]
    y_counter = Tensor.arange(num_classes, requires_grad=False).unsqueeze(0).expand(Y.numel(), num_classes)
    y = (y_counter == Y.flatten().reshape(-1, 1)).where(-1.0 * num_classes, 0)
    y = y.reshape(*Y.shape, num_classes)
    return (1 - label_smoothing) * out.mul(y).mean() + (-1 * label_smoothing * out.mean())
  
  @TinyJit
  def train_step(X, Y):
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
  model = ResNet50(num_classes)
  parameters = get_parameters(model)

  BS = 16
  lr = 0.256 * (BS / 256)  # Linearly scale from BS=256, lr=0.256
  epochs = 100
  optimizer = optim.SGD(parameters, lr, momentum=.875, weight_decay=1/2**15)
  scheduler = CosineAnnealingLR(optimizer, epochs)
  print(f"training with batch size {BS} for {epochs} epochs")

  steps_in_train_epoch = (len(get_train_files()) // BS) - 1
  steps_in_val_epoch = (len(get_val_files()) // BS) - 1
  for e in range(epochs):
    # train loop
    Tensor.training = True
    for X, Y in (t := tqdm(iterate(bs=BS, val=False, num_workers=16), total=steps_in_train_epoch)):
      X, Y = Tensor(X, requires_grad=False), Tensor(Y, requires_grad=False)
      st = time.time()
      loss, out = train_step(X, Y)
      et = time.time()

      t.set_description(f"loss: {loss.numpy().item():.3f}")
      wandb.log({"train/loss": loss.numpy().item(),
                 "train/forward_time": et - st,
                 "lr": scheduler.get_lr().cpu().numpy().item(),
      })
    
    # "eval" loop. Evaluate every 4 epochs, starting with epoch 1
    if e % 4 == 1:
      eval_loss = []
      eval_times = []
      eval_top_1_acc = []
      eval_top_5_acc = []
      Tensor.training = False
      for X, Y in (t := tqdm(iterate(bs=BS, val=True, num_workers=16), total=steps_in_val_epoch)):
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
      })

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
  Tensor.training = True

  for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
    nm = f"train_{m}"
    if nm in globals():
      print(f"training {m}")
      globals()[nm]()
