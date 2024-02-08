from extra import dist
from tinygrad import GlobalCounters, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn.state import get_parameters, get_state_dict
from tinygrad.nn import optim, state
from tqdm import tqdm
import numpy as np
import random
import wandb
import time
import os

FP16 = getenv("FP16", 0)
BS = getenv("BS", 64)
EVAL_BS = getenv('EVAL_BS', BS)

# fp32 GPUS<=6 7900xtx can fit BS=112

def train_resnet():
  # TODO: Resnet50-v1.5
  from extra.models.resnet import ResNet50, ResNet18
  from examples.mlperf.dataloader import batch_load_resnet
  from extra.datasets.imagenet import get_train_files, get_val_files
  from extra.lr_scheduler import MultiStepLR, LR_Scheduler, Optimizer, PolynomialLR

  def sparse_categorical_crossentropy(out, Y, label_smoothing=0):
    num_classes = out.shape[-1]
    y_counter = Tensor.arange(num_classes, requires_grad=False, device=Y.device).unsqueeze(0).expand(Y.numel(), num_classes)
    y = (y_counter == Y.reshape(-1, 1)).where(-1.0 * num_classes, 0)
    y = y.reshape(*Y.shape, num_classes)
    return (1 - label_smoothing) * out.mul(y).mean() + (-1 * label_smoothing * out.mean())


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

  if FP16: dtypes.default_float = dtypes.float16

  if getenv("MOCKGPUS", 0):
    GPUS = tuple([f'{Device.DEFAULT}:{0}' for i in range(getenv("MOCKGPUS", 0))])
  else:
    GPUS = tuple([f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 1))])
  print(f"Training on {GPUS}")
  for x in GPUS: Device[x]

  num_classes = getenv("TEST_CATS", 1000)
  model = ResNet50(num_classes) if not getenv("SMALL") else ResNet18(num_classes)

  for v in get_parameters(model):
    v.to_(GPUS)
  parameters = get_parameters(model)

  @TinyJit
  def forward_step(X, Y):
    optimizer.zero_grad()
    X = normalize(X)
    out = model.forward(X)
    loss = sparse_categorical_crossentropy(out, Y, label_smoothing=0.1) * lr_scaler
    return loss.realize(), out.realize(), (out.argmax(-1) == Y).sum()
  # ** need to pass X and Y into backward step so that tinyjit can correctly replace the buffers in the backward tree!!! **
  # otherwise it will use stale buffer from the second call to forward forever in the backward pass
  @TinyJit
  def backward_step(X, Y, loss):
    scheduler.step()
    loss.backward()
    optimizer.step()
    pass

  input_mean = Tensor([0.485, 0.456, 0.406], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  input_std = Tensor([0.229, 0.224, 0.225], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)

  def normalize(x):
    x = x.permute([0, 3, 1, 2]).cast(dtypes.float32) / 255.0
    x -= input_mean
    x /= input_std
    return x.cast(dtypes.default_float)

  target = getenv("TARGET", 0.759)
  achieved = False
  lr_scaler = 8 if FP16 else 1
  lr_gamma = 0.1
  lr_steps = [30, 60, 80]
  lr_warmup_epochs = 5
  base_lr = 0.256 * (BS / 256)  # Linearly scale from BS=256, lr=0.256
  base_lr = 8.4 * (BS/2048)
  epochs = getenv("EPOCHS", 45)
  if getenv("LARS", 1):
    optimizer = optim.LARS(parameters, base_lr / lr_scaler, momentum=.9, weight_decay=2e-4)
  else:
    optimizer = optim.SGD(parameters, base_lr / lr_scaler, momentum=.9, weight_decay=2e-4)
  steps_in_train_epoch = (len(get_train_files()) // BS)
  steps_in_val_epoch = (len(get_val_files()) // EVAL_BS)
  #scheduler = MultiStepLR(optimizer, [m for m in lr_steps], gamma=lr_gamma, warmup=lr_warmup)
  scheduler = PolynomialLR(optimizer, base_lr, 1e-4, epochs=epochs * steps_in_train_epoch, warmup=lr_warmup_epochs * steps_in_train_epoch)
  print(f"training with batch size {BS} for {epochs} epochs")
  if getenv("TEST_LR", 0):
    for epoch in range(epochs):
      for i in range(steps_in_train_epoch):
        scheduler.step()
        if i % 1000 == 0: print(epoch, i, scheduler.get_lr().numpy())
    exit(0)

  train_state = {'model':model, 'scheduler':scheduler, 'optimizer':optimizer}
  def _get_state_dict():
    big_dict = state.get_state_dict(train_state)
    deduped = {}
    seen = set()
    for k, v in big_dict.items():
      if v in seen: continue
      seen.add(v)
      deduped[k] = v
    return deduped
  def _load_state_dict(state_dict):
    big_dict = state.get_state_dict(train_state)
    # hack: put back the dupes
    dupe_names = {}
    for k, v in big_dict.items():
      if v not in dupe_names:
        dupe_names[v] = k
        assert k in state_dict
      state_dict[k] = state_dict[dupe_names[v]]
    state.load_state_dict(train_state, state_dict)
  start_epoch = 0
  if ckpt:=getenv("RESUME", ""):
    print(f"resuming from {ckpt}")
    resume_dict = state.safe_load(ckpt)
    #print((resume_dict.keys()))
    _load_state_dict(resume_dict)
    start_epoch = int(scheduler.epoch_counter.numpy().item() / steps_in_train_epoch)
    print(f"resuming at epoch {start_epoch}")
  elif getenv("TESTEVAL"): model.load_from_pretrained()
  for v in get_parameters(model):
    # can't use mlb.to(tuple)...
    if not isinstance(v.device, tuple):
      v.to_(GPUS)

  # ** init wandb **
  wandb_config = {
    'BS': BS,
    'EVAL_BS': EVAL_BS,
    'base_lr': base_lr,
    'epochs': epochs,
    'classes': num_classes,
    'train_files': len(get_train_files()),
    'eval_files': len(get_train_files()),
    'steps_in_train_epoch': steps_in_train_epoch,
    'GPUS': GPUS,
    'FP16': FP16,
    'BEAM': getenv('BEAM'),
    'TEST_TRAIN': getenv('TEST_TRAIN'),
    'TEST_EVAL': getenv('TEST_EVAL'),
    'model': 'resnet50' if not getenv('SMALL') else 'resnet18',
    'optimizer': optimizer.__class__.__name__,
    'scheduler': scheduler.__class__.__name__,
  }
  wandb_tags = []
  wandb_tags.append(f'cats{num_classes}')
  wandb_tags.append('cats')
  if getenv("WANDB_RESUME", ""):
    wandb.init(id=getenv("WANDB_RESUME", ""), resume="must", config=wandb_config, tags=wandb_tags)
  else:
    wandb.init(config=wandb_config, tags=wandb_tags)

  for e in range(start_epoch, epochs):
    # train loop
    Tensor.training = True
    dt = time.perf_counter()

    iterator = iter(tqdm(t := batch_load_resnet(batch_size=BS, val=False, shuffle=True), total=steps_in_train_epoch))
    def data_get():
      x, y, cookie = next(iterator)
      return x.shard(GPUS, axis=0).realize(), Tensor(y).shard(GPUS, axis=0), cookie

    i, proc = 0, data_get()
    st = time.perf_counter()
    while proc is not None:
      if getenv("TESTEVAL"): break

      GlobalCounters.reset()
      proc = (forward_step(proc[0], proc[1]), (proc[0], proc[1]), proc[2])
      # the backward step should be realized by loss.numpy(), even though it doesn't depend on this.
      # doing this uses 16.38gb vs 15.55gb? why? because the grads get realized in optimizer.step, and the backward buffers are freed?
      fwet = time.perf_counter()
      backward_step(*proc[1], proc[0][0])

      et = time.perf_counter()
      dt = time.perf_counter()

      try:
        next_proc = data_get()
      except StopIteration:
        next_proc = None

      dte = time.perf_counter()

      proc, top_1_acc = proc[0][0].numpy(), proc[0][2].numpy().item() / BS  # return cookie
      loss_cpu = proc / lr_scaler
      cl = time.perf_counter()
      new_st = time.perf_counter()

      tqdm.write(
        f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(fwet - st) * 1000.0:7.2f}fw {(et - fwet) * 1000.0:7.2f}bw ms python, {(cl - dte) * 1000.0:7.2f} ms CL, {(dte - dt) * 1000.0:6.2f} ms fetch data, {loss_cpu:5.2f} loss, {top_1_acc:3.2f} acc, {optimizer.lr.numpy()[0] * lr_scaler:.6f} LR, {GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
      wandb.log({"lr": optimizer.lr.numpy() * lr_scaler,
                 "train/data_time": dte-dt,
                 "train/step_time": cl - st,
                 "train/python_time": et - st,
                 "train/cl_time": cl - dte,
                 "train/loss": loss_cpu,
                 "train/top_1_acc": top_1_acc,
                 "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st),
                 })

      st = new_st

      proc, next_proc = next_proc, None

      i += 1



    # "eval" loop. Evaluate every 4 epochs, starting with epoch 0
    if (e+1) % 1 == 0:
      eval_loss = []
      eval_times = []
      eval_top_1_acc = []
      eval_top_5_acc = []
      #Tensor.training = False  # disable to make kernels as similar as possible

      iterator = iter(tqdm(t := batch_load_resnet(batch_size=EVAL_BS, val=True, shuffle=False), total=steps_in_val_epoch))

      proc = data_get()

      while proc is not None:
        GlobalCounters.reset()
        st = time.time()

        proc = (forward_step(proc[0], proc[1]), proc[1], proc[2])

        try:
          next_proc = data_get()
        except StopIteration:
          next_proc = None

        top_1_acc = calculate_accuracy(proc[0][1], proc[1], 1)
        top_5_acc = calculate_accuracy(proc[0][1], proc[1], 5)

        eval_loss.append(proc[0][0].numpy().item())
        eval_top_1_acc.append(top_1_acc)
        eval_top_5_acc.append(top_5_acc)

        et = time.time()

        eval_times.append(et - st)

        proc, next_proc = next_proc, None  # drop cookie

      total_top_1 = sum(eval_top_1_acc) / len(eval_top_1_acc)
      tqdm.write(f"eval loss: {sum(eval_loss) / len(eval_loss):.2f}, eval time: {sum(eval_times) / len(eval_times):.2f}, eval top 1 acc: {sum(eval_top_1_acc) / len(eval_top_1_acc):.2f}, eval top 5 acc: {sum(eval_top_5_acc) / len(eval_top_5_acc):.2f}")
      wandb.log({"eval/loss": sum(eval_loss) / len(eval_loss),
                "eval/forward_time": sum(eval_times) / len(eval_times),
                "eval/top_1_acc": total_top_1,
                "eval/top_5_acc": sum(eval_top_5_acc) / len(eval_top_5_acc),
      })

      if not achieved and total_top_1 >= target:
        fn = f"./ckpts/{wandb_config['model']}_cats{num_classes}.safe"
        state.safe_save(state.get_state_dict(model), fn)
        print(f" *** WGMI {fn} ***")
        achieved = True

      if (e+1) % getenv("CKPT_EPOCHS", 4) == 0 or e + 1 == epochs:
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        if wandb.run is not None:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_{wandb.run.id}_e{e}.safe"
        else:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_e{e}.safe"
        print(f"saving ckpt to {fn}")
        state.safe_save(_get_state_dict(), fn)

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
      train_func = globals()[nm]
      break
  else:
    print("please specify MODEL")
    exit(1)

  train_func()
