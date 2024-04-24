import functools
import os, sys
import time
from tqdm import tqdm
import multiprocessing

from tinygrad import Device, GlobalCounters, Tensor, TinyJit, dtypes
from tinygrad.helpers import colored, getenv, BEAM, WINO, Context
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save
from tinygrad.nn.optim import LARS, SGD, OptimizerGroup

from extra.lr_scheduler import LRSchedulerGroup
from examples.mlperf.helpers import get_training_state, load_training_state

def train_resnet():
  from extra.models import resnet
  from examples.mlperf.dataloader import batch_load_resnet
  from extra.datasets.imagenet import get_train_files, get_val_files
  from examples.mlperf.lr_schedulers import PolynomialDecayWithWarmup
  from examples.mlperf.initializers import Conv2dHeNormal, Linear
  from examples.hlb_cifar10 import UnsyncedBatchNorm

  config = {}
  seed = config["seed"] = getenv("SEED", 42)
  Tensor.manual_seed(seed)  # seed for weight initialization

  GPUS = config["GPUS"] = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  print(f"Training on {GPUS}")
  for x in GPUS: Device[x]

  TRAIN_BEAM = getenv("TRAIN_BEAM", BEAM.value)
  EVAL_BEAM = getenv("EVAL_BEAM", BEAM.value)

  # ** model definition and initializers **
  num_classes = 1000
  resnet.Conv2d = Conv2dHeNormal
  resnet.Linear = Linear
  if not getenv("SYNCBN"): resnet.BatchNorm = functools.partial(UnsyncedBatchNorm, num_devices=len(GPUS))
  model = resnet.ResNet50(num_classes)

  # shard weights and initialize in order
  for k, x in get_state_dict(model).items():
    if not getenv("SYNCBN") and ("running_mean" in k or "running_var" in k):
      x.realize().shard_(GPUS, axis=0)
    else:
      x.realize().to_(GPUS)
  parameters = get_parameters(model)

  # ** hyperparameters **
  epochs            = config["epochs"]            = getenv("EPOCHS", 37)
  BS                = config["BS"]                = getenv("BS", 104 * len(GPUS))  # fp32 GPUS<=6 7900xtx can fit BS=112
  EVAL_BS           = config["EVAL_BS"]           = getenv("EVAL_BS", BS)
  base_lr           = config["base_lr"]           = getenv("LR", 7.4 * (BS/1536))
  lr_warmup_epochs  = config["lr_warmup_epochs"]  = getenv("WARMUP_EPOCHS", 2)
  decay             = config["decay"]             = getenv("DECAY", 5e-5)

  loss_scaler       = config["LOSS_SCALER"]       = getenv("LOSS_SCALER", 128.0 if dtypes.default_float == dtypes.float16 else 1.0)

  target, achieved  = getenv("TARGET", 0.759), False
  eval_start_epoch  = getenv("EVAL_START_EPOCH", 0)
  eval_epochs       = getenv("EVAL_EPOCHS", 1)

  steps_in_train_epoch  = config["steps_in_train_epoch"]  = (len(get_train_files()) // BS)
  steps_in_val_epoch    = config["steps_in_val_epoch"]    = (len(get_val_files()) // EVAL_BS)

  config["DEFAULT_FLOAT"] = dtypes.default_float.name
  config["BEAM"]          = BEAM.value
  config["TRAIN_BEAM"]    = TRAIN_BEAM
  config["EVAL_BEAM"]     = EVAL_BEAM
  config["WINO"]          = WINO.value
  config["SYNCBN"]        = getenv("SYNCBN")

  # ** Optimizer **
  skip_list = [v for k, v in get_state_dict(model).items() if "bn" in k or "bias" in k or "downsample.1" in k]
  parameters = [x for x in parameters if x not in set(skip_list)]
  optimizer = LARS(parameters, base_lr, momentum=.9, weight_decay=decay)
  optimizer_skip = SGD(skip_list, base_lr, momentum=.9, weight_decay=0.0, classic=True)
  optimizer_group = OptimizerGroup(optimizer, optimizer_skip)

  # ** LR scheduler **
  scheduler = PolynomialDecayWithWarmup(optimizer, initial_lr=base_lr, end_lr=1e-4,
                                        train_steps=epochs * steps_in_train_epoch,
                                        warmup=lr_warmup_epochs * steps_in_train_epoch)
  scheduler_skip = PolynomialDecayWithWarmup(optimizer_skip, initial_lr=base_lr, end_lr=1e-4,
                                             train_steps=epochs * steps_in_train_epoch,
                                             warmup=lr_warmup_epochs * steps_in_train_epoch)
  scheduler_group = LRSchedulerGroup(scheduler, scheduler_skip)
  print(f"training with batch size {BS} for {epochs} epochs")

  # ** resume from checkpointing **
  start_epoch = 0
  if ckpt:=getenv("RESUME", ""):
    load_training_state(model, optimizer_group, scheduler_group, safe_load(ckpt))
    start_epoch = int(scheduler.epoch_counter.numpy().item() / steps_in_train_epoch)
    print(f"resuming from {ckpt} at epoch {start_epoch}")

  # ** init wandb **
  WANDB = getenv("WANDB")
  if WANDB:
    import wandb
    wandb_args = {"id": wandb_id, "resume": "must"} if (wandb_id := getenv("WANDB_RESUME", "")) else {}
    wandb.init(config=config, **wandb_args)

  BENCHMARK = getenv("BENCHMARK")

  # ** jitted steps **
  input_mean = Tensor([123.68, 116.78, 103.94], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  # mlperf reference resnet does not divide by input_std for some reason
  # input_std = Tensor([0.229, 0.224, 0.225], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  def normalize(x): return (x.permute([0, 3, 1, 2]) - input_mean).cast(dtypes.default_float)
  @TinyJit
  def train_step(X, Y):
    optimizer_group.zero_grad()
    X = normalize(X)
    out = model.forward(X)
    loss = out.cast(dtypes.float32).sparse_categorical_crossentropy(Y, label_smoothing=0.1)
    top_1 = (out.argmax(-1) == Y).sum()
    (loss * loss_scaler).backward()
    for t in optimizer_group.params: t.grad = t.grad.contiguous() / loss_scaler
    optimizer_group.step()
    scheduler_group.step()
    return loss.realize(), top_1.realize()

  @TinyJit
  def eval_step(X, Y):
    X = normalize(X)
    out = model.forward(X)
    loss = out.cast(dtypes.float32).sparse_categorical_crossentropy(Y, label_smoothing=0.1)
    top_1 = (out.argmax(-1) == Y).sum()
    return loss.realize(), top_1.realize()

  def data_get(it):
    x, y, cookie = next(it)
    return x.shard(GPUS, axis=0).realize(), Tensor(y, requires_grad=False).shard(GPUS, axis=0), cookie

  # ** epoch loop **
  step_times = []
  for e in range(start_epoch, epochs):
    # ** train loop **
    Tensor.training = True
    BEAM.value = TRAIN_BEAM
    batch_loader = batch_load_resnet(batch_size=BS, val=False, shuffle=True, seed=seed*epochs + e)
    it = iter(tqdm(batch_loader, total=steps_in_train_epoch, desc=f"epoch {e}", disable=BENCHMARK))
    i, proc = 0, data_get(it)
    st = time.perf_counter()
    while proc is not None:
      GlobalCounters.reset()
      (loss, top_1_acc), proc = train_step(proc[0], proc[1]), proc[2]

      pt = time.perf_counter()

      try:
        next_proc = data_get(it)
      except StopIteration:
        next_proc = None

      dt = time.perf_counter()

      device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
      loss, top_1_acc = loss.numpy().item(), top_1_acc.numpy().item() / BS

      cl = time.perf_counter()
      if BENCHMARK:
        step_times.append(cl - st)

      tqdm.write(
        f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms fetch data, "
        f"{(cl - dt) * 1000.0:7.2f} ms {device_str}, {loss:5.2f} loss, {top_1_acc:3.2f} acc, {optimizer.lr.numpy()[0]:.6f} LR, "
        f"{GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
      if WANDB:
        wandb.log({"lr": optimizer.lr.numpy(), "train/loss": loss, "train/top_1_acc": top_1_acc, "train/step_time": cl - st,
                   "train/python_time": pt - st, "train/data_time": dt - pt, "train/cl_time": cl - dt,
                   "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st), "epoch": e + (i + 1) / steps_in_train_epoch})

      st = cl
      proc, next_proc = next_proc, None  # return old cookie
      i += 1

      if i == BENCHMARK:
        median_step_time = sorted(step_times)[(BENCHMARK + 1) // 2]  # in seconds
        estimated_total_minutes = int(median_step_time * steps_in_train_epoch * epochs / 60)
        print(f"Estimated training time: {estimated_total_minutes // 60}h{estimated_total_minutes % 60}m")
        print(f"epoch global_ops: {steps_in_train_epoch * GlobalCounters.global_ops:_}, "
              f"epoch global_mem: {steps_in_train_epoch * GlobalCounters.global_mem:_}")
        # if we are doing beam search, run the first eval too
        if (TRAIN_BEAM or EVAL_BEAM) and e == start_epoch: break
        return

    # ** eval loop **
    if (e + 1 - eval_start_epoch) % eval_epochs == 0 and steps_in_val_epoch > 0:
      if getenv("RESET_STEP", 1): train_step.reset()  # free the train step memory :(
      eval_loss = []
      eval_times = []
      eval_top_1_acc = []
      Tensor.training = False
      BEAM.value = EVAL_BEAM

      it = iter(tqdm(batch_load_resnet(batch_size=EVAL_BS, val=True, shuffle=False), total=steps_in_val_epoch))
      i, proc = 0, data_get(it)
      while proc is not None:
        GlobalCounters.reset()
        st = time.time()

        (loss, top_1_acc), proc = eval_step(proc[0], proc[1]), proc[2]  # drop inputs, keep cookie

        try:
          next_proc = data_get(it)
        except StopIteration:
          next_proc = None

        loss, top_1_acc = loss.numpy().item(), top_1_acc.numpy().item() / EVAL_BS
        eval_loss.append(loss)
        eval_top_1_acc.append(top_1_acc)
        proc, next_proc = next_proc, None  # return old cookie
        i += 1
        if i == BENCHMARK: return

        et = time.time()
        eval_times.append(et - st)

      if getenv("RESET_STEP", 1): eval_step.reset()
      total_loss = sum(eval_loss) / len(eval_loss)
      total_top_1 = sum(eval_top_1_acc) / len(eval_top_1_acc)
      total_fw_time = sum(eval_times) / len(eval_times)
      tqdm.write(f"eval loss: {total_loss:.2f}, eval time: {total_fw_time:.2f}, eval top 1 acc: {total_top_1:.3f}")
      if WANDB:
        wandb.log({"eval/loss": total_loss, "eval/top_1_acc": total_top_1, "eval/forward_time": total_fw_time, "epoch": e + 1})

      # save model if achieved target
      if not achieved and total_top_1 >= target:
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        fn = f"./ckpts/resnet50.safe"
        safe_save(get_state_dict(model), fn)
        print(f" *** Model saved to {fn} ***")
        achieved = True

      # checkpoint every time we eval
      if getenv("CKPT"):
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        if WANDB and wandb.run is not None:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_{wandb.run.id}_e{e}.safe"
        else:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_e{e}.safe"
        print(f"saving ckpt to {fn}")
        safe_save(get_training_state(model, optimizer_group, scheduler_group), fn)

def train_retinanet():
  from contextlib import redirect_stdout
  import numpy as np
  import math
  WANDB = getenv('WANDB')
  # WANDB = False
  HOSTNAME = getenv('SLURM_STEP_NODELIST', '3080')
  EPOCHS = 100

  BS = getenv('BS', 52)

  BS_EVAL = getenv('BS_EVAL', 52)
  
  WARMUP_EPOCHS = 1
  WARMUP_FACTOR = 0.001
  LR = 0.0001
  MAP_TARGET = 0.34
  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  SYNCBN = False
  SYNCBN = True
  TRAIN_BEAM = getenv("TRAIN_BEAM", BEAM.value)
  EVAL_BEAM = getenv("EVAL_BEAM", BEAM.value)
  # dtypes.default_float = dtypes.bfloat16
  loss_scaler = 128.0 if dtypes.default_float in [dtypes.float16, dtypes.bfloat16] else 1.0
  print('LOSS_SCALER', loss_scaler)
  if WANDB:
    import wandb
    wandb.init(project='RetinaNet')
  print(f"Training on {GPUS}")
  for x in GPUS: Device[x]
  from extra.models import retinanet
  from examples.mlperf.initializers import Conv2dNormal, Conv2dKaiming

  prior_probability=0.01
  retinanet.Conv2dNormal = Conv2dNormal
  retinanet.Conv2dNormal_prior_prob = functools.partial(Conv2dNormal, b=-math.log((1 - prior_probability) / prior_probability))
  retinanet.Conv2dKaiming = Conv2dKaiming

  from extra.models.retinanet import AnchorGenerator
  from examples.mlperf.lr_schedulers import Retina_LR
  anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
  aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
  anchor_generator = AnchorGenerator(
      anchor_sizes, aspect_ratios
  )

  from extra.models import resnet
  from examples.hlb_cifar10 import UnsyncedBatchNorm
  from tinygrad.nn.optim import Adam
  from extra.datasets.openimages_new import get_openimages, batch_load_retinanet, batch_load_retinanet_val
  from pycocotools.cocoeval import COCOeval
  from pycocotools.coco import COCO
  ROOT = 'extra/datasets/open-images-v6TEST'
  NAME = 'openimages-mlperf'
  coco_train = get_openimages(NAME,ROOT, 'train')
  coco_val = get_openimages(NAME,ROOT, 'val')

  if not SYNCBN: resnet.BatchNorm = functools.partial(UnsyncedBatchNorm, num_devices=len(GPUS))
  model = retinanet.RetinaNet(resnet.ResNeXt50_32X4D(), num_anchors=anchor_generator.num_anchors_per_location()[0])
  model.backbone.body.fc = None

  parameters = []

  for k, v in get_state_dict(model).items():
    # print(k, v.shape)
    # print(v.dtype)
    # print(k)
    if 'head' in k and ('clas' in k or 'reg' in k ):
      v.requires_grad = True
    elif k.split('.')[2] in ["layer4", "layer3", "layer2"] and 'bn' not in k and 'weight' in k:
    # elif k.split('.')[2] in ["layer4"] and 'bn' not in k and 'down' not in k:
      if 'downsample' in k:
        if 'downsample.0' in k:
          # print(k)
          v.requires_grad = True
        else:
          v.requires_grad = False
      else:
        # print(k)
        v.requires_grad = True
    elif 'fpn' in k:
      # print(k)
      v.requires_grad = True
    else:
      v.requires_grad = False
    if not SYNCBN and ("running_mean" in k or "running_var" in k):
      v.shard_(GPUS, axis=0)
    else:
      v.to_(GPUS)
  # model.load_checkpoint("./ckpts/retinanet_4xgpu020_B100_E0_11703.safe")

  # model.load_checkpoint("./ckpts/retinanet_4xgpu020_B52_E0x75.safe")
  # model.load_from_pretrained()
  for k, v in get_state_dict(model).items():
    if v.requires_grad:
      print(k)
  parameters = get_parameters(model)
  optimizer = Adam(parameters, lr=LR)

  image_std = Tensor([0.229, 0.224, 0.225], device=GPUS, dtype=dtypes.float32).reshape(1,-1,1,1)
  image_mean = Tensor([0.485, 0.456, 0.406], device=GPUS, dtype=dtypes.float32).reshape(1,-1,1,1)
  def normalize(x):
    return (((x.permute((0,3,1,2)) / 255.0) - image_mean)/image_std).cast(dtypes.default_float)
    x = x.permute((0,3,1,2)) / 255.0
    x -= image_mean
    x /= image_std
    return x.cast(dtypes.default_float)#.realize()
  @TinyJit
  def train_step(X, boxes_temp, labels_temp, matched_idxs):
    Tensor.training = True
    optimizer.zero_grad()
    b,r,c = model(normalize(X), True)
    loss_reg = mdl_reg_loss(r, matched_idxs, boxes_temp)
    loss_class = mdl_class_loss(c, matched_idxs, labels_temp)
    loss = loss_reg+loss_class
    (loss*loss_scaler).backward()
    for t in optimizer.params: t.grad = t.grad.contiguous() / loss_scaler
    optimizer.step()
    return loss.realize()
  @TinyJit
  def val_step(X):
    Tensor.training = False
    out = model(normalize(X)).cast(dtypes.float32)
    return out.realize()

  feature_shapes = [(100, 100), (50, 50), (25, 25), (13, 13), (7, 7)]
  ANCHORS = anchor_generator((BS,3,800,800), feature_shapes)
  # ANCHORS = [a.realize() for a in ANCHORS]
  ANCHORS_STACK = Tensor.stack(ANCHORS)
  ANCHORS_STACK = ANCHORS_STACK.shard(GPUS, axis=0)
  ANCHOR_NP = ANCHORS[0].numpy()
  mdl_reg_loss = lambda r, m, b_t: model.head.regression_head.loss(r,ANCHORS_STACK, m, b_t)
  mdl_class_loss = lambda c, m, l_t: model.head.classification_head.loss(c,m, l_t)
  def data_get(it):
    x, yb, yl, ym, cookie = next(it)
    return x.shard(GPUS, axis=0), yb.shard(GPUS, axis=0), yl.shard(GPUS, axis=0), ym.shard(GPUS, axis=0), cookie 
  def data_get_val(it):
    x, Y_idx, cookie = next(it)
    return x.shard(GPUS, axis=0), Y_idx, cookie
  # b SHAPE for anchor_gen
  # (44, 256, 100, 100)
  # (44, 256, 50, 50)
  # (44, 256, 25, 25)
  # (44, 256, 13, 13)
  # (44, 256, 7, 7)

  # (52, 256, 100, 100)
  # (52, 256, 50, 50)
  # (52, 256, 25, 25)
  # (52, 256, 13, 13)
  # (52, 256, 7, 7)
  for epoch in range(EPOCHS):
    
    print(colored(f'EPOCH {epoch}/{EPOCHS}:', 'cyan'))

    # **********TRAIN***************
    Tensor.training = True
    BEAM.value = TRAIN_BEAM
    # train_step.reset()
    # mdlrun_false.reset()
    lr_sched = None
    # if epoch < WARMUP_EPOCHS:
    #   start_iter = epoch*len(coco_train.ids)//BS
    #   warmup_iters = WARMUP_EPOCHS*len(coco_train.ids)//BS
    #   lr_sched = Retina_LR(optimizer, start_iter, warmup_iters, WARMUP_FACTOR, LR)
    # else:
    #   optimizer.lr.assign(Tensor([LR], device=GPUS))
    batch_loader = batch_load_retinanet(coco_train, bs=BS, val=False, shuffle=False, anchor_np=ANCHOR_NP)
    it = iter(tqdm(batch_loader, total=len(coco_train)//BS, desc=f"epoch {epoch}"))
    cnt, proc = 0, data_get(it)

    st = time.perf_counter()
    while proc is not None:
      GlobalCounters.reset()
      loss, proc = train_step(proc[0], proc[1], proc[2], proc[3]), proc[4]

      pt = time.perf_counter()
      try:
        next_proc = data_get(it)
      except StopIteration:
        next_proc = None

      dt = time.perf_counter()

      device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
      loss = loss.numpy().item()
      # if lr_sched is not None:
      #   lr_sched.step()

      cl = time.perf_counter()

      tqdm.write(
        f"{cnt:5} {((cl - st)) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms fetch data, "
        f"{(cl - dt) * 1000.0:7.2f} ms {device_str}, {loss:5.2f} loss, {optimizer.lr.numpy()[0]:.6f} LR, "
        f"{GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
      if WANDB:
        wandb.log({"lr": optimizer.lr.numpy(), "train/loss": loss, "train/step_time": cl - st,
                   "train/python_time": pt - st, "train/data_time": dt - pt, "train/cl_time": cl - dt,
                   "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st), "epoch": epoch + (cnt + 1) / (len(coco_train)//BS)})

      st = cl
      proc, next_proc = next_proc, None  # return old cookie
      cnt+=1

      if cnt%1000==0:
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        fn = f"./ckpts/retinanet_{len(GPUS)}x{HOSTNAME}_B{BS}_E{epoch}_{cnt}.safe"
        state_dict = get_state_dict(model)
        # print(state_dict.keys())
        safe_save(state_dict, fn)
        print(f" *** Model saved to {fn} ***")
    if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
    fn = f"./ckpts/retinanet_{len(GPUS)}x{HOSTNAME}_B{BS}_E{epoch}_{cnt}.safe"
    state_dict = get_state_dict(model)
    # print(state_dict.keys())
    safe_save(state_dict, fn)
    print(f" *** Model saved to {fn} ***")
    # ***********EVAL******************
    bt = time.time()
    train_step.reset()
    Tensor.training = False
    BEAM.value = EVAL_BEAM
    print(colored(f'{epoch} START EVAL', 'cyan'))
    coco_eval = COCOeval(coco_val.coco, iouType="bbox")
    eval_times = []
    coco_evalimgs, evaluated_imgs, ncats, narea = [], [], len(coco_eval.params.catIds), len(coco_eval.params.areaRng)

    batch_loader = batch_load_retinanet_val(coco_val, bs=BS_EVAL, val=True, shuffle=False)
    it = iter(tqdm(batch_loader, total=len(coco_val)//BS_EVAL, desc=f"epoch_val {epoch}"))
    cnt, proc = 0, data_get_val(it)

    while proc is not None:
      GlobalCounters.reset()
      st = time.time()
      y_idxs = proc[1]
      orig_shapes = []
      img_ids = []
      for i in y_idxs:
        img_ids.append(coco_val.ids[i])
        orig_shapes.append(coco_val.__getitem__(i)[0].size[::-1])

      out, proc = val_step(proc[0]), proc[2]

      try:
        next_proc = data_get_val(it)
      except StopIteration:
        next_proc = None

      out = out.numpy()
      predictions = model.postprocess_detections(out, orig_image_sizes=orig_shapes)

      coco_results  = [{"image_id": img_ids[i], "category_id": label, "bbox": box.tolist(),
                         "score": score} for i, prediction in enumerate(predictions) 
                         for box, score, label in zip(*prediction.values())]
      # print('len_reults', len(coco_results))
      # IF COCO_RESULTS LOWER THAN THRESH, ERROR IN EVAL
      # REFERNCE PUSHES EMPTY COCO OBJ
      with redirect_stdout(None):
        coco_eval.cocoDt = coco_val.coco.loadRes(coco_results) if coco_results else COCO()
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
      evaluated_imgs.extend(img_ids)
      coco_evalimgs.append(np.array(coco_eval.evalImgs).reshape(ncats, narea, len(img_ids)))
      print(colored(f'{cnt} EVAL_STEP || {time.time()-st}', 'red'))
      cnt=cnt+1
      proc, next_proc = next_proc, None  # return old cookie
      if cnt>30: break
    coco_eval.params.imgIds = evaluated_imgs
    coco_eval._paramsEval.imgIds = evaluated_imgs
    coco_eval.evalImgs = list(np.concatenate(coco_evalimgs, -1).flatten())
    coco_eval.accumulate()
    coco_eval.summarize()
    eval_acc = coco_eval.stats[0]
    print(colored(f'{epoch} EVAL_ACC {eval_acc} || {time.time()-bt}', 'green'))
    val_step.reset()

      
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
  multiprocessing.set_start_method('spawn')
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()