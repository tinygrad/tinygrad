import functools
import os, sys
import time
from tqdm import tqdm
import multiprocessing

from tinygrad import Device, GlobalCounters, Tensor, TinyJit, dtypes
from tinygrad.helpers import colored, getenv, BEAM, WINO
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
  epochs            = config["epochs"]            = getenv("EPOCHS", 41)
  BS                = config["BS"]                = getenv("BS", 104 * len(GPUS))  # fp32 GPUS<=6 7900xtx can fit BS=112
  EVAL_BS           = config["EVAL_BS"]           = getenv("EVAL_BS", BS)
  base_lr           = config["base_lr"]           = getenv("LR", 8.5 * (BS/2048))
  lr_warmup_epochs  = config["lr_warmup_epochs"]  = getenv("WARMUP_EPOCHS", 5)
  decay             = config["decay"]             = getenv("DECAY", 2e-4)

  loss_scaler       = config["LOSS_SCALER"]       = getenv("LOSS_SCALER", 128.0 if dtypes.default_float == dtypes.float16 else 1.0)

  target, achieved  = getenv("TARGET", 0.759), False
  eval_start_epoch  = getenv("EVAL_START_EPOCH", 0)
  eval_epochs       = getenv("EVAL_EPOCHS", 1)

  steps_in_train_epoch  = config["steps_in_train_epoch"]  = (len(get_train_files()) // BS)
  steps_in_val_epoch    = config["steps_in_val_epoch"]    = (len(get_val_files()) // EVAL_BS)

  config["DEFAULT_FLOAT"] = dtypes.default_float.name
  config["BEAM"]    = BEAM.value
  config["WINO"]    = WINO.value
  config["SYNCBN"]  = getenv("SYNCBN")

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
        estimated_total_hours = median_step_time * steps_in_train_epoch * epochs / 60 / 60
        print(f"Estimated training time: {estimated_total_hours:.0f}h{(estimated_total_hours - int(estimated_total_hours)) * 60:.0f}m")
        # if we are doing beam search, run the first eval too
        if BEAM.value and e == start_epoch: break
        return

    # ** eval loop **
    if (e + 1 - eval_start_epoch) % eval_epochs == 0 and steps_in_val_epoch > 0:
      train_step.reset()  # free the train step memory :(
      eval_loss = []
      eval_times = []
      eval_top_1_acc = []
      Tensor.training = False

      it = iter(tqdm(batch_load_resnet(batch_size=EVAL_BS, val=True, shuffle=False), total=steps_in_val_epoch))
      proc = data_get(it)
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

        et = time.time()
        eval_times.append(et - st)

      eval_step.reset()
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
  EPOCHS = 10
  BS = 2
  BS_EVAL = 2
  WARMUP_EPOCHS = 1
  WARMUP_FACTOR = 0.001
  LR = 0.0001
  MAP_TARGET = 0.34
  from extra.models.retinanetNew import RetinaNet, AnchorGenerator
  from examples.mlperf.lr_schedulers import Retina_LR
  anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
  aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
  anchor_generator = AnchorGenerator(
      anchor_sizes, aspect_ratios
  )
  from extra.models.resnet import ResNeXt50_32X4D
  from tinygrad.nn.optim import Adam
  from extra.datasets.openimages_new import iterate, iterate_val
  from extra.datasets.openimages_new import get_openimages
  from pycocotools.cocoeval import COCOeval
  ROOT = 'extra/datasets/open-images-v6TEST'
  NAME = 'openimages-mlperf'
  coco_train = get_openimages(NAME,ROOT, 'train')
  coco_val = get_openimages(NAME,ROOT, 'val')
  # coco_eval = COCOeval(coco_val.coco, iouType="bbox")

  model = RetinaNet(ResNeXt50_32X4D(), num_anchors=anchor_generator.num_anchors_per_location()[0])
  mdlrun = TinyJit(lambda x: model(x, True))
  mdlrun_false = TinyJit(lambda x: model(x, False).realize())

  parameters = []
  for k, x in get_state_dict(model).items():
    if 'head' in k and ('clas' in k or 'reg' in k ):
      print(k)
      x.requires_grad = True
      parameters.append(x)
    else:
      x.requires_grad = False

  optimizer = Adam(parameters, lr=LR)

  @TinyJit
  def train_step(X, Y_b_P, Y_l_P, matched_idxs):
    Tensor.training = True
    optimizer.zero_grad()

    # b,r,c = mdlrun(X)
    b,r,c = model(X, True)

    loss_reg = mdl_reg_loss(r, Y_b_P, matched_idxs)
    loss_class = mdl_class_loss(c, Y_l_P, matched_idxs)
    loss = loss_reg+loss_class

    print(colored(f'loss_reg {loss_reg.numpy()}', 'green'))
    print(colored(f'loss_class {loss_class.numpy()}', 'green'))

    loss.backward()
    optimizer.step()
    return loss.realize()

  for epoch in range(EPOCHS):
    print(colored(f'EPOCH {epoch}/{EPOCHS}:', 'cyan'))
    train_step.reset()
    mdlrun_false.reset()
    lr_sched = None
    if epoch < WARMUP_EPOCHS:
      start_iter = epoch*len(coco_train.ids)//BS
      warmup_iters = WARMUP_EPOCHS*len(coco_train.ids)//BS
      lr_sched = Retina_LR(optimizer, start_iter, warmup_iters, WARMUP_FACTOR, LR)
    else:
      optimizer.lr.assign(Tensor([LR]))
    cnt = 0

    for X, Y_boxes, Y_labels, Y_boxes_p, Y_labels_p in iterate(coco_train, BS):
      if(cnt==0 and epoch==0):
        # INIT LOSS FUNC
        b,_,_ = mdlrun(X)
        ANCHORS = anchor_generator(X, b)
        ANCHORS = [a.realize() for a in ANCHORS]
        ANCHORS = Tensor.stack(ANCHORS)
        mdlrun.reset()
        # mdl_reg_loss_jit = TinyJit(lambda r, y, m: model.head.regression_head.loss(r,y,ANCHORS, m).realize())
        # mdl_class_loss_jit = TinyJit(lambda c, y,m: model.head.classification_head.loss(c,y,m).realize())
        mdl_reg_loss = lambda r, y, m: model.head.regression_head.loss(r,y,ANCHORS, m)
        mdl_class_loss = lambda c, y,m: model.head.classification_head.loss(c,y,m)

      st = time.time()
      cnt+=1
      # matcher_gen not jittable for now
      matched_idxs = model.matcher_gen(ANCHORS, Y_boxes).realize()
      loss = train_step(X, Y_boxes_p, Y_labels_p, matched_idxs)
      if lr_sched is not None:
        lr_sched.step()

      print(colored(f'{cnt} STEP {loss.numpy()} || {time.time()-st} || LR: {optimizer.lr.item()}', 'magenta'))
      if cnt>5: 
        train_step.reset()
        break
    # ****EVAL STEP
    print(colored(f'{epoch} START EVAL', 'cyan'))
    coco_eval = COCOeval(coco_val.coco, iouType="bbox")
    Tensor.training = False
    train_step.reset()
    st = time.time()
    coco_evalimgs, evaluated_imgs, ncats, narea = [], [], len(coco_eval.params.catIds), len(coco_eval.params.areaRng)
    cnt = 0
    for X, targets in iterate_val(coco_val, BS_EVAL):
      orig_shapes= []
      for tt in targets:
        orig_shapes.append(list(tt['image_size']))
      # print(orig_shapes)
      sub_t = time.time()
      b,r,c = mdlrun(X)
      m_t = time.time()-sub_t
      print('MODEL_RUN_JIT', m_t)
      if cnt==0 and epoch==0:
        ANCHORS_VAL = anchor_generator(X, b)
        ANCHORS_VAL = [a.realize() for a in ANCHORS_VAL]
        ANCHORS_VAL = Tensor.stack(ANCHORS_VAL)
        num_anchors_per_level = [xx.shape[2] * xx.shape[3] for xx in b]
        HW = 0
        for v in num_anchors_per_level:
          HW += v
        HWA = c.shape[1]
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]
        split_anchors = [list(a.split(num_anchors_per_level)) for a in ANCHORS_VAL]
        sa = []
        for aa in split_anchors:
          s_temp = []
          for a in aa:
            s_temp.append(a.numpy())
          sa.append(s_temp)
        split_anchors = sa
      c_split = list(c.sigmoid().split(num_anchors_per_level, dim=1))
      r_split = list(r.split(num_anchors_per_level, dim=1))
      c_split = [cc.numpy() for cc in c_split]
      r_split = [rr.numpy() for rr in r_split]
      ps_t = time.time() - m_t-sub_t
      print('POST_SPLITS', ps_t)
      # print(split_anchors)
      # print('PRE_POST_PROCESS')
      predictions = model.postprocess_detections_val(c_split, r_split, split_anchors, orig_shapes)
      # out = mdlrun_false(X).numpy()
      # out = r.cat(c.sigmoid(), dim=-1).numpy()
      # predictions = model.postprocess_detections(out, orig_image_sizes=[t["image_size"] for t in targets])
      # print(predictions)
      img_ids = [t["image_id"] for t in targets]
      coco_results  = [{"image_id": targets[i]["image_id"], "category_id": label, "bbox": box.tolist(), "score": score} for i, prediction in enumerate(predictions) for box, score, label in zip(*prediction.values())]

      # print('coco_results',len(coco_results))
      with redirect_stdout(None):
        coco_eval.cocoDt = coco_val.coco.loadRes(coco_results)
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
      evaluated_imgs.extend(img_ids)
      coco_evalimgs.append(np.array(coco_eval.evalImgs).reshape(ncats, narea, len(img_ids)))
      print(colored(f'{cnt} EVAL_STEP || {time.time()-sub_t}', 'red'))
      cnt=cnt+1
    coco_eval.params.imgIds = evaluated_imgs
    coco_eval._paramsEval.imgIds = evaluated_imgs
    coco_eval.evalImgs = list(np.concatenate(coco_evalimgs, -1).flatten())
    coco_eval.accumulate()
    coco_eval.summarize()
    eval_acc = coco_eval.stats[0]
    print(colored(f'{epoch} EVAL_ACC {eval_acc} || {time.time()-st}', 'green'))
    mdlrun.reset()

      
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
