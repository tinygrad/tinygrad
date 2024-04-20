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
    with Context(BEAM=TRAIN_BEAM):
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
    with Context(BEAM=EVAL_BEAM):
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
        estimated_total_minutes = int(median_step_time * steps_in_train_epoch * epochs / 60)
        print(f"Estimated training time: {estimated_total_minutes // 60}h{estimated_total_minutes % 60}m")
        # if we are doing beam search, run the first eval too
        if (TRAIN_BEAM or EVAL_BEAM) and e == start_epoch: break
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
  WANDB = True
  HOSTNAME = getenv('SLURM_STEP_NODELIST', '3080')
  EPOCHS = 100
  BS = 16 # A100x2
  BS=2*4*4
  BS=44
  # BS = 3*6
  # BS = 5*8
  # BS = 2*4
  BS_EVAL = 16
  # BS_EVAL = 2*4
  WARMUP_EPOCHS = 1
  WARMUP_FACTOR = 0.001
  LR = 0.0001
  MAP_TARGET = 0.34
  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  SYNCBN = False
  SYNCBN = True
  # dtypes.default_float = dtypes.bfloat16
  loss_scaler = 128.0 if dtypes.default_float in [dtypes.float16, dtypes.bfloat16] else 1.0
  if WANDB:
    import wandb
    wandb.init(project='RetinaNet')
  print(f"Training on {GPUS}")
  for x in GPUS: Device[x]
  from extra.models.retinanetNew import RetinaNet, AnchorGenerator
  from examples.mlperf.lr_schedulers import Retina_LR
  anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
  aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
  anchor_generator = AnchorGenerator(
      anchor_sizes, aspect_ratios
  )
  # from extra.models.resnet import ResNeXt50_32X4D
  from extra.models import resnet
  from examples.hlb_cifar10 import UnsyncedBatchNorm
  from tinygrad.nn.optim import Adam
  from extra.datasets.openimages_new import iterate, iterate_val, MATCHER_FUNC
  from extra.datasets.openimages_new import get_openimages
  from pycocotools.cocoeval import COCOeval
  from pycocotools.coco import COCO
  ROOT = 'extra/datasets/open-images-v6TEST'
  NAME = 'openimages-mlperf'
  coco_train = get_openimages(NAME,ROOT, 'train')
  coco_val = get_openimages(NAME,ROOT, 'val')

  if not SYNCBN: resnet.BatchNorm = functools.partial(UnsyncedBatchNorm, num_devices=len(GPUS))
  model = RetinaNet(resnet.ResNeXt50_32X4D(), num_anchors=anchor_generator.num_anchors_per_location()[0])
  model.backbone.body.fc = None
  mdlrun = TinyJit(lambda x: model(x, True))
  # mdlrun_false = TinyJit(lambda x: model(x, False).realize())

  parameters = []
  # model.load_from_pretrained()
  # model.load_checkpoint("./ckpts/retinanet_B16_E10.safe")

  for k, v in get_state_dict(model).items():
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
      v.realize().shard_(GPUS, axis=0)
    else:
      v.realize().to_(GPUS)

  for k, v in get_state_dict(model).items():
    if v.requires_grad:
      print(k)
  parameters = get_parameters(model)
  optimizer = Adam(parameters, lr=LR)

  @TinyJit
  def train_step(X, Y_b_P, Y_l_P, matched_idxs, boxes_temp, labels_temp):
    Tensor.training = True
    optimizer.zero_grad()

    # b,r,c = mdlrun(X)
    b,r,c = model(X, True)

    loss_reg = mdl_reg_loss(r, Y_b_P, matched_idxs, boxes_temp)
    loss_class = mdl_class_loss(c, Y_l_P, matched_idxs, labels_temp)
    loss = (loss_reg+loss_class)*loss_scaler

    # print(colored(f'loss_reg {loss_reg.numpy()}', 'green'))
    # print(colored(f'loss_class {loss_class.numpy()}', 'green'))

    loss.backward()
    for t in optimizer.params: t.grad = t.grad.contiguous() / loss_scaler
    optimizer.step()
    return loss.realize()
  @TinyJit
  def val_step(X):
    Tensor.training = False
    out = model(X)
    return out.realize()
  func = None
  for epoch in range(EPOCHS):
    print(colored(f'EPOCH {epoch}/{EPOCHS}:', 'cyan'))
    # train_step.reset()
    # mdlrun_false.reset()
    lr_sched = None
    if epoch < WARMUP_EPOCHS:
      start_iter = epoch*len(coco_train.ids)//BS
      warmup_iters = WARMUP_EPOCHS*len(coco_train.ids)//BS
      lr_sched = Retina_LR(optimizer, start_iter, warmup_iters, WARMUP_FACTOR, LR)
    else:
      optimizer.lr.assign(Tensor([LR], device=GPUS))
    cnt = 0
    data_end = time.perf_counter()
    for X, Y_boxes, Y_labels, Y_boxes_p, Y_labels_p, M_IDXS in iterate(coco_train, BS, func):
      # print('Global reset')
      GlobalCounters.reset()
      data_time = time.perf_counter() - data_end
      st = time.perf_counter()
      X = X.shard(GPUS, axis=0)
      if(cnt==0 and epoch==0):
        # o_x, oyb, oyl, omi = X, Y_boxes_p, Y_labels_p, M_IDXS
        # INIT LOSS FUNC
        b,_,_ = mdlrun(X)
        for bb in b:
          print(bb.shape)
        ANCHORS = anchor_generator(X.shape, b)
        ANCHORS = [a.realize() for a in ANCHORS]
        ANCHORS_STACK = Tensor.stack(ANCHORS)
        # print('ANCHOR_STACK', ANCHORS[0].shape)
        # sys.exit()
        ANCHORS_STACK = ANCHORS_STACK.shard(GPUS, axis=0)
        func = lambda x: model.matcher_gen_per_img(ANCHORS[0], x)
        # print(func)
        mdlrun.reset()
        # mdl_reg_loss_jit = TinyJit(lambda r, y, m: model.head.regression_head.loss(r,y,ANCHORS, m).realize())
        # mdl_class_loss_jit = TinyJit(lambda c, y,m: model.head.classification_head.loss(c,y,m).realize())
        mdl_reg_loss = lambda r, y, m, b_t: model.head.regression_head.loss(r,y,ANCHORS_STACK, m, b_t)
        mdl_class_loss = lambda c, y,m, l_t: model.head.classification_head.loss(c,y,m, l_t)
        break

      else:
        # print('ANCHORS', len(ANCHORS), ANCHORS[0].mean().item(), ANCHORS[1].mean().item())
        # matcher_gen not jittable for now
        pre_match = time.perf_counter()
        # matched_idxs = model.matcher_gen(ANCHORS, Y_boxes) #.realize()
        # matched_idxs = Tensor.stack(M_IDXS).realize()
        matched_idxs = M_IDXS
        # print('MATCHED_IDX', matched_idxs.mean().item())
        pre_zip = time.perf_counter()

        # Work around for zipping a sharded tensor
        # Need to think of clean way to execute
        # boxes_temp = []
        # for tb, m in zip(Y_boxes_p, matched_idxs):
        #   boxes_temp.append(tb[m].realize())
        # boxes_temp = Tensor.stack(boxes_temp)
        # labels_temp = []
        # for tl, m in zip(Y_labels_p, matched_idxs):
        #   labels_temp.append(tl[m].realize())
        # labels_temp = Tensor.stack(labels_temp)
        # boxes_temp = Tensor.stack(Y_boxes_p).realize()
        # labels_temp = Tensor.stack(Y_labels_p).realize()
        boxes_temp = Y_boxes_p
        labels_temp = Y_labels_p
        # print('matched_IDXS', matched_idxs.shape, len(Y_boxes))
        pre_shard = time.perf_counter()
        # print('MATH_DEVICE', matched_idxs.device)
        matched_idxs = matched_idxs.shard(GPUS, axis=0)
        # print('POST_MATH_DEVICE', matched_idxs.device)
        # # Y_boxes_p.shard_(GPUS, axis=0)
        # # Y_labels_p.shard_(GPUS, axis=0)
        boxes_temp = boxes_temp.shard(GPUS, axis=0)
        labels_temp = labels_temp.shard(GPUS, axis=0)
        post_shard = time.perf_counter() - pre_shard
        jit_t = time.perf_counter()
        loss = train_step(X, None, None, matched_idxs, boxes_temp, labels_temp)
        jitted_time = time.perf_counter()-jit_t
        if lr_sched is not None:
          lr_sched.step()
        et = time.perf_counter()-st
        if cnt%1==0:
          # print(f'hit {(pre_zip-pre_match)*1000.0} {(pre_shard-pre_zip)*1000.0}'
          #       f' {jitted_time*1000.0} {post_shard*1000.0}')
          # print(X.shape, matched_idxs.shape, boxes_temp.shape, labels_temp.shape)
          # print(colored(f'{cnt} STEP {loss.item():.5f}, time: {et*1000.0:7.2f} ms run, '
          #               f'data: {data_time*1000.0:7.2f} ms|| LR: {optimizer.lr.numpy().item():.6f}, '
          #               f'mem: {GlobalCounters.mem_used / 1e9:.4f} GB used, '
          #               f'GFLOPS: {GlobalCounters.global_ops * 1e-9 / et:7.2f}', 'magenta'))
          print(colored(f'{cnt} STEP {loss.shape}, time: {et*1000.0:7.2f} ms run, '
                        f'data: {data_time*1000.0:7.2f} ms|| '
                        f'mem: {GlobalCounters.mem_used / 1e9:.4f} GB used, '
                        f'GFLOPS: {GlobalCounters.global_ops * 1e-9 / et:7.2f}', 'magenta'))
        if cnt%5==0:
          ll = loss.item()
          print(f'LOSS: {ll:.5f}')
          wandb.log({'train/loss':ll})
        data_end = time.perf_counter()
        # if cnt>5 and epoch==0: 
        #   # train_step.reset()
        #   break
      cnt+=1
      if cnt%50==0:
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        fn = f"./ckpts/retinanet_{len(GPUS)}x{HOSTNAME}_B{BS}_E{epoch}_{cnt}.safe"
        state_dict = get_state_dict(model)
        # print(state_dict.keys())
        safe_save(state_dict, fn)
        print(f" *** Model saved to {fn} ***")

    # # ****EVAL STEP
    # # train_step.reset()
    # print(colored(f'{epoch} START EVAL', 'cyan'))
    # coco_eval = COCOeval(coco_val.coco, iouType="bbox")

    # Tensor.training = False
    # # print('rand',model.head.regression_head.bbox_reg.weight.mean().numpy(),model.head.regression_head.bbox_reg.weight.shape)
    # # model.load_from_pretrained()
    # # model.backbone.body.fc = None
    # # print('pre_train',model.head.regression_head.bbox_reg.weight.mean().numpy(),model.head.regression_head.bbox_reg.weight.shape)
    # # model.load_checkpoint("./ckpts/retinanet_B16_E10.safe")
    # # print('ckpt',model.head.regression_head.bbox_reg.weight.mean().numpy(), model.head.regression_head.bbox_reg.weight.shape)

    # st = time.time()
    # coco_evalimgs, evaluated_imgs, ncats, narea = [], [], len(coco_eval.params.catIds), len(coco_eval.params.areaRng)
    # cnt = 0
    # for X, targets in iterate_val(coco_val, BS_EVAL):
    #   X.shard_(GPUS, axis=0)
    #   orig_shapes= []
    #   for tt in targets:
    #     orig_shapes.append(tt['image_size'])
    #   # print('orig_shapes', orig_shapes)
    #   # print(orig_shapes)
    #   sub_t = time.time()
    #   # out = mdlrun_false(X).numpy()
    #   out = val_step(X).numpy()
    #   predictions = model.postprocess_detections(out, orig_image_sizes=orig_shapes)
    #   # print(predictions)
    #   img_ids = [t["image_id"] for t in targets]
    #   # print('img_ids', img_ids)
    #   coco_results  = [{"image_id": targets[i]["image_id"], "category_id": label, "bbox": box.tolist(),
    #                      "score": score} for i, prediction in enumerate(predictions) 
    #                      for box, score, label in zip(*prediction.values())]
    #   print('len_reults', len(coco_results))
    #   # IF COCO_RESULTS LOWER THAN THRESH, ERROR IN EVAL
    #   # REFERNCE PUSHES EMPTY COCO OBJ
    #   with redirect_stdout(None):
    #     coco_eval.cocoDt = coco_val.coco.loadRes(coco_results) if coco_results else COCO()
    #     coco_eval.params.imgIds = img_ids
    #     coco_eval.evaluate()
    #   evaluated_imgs.extend(img_ids)
    #   coco_evalimgs.append(np.array(coco_eval.evalImgs).reshape(ncats, narea, len(img_ids)))
    #   print(colored(f'{cnt} EVAL_STEP || {time.time()-sub_t}', 'red'))
    #   cnt=cnt+1
    # coco_eval.params.imgIds = evaluated_imgs
    # coco_eval._paramsEval.imgIds = evaluated_imgs
    # coco_eval.evalImgs = list(np.concatenate(coco_evalimgs, -1).flatten())
    # coco_eval.accumulate()
    # coco_eval.summarize()
    # eval_acc = coco_eval.stats[0]
    # print(colored(f'{epoch} EVAL_ACC {eval_acc} || {time.time()-st}', 'green'))

    # # val_step.reset()
    # # sys.exit()

      
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