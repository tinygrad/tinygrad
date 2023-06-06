from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn import optim
import numpy as np
from models.resnet import ResNeXt50_32X4D
from models.retinanet import RetinaNet
from typing import List
from examples.mlperf.train_retinanet.utils.bbox_transformations import bbox_transform, resize_box_based_on_new_image_size
from extra.training import smooth_l1_loss, focal_loss
import random
from datasets.openimages import openimages, iterate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from contextlib import redirect_stdout
import time
from itertools import islice

NUM_CLASSES = 264
INPUT_IMG_SHAPE = (800, 800, 3)
TRAIN_BS = 8


# Compute the IoU between all <box,query_box> pairs. A box is represented as (x1, y1, x2, y2).
# boxes: (N, 4), query_boxes: (K, 4), returns: (N, K)
def compute_overlap(boxes: np.ndarray, query_boxes: np.ndarray):
  overlaps = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float32)
  for k in range(query_boxes.shape[0]):
    query_box_area = ((query_boxes[k, 2] - query_boxes[k, 0]) * (query_boxes[k, 3] - query_boxes[k, 1]))
    for n in range(boxes.shape[0]):
      intersection_width = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]))
      if intersection_width <= 0: continue
      intersection_height = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]))
      if intersection_height <= 0: continue
      box_area = np.float32((boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1]))
      union_area = box_area + query_box_area - intersection_width * intersection_height
      overlaps[n, k] = intersection_width * intersection_height / union_area
  return overlaps

# anchors: (A, 4). Every anchor is represented as (x1, y1, x2, y2). The anchors are always the same.
# `annotations` is a list of length BS. One annotation for every image in the batch. One annotation can have multiple labeled boxes.
def compute_batch_targets(anchors: np.ndarray, annotations: List, negative_overlap=0.4, positive_overlap=0.5):
  batch_size = len(annotations)
  regression_targets = np.zeros((batch_size, len(anchors), 4), dtype=np.float32)
  classif_targets = np.zeros((batch_size, len(anchors), NUM_CLASSES), dtype=np.float32)
  positive_anchors_mask = np.zeros((batch_size, len(anchors)), dtype=bool)  # Anchors that shouldn't be used for training.
  negative_anchors_mask = np.zeros((batch_size, len(anchors)), dtype=bool)  # Anchors that should only be used for learning classification, not bbox regression.
  for i in range(batch_size):
    ann_boxes, ann_labels = annotations[i]['boxes'], annotations[i]['labels']
    assert len(ann_boxes) > 0
    ann_boxes = np.array([resize_box_based_on_new_image_size(box, img_old_size=annotations[i]['image_size'], img_new_size=INPUT_IMG_SHAPE[:2]) for box in ann_boxes])
    overlaps = compute_overlap(anchors, ann_boxes)
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    positive_anchors_idxs = max_overlaps >= positive_overlap  # The object might be in one of these anchors, so we'll learn to regress to them and to classify the label as such.
    negative_anchors_idxs = (max_overlaps < negative_overlap) & ~positive_anchors_idxs  # There are definitely no objects in these anchors, so we use them for learning to classify object absence (i.e., all classes to 0).

    regression_targets[i, :, :] = bbox_transform(anchors, ann_boxes[argmax_overlaps_inds, :])  # Calculate the spatial transformations from the anchors to the GT boxes.
    classif_targets[i, positive_anchors_idxs, ann_labels[argmax_overlaps_inds[positive_anchors_idxs]].astype(int)] = 1
    positive_anchors_mask[i, positive_anchors_idxs] = True
    negative_anchors_mask[i, negative_anchors_idxs] = True
  return {"regression_targets": regression_targets, "classif_targets": classif_targets, "positive_anchors_mask": positive_anchors_mask, "negative_anchors_mask": negative_anchors_mask}

def train_retinanet():
  model = RetinaNet(ResNeXt50_32X4D(), num_classes=NUM_CLASSES, num_anchors=9)
  model.backbone.body.fc = None  # it's not used by RetinaNet and would break the training loop because of .requires_grad

  anchors = model.anchor_gen(input_size=INPUT_IMG_SHAPE[:2])
  def flatten(l): return [item for sublist in l for item in sublist]
  anchors = np.array(flatten(anchors))
  print('Number of anchors:', len(anchors))

  input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
  input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
  def input_fixup(x):
    x = x.permute([0,3,1,2]) / 255.0
    x -= input_mean
    x /= input_std
    return x

  coco = COCO(openimages())

  optimizer = optim.SGD(optim.get_parameters(model), lr=0.001, weight_decay=0.0001, momentum=0.9)

  def evaluate_model():
    print('Evaluating model...')
    Tensor.training = False
    coco_eval = COCOeval(coco, iouType="bbox")
    coco_evalimgs, evaluated_imgs, ncats, narea = [], [], len(coco_eval.params.catIds), len(coco_eval.params.areaRng)
    mdlrun = TinyJit(lambda x: model(input_fixup(x)).realize())
    eval_bs = TRAIN_BS*2
    n = 0
    st = time.perf_counter()
    for x, targets in islice(iterate(coco, eval_bs, shuffle=True), 120):
      dat = Tensor(x.astype(np.float32))
      mt = time.perf_counter()
      if dat.shape[0] == eval_bs:
        outs = mdlrun(dat).numpy()
      else:
        mdlrun.jit_cache = None
        outs =  model(input_fixup(dat)).numpy()
      et = time.perf_counter()
      predictions = model.postprocess_detections(outs, input_size=dat.shape[1:3], orig_image_sizes=[t["image_size"] for t in targets])
      ext = time.perf_counter()
      n += len(targets)
      print(f"[{n}/{len(coco.imgs)}] == {(mt-st)*1000:.2f} ms loading data, {(et-mt)*1000:.2f} ms to run model, {(ext-et)*1000:.2f} ms for postprocessing")
      img_ids = [t["image_id"] for t in targets]
      coco_results  = [{"image_id": targets[i]["image_id"], "category_id": label, "bbox": box, "score": score}
        for i, prediction in enumerate(predictions) for box, score, label in zip(*prediction.values())]
      with redirect_stdout(None):
        coco_eval.cocoDt = coco.loadRes(coco_results)
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
      evaluated_imgs.extend(img_ids)
      coco_evalimgs.append(np.array(coco_eval.evalImgs).reshape(ncats, narea, len(img_ids)))
      st = time.perf_counter()

    coco_eval.params.imgIds = evaluated_imgs
    coco_eval._paramsEval.imgIds = evaluated_imgs
    coco_eval.evalImgs = list(np.concatenate(coco_evalimgs, -1).flatten())
    coco_eval.accumulate()
    coco_eval.summarize()

  @TinyJit
  def train_step(dat, annotations):
    optimizer.zero_grad()

    outs = model(input_fixup(dat))
    regression_preds, classif_preds = outs[:, :, :4], outs[:, :, 4:]

    target = compute_batch_targets(anchors, annotations, model.num_classes)
    regression_targets, classif_targets = Tensor(target['regression_targets'], requires_grad=False), Tensor(target['classif_targets'], requires_grad=False)

    regression_mask = Tensor(np.repeat(target['positive_anchors_mask'][:, :, np.newaxis], repeats=4, axis=2))
    regression_preds = regression_mask.where(regression_preds, regression_targets)
    regression_losses = [smooth_l1_loss(regression_preds[img_idx], regression_targets[img_idx], reduction='sum') for img_idx in range(TRAIN_BS)]
    regression_loss = Tensor.stack(regression_losses).mean()

    classif_mask = Tensor(np.repeat((target['positive_anchors_mask'] + target['negative_anchors_mask'])[:, :, np.newaxis], repeats=NUM_CLASSES, axis=2))
    classif_preds = classif_mask.where(classif_preds, classif_targets)
    classif_losses = [focal_loss(classif_preds[img_idx], classif_targets[img_idx], reduction='sum') / max(1, np.count_nonzero(target['positive_anchors_mask'][img_idx])) for img_idx in range(TRAIN_BS)]
    classif_loss = Tensor.stack(classif_losses).mean()

    loss = regression_loss + classif_loss
    loss.backward()

    optimizer.step()
    return loss.realize()

  n = 0
  for epoch in range(1, 11):
    Tensor.training = True
    train_losses = []
    for imgs, annotations in iterate(coco, TRAIN_BS, shuffle=True):
      if imgs.shape[0] != TRAIN_BS: break  # for JIT
      loss = train_step(Tensor(imgs.astype(np.float32)), annotations)
      train_losses.append(loss.numpy().item())
      n += len(annotations)
      print(f"[Epoch {epoch}, {n}/{len(coco.imgs)}]")
      if len(train_losses) >= 10:
        print('Training loss (per batch, last 10):', (sum(train_losses) / len(train_losses)))
        train_losses = []
    evaluate_model()

if __name__ == "__main__":
  Tensor.manual_seed(0)
  np.random.seed(0)
  random.seed(0)
  print('training retinanet')
  train_retinanet()
