import copy,sys
import os
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
# import torch

from tinygrad import Tensor, dtypes

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from tinygrad.helpers import colored
class ConvertCocoPolysToMask(object):
  def __init__(self, filter_iscrowd=True):
    self.filter_iscrowd = filter_iscrowd

  def __call__(self, image, target):
    # w, h = image.size
    w,h = image.size

    image_id = target["image_id"]
    # image_id = image_id

    anno = target["annotations"]

    if self.filter_iscrowd:
      anno = [obj for obj in anno if obj['iscrowd'] == 0]

    boxes = [obj["bbox"] for obj in anno]
    # print('BOXES:HHHHH', boxes)
    # guard against no boxes via resizing
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    # print('BOXES:POSTT', boxes)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
    # print('BOXES:POSTPOST', boxes)
    classes = [obj["category_id"] for obj in anno]
    classes = np.array(classes, dtype=np.int16)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    # print('BOXES:KEPPPOST', boxes)
    classes = classes[keep]

    target = {}
    # print('CONVERTINGTTTT')
    target["boxes"] = resize_boxes_np(boxes, image.size[::-1], (800, 800))
    # print('BOXES:TENSCONV', target["boxes"].numpy())
    target["labels"] = classes
    target["image_id"] = image_id
    # print('FINISHED_CONVERTING')

    return image, target

class CocoDetection:
  def __init__(self, img_folder, ann_file, transforms=None):
    self.root = img_folder
    self.coco = COCO(ann_file)
    self.ids = list(sorted(self.coco.imgs.keys()))
    # super(CocoDetection, self).__init__(img_folder, ann_file)
    self._transforms = transforms
  def _load_image(self, id: int) -> Image.Image:
    path = self.coco.loadImgs(id)[0]["file_name"]
    return Image.open(os.path.join(self.root, path)).convert("RGB")

  def _load_target(self, id: int) -> List[Any]:
    return self.coco.loadAnns(self.coco.getAnnIds(id))
  def __len__(self) -> int:
    return len(self.ids)
  def __getitem__(self, index):

    id = self.ids[index]
    img = self._load_image(id)
    orig_size = img.size

    target = self._load_target(id)

    image_id = self.ids[index]
    target = dict(image_id=image_id, annotations=target)
    if self._transforms is not None:
      # print('HIT')
      img, target = self._transforms(img, target)
      target['image_size'] = orig_size
    return img, target
  
def get_openimages(name, root, image_set, transforms=None):
  PATHS = {
      "train": os.path.join(root, "train"),
      "val":   os.path.join(root, "validation"),
  }

  t = ConvertCocoPolysToMask(filter_iscrowd=False)

  img_folder = os.path.join(PATHS[image_set], "data")
  ann_file = os.path.join(PATHS[image_set], "labels", f"{name}.json")

  dataset = CocoDetection(img_folder, ann_file, transforms=t)

  return dataset

SIZE = (800, 800)

def resize_img(img:Image.Image):
  return img.resize(SIZE, resample = Image.BILINEAR)
def resize_boxes_np(boxes, original_size: List[int], new_size: List[int]):
  ratios = [
      s / s_orig
      for s, s_orig in zip(new_size, original_size)
  ]
  ratio_height, ratio_width = ratios

  xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], \
                          boxes[:, 2], boxes[:, 3]

  xmin = xmin * ratio_width
  xmax = xmax * ratio_width
  ymin = ymin * ratio_height
  ymax = ymax * ratio_height
  # print('UNBIND SHAPE_POST:', xmin.shape, xmax.shape, ymin.shape, ymax.shape)
  return np.stack((xmin, ymin, xmax, ymax), axis=1)

def box_iou_np(boxes1:np.ndarray, boxes2:np.ndarray):
  def box_area(boxes): return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
  area1 = box_area(boxes1)
  area2 = box_area(boxes2)
  
  lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
  rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
  wh = (rb-lt).clip(min=0)
  inter = wh[:, :, 0] * wh[:, :, 1]
  union = area1[:, None] + area2 - inter
  return inter / union
class Matcher_np(object):
  BELOW_LOW_THRESHOLD = -1
  BETWEEN_THRESHOLDS = -2
  def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):

    self.BELOW_LOW_THRESHOLD = -1
    self.BETWEEN_THRESHOLDS = -2
    assert low_threshold <= high_threshold
    self.high_threshold = high_threshold
    self.low_threshold = low_threshold
    self.allow_low_quality_matches = allow_low_quality_matches

  def __call__(self, match_quality_matrix:np.ndarray):

    if match_quality_matrix.size == 0:
      # empty targets or proposals not supported during training
      if match_quality_matrix.shape[0] == 0:
        raise ValueError(
            "No ground-truth boxes available for one of the images "
            "during training")
      else:
        raise ValueError(
            "No proposal boxes available for one of the images "
            "during training")

    # match_quality_matrix is M (gt) x N (predicted)
    # Max over gt elements (dim 0) to find best gt candidate for each prediction
    matched_vals, matches = match_quality_matrix.max(axis=0), match_quality_matrix.argmax(axis=0)

    if self.allow_low_quality_matches:
      all_matches = np.copy(matches)
    else:
      all_matches = None

    # Assign candidate matches with low quality to negative (unassigned) values
    below_low_threshold = matched_vals < self.low_threshold
    between_thresholds = (matched_vals >= self.low_threshold) & (
        matched_vals < self.high_threshold
    )

    matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
    matches[between_thresholds] = self.BETWEEN_THRESHOLDS
    if self.allow_low_quality_matches:
      assert all_matches is not None
      self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)
    return matches

  def set_low_quality_matches_(self, matches:np.ndarray, all_matches:np.ndarray, match_quality_matrix:np.ndarray):
    # For each gt, find the prediction with which it has highest quality
    highest_quality_foreach_gt= match_quality_matrix.max(axis=1)
    # Find highest quality match available, even if it is low, including ties
    gt_pred_pairs_of_highest_quality = np.nonzero(
        match_quality_matrix == highest_quality_foreach_gt[:, None]
    )
    pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
    matches[pred_inds_to_update] = all_matches[pred_inds_to_update]

def matcher_iou_func(box, anchor):
  M_NP = Matcher_np(0.5, 0.4, True)
  return M_NP(box_iou_np(box, anchor))

from multiprocessing import Queue, Process, shared_memory, connection, Lock, cpu_count
import pickle, random
from tinygrad.helpers import getenv, prod, Timing, Context
from tqdm import tqdm
class MyQueue:
  def __init__(self, multiple_readers=True, multiple_writers=True):
    self._reader, self._writer = connection.Pipe(duplex=False)
    self._rlock = Lock() if multiple_readers else None
    self._wlock = Lock() if multiple_writers else None
  def get(self):
    if self._rlock: self._rlock.acquire()
    ret = pickle.loads(self._reader.recv_bytes())
    if self._rlock: self._rlock.release()
    return ret
  def put(self, obj):
    if self._wlock: self._wlock.acquire()
    self._writer.send_bytes(pickle.dumps(obj))
    if self._wlock: self._wlock.release()

def shuffled_indices(n, seed=None):
  rng = random.Random(seed)
  indices = {}
  for i in range(n-1, -1, -1):
    j = rng.randint(0, i)
    if i not in indices: indices[i] = i
    if j not in indices: indices[j] = j
    indices[i], indices[j] = indices[j], indices[i]
    yield indices[i]
    del indices[i]


def loader_process(q_in, q_out, X:Tensor, seed, coco, YB:Tensor, YL:Tensor, YM:Tensor, Anchor):
  import signal
  signal.signal(signal.SIGINT, lambda _, __: exit(0))

  with Context(DEBUG=0):
    while (_recv := q_in.get()) is not None:
      idx, img_idx, val = _recv
      img, target = coco.__getitem__(img_idx)

      if val:
        pass
      else:

        img = np.array(resize_img(img))
        midx = matcher_iou_func(target['boxes'], Anchor)
        m_temp = np.clip(midx, 0, None)
        tb = target['boxes'][m_temp]
        tl = target['labels'][m_temp]
        del target, m_temp

      # broken out
      #img_tensor = Tensor(img.tobytes(), device='CPU')
      #storage_tensor = X[idx].contiguous().realize().lazydata.realized
      #storage_tensor._copyin(img_tensor.numpy())

      # faster
      X[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = img.tobytes()
      YB[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = tb.tobytes()
      YL[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = tl.tobytes()
      YM[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = midx.tobytes()

      # ideal
      #X[idx].assign(img.tobytes())   # NOTE: this is slow!
      q_out.put(idx)
    q_out.put(None)
def batch_load_retinanet(coco, bs=8, shuffle=False, seed=None, val = False, anchor_np=[1,2,3,4]):
  DATA_LEN = len(coco)
  BATCH_COUNT = min(32, DATA_LEN//bs)
  gen = shuffled_indices(DATA_LEN, seed=seed) if shuffle else iter(range(DATA_LEN))

  def enqueue_batch(num):
    for idx in range(num*bs, (num+1)*bs):
      img_idx = next(gen)
      q_in.put((idx, img_idx, val))

  shutdown = False
  class Cookie:
    def __init__(self, num): self.num = num
    def __del__(self):
      if not shutdown:
        try: enqueue_batch(self.num)
        except StopIteration: pass

  gotten = [0]*BATCH_COUNT
  def receive_batch():
    while 1:
      num = q_out.get()//bs
      gotten[num] += 1
      if gotten[num] == bs: break
    gotten[num] = 0
    return X[num*bs:(num+1)*bs], YB[num*bs:(num+1)*bs], YL[num*bs:(num+1)*bs], YM[num*bs:(num+1)*bs], Cookie(num)
  
  q_in, q_out = Queue(), Queue()
  sz = (bs*BATCH_COUNT, 800, 800, 3)
  if os.path.exists("/dev/shm/retinanet_X"): os.unlink("/dev/shm/retinanet_X")
  shm = shared_memory.SharedMemory(name="retinanet_X", create=True, size=prod(sz))
  bsz = (bs*BATCH_COUNT, 120087, 4)
  if os.path.exists("/dev/shm/retinanet_YB"): os.unlink("/dev/shm/retinanet_YB")
  bshm = shared_memory.SharedMemory(name="retinanet_YB", create=True, size=prod(bsz))
  lsz = (bs*BATCH_COUNT, 120087)
  if os.path.exists("/dev/shm/retinanet_YL"): os.unlink("/dev/shm/retinanet_YL")
  lshm = shared_memory.SharedMemory(name="retinanet_YL", create=True, size=prod(lsz))
  msz = (bs*BATCH_COUNT, 120087)
  if os.path.exists("/dev/shm/retinanet_YM"): os.unlink("/dev/shm/retinanet_YM")
  mshm = shared_memory.SharedMemory(name="retinanet_YM", create=True, size=prod(msz))
  procs = []

  try:
    # disk:shm is slower
    #X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:shm:{shm.name}")
    X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/retinanet_X")
    YB = Tensor.empty(*bsz, dtype=dtypes.float32, device=f"disk:/dev/shm/retinanet_YB")
    YL = Tensor.empty(*lsz, dtype=dtypes.int16, device=f"disk:/dev/shm/retinanet_YL")
    YM = Tensor.empty(*msz, dtype=dtypes.int64, device=f"disk:/dev/shm/retinanet_YM")


    # for _ in range(cpu_count()):
    for _ in range(4):
      print('hit*************')
      p = Process(target=loader_process, args=(q_in, q_out, X, seed, coco, YB, YL, YM, anchor_np))
      p.daemon = True
      p.start()
      procs.append(p)

    for bn in range(BATCH_COUNT): enqueue_batch(bn)

    # NOTE: this is batch aligned, last ones are ignored
    for _ in range(0, DATA_LEN//bs): yield receive_batch()
  finally:
    shutdown = True
    # empty queues
    for _ in procs: q_in.put(None)
    q_in.close()
    for _ in procs:
      while q_out.get() is not None: pass
    q_out.close()
    # shutdown processes
    for p in procs: p.join()
    shm.close()
    shm.unlink()
    bshm.close()
    bshm.unlink()
    lshm.close()
    lshm.unlink()
    mshm.close()
    mshm.unlink()

def loader_process_val(q_in, q_out, X:Tensor, seed, coco):
  import signal
  signal.signal(signal.SIGINT, lambda _, __: exit(0))

  with Context(DEBUG=0):
    while (_recv := q_in.get()) is not None:
      idx, img_idx, val = _recv
      img, target = coco.__getitem__(img_idx)
      if val:
        img = np.array(resize_img(img))
      else:
        pass

      # broken out
      #img_tensor = Tensor(img.tobytes(), device='CPU')
      #storage_tensor = X[idx].contiguous().realize().lazydata.realized
      #storage_tensor._copyin(img_tensor.numpy())

      # faster
      X[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = img.tobytes()


      # ideal
      #X[idx].assign(img.tobytes())   # NOTE: this is slow!
      q_out.put(idx)
    q_out.put(None)
def batch_load_retinanet_val(coco, bs=8, shuffle=False, seed=None, val = True):
  DATA_LEN = len(coco)
  BATCH_COUNT = min(32, DATA_LEN//bs)
  gen = shuffled_indices(DATA_LEN, seed=seed) if shuffle else iter(range(DATA_LEN))

  def enqueue_batch(num):
    for idx in range(num*bs, (num+1)*bs):
      img_idx = next(gen)
      q_in.put((idx, img_idx, val))
      Y_IDX[idx] = img_idx

  shutdown = False
  class Cookie:
    def __init__(self, num): self.num = num
    def __del__(self):
      if not shutdown:
        try: enqueue_batch(self.num)
        except StopIteration: pass

  gotten = [0]*BATCH_COUNT
  def receive_batch():
    while 1:
      num = q_out.get()//bs
      gotten[num] += 1
      if gotten[num] == bs: break
    gotten[num] = 0
    return X[num*bs:(num+1)*bs], Y_IDX[num*bs:(num+1)*bs], Cookie(num)
  
  q_in, q_out = Queue(), Queue()
  sz = (bs*BATCH_COUNT, 800, 800, 3)
  if os.path.exists("/dev/shm/retinanet_X_val"): os.unlink("/dev/shm/retinanet_X_val")
  shm = shared_memory.SharedMemory(name="retinanet_X_val", create=True, size=prod(sz))
  procs = []

  try:
    # disk:shm is slower
    #X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:shm:{shm.name}")
    X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/retinanet_X_val")
    Y_IDX = [None] * (bs*BATCH_COUNT)

    # for _ in range(cpu_count()):
    for _ in range(4):
      print('hit*************')
      p = Process(target=loader_process_val, args=(q_in, q_out, X, seed, coco))
      p.daemon = True
      p.start()
      procs.append(p)

    for bn in range(BATCH_COUNT): enqueue_batch(bn)

    # NOTE: this is batch aligned, last ones are ignored
    for _ in range(0, DATA_LEN//bs): yield receive_batch()
  finally:
    shutdown = True
    # empty queues
    for _ in procs: q_in.put(None)
    q_in.close()
    for _ in procs:
      while q_out.get() is not None: pass
    q_out.close()
    # shutdown processes
    for p in procs: p.join()
    shm.close()
    shm.unlink()


if __name__ == '__main__':
  from extra.datasets.openimages_new import get_openimages, iterate
  ROOT = 'extra/datasets/open-images-v6TEST'
  NAME = 'openimages-mlperf'
  coco_train = get_openimages(NAME,ROOT, 'train')
  with tqdm(total=len(coco_train)) as pbar:
    for x,y,c in batch_load_retinanet(coco_train):
      pbar.update(x.shape[0])
      print(x.shape, x.dtype, x.device)
      print(y)
      # for i in y:


  # with tqdm(total=len(coco_train)) as pbar:
  #   for x in iterate(coco_train):
  #     pbar.update(x[0].shape[0])
  #     print(x[0].shape, x[0].dtype, x[0].device)
  
