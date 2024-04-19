import copy,sys
import os
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
# import torch

from tinygrad import Tensor, dtypes, Device

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from tinygrad.helpers import colored
class ConvertCocoPolysToMask(object):
  def __init__(self, filter_iscrowd=True):
    self.filter_iscrowd = filter_iscrowd

  def __call__(self, image, target):
    # w, h = image.size
    h, w = image.size

    image_id = target["image_id"]
    image_id = Tensor([image_id], requires_grad=False)

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
    # target["boxes"] = Tensor(boxes, requires_grad=False)#.realize()
    # # print('BOXES:TENSCONV', target["boxes"].numpy())
    # target["labels"] = Tensor(classes, requires_grad=False)#.realize()
 # max_pad = 500
        # n = bbox.shape[0]
        # boxes_padded = bbox.pad((((0,max_pad-n), None)),0)
        # labels_padded = t['labels'].reshape(-1,1).pad((((0,max_pad-n), None)),-1).reshape(-1)
    # print(boxes.shape, classes.shape)
    boxes = resize_boxes_np(boxes,image.size[::-1], (800,800))
    pad_sz = MAX_PAD - boxes.shape[0]
    boxes = np.pad(boxes, pad_width=((0,pad_sz), (0,0)))
    classes = np.pad(classes, pad_width=(0,pad_sz), constant_values=-1)
    target["boxes"] = boxes
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
    # print('iterate orig size', orig_size, id)
    # sys.exit()
    target = self._load_target(id)

    # if self.transforms is not None:
    #   img, target = self.transforms(img, target)

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

  # if transforms is not None:
  #     t.append(transforms)
  # transforms = T.Compose(t)

  img_folder = os.path.join(PATHS[image_set], "data")
  ann_file = os.path.join(PATHS[image_set], "labels", f"{name}.json")

  dataset = CocoDetection(img_folder, ann_file, transforms=t)

  return dataset

def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
  ratios = [
      Tensor(s, dtype=dtypes.float32) / Tensor(s_orig, dtype=dtypes.float32)
      for s, s_orig in zip(new_size, original_size)
  ]
  ratio_height, ratio_width = ratios

  # print('resize_boxes boxes.shape:',boxes.shape, original_size, new_size)
  # print(boxes.numpy())

  # bnp = boxes.numpy()
  # xmin, ymin, xmax, ymax = Tensor(bnp[:, 0]), Tensor(bnp[:, 1]), \
  #                         Tensor(bnp[:, 2]), Tensor(bnp[:, 3])
  xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], \
                          boxes[:, 2], boxes[:, 3]
  # print('temp t LEN:',t)
  # print('UNBIND SHAPE_PRE:', xmin.shape, xmin.numpy())

  xmin = xmin * ratio_width
  xmax = xmax * ratio_width
  ymin = ymin * ratio_height
  ymax = ymax * ratio_height
  # print('UNBIND SHAPE_POST:', xmin.shape, xmax.shape, ymin.shape, ymax.shape)
  ans = Tensor.stack((xmin, ymin, xmax, ymax), dim=1)#.cast(dtypes.float32)
  # print('RESIZE_ANS', ans.shape)
  ans.requires_grad = False
  return ans
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
  # ans = Tensor.stack((xmin, ymin, xmax, ymax), dim=1)#.cast(dtypes.float32)
  ans = np.stack((xmin, ymin, xmax, ymax), axis=1)

  # print('RESIZE_ANS', ans.shape)
  # ans.requires_grad = False
  return ans

import torchvision.transforms.functional as F
# SIZE = (400,400)
SIZE = (800, 800)

image_std = Tensor([0.229, 0.224, 0.225]).reshape(-1,1,1)
image_mean = Tensor([0.485, 0.456, 0.406]).reshape(-1,1,1)
def normalize(x):
  x = x.permute((2,0,1)) / 255.0
  # x = x /255.0
  # x = x.permute((2,1,0)) / 255.0
  x -= image_mean
  x /= image_std
  return x#.realize()
MATCHER_FUNC = None  
def iterate(coco, bs=8, func =None):
  i = 0
  i_sub = 0
  rem =0
  while(i+bs+rem<len(coco.ids)):
    # print('iterate', i)
    i_sub = 0
    rem =0
    X, target_boxes, target_labels, target_boxes_padded, target_labels_padded, matched_idxs = [], [], [], [], [], []
    while(i_sub<bs and i+bs+rem<len(coco.ids)):
      # print(i_sub)
      x_orig,t = coco.__getitem__(i+i_sub+rem)
      # x_tens = Tensor.empty((SIZE[0], SIZE[1], 3), dtype=dtypes.uint8)
      # print('X_ORIG_SIZE', x_orig.size)
      # print('DATLOAD_ITER', t['boxes'].shape)

      # Training not done on empty targets
      if(t['boxes'].shape[0]<=0):
        print(colored(f'EMPTY BOZES {i+i_sub+rem}', 'cyan'))
        rem+=1
      else:
        # xNew_pre = normalize(Tensor(np.array(x_orig)))
        # print(np.array(x_orig).shape)
        # xNew_tor = F.resize(torch.tensor(np.array(x_orig), device='cpu').permute(2,0,1), size=SIZE)
        xNew_tor = x_orig.resize(SIZE, resample = Image.BILINEAR)
        # print('xNEW_TOR', xNew_tor.shape)
        # print('X_NEW', xNew.shape)
        # print('PIL_TO_NP_CONV')
        x_np = np.array(xNew_tor)#.reshape(800,800,3)
        # print('NP_TENS_CONV', x_np.dtype)

        x_tens = Tensor(x_np).realize()
        # x_tens.contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = xNew_tor.tobytes()
        # x_tens.assign(x_np).realize()
        # x_tens.contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = x_np.tobytes()
        # print('NORMALIZIMG', x_tens.dtype)
        xNew = normalize(x_tens).realize()
        # print('Finished_NORMAL')
        # xNew = Tensor(np.array(xNew_tor)).permute(2,0,1)

        # print('X_MEAN_NORM',xNew.shape, xNew.mean().numpy())
        X.append(xNew)
        bbox = t['boxes']
        # print('ITERATE_PRE_RESIZE', bbox.shape)
        # bbox = resize_boxes(bbox, (x_orig.size[1],x_orig.size[0]), SIZE)
        bbox = resize_boxes(bbox, x_orig.size[::-1], SIZE) #.realize()
        # print('ITERATE_POST_RESIZE', bbox.shape)
        # t['boxes'] = bbox.realize()
        # max_pad = 120087

        # max_pad = 500
        # n = bbox.shape[0]
        # boxes_padded = bbox.pad((((0,max_pad-n), None)),0)
        # labels_padded = t['labels'].reshape(-1,1).pad((((0,max_pad-n), None)),-1).reshape(-1)

        # print('ITERATE', xNew.shape, t['boxes'].shape, t['labels'].shape)
        # print('ITERATE_PADDED', xNew.shape, boxes_padded.shape, labels_padded.shape)

        # target_boxes.append(bbox) #.realize())
        # target_labels.append(t['labels']) #.realize())
        
        # print('lABEL LOAD CHEK', target_labels[-1].shape)
        # print('PADDING LOGIC', labels_padded.numpy())
        # print(boxes_padded.numpy())
        # target_boxes_padded.append(boxes_padded.realize())
        # target_labels_padded.append(labels_padded.realize())
        i_sub+=1
        # if MATCHER_FUNC is not None:
        # if i>0:
        if func is not None:
          # print('MATCHER_FUNC_HIT')
          # print(func)
          m_idx = func(bbox) #.realize()
          # tbm = boxes_padded[m_idx] #.realize()
          # tlm = labels_padded[m_idx] #.realize()
          tbm = bbox[m_idx] #.realize()
          tlm = t['labels'][m_idx] #.realize()
          target_boxes_padded.append(tbm)
          target_labels_padded.append(tlm)
          matched_idxs.append(m_idx)
      # x_orig.close()
      # del t

    # yield Tensor.stack(X), target_boxes, target_labels, target_boxes_padded, target_labels_padded, matched_idxs
    if func is not None:
      yield Tensor.stack(X).realize(), None, None, Tensor.stack(target_boxes_padded).realize(), \
        Tensor.stack(target_labels_padded).realize(), Tensor.stack(matched_idxs).realize()
    else:
      yield Tensor.stack(X), None, None, target_boxes_padded, \
        target_labels_padded, matched_idxs
    # yield Tensor.stack(X), None, None, target_boxes_padded, \
    #     target_labels_padded, matched_idxs
    i= i+bs+rem

def iterate_val(coco, bs=8):
  for i in range(0, len(coco.ids)-bs, bs):
  # for i in range(0, 10, bs):
    X, targets = [], []
    for img_id in coco.ids[i:i+bs]:
      x_orig, t = coco.__getitem__(img_id)
      xNew_tor = F.resize(x_orig, size=SIZE)
      xNew = normalize(Tensor(np.array(xNew_tor)))
      X.append(xNew)
      # bbox = t['boxes']
      # t['boxes'] = resize_boxes(bbox, x_orig.size, SIZE).numpy()
      # t['labels'] = t['labels'].numpy()
      # t['image_id'] = t['image_id'].item()
      tNew = {'image_size' : t['image_size'][::-1], 
              'image_id' : t['image_id'].item()}
      targets.append(tNew)
    yield Tensor.stack(X), targets

from multiprocessing import Queue, Process, shared_memory, connection, Lock, cpu_count
import pickle, random
from tinygrad.helpers import getenv, prod, Timing, Context
from tqdm import tqdm
MAX_PAD = 500
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

def resize_img(img):
  return img.resize(SIZE, resample = Image.BILINEAR)
def loader_process(q_in, q_out, X:Tensor, seed, coco, YB:Tensor, YL:Tensor):
  import signal
  signal.signal(signal.SIGINT, lambda _, __: exit(0))

  # from extra.datasets.imagenet import center_crop, preprocess_train

  with Context(DEBUG=0):
    while (_recv := q_in.get()) is not None:
      idx, img_idx, val = _recv
      img, target = coco.__getitem__(img_idx)
      # img = Image.open(fn)
      # img = img.convert('RGB') if img.mode != "RGB" else img

      if val:
        pass
        # eval: 76.08%, load in 0m7.366s (0m5.301s with simd)
        # sudo apt-get install libjpeg-dev
        # CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
        # img = center_crop(img)
        # img = np.array(img)
      else:
        # reseed rng for determinism
        # if seed is not None:
        #   np.random.seed(seed * 2 ** 20 + idx)
        #   random.seed(seed * 2 ** 20 + idx)
        # img = preprocess_train(img)
        img = np.array(resize_img(img))

      # broken out
      #img_tensor = Tensor(img.tobytes(), device='CPU')
      #storage_tensor = X[idx].contiguous().realize().lazydata.realized
      #storage_tensor._copyin(img_tensor.numpy())

      # faster
      X[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = img.tobytes()
      YB[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = target['boxes'].tobytes()
      YL[idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = target['labels'].tobytes()
      # ideal
      #X[idx].assign(img.tobytes())   # NOTE: this is slow!
      q_out.put(idx)
    q_out.put(None)
def batch_load_retinanet(coco, bs=8, shuffle=False, seed=None, val = False):
  DATA_LEN = len(coco)
  BATCH_COUNT = DATA_LEN//bs
  gen = shuffled_indices(DATA_LEN, seed=seed) if shuffle else iter(range(DATA_LEN))

  def enqueue_batch(num):
    for idx in range(num*bs, (num+1)*bs):
      img_idx = next(gen)
      img,target = coco.__getitem__(img_idx)
      if target['boxes'].size == 0:
        # need to skip for train loops
        img_idx = 7
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
    return X[num*bs:(num+1)*bs], Y_IDX[num*bs:(num+1)*bs], YB[num*bs:(num+1)*bs], YL[num*bs:(num+1)*bs], Cookie(num)
    # return X[num*bs:(num+1)*bs], Cookie(num)
  
  q_in, q_out = Queue(), Queue()
  sz = (bs*BATCH_COUNT, 800, 800, 3)
  bsz = (bs*BATCH_COUNT, MAX_PAD, 4)
  lsz = (bs*BATCH_COUNT, MAX_PAD)
  if os.path.exists("/dev/shm/retinanet_X"): os.unlink("/dev/shm/retinanet_X")
  shm = shared_memory.SharedMemory(name="retinanet_X", create=True, size=prod(sz))
  if os.path.exists("/dev/shm/retinanet_YB"): os.unlink("/dev/shm/retinanet_YB")
  bshm = shared_memory.SharedMemory(name="retinanet_YB", create=True, size=prod(bsz))
  if os.path.exists("/dev/shm/retinanet_YL"): os.unlink("/dev/shm/retinanet_YL")
  lshm = shared_memory.SharedMemory(name="retinanet_YL", create=True, size=prod(lsz))

  procs = []

  try:
    # disk:shm is slower
    #X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:shm:{shm.name}")
    X = Tensor.empty(*sz, dtype=dtypes.uint8, device=f"disk:/dev/shm/retinanet_X")
    YB = Tensor.empty(*bsz, dtype=dtypes.default_float, device=f"disk:/dev/shm/retinanet_YB")
    YL = Tensor.empty(*lsz, dtype=dtypes.int16, device=f"disk:/dev/shm/retinanet_YL")
    Y_IDX = [None] * (bs*BATCH_COUNT)
    # Y = [None] * (bs*BATCH_COUNT)

    for _ in range(cpu_count()):
      p = Process(target=loader_process, args=(q_in, q_out, X, seed, coco, YB, YL))
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
if __name__ == '__main__':
  from extra.datasets.openimages_new import get_openimages, iterate
  ROOT = 'extra/datasets/open-images-v6TEST'
  NAME = 'openimages-mlperf'
  coco_train = get_openimages(NAME,ROOT, 'train')
  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  print(f"Training on {GPUS}")
  for x in GPUS: Device[x]
  with tqdm(total=len(coco_train)) as pbar:
    for x, y, yb, yl, c in batch_load_retinanet(coco_train, bs=2):
      x = x.shard(GPUS,axis=0).realize()
      yb = yb.shard(GPUS,axis=0).realize()
      pbar.update(x.shape[0])
      # print(x.shape, x.dtype, x.device)
      # print(x.device)
      # print(y)
      # for i in y:


  # with tqdm(total=len(coco_train)) as pbar:
  #   for x in iterate(coco_train):
  #     pbar.update(x[0].shape[0])
  #     print(x[0].shape, x[0].dtype, x[0].device)
  
