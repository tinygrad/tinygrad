from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, Context

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  from typing import Any
  from tinygrad.tensor import Tensor
  from tinygrad.helpers import getenv, Context, dtypes
  from models.retinanet import RetinaNet
  from models.resnet import ResNeXt50_32X4D
  from extra.datasets import openimages
  from PIL import Image
  from pycocotools.coco import COCO
  from pycocotools.cocoeval import COCOeval
  from tinygrad.nn import optim
  from tinygrad.ops import GlobalCounters
  from tinygrad.state import get_parameters
  from tqdm import trange
  import os, time
  import numpy as np
  import fiftyone as fo
  from examples.hlb_cifar10 import cross_entropy

  NUM = getenv("NUM", 2)
  BS = getenv("BS", 8)
  CNT = getenv("CNT", 10)
  BACKWARD = getenv("BACKWARD", 0)
  TRAINING = getenv("TRAINING", 1)
  ADAM = getenv("ADAM", 0)
  CLCACHE = getenv("CLCACHE", 0)
  backbone = ResNeXt50_32X4D()
  retina = RetinaNet(backbone)
  #retina.load_from_pretrained()

  #example_inference(np, fo, os, input_fixup, retina)
  #dataset = COCO("/tmp/coco.json")
  
  
  params = get_parameters(retina)
  for p in params: p.realize()
  optimizer = optim.SGD(params, lr=0.001)

  Tensor.training = TRAINING
  Tensor.no_grad = not BACKWARD
  # TODO adapt for training.py
  for i in trange(CNT):
    GlobalCounters.reset()
    cpy = time.monotonic()
    x_train = Tensor.randn(BS, 3, 224, 224, requires_grad=False).realize()
    y_train = Tensor.randn(BS, 1000, requires_grad=False).realize()
  
  # TODO: replace with TinyJit
    if i < 3 or not CLCACHE:
      st = time.monotonic()
      head_outputs = retina(x_train)
      breakpoint()
      loss = retina.loss(head_outputs,y_train)
      if i == 2 and CLCACHE: GlobalCounters.cache = []
      if BACKWARD:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      mt = time.monotonic()
      loss.realize()
      for p in params:
        p.realize()
      et = time.monotonic()
    else:
      st = mt = time.monotonic()
      for prg, args in cl_cache: prg(*args)
      et = time.monotonic()
    if i == 2 and CLCACHE:
        cl_cache = GlobalCounters.cache
        GlobalCounters.cache = None

    mem_used = GlobalCounters.mem_used
    loss_cpu = loss.detach().numpy()[0]
    cl = time.monotonic()

def example_inference(np, fo, os, input_fixup, retina):
    #TODO delete this once training loop works fine
    images = load_openimages_as_tg_tensor(np, fo, os, input_fixup)
    images = input_fixup(images)

    model_detection_embeds = retina(images).numpy()
    model_detections = retina.postprocess_detections(model_detection_embeds)
    return model_detections

def load_openimages_as_tg_tensor(np, fo, os, input_fixup, n_images=24, img_reshape = (100,100)):
    Warning("Enhance with DataLoader") #TODO
    train_16_batch = fo.zoo.load_zoo_dataset("open-images-v6",split="train",label_types="detections", max_samples=n_images)
    classes = train_16_batch.distinct("ground_truth.detections.label")
    IMAGES_DIR = os.path.dirname(train_16_batch.first().filepath)

    #IMAGES_DIR = os.path.dirname(r"C:\Users\msoro\fiftyone\open-images-v6\train\data\000002b66c9c498e.jpg'")
    train_16_batch.take(n_images).export(
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
    labels_path="/tmp/coco.json",
    classes=classes,)
  
    coco_dataset = fo.Dataset.from_dir(dataset_type=fo.types.COCODetectionDataset,data_path=IMAGES_DIR,labels_path="/tmp/coco.json",include_id=True)
    #tiny images for debugging...
    images = [Image.open(sample.filepath).resize(img_reshape) for sample in coco_dataset]
    images = Tensor([np.asarray(im) for im in images])
    return images
  #REMEMBER: Quality target = 34.0% mAP
  #TODO 3: adapt for mlperf-retinanet training standard 
  # reference torch implementation https://github.com/mlcommons/training/blob/master/object_detection/pytorch/maskrcnn_benchmark/modeling/rpn/retinanet


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


