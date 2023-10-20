import random
import time
import datetime

from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
from torchvision import transforms as T
from models.mask_rcnn import MaskRCNN, to_image_list
from infer import Resize, Normalize
from models.resnet import ResNet
from util import FileLoader
from PIL import Image
from models.mask_rcnn import *

def build_transforms(is_train=True):
  pixel_mean = [102.9801, 115.9465, 122.7717]
  pixel_std = [1., 1., 1.]
  to_bgr255 = True
  if is_train:
    min_size = (800,)
    max_size = 1333
    flip_prob = 0.5
  else:
    min_size = (800,)
    max_size = 1333
    flip_prob = 0

  normalize_transform = Normalize(
    mean=pixel_mean, std=pixel_std, to_bgr255=to_bgr255
  )

  return T.Compose(
    [
      Resize(min_size, max_size),
      T.ToTensor(), ## TODO: RandomHorizontalFlip (flip_prob) helps here for training
      Normalize(
          mean=pixel_mean, std=pixel_std, to_bgr255=True
      ),
      normalize_transform
    ]
  )

def do_train(
    model: MaskRCNN,
    file_loader: FileLoader
):
    print("Start training")
    start_training_time = time.time()
    iteration = 0

    transform = build_transforms(is_train=True)

    # Consider using only backbone and RPN parameters
    params_to_optimize = list(model.backbone.parameters()) + list(model.rpn.parameters())
    optimizer = optim.SGD(params_to_optimize, lr=0.001, momentum=0.9, weight_decay=0.0005)

    for batch in file_loader:
        iteration = iteration + len(batch)

        imgs = []
        targets = []
        for img, tg in batch:
            imgs.append(Tensor(transform(img).numpy(), requires_grad=False))
            targets.append(tg)

        # Forward pass only through backbone and RPN
        images = to_image_list(imgs)
        features = model.backbone(images.tensors)
        proposals, proposal_losses = model.rpn(images, features, targets)

        # Compute losses
        losses = sum(loss for loss in proposal_losses.values())
        
        # Backward pass and optimize
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Release unneeded variables to save memory
        del imgs
        del targets
        del features
        del proposals

        if iteration % 20 == 0:
            print(f"iter {iteration}")

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    print(f"Total training time: {total_time_str} ({total_training_time / file_loader.num_files:.4f} s / it)")

## loads the image and the target
def load_image(image_path):
  return Image.open(image_path).convert("RGB"), {}

def train():
  model = MaskRCNN(ResNet(50, num_classes=None, stride_in_1x1=True), training=True)
  file_loader = FileLoader()
  start_train_time = time.time()
  success = do_train(
      model,
      file_loader
  )
  end_train_time = time.time()
  total_training_time = end_train_time - start_train_time

  return model, success

def generate_seeds(rng, size):
  seeds = [rng.randint(0, 2**32 - 1) for _ in range(size)]
  return seeds

def main():
  # todo make distributed
  num_gpus = 1
  distributed = num_gpus > 1

  # random master seed, random.SystemRandom() uses /dev/urandom on Unix
  master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
  # random number generator with seed set to master_seed
  random_number_generator = random.Random(master_seed)


  model, success = train(0, distributed, random_number_generator)

  if success is not None:
    if success:
      print("&&&& MLPERF METRIC STATUS=SUCCESS")
    else:
      print("&&&& MLPERF METRIC STATUS=ABORTED")

def simple():
  from loss import make_match_fn,make_balanced_sampler_fn,generate_rpn_labels,RPNLossComputation, print_gpu_memory
  from extra.datasets.coco import BASEDIR

  hq_fn, _ = make_match_fn(0.7, 0.4)
  sampler = make_balanced_sampler_fn(256, 0.5)
  coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
  loss = RPNLossComputation(hq_fn, sampler, coder, generate_rpn_labels)
  channels=256
  anchor_generator = AnchorGenerator()
  backbone = ResNetFPN(ResNet(50, num_classes=None, stride_in_1x1=True), out_channels=channels)
  rpn = RPNHead(
    channels, anchor_generator.num_anchors_per_location()[0]
  )
  # optimizer = optim.SGD(backbone.parameters() + rpn.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

  import random
  from pycocotools.coco import COCO
  import os
  from PIL import Image
  import gc

  NUM_EPOCHS = 100  # or however many epochs you'd like

  # Load COCO annotations
  coco = COCO(os.path.join(BASEDIR, 'annotations', 'instances_train2017.json'))
  img_ids = coco.getImgIds()

  for epoch in range(NUM_EPOCHS):
    # Select a random image ID
    random_img_id = random.choice(img_ids)
    img_metadata = coco.loadImgs(random_img_id)[0]
    img_filename = os.path.join(BASEDIR, 'train2017', img_metadata['file_name'])
    print("training", img_filename)
    img = [Tensor(build_transforms()(Image.open(img_filename).convert("RGB")).numpy(), requires_grad=True)]
    images = to_image_list(img)
    print_gpu_memory("before backbone")
    features = backbone(images.tensors)
    features[0].realize()
    print_gpu_memory("after backbone realize")
    objectness, rpn_box_regression = rpn(features)
    print_gpu_memory("before objectness realize")
    objectness[0].realize()
    print_gpu_memory("after objectness realize")
    del objectness
    del features
    gc.collect()
    print_gpu_memory("after stuff del")
    anchors = [anchor for anchor in anchor_generator(images, features)]
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=[random_img_id]))
    gt = []
    for annotation in annotations:
        bbox = annotation['bbox']  # [x,y,width,height]
        x, y, width, height = bbox
        gt.append([x, y, x + width, y + height])
    if len(gt) == 0: continue
    targets = [BoxList(Tensor(gt), image_size=anchors[0][0].size)]
    objectness_loss, regression_loss = None, None
    try:
      objectness_loss, regression_loss = loss(anchors, objectness, rpn_box_regression, targets)
      if objectness_loss is None or regression_loss is None: continue # todo negative mine
    except Exception as e:
      print("forward error")
      print(e)
      continue
    total_loss = objectness_loss + regression_loss
    # optimizer.zero_grad()
    try:
      total_loss.backward()
    except Exception as e:
      print("backward error")
      print(e) # some backwards overflow cuda blocks
      continue
    # optimizer.step()
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {total_loss.numpy()}")
    mem_info = print_gpu_memory("epoch")
    del total_loss, images, img, features, objectness, rpn_box_regression, anchors, targets, objectness_loss, regression_loss, img_metadata, annotations

if __name__ == "__main__":
  start = time.time()
  simple()
  print("&&&& MLPERF METRIC TIME=", time.time() - start)
