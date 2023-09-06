import random
import time
import datetime

from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
from torchvision import transforms as T
from models.mask_rcnn import MaskRCNN, Resize, Normalize, to_image_list
from infer_mask_rcnn import Resize, Normalize, to_image_list
from models.resnet import ResNet
from util import FileLoader
from PIL import Image

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


if __name__ == "__main__":
  start = time.time()
  main()
  print("&&&& MLPERF METRIC TIME=", time.time() - start)
