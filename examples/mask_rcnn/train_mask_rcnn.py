import random
import time
import datetime

from torchvision import transforms as T
from models.mask_rcnn import MaskRCNN, Resize, Normalize
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
  for image in file_loader: 
    iteration = iteration + 1

    # TODO is there a tiny scheduler equiv?
    scheduler.step()

    # todo convert to tiny
    images = image.to(device )## TODO: multi image batches  
    targets = [target.to(device) for target in targets]

    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())

    losses.backward()

    optimizer.step()
    optimizer.zero_grad()

    if iteration % 20 == 0:
      print(f"iter {iteration}")
    # TODO tiny checkpointer
    # TODO: Implement early-exit -- nice for testing
  total_training_time = time.time() - start_training_time
  total_time_str = str(datetime.timedelta(seconds=total_training_time))
  print("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / (file_loader.num_files)))

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
  print(
          "&&&& MLPERF METRIC THROUGHPUT per GPU={:.4f} iterations / s".format((arguments["iteration"] * 1.0) / total_training_time)
  )

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
