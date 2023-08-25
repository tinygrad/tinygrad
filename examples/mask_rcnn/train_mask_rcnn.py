import functools
import random
import time
import datetime

from torchvision import transforms as T
from models.mask_rcnn import MaskRCNN, Resize, Normalize
from models.resnet import ResNet

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


def mlperf_log_epoch_start(iteration, iters_per_epoch):
  # First iteration:
  #     Note we've started training & tag first epoch start
  if iteration == 0:
    print("epoch start epoch 0")
    return
  if iteration % iters_per_epoch == 0:
    epoch = iteration // iters_per_epoch
    print("epoch start epoch {}".format(epoch))

def make_data_sampler(dataset, shuffle, distributed):
  # TODO Implement for tiny, this is an import data sampler
  # if distributed:
  #     return samplers.DistributedSampler(dataset, shuffle=shuffle)
  # if shuffle:
  #     sampler = torch.utils.data.sampler.RandomSampler(dataset)
  # else:
  #     sampler = torch.utils.data.sampler.SequentialSampler(dataset)
  return None

def make_data_loader(cfg,
                num_gpus=1,
                images_per_batch=1,
                start_iter=0, random_number_generator=None):
  images_per_batch = cfg.SOLVER.IMS_PER_BATCH
  assert (
    images_per_batch % num_gpus == 0
  ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
  "of GPUs ({}) used.".format(images_per_batch, num_gpus)
  images_per_gpu = images_per_batch // num_gpus
  shuffle = True
  num_iters = 90000

  # group images which have similar aspect ratio. In this case, we only
  # group in two cases: those with width / height > 1, and the other way around,
  # but the code supports more general grouping strategy
  aspect_grouping = [1]

  # todo reintroduce test 
  dataset_list = ["/path/to/train"]

  transforms = build_transforms(is_train=True)

  ## TODO, this needs a tiny equivalent
  # datasets, epoch_size = build_dataset(dataset_list, transforms, DatasetCatalog, is_train)

  # data_loaders = []
  # for dataset in datasets:
  #     sampler = make_data_sampler(dataset, shuffle, is_distributed)
  #     batch_sampler = make_batch_data_sampler(
  #         dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter, random_number_generator,
  #     )
  #     collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
  #     num_workers = cfg.DATALOADER.NUM_WORKERS
  #     data_loader = torch.utils.data.DataLoader(
  #         dataset,
  #         num_workers=num_workers,
  #         batch_sampler=batch_sampler,
  #         collate_fn=collator,
  #     )
  #     data_loaders.append(data_loader)
  # if is_train:
  #     # during training, a single (possibly concatenated) data_loader is returned
  #     assert len(data_loaders) == 1
  #     iterations_per_epoch = epoch_size // images_per_batch + 1
  #     return data_loaders[0], iterations_per_epoch
  # return data_loaders

def do_train(
  model: MaskRCNN,
  data_loader,
  scheduler,
  checkpointer,
  device,
  checkpoint_period,
  save_checkpoints,
  iteration,
  per_iter_end_callback_fn
):
  print("Start training")
  max_iter = len(data_loader)
  start_iter = iteration
  model.train()
  start_training_time = time.time()
  end = time.time()
  for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
    data_time = time.time() - end ## could be useful to log data load time
    iteration = iteration + 1

    # TODO is there a tiny scheduler equiv?
    scheduler.step()

    # todo convert to tiny
    images = images.to(device)
    targets = [target.to(device) for target in targets]

    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())

    losses.backward()

    optimizer.step()
    optimizer.zero_grad()

    
    end = time.time()

    if iteration % 20 == 0 or iteration == max_iter:
      print(
          [
              "eta: {eta}",
              "iter: {iter}",
              "{meters}",
              "lr: {lr:.6f}",
              "max mem: {memory:.0f}",
          ].join("\t")
      )
    # TODO tiny checkpointer
    # if iteration % checkpoint_period == 0 and save_checkpoints:
    #     checkpointer.save("model_{:07d}".format(iteration), **[iteration])
    # if iteration == max_iter and save_checkpoints:
    #     checkpointer.save("model_final", **[iteration])

    ## TODO: Implement early-exit -- nice for testing
    if per_iter_end_callback_fn is not None:
      # Note: iteration has been incremented previously for
      # human-readable checkpoint names (i.e. 60000 instead of 59999)
      # so need to adjust again here
      early_exit = per_iter_end_callback_fn(iteration=iteration-1)
      if early_exit:
        break

  total_training_time = time.time() - start_training_time
  total_time_str = str(datetime.timedelta(seconds=total_training_time))
  print("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / (max_iter)))
  if per_iter_end_callback_fn is not None:
    if early_exit:
      return True
    else:
      return False
  else:
    return None

def train(local_rank, distributed, random_number_generator):
  model = MaskRCNN(ResNet(50, num_classes=None, stride_in_1x1=True), training=True)
    
    
  # if distributed:
  #     model = torch.nn.parallel.DistributedDataParallel(
  #         model, device_ids=[local_rank], output_device=local_rank,
  #         # this should be removed if we update BatchNorm stats
  #         broadcast_buffers=False,
  #     )


  data_loader, iters_per_epoch = make_data_loader(
      is_train=True,
      is_distributed=distributed,
      start_iter=0,
      random_number_generator=random_number_generator
  )

  checkpoint_period = 2500

  # set the callback function to evaluate and potentially
  # early exit each epoch
  # TODO Implement early exit, Swap with tiny tester
  start_train_time = time.time()
  success = do_train(
      model,
      data_loader,
      checkpoint_period,
      per_iter_start_callback_fn=functools.partial(mlperf_log_epoch_start, iters_per_epoch=iters_per_epoch)
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
