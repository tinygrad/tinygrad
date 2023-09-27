import time

from diskcache import Cache
from tqdm import tqdm

from examples.mlperf.metrics import get_dice_score_np
from extra.datasets.kits19 import sliding_window_inference
from tinygrad.helpers import getenv

cache = Cache('/home/gijs/code_projects/tinygrad/model_cache_inference')


# @profile
def evaluate(flags, model, loader, score_fn=get_dice_score_np, epoch=0):
    s, i = 0, 0
    eval_cpu = getenv("EVAL_CPU", 1)
    for i, batch in enumerate(tqdm(loader, disable=not flags.verbose)):
      image, label = batch

      image, label = image.numpy(), label.numpy()

      # print('eval image shape',image.shape)
      # image (1, 1, 168, 365, 365)
      # eval: image (1, 1, 190, 392, 392)
      start_time = time.time()

      # @cache.memoize()
      # def slide_cache(image, label):
      #   print("SLIDE CACHE")
      #   pred, label = sliding_window_inference(model, image, label)
      #   return pred, label
      #
      # output, label = slide_cache(image, label)
      output, label = sliding_window_inference(model, image, label)
      label = label.squeeze(axis=1)
      print('label shape', label.shape)

      score = score_fn(output, label).mean()
      s += score  # to cpu saves a lot of memory
      print('score', score)
      print('eval time2', time.time() - start_time)

      del output, label

    val_dice_score = s / (i+1)

    eval_metrics = {"epoch": epoch,
                    "mean_dice": val_dice_score}

    return eval_metrics


def pad_input(volume, roi_shape, strides, padding_val, dim=3):
    """
    mode: constant, reflect, replicate, circular
    """
    bounds = [(strides[i] - volume.shape[2:][i] % strides[i]) % strides[i] for i in range(dim)]
    bounds = [bounds[i] if (volume.shape[2:][i] + bounds[i]) >= roi_shape[i] else bounds[i] + strides[i]
              for i in range(dim)]
    paddings = ((bounds[2] // 2, bounds[2] - bounds[2] // 2),
                (bounds[1] // 2, bounds[1] - bounds[1] // 2),
                (bounds[0] // 2, bounds[0] - bounds[0] // 2),
                (0, 0),
                (0, 0))

    return volume.pad(paddings, value=padding_val), paddings