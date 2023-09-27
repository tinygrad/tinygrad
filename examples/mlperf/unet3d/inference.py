import time

from scipy import signal
from tqdm import tqdm
import numpy as np

from examples.mlperf.unet3d.losses import DiceCELoss, DiceScore
from extra.datasets.kits19 import sliding_window_inference
from tinygrad.helpers import dtypes, getenv
from tinygrad.tensor import Tensor
# def evaluate(flags, model, loader, loss_fn, score_fn, device, epoch=0, is_distributed=False):
#     rank = get_rank()
#     world_size = get_world_size()
#     model.to(device)
#     if flags.load_ckpt_path:
#         map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
#         checkpoint = torch.load(flags.load_ckpt_path, map_location=map_location)
#         epoch = checkpoint['epoch']
#         model.load_state_dict(checkpoint['best_model_state_dict'])
#         if is_distributed:
#             model = torch.nn.parallel.DistributedDataParallel(model,
#                                                               device_ids=[flags.local_rank],
#                                                               output_device=flags.local_rank)
#
#     model.eval()
#
#     eval_loss = []
#     scores = []
#     with torch.no_grad():
#         for i, batch in enumerate(tqdm(loader, disable=(rank != 0) or not flags.verbose)):
#             image, label = batch
#             image, label = image.to(device), label.to(device)
#             if image.numel() == 0:
#                 continue
#             with autocast(enabled=flags.amp):
#                 output, label = sliding_window_inference(
#                     inputs=image,
#                     labels=label,
#                     roi_shape=flags.val_input_shape,
#                     model=model,
#                     overlap=flags.overlap,
#                     mode="gaussian",
#                     padding_val=-2.2
#                 )
#                 eval_loss_value = loss_fn(output, label)
#                 scores.append(score_fn(output, label))
#             eval_loss.append(eval_loss_value)
#             del output
#             del label
#
#     scores = reduce_tensor(torch.mean(torch.stack(scores, dim=0), dim=0), world_size)
#     eval_loss = reduce_tensor(torch.mean(torch.stack(eval_loss, dim=0), dim=0), world_size)
#     # scores = torch.mean(torch.stack(scores, dim=0), dim=0)
#     # eval_loss = torch.mean(torch.stack(eval_loss, dim=0), dim=0)
#
#     scores, eval_loss = scores.cpu().numpy(), float(eval_loss.cpu().numpy())
#     eval_metrics = {"epoch": epoch,
#                     "L1 dice": scores[-2],
#                     "L2 dice": scores[-1],
#                     "mean_dice": (scores[-1] + scores[-2]) / 2,
#                     "eval_loss": eval_loss}
#
#     return eval_metrics
from line_profiler_pycharm import profile
@profile
def evaluate(flags, model, loader, score_fn, epoch=0):
    s, i = 0, 0
    eval_cpu = getenv("EVAL_CPU", 1)
    for i, batch in enumerate(tqdm(loader, disable=not flags.verbose)):
      # print("eval batch", i)
      image, label = batch
      dtype_img = dtypes.half if getenv("FP16") else dtypes.float

      image, label = Tensor(image.numpy()[:1], dtype=dtype_img), Tensor(label.numpy()[:1], dtype=dtype_img) # todo added [:1] for overfitting

      # print('eval image shape',image.shape)
      start_time = time.time()
      output, label = sliding_window_inference(model, image, label, flags.val_input_shape)
      del image
      if eval_cpu:
        output = output.cpu().realize()
        label = label.cpu().realize()
      # print('output.shape', output.shape) #~ (1, 3, 190, 384, 384)
      s += score_fn(output, label).mean().numpy() # to cpu saves a lot of memory
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


def gaussian_kernel(n, std):
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian3D = np.outer(gaussian2D, gaussian1D)
    gaussian3D = gaussian3D.reshape(n, n, n)
    gaussian3D = np.cbrt(gaussian3D)
    gaussian3D /= gaussian3D.max()
    return Tensor(gaussian3D)