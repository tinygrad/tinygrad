# Note: The one change we need to make if we're in Colab is to uncomment this below block.
# If we are in an ipython session or a notebook, clear the state to avoid bugs
"""
try:
  _ = get_ipython().__class__.__name__
  ## we set -f below to avoid prompting the user before clearing the notebook state
  %reset -f
except NameError:
  pass ## we're still good
"""

import copy
import math
import os
from functools import partial

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import transforms

import tinygrad.frontend.torch

torch.set_default_device("tiny")

print(tinygrad.frontend.torch)
## <-- teaching comments
# <-- functional comments
# You can run 'sed -i.bak '/\#\#/d' ./main.py' to remove the teaching comments if they are in the way of your work. <3

# This can go either way in terms of actually being helpful when it comes to execution speed.
# torch.backends.cudnn.benchmark = True

# This code was built from the ground up to be directly hackable and to support rapid experimentation, which is something you might see
# reflected in what would otherwise seem to be odd design decisions. It also means that maybe some cleaning up is required before moving
# to production if you're going to use this code as such (such as breaking different section into unique files, etc). That said, if there's
# ways this code could be improved and cleaned up, please do open a PR on the GitHub repo. Your support and help is much appreciated for this
# project! :)


# This is for testing that certain changes don't exceed some X% portion of the reference GPU (here an A100)
# so we can help reduce a possibility that future releases don't take away the accessibility of this codebase.
# torch.cuda.set_per_process_memory_fraction(fraction=6.5/40., device=0) ## 40. GB is the maximum memory of the base A100 GPU

# set global defaults (in this particular file) for convolutions
default_conv_kwargs = {"kernel_size": 3, "padding": "same", "bias": False}

batchsize = 32
bias_scaler = 64
# To replicate the ~95.79%-accuracy-in-110-seconds runs, you can change the base_depth from 64->128, train_epochs from 12.1->90, ['ema'] epochs 10->80, cutmix_size 3->10, and cutmix_epochs 6->80
hyp = {
  "opt": {
    "bias_lr": 1.525 * bias_scaler / 512,  # TODO: Is there maybe a better way to express the bias and batchnorm scaling? :'))))
    "non_bias_lr": 1.525 / 512,
    "bias_decay": 6.687e-4 * batchsize / bias_scaler,
    "non_bias_decay": 6.687e-4 * batchsize,
    "scaling_factor": 1.0 / 9,
    "percent_start": 0.23,
    "loss_scale_scaler": 1.0
    / 32,  # * Regularizer inside the loss summing (range: ~1/512 - 16+). FP8 should help with this somewhat too, whenever it comes out. :)
  },
  "net": {
    "whitening": {
      "kernel_size": 2,
      "num_examples": 50000,
    },
    "batch_norm_momentum": 0.4,  # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
    "cutmix_size": 3,
    "cutmix_epochs": 6,
    "pad_amount": 2,
    "base_depth": 64,  ## This should be a factor of 8 in some way to stay tensor core friendly
  },
  "misc": {
    "ema": {
      "epochs": 10,  # Slight bug in that this counts only full epochs and then additionally runs the EMA for any fractional epochs at the end too
      "decay_base": 0.95,
      "decay_pow": 3.0,
      "every_n_steps": 5,
    },
    "train_epochs": 12.1,
    "device": "tiny",
    "data_location": "data.pt",
  },
}

#############################################
#                Dataloader                 #
#############################################

if not os.path.exists(hyp["misc"]["data_location"]):
  transform = transforms.Compose([transforms.ToTensor()])

  cifar10 = torchvision.datasets.CIFAR10("cifar10/", download=True, train=True, transform=transform)
  cifar10_eval = torchvision.datasets.CIFAR10("cifar10/", download=False, train=False, transform=transform)

  # use the dataloader to get a single batch of all of the dataset items at once.
  train_dataset_gpu_loader = torch.utils.data.DataLoader(
    cifar10, batch_size=len(cifar10), drop_last=True, shuffle=True, num_workers=0, persistent_workers=False
  )
  eval_dataset_gpu_loader = torch.utils.data.DataLoader(
    cifar10_eval, batch_size=len(cifar10_eval), drop_last=True, shuffle=False, num_workers=0, persistent_workers=False
  )

  train_dataset_gpu = {}
  eval_dataset_gpu = {}

  train_dataset_gpu["images"], train_dataset_gpu["targets"] = [
    item.to(device=hyp["misc"]["device"], non_blocking=True) for item in next(iter(train_dataset_gpu_loader))
  ]
  eval_dataset_gpu["images"], eval_dataset_gpu["targets"] = [
    item.to(device=hyp["misc"]["device"], non_blocking=True) for item in next(iter(eval_dataset_gpu_loader))
  ]

  cifar10_std, cifar10_mean = torch.std_mean(
    train_dataset_gpu["images"], dim=(0, 2, 3)
  )  # dynamically calculate the std and mean from the data. this shortens the code and should help us adapt to new datasets!

  def batch_normalize_images(input_images, mean, std):
    return (input_images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

  # preload with our mean and std
  batch_normalize_images = partial(batch_normalize_images, mean=cifar10_mean, std=cifar10_std)

  ## Batch normalize datasets, now. Wowie. We did it! We should take a break and make some tea now.
  train_dataset_gpu["images"] = batch_normalize_images(train_dataset_gpu["images"])
  eval_dataset_gpu["images"] = batch_normalize_images(eval_dataset_gpu["images"])

  data = {"train": train_dataset_gpu, "eval": eval_dataset_gpu}

  ## Convert dataset to FP16 now for the rest of the process....
  data["train"]["images"] = data["train"]["images"].half().requires_grad_(False)
  data["eval"]["images"] = data["eval"]["images"].half().requires_grad_(False)

  # Convert this to one-hot to support the usage of cutmix (or whatever strange label tricks/magic you desire!)
  data["train"]["targets"] = F.one_hot(data["train"]["targets"]).half()
  data["eval"]["targets"] = F.one_hot(data["eval"]["targets"]).half()

  torch.save(data, hyp["misc"]["data_location"])

else:
  ## This is effectively instantaneous, and takes us practically straight to where the dataloader-loaded dataset would be. :)
  ## So as long as you run the above loading process once, and keep the file on the disc it's specified by default in the above
  ## hyp dictionary, then we should be good. :)
  data = torch.load(hyp["misc"]["data_location"])
  data["eval"]["targets"] = data["eval"]["targets"][:1024]
  data["train"]["targets"] = data["train"]["targets"][:1024]
  data["eval"]["images"] = data["eval"]["images"][:1024]
  data["train"]["images"] = data["train"]["images"][:1024]
  print(data["eval"]["targets"].shape)

## As you'll note above and below, one difference is that we don't count loading the raw data to GPU since it's such a variable operation, and can sort of get in the way
## of measuring other things. That said, measuring the preprocessing (outside of the padding) is still important to us.

# Pad the GPU training dataset
if hyp["net"]["pad_amount"] > 0:
  ## Uncomfortable shorthand, but basically we pad evenly on all _4_ sides with the pad_amount specified in the original dictionary
  data["train"]["images"] = F.pad(data["train"]["images"], (hyp["net"]["pad_amount"],) * 4, "reflect")

#############################################
#            Network Components             #
#############################################


# We might be able to fuse this weight and save some memory/runtime/etc, since the fast version of the network might be able to do without somehow....
class BatchNorm(nn.BatchNorm2d):
  def __init__(self, num_features, eps=1e-12, momentum=hyp["net"]["batch_norm_momentum"], weight=False, bias=True):
    super().__init__(num_features, eps=eps, momentum=momentum)
    self.weight.data.fill_(1.0)
    self.bias.data.fill_(0.0)
    self.weight.requires_grad = weight
    self.bias.requires_grad = bias


# Allows us to set default arguments for the whole convolution itself.
# Having an outer class like this does add space and complexity but offers us
# a ton of freedom when it comes to hacking in unique functionality for each layer type
class Conv(nn.Conv2d):
  def __init__(self, *args, **kwargs):
    kwargs = {**default_conv_kwargs, **kwargs}
    super().__init__(*args, **kwargs)
    self.kwargs = kwargs


class Linear(nn.Linear):
  def __init__(self, *args, temperature=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.kwargs = kwargs
    self.temperature = temperature

  def forward(self, x):
    if self.temperature is not None:
      weight = self.weight * self.temperature
    else:
      weight = self.weight
    return x @ weight.T


# can hack any changes to each convolution group that you want directly in here
class ConvGroup(nn.Module):
  def __init__(self, channels_in, channels_out):
    super().__init__()
    self.channels_in = channels_in
    self.channels_out = channels_out

    self.pool1 = nn.MaxPool2d(2)
    self.conv1 = Conv(channels_in, channels_out)
    self.conv2 = Conv(channels_out, channels_out)

    self.norm1 = BatchNorm(channels_out)
    self.norm2 = BatchNorm(channels_out)

    self.activ = nn.GELU()

  def forward(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.norm1(x)
    x = self.activ(x)
    x = self.conv2(x)
    x = self.norm2(x)
    x = self.activ(x)

    return x


class FastGlobalMaxPooling(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    # Previously was chained torch.max calls.
    # requires less time than AdaptiveMax2dPooling -- about ~.3s for the entire run, in fact (which is pretty significant! :O :D :O :O <3 <3 <3 <3)
    return torch.amax(x, dim=(2, 3))  # Global maximum pooling


#############################################
#          Init Helper Functions            #
#############################################


def get_patches(x, patch_shape=(3, 3), dtype=torch.float32):
  # This uses the unfold operation (https://pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html?highlight=unfold#torch.nn.functional.unfold)
  # to extract a _view_ (i.e., there's no data copied here) of blocks in the input tensor. We have to do it twice -- once horizontally, once vertically. Then
  # from that, we get our kernel_size*kernel_size patches to later calculate the statistics for the whitening tensor on :D
  c, (h, w) = x.shape[1], patch_shape
  return x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).to(dtype)  # TODO: Annotate?


def get_whitening_parameters(patches):
  # As a high-level summary, we're basically finding the high-dimensional oval that best fits the data here.
  # We can then later use this information to map the input information to a nicely distributed sphere, where also
  # the most significant features of the inputs each have their own axis. This significantly cleans things up for the
  # rest of the neural network and speeds up training.
  n, c, h, w = patches.shape
  est_covariance = torch.cov(patches.view(n, c * h * w).t())
  eigenvalues, eigenvectors = torch.linalg.eigh(
    est_covariance, UPLO="U"
  )  # this is the same as saying we want our eigenvectors, with the specification that the matrix be an upper triangular matrix (instead of a lower-triangular matrix)
  return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.t().reshape(c * h * w, c, h, w).flip(0)


# Run this over the training set to calculate the patch statistics, then set the initial convolution as a non-learnable 'whitening' layer
def init_whitening_conv(layer, train_set=None, num_examples=None, previous_block_data=None, pad_amount=None, freeze=True, whiten_splits=None):
  if train_set is not None and previous_block_data is None:
    if pad_amount > 0:
      previous_block_data = train_set[
        :num_examples, :, pad_amount:-pad_amount, pad_amount:-pad_amount
      ]  # if it's none, we're at the beginning of our network.
    else:
      previous_block_data = train_set[:num_examples, :, :, :]

  # chunking code to save memory for smaller-memory-size (generally consumer) GPUs
  if whiten_splits is None:
    previous_block_data_split = [previous_block_data]  # If we're whitening in one go, then put it in a list for simplicity to reuse the logic below
  else:
    previous_block_data_split = previous_block_data.split(
      whiten_splits, dim=0
    )  # Otherwise, we split this into different chunks to keep things manageable

  eigenvalue_list, eigenvector_list = [], []
  for data_split in previous_block_data_split:
    eigenvalues, eigenvectors = get_whitening_parameters(get_patches(data_split, patch_shape=layer.weight.data.shape[2:]))
    eigenvalue_list.append(eigenvalues)
    eigenvector_list.append(eigenvectors)

  eigenvalues = torch.stack(eigenvalue_list, dim=0).mean(0)
  eigenvectors = torch.stack(eigenvector_list, dim=0).mean(0)
  # i believe the eigenvalues and eigenvectors come out in float32 for this because we implicitly cast it to float32 in the patches function (for numerical stability)
  set_whitening_conv(layer, eigenvalues.to(dtype=layer.weight.dtype), eigenvectors.to(dtype=layer.weight.dtype), freeze=freeze)
  data = layer(previous_block_data.to(device="tiny", dtype=layer.weight.dtype))
  return data


def set_whitening_conv(conv_layer, eigenvalues, eigenvectors, eps=1e-2, freeze=True):
  shape = conv_layer.weight.data.shape
  eigenvectors_sliced = (eigenvectors / torch.sqrt(eigenvalues + eps))[
    -shape[0] :, :, :, :
  ]  # set the first n filters of the weight data to the top n significant (sorted by importance) filters from the eigenvectors
  eigenvectors_sliced = eigenvectors_sliced.to("tiny")
  conv_layer.weight.data = torch.cat((eigenvectors_sliced, -eigenvectors_sliced), dim=0)
  ## We don't want to train this, since this is implicitly whitening over the whole dataset
  ## For more info, see David Page's original blogposts (link in the README.md as of this commit.)
  if freeze:
    conv_layer.weight.requires_grad = False


#############################################
#            Network Definition             #
#############################################

scaler = 2.0  ## You can play with this on your own if you want, for the first beta I wanted to keep things simple (for now) and leave it out of the hyperparams dict
depths = {
  "init": round(scaler**-1 * hyp["net"]["base_depth"]),  # 32  w/ scaler at base value
  "block1": round(scaler**0 * hyp["net"]["base_depth"]),  # 64  w/ scaler at base value
  "block2": round(scaler**2 * hyp["net"]["base_depth"]),  # 256 w/ scaler at base value
  "block3": round(scaler**3 * hyp["net"]["base_depth"]),  # 512 w/ scaler at base value
  "num_classes": 10,
}


class SpeedyConvNet(nn.Module):
  def __init__(self, network_dict):
    super().__init__()
    self.net_dict = network_dict  # flexible, defined in the make_net function

  # This allows you to customize/change the execution order of the network as needed.
  def forward(self, x):
    if not self.training:
      x = torch.cat((x, torch.flip(x, (-1,))))
    x = self.net_dict["initial_block"]["whiten"](x)
    x = self.net_dict["initial_block"]["activation"](x)
    x = self.net_dict["conv_group_1"](x)
    x = self.net_dict["conv_group_2"](x)
    x = self.net_dict["conv_group_3"](x)
    x = self.net_dict["pooling"](x)
    x = self.net_dict["linear"](x)
    if not self.training:
      # Average the predictions from the lr-flipped inputs during eval
      orig, flipped = x.split(x.shape[0] // 2, dim=0)
      x = 0.5 * orig + 0.5 * flipped
    return x


def make_net():
  # TODO: A way to make this cleaner??
  # Note, you have to specify any arguments overlapping with defaults (i.e. everything but in/out depths) as kwargs so that they are properly overridden (TODO cleanup somehow?)
  whiten_conv_depth = 3 * hyp["net"]["whitening"]["kernel_size"] ** 2
  network_dict = nn.ModuleDict({
    "initial_block": nn.ModuleDict({
      "whiten": Conv(3, whiten_conv_depth, kernel_size=hyp["net"]["whitening"]["kernel_size"], padding=0),
      "activation": nn.GELU(),
    }),
    "conv_group_1": ConvGroup(2 * whiten_conv_depth, depths["block1"]),
    "conv_group_2": ConvGroup(depths["block1"], depths["block2"]),
    "conv_group_3": ConvGroup(depths["block2"], depths["block3"]),
    "pooling": FastGlobalMaxPooling(),
    "linear": Linear(depths["block3"], depths["num_classes"], bias=False, temperature=hyp["opt"]["scaling_factor"]),
  })

  net = SpeedyConvNet(network_dict)
  net = net.to(hyp["misc"]["device"])
  net = net.to(memory_format=torch.channels_last)  # to appropriately use tensor cores/avoid thrash while training
  net.train()
  net.half()  # Convert network to half before initializing the initial whitening layer.

  ## Initialize the whitening convolution
  with torch.no_grad():
    # Initialize the first layer to be fixed weights that whiten the expected input values of the network be on the unit hypersphere. (i.e. their...average vector length is 1.?, IIRC)
    init_whitening_conv(
      net.net_dict["initial_block"]["whiten"],
      data["train"]["images"].index_select(0, torch.randperm(data["train"]["images"].shape[0], device=data["train"]["images"].device)),
      num_examples=hyp["net"]["whitening"]["num_examples"],
      pad_amount=hyp["net"]["pad_amount"],
      whiten_splits=5000,
    )  ## Hardcoded for now while we figure out the optimal whitening number
    ## If you're running out of memory (OOM) feel free to decrease this, but
    ## the index lookup in the dataloader may give you some trouble depending
    ## upon exactly how memory-limited you are

    for layer_name in net.net_dict.keys():
      if "conv_group" in layer_name:
        # Create an implicit residual via a dirac-initialized tensor
        dirac_weights_in = torch.nn.init.dirac_(torch.empty_like(net.net_dict[layer_name].conv1.weight))

        # Add the implicit residual to the already-initialized convolutional transition layer.
        # One can use more sophisticated initializations, but this one appeared worked best in testing.
        # What this does is brings up the features from the previous residual block virtually, so not only
        # do we have residual information flow within each block, we have a nearly direct connection from
        # the early layers of the network to the loss function.
        std_pre, mean_pre = torch.std_mean(net.net_dict[layer_name].conv1.weight.data)
        net.net_dict[layer_name].conv1.weight.data = net.net_dict[layer_name].conv1.weight.data + dirac_weights_in
        std_post, mean_post = torch.std_mean(net.net_dict[layer_name].conv1.weight.data)

        # Renormalize the weights to match the original initialization statistics
        net.net_dict[layer_name].conv1.weight.data.sub_(mean_post).div_(std_post).mul_(std_pre).add_(mean_pre)

        ## We do the same for the second layer in each convolution group block, since this only
        ## adds a simple multiplier to the inputs instead of the noise of a randomly-initialized
        ## convolution. This can be easily scaled down by the network, and the weights can more easily
        ## pivot in whichever direction they need to go now.
        ## The reason that I believe that this works so well is because a combination of MaxPool2d
        ## and the nn.GeLU function's positive bias encouraging values towards the nearly-linear
        ## region of the GeLU activation function at network initialization. I am not currently
        ## sure about this, however, it will require some more investigation. For now -- it works! D:
        torch.nn.init.dirac_(net.net_dict[layer_name].conv2.weight)

  return net


#############################################
#            Data Preprocessing             #
#############################################


## This is actually (I believe) a pretty clean implementation of how to do something like this, since shifted-square masks unique to each depth-channel can actually be rather
## tricky in practice. That said, if there's a better way, please do feel free to submit it! This can be one of the harder parts of the code to understand (though I personally get
## stuck on the fold/unfold process for the lower-level convolution calculations.
def make_random_square_masks(inputs, mask_size):
  ##### TODO: Double check that this properly covers the whole range of values. :'( :')
  if mask_size == 0:
    return None  # no need to cutout or do anything like that since the patch_size is set to 0
  is_even = int(mask_size % 2 == 0)
  in_shape = inputs.shape

  # seed centers of squares to cutout boxes from, in one dimension each
  mask_center_y = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(
    mask_size // 2 - is_even, in_shape[-2] - mask_size // 2 - is_even
  )
  mask_center_x = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(
    mask_size // 2 - is_even, in_shape[-1] - mask_size // 2 - is_even
  )

  # measure distance, using the center as a reference point
  to_mask_y_dists = torch.arange(in_shape[-2], device=inputs.device).view(1, 1, in_shape[-2], 1) - mask_center_y.view(-1, 1, 1, 1)
  to_mask_x_dists = torch.arange(in_shape[-1], device=inputs.device).view(1, 1, 1, in_shape[-1]) - mask_center_x.view(-1, 1, 1, 1)

  to_mask_y = (to_mask_y_dists >= (-(mask_size // 2) + is_even)) * (to_mask_y_dists <= mask_size // 2)
  to_mask_x = (to_mask_x_dists >= (-(mask_size // 2) + is_even)) * (to_mask_x_dists <= mask_size // 2)

  final_mask = (
    to_mask_y * to_mask_x
  )  ## Turn (y by 1) and (x by 1) boolean masks into (y by x) masks through multiplication. Their intersection is square, hurray! :D

  return final_mask


def batch_cutmix(inputs, targets, patch_size):
  with torch.no_grad():
    batch_permuted = torch.randperm(inputs.shape[0], device="tiny")
    cutmix_batch_mask = make_random_square_masks(inputs, patch_size)
    if cutmix_batch_mask is None:
      return inputs, targets  # if the mask is None, then that's because the patch size was set to 0 and we will not be using cutmix today.
    # We draw other samples from inside of the same batch
    cutmix_batch = torch.where(cutmix_batch_mask, torch.index_select(inputs, 0, batch_permuted), inputs)
    cutmix_targets = torch.index_select(targets, 0, batch_permuted)
    # Get the percentage of each target to mix for the labels by the % proportion of pixels in the mix
    portion_mixed = float(patch_size**2) / (inputs.shape[-2] * inputs.shape[-1])
    cutmix_labels = portion_mixed * cutmix_targets + (1.0 - portion_mixed) * targets
    return cutmix_batch, cutmix_labels


def batch_crop(inputs, crop_size):
  with torch.no_grad():
    crop_mask_batch = make_random_square_masks(inputs, crop_size)
    cropped_batch = torch.masked_select(inputs, crop_mask_batch).view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)
    return cropped_batch


def batch_flip_lr(batch_images, flip_chance=0.5):
  with torch.no_grad():
    # TODO: Is there a more elegant way to do this? :') :'((((
    return torch.where(torch.rand_like(batch_images[:, 0, 0, 0].view(-1, 1, 1, 1)) < flip_chance, torch.flip(batch_images, (-1,)), batch_images)


########################################
#          Training Helpers            #
########################################


class NetworkEMA(nn.Module):
  def __init__(self, net):
    super().__init__()  # init the parent module so this module is registered properly
    self.net_ema = copy.deepcopy(net).eval().requires_grad_(False)  # copy the model

  def update(self, current_net, decay):
    with torch.no_grad():
      for ema_net_parameter, (parameter_name, incoming_net_parameter) in zip(
        self.net_ema.state_dict().values(), current_net.state_dict().items()
      ):  # potential bug: assumes that the network architectures don't change during training (!!!!)
        if incoming_net_parameter.dtype in (torch.half, torch.float):
          ema_net_parameter.mul_(decay).add_(
            incoming_net_parameter.detach().mul(1.0 - decay)
          )  # update the ema values in place, similar to how optimizer momentum is coded
          # And then we also copy the parameters back to the network, similarly to the Lookahead optimizer (but with a much more aggressive-at-the-end schedule)
          if not ("norm" in parameter_name and "weight" in parameter_name) and not "whiten" in parameter_name:
            incoming_net_parameter.copy_(ema_net_parameter.detach())

  def forward(self, inputs):
    with torch.no_grad():
      return self.net_ema(inputs)


# TODO: Could we jit this in the (more distant) future? :)
@torch.no_grad()
def get_batches(data_dict, key, batchsize, epoch_fraction=1.0, cutmix_size=None):
  num_epoch_examples = len(data_dict[key]["images"])
  shuffled = torch.randperm(num_epoch_examples, device="tiny")
  if epoch_fraction < 1:
    shuffled = shuffled[
      : batchsize * round(epoch_fraction * shuffled.shape[0] / batchsize)
    ]  # TODO: Might be slightly inaccurate, let's fix this later... :) :D :confetti: :fireworks:
    num_epoch_examples = shuffled.shape[0]
  crop_size = 32

  ## Here, we prep the dataset by applying all data augmentations in batches ahead of time before each epoch, then we return an iterator below
  ## that iterates in chunks over with a random derangement (i.e. shuffled indices) of the individual examples. So we get perfectly-shuffled
  ## batches (which skip the last batch if it's not a full batch), but everything seems to be (and hopefully is! :D) properly shuffled. :)
  if key == "train":
    images = batch_crop(data_dict[key]["images"], crop_size)  # TODO: hardcoded image size for now?
    images = batch_flip_lr(images)
    images, targets = batch_cutmix(images, data_dict[key]["targets"], patch_size=cutmix_size)
  else:
    images = data_dict[key]["images"]
    targets = data_dict[key]["targets"]

  # Send the images to an (in beta) channels_last to help improve tensor core occupancy (and reduce NCHW <-> NHWC thrash) during training
  images = images.to(device="tiny", memory_format=torch.channels_last)
  targets = targets.to(device="tiny")

  for idx in range(num_epoch_examples // batchsize):
    if not (idx + 1) * batchsize > num_epoch_examples:  ## Use the shuffled randperm to assemble individual items into a minibatch
      yield (
        images.index_select(0, shuffled[idx * batchsize : (idx + 1) * batchsize]),
        targets.index_select(0, shuffled[idx * batchsize : (idx + 1) * batchsize]),
      )  ## Each item is only used/accessed by the network once per epoch. :D


def init_split_parameter_dictionaries(network):
  params_non_bias = {
    "params": [],
    "lr": hyp["opt"]["non_bias_lr"],
    "momentum": 0.85,
    "nesterov": True,
    "weight_decay": hyp["opt"]["non_bias_decay"],
    "foreach": True,
  }
  params_bias = {
    "params": [],
    "lr": hyp["opt"]["bias_lr"],
    "momentum": 0.85,
    "nesterov": True,
    "weight_decay": hyp["opt"]["bias_decay"],
    "foreach": True,
  }

  for name, p in network.named_parameters():
    if p.requires_grad:
      if "bias" in name:
        params_bias["params"].append(p)
      else:
        params_non_bias["params"].append(p)
  return params_non_bias, params_bias


## Hey look, it's the soft-targets/label-smoothed loss! Native to PyTorch. Now, _that_ is pretty cool, and simplifies things a lot, to boot! :D :)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, reduction="none")

logging_columns_list = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "ema_val_acc", "total_time_seconds"]


# define the printing function and print the column heads
def print_training_details(columns_list, separator_left="|  ", separator_right="  ", final="|", column_heads_only=False, is_final_entry=False):
  print_string = ""
  if column_heads_only:
    for column_head_name in columns_list:
      print_string += separator_left + column_head_name + separator_right
    print_string += final
    print("-" * (len(print_string)))  # print the top bar
    print(print_string)
    print("-" * (len(print_string)))  # print the bottom bar
  else:
    for column_value in columns_list:
      print_string += separator_left + column_value + separator_right
    print_string += final
    print(print_string)
  if is_final_entry:
    print("-" * (len(print_string)))  # print the final output bar


print_training_details(
  logging_columns_list, column_heads_only=True
)  ## print out the training column heads before we print the actual content for each run.

########################################
#           Train and Eval             #
########################################


def main():
  # Initializing constants for the whole run.
  net_ema = None  ## Reset any existing network emas, we want to have _something_ to check for existence so we can initialize the EMA right from where the network is during training
  ## (as opposed to initializing the network_ema from the randomly-initialized starter network, then forcing it to play catch-up all of a sudden in the last several epochs)

  total_time_seconds = 0.0
  current_steps = 0.0

  # TODO: Doesn't currently account for partial epochs really (since we're not doing "real" epochs across the whole batchsize)....
  num_steps_per_epoch = len(data["train"]["images"]) // batchsize
  total_train_steps = math.ceil(num_steps_per_epoch * hyp["misc"]["train_epochs"])
  ema_epoch_start = math.floor(hyp["misc"]["train_epochs"]) - hyp["misc"]["ema"]["epochs"]

  ## I believe this wasn't logged, but the EMA update power is adjusted by being raised to the power of the number of "every n" steps
  ## to somewhat accomodate for whatever the expected information intake rate is. The tradeoff I believe, though, is that this is to some degree noisier as we
  ## are intaking fewer samples of our distribution-over-time, with a higher individual weight each. This can be good or bad depending upon what we want.
  projected_ema_decay_val = hyp["misc"]["ema"]["decay_base"] ** hyp["misc"]["ema"]["every_n_steps"]

  # Adjust pct_start based upon how many epochs we need to finetune the ema at a low lr for
  pct_start = hyp["opt"]["percent_start"]  # * (total_train_steps/(total_train_steps - num_low_lr_steps_for_ema))

  # Get network
  net = make_net()

  ## Stowing the creation of these into a helper function to make things a bit more readable....
  non_bias_params, bias_params = init_split_parameter_dictionaries(net)

  # One optimizer for the regular network, and one for the biases. This allows us to use the superconvergence onecycle training policy for our networks....
  opt = torch.optim.SGD(**non_bias_params)
  opt_bias = torch.optim.SGD(**bias_params)

  ## Not the most intuitive, but this basically takes us from ~0 to max_lr at the point pct_start, then down to .1 * max_lr at the end (since 1e16 * 1e-15 = .1 --
  ##   This quirk is because the final lr value is calculated from the starting lr value and not from the maximum lr value set during training)
  initial_div_factor = 1e16  # basically to make the initial lr ~0 or so :D
  final_lr_ratio = 0.07  # Actually pretty important, apparently!
  lr_sched = torch.optim.lr_scheduler.OneCycleLR(
    opt,
    max_lr=non_bias_params["lr"],
    pct_start=pct_start,
    div_factor=initial_div_factor,
    final_div_factor=1.0 / (initial_div_factor * final_lr_ratio),
    total_steps=total_train_steps,
    anneal_strategy="linear",
    cycle_momentum=False,
  )
  lr_sched_bias = torch.optim.lr_scheduler.OneCycleLR(
    opt_bias,
    max_lr=bias_params["lr"],
    pct_start=pct_start,
    div_factor=initial_div_factor,
    final_div_factor=1.0 / (initial_div_factor * final_lr_ratio),
    total_steps=total_train_steps,
    anneal_strategy="linear",
    cycle_momentum=False,
  )

  ## For accurately timing GPU code
  # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  # torch.cuda.synchronize()  ## clean up any pre-net setup operations

  if True:  ## Sometimes we need a conditional/for loop here, this is placed to save the trouble of needing to indent
    for epoch in range(math.ceil(hyp["misc"]["train_epochs"])):
      #################
      # Training Mode #
      #################
      # torch.cuda.synchronize()
      # starter.record()
      net.train()

      loss_train = None
      accuracy_train = None

      cutmix_size = hyp["net"]["cutmix_size"] if epoch >= hyp["misc"]["train_epochs"] - hyp["net"]["cutmix_epochs"] else 0
      epoch_fraction = (
        1 if epoch + 1 < hyp["misc"]["train_epochs"] else hyp["misc"]["train_epochs"] % 1
      )  # We need to know if we're running a partial epoch or not.

      for epoch_step, (inputs, targets) in enumerate(
        get_batches(data, key="train", batchsize=batchsize, epoch_fraction=epoch_fraction, cutmix_size=cutmix_size)
      ):
        ## Run everything through the network
        outputs = net(inputs)

        loss_batchsize_scaler = (
          512 / batchsize
        )  # to scale to keep things at a relatively similar amount of regularization when we change our batchsize since we're summing over the whole batch
        ## If you want to add other losses or hack around with the loss, you can do that here.
        loss = (
          loss_fn(outputs, targets).mul(hyp["opt"]["loss_scale_scaler"] * loss_batchsize_scaler).sum().div(hyp["opt"]["loss_scale_scaler"])
        )  ## Note, as noted in the original blog posts, the summing here does a kind of loss scaling
        ## (and is thus batchsize dependent as a result). This can be somewhat good or bad, depending...

        # we only take the last-saved accs and losses from train
        if epoch_step % 50 == 0:
          train_acc = (outputs.detach().argmax(-1) == targets.argmax(-1)).float().mean().item()
          train_loss = loss.detach().cpu().item() / (batchsize * loss_batchsize_scaler)

        loss.backward()

        ## Step for each optimizer, in turn.
        opt.step()
        opt_bias.step()

        # We only want to step the lr_schedulers while we have training steps to consume. Otherwise we get a not-so-friendly error from PyTorch
        lr_sched.step()
        lr_sched_bias.step()

        ## Using 'set_to_none' I believe is slightly faster (albeit riskier w/ funky gradient update workflows) than under the default 'set to zero' method
        opt.zero_grad(set_to_none=True)
        opt_bias.zero_grad(set_to_none=True)
        current_steps += 1

        if epoch >= ema_epoch_start and current_steps % hyp["misc"]["ema"]["every_n_steps"] == 0:
          ## Initialize the ema from the network at this point in time if it does not already exist.... :D
          if net_ema is None:  # don't snapshot the network yet if so!
            net_ema = NetworkEMA(net)
            continue
          # We warm up our ema's decay/momentum value over training exponentially according to the hyp config dictionary (this lets us move fast, then average strongly at the end).
          net_ema.update(net, decay=projected_ema_decay_val * (current_steps / total_train_steps) ** hyp["misc"]["ema"]["decay_pow"])

      # ender.record()
      # torch.cuda.synchronize()
      total_time_seconds += 1e-3  # * starter.elapsed_time(ender)

      ####################
      # Evaluation  Mode #
      ####################
      net.eval()

      eval_batchsize = 32
      assert data["eval"]["images"].shape[0] % eval_batchsize == 0, (
        "Error: The eval batchsize must evenly divide the eval dataset (for now, we don't have drop_remainder implemented yet)."
      )
      loss_list_val, acc_list, acc_list_ema = [], [], []

      with torch.no_grad():
        for inputs, targets in get_batches(data, key="eval", batchsize=eval_batchsize):
          if epoch >= ema_epoch_start:
            outputs = net_ema(inputs)
            acc_list_ema.append((outputs.argmax(-1) == targets.argmax(-1)).float().mean())
          outputs = net(inputs)
          loss_list_val.append(loss_fn(outputs, targets).float().mean())
          acc_list.append((outputs.argmax(-1) == targets.argmax(-1)).float().mean())

        val_acc = torch.stack(acc_list).mean().item()
        ema_val_acc = None
        # TODO: We can fuse these two operations (just above and below) all-together like :D :))))
        if epoch >= ema_epoch_start:
          ema_val_acc = torch.stack(acc_list_ema).mean().item()

        val_loss = torch.stack(loss_list_val).mean().item()
      # We basically need to look up local variables by name so we can have the names, so we can pad to the proper column width.
      ## Printing stuff in the terminal can get tricky and this used to use an outside library, but some of the required stuff seemed even
      ## more heinous than this, unfortunately. So we switched to the "more simple" version of this!
      format_for_table = (
        lambda x, locals: (f"{locals[x]}".rjust(len(x)))
        if type(locals[x]) == int
        else "{:0.4f}".format(locals[x]).rjust(len(x))
        if locals[x] is not None
        else " " * len(x)
      )

      # Print out our training details (sorry for the complexity, the whole logging business here is a bit of a hot mess once the columns need to be aligned and such....)
      ## We also check to see if we're in our final epoch so we can print the 'bottom' of the table for each round.
      print_training_details(
        list(map(partial(format_for_table, locals=locals()), logging_columns_list)),
        is_final_entry=(epoch >= math.ceil(hyp["misc"]["train_epochs"] - 1)),
      )
  return ema_val_acc  # Return the final ema accuracy achieved (not using the 'best accuracy' selection strategy, which I think is okay here....)


if __name__ == "__main__":
  acc_list = []
  for run_num in range(25):
    acc_list.append(torch.tensor(main()))
  print("Mean and variance:", (torch.mean(torch.stack(acc_list)).item(), torch.var(torch.stack(acc_list)).item()))
