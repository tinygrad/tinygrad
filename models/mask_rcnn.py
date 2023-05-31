import re
import numpy as np
from pathlib import Path
from tinygrad import nn
from tinygrad.tensor import Tensor
from extra.utils import get_child, download_file, fake_torch_load
from models.resnet import ResNet
from torch.nn import functional as F
import torch

def upsample(x):
  bs, c, py, px = x.shape
  return x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py * 2, px * 2)


class LastLevelMaxPool:
  def __call__(self, x):
    return [Tensor.max_pool2d(x, 1, 2)]


class FPN:
  def __init__(self, in_channels_list, out_channels):
    self.inner_blocks, self.layer_blocks = [], []
    for in_channels in in_channels_list:
      self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
      self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    self.top_block = LastLevelMaxPool()

  def __call__(self, x: Tensor):
    """
    Arguments:
        x (list[Tensor]): feature maps for each feature level.
    Returns:
        results (tuple[Tensor]): feature maps after FPN layers.
            They are ordered from highest resolution first.
    """
    last_inner = self.inner_blocks[-1](x[-1])
    results = []
    results.append(self.layer_blocks[-1](last_inner))
    for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
    ):
      if not inner_block:
        continue
      inner_top_down = Tensor(F.interpolate(torch.tensor(last_inner.numpy()), scale_factor=2, mode="nearest").numpy())
      inner_lateral = inner_block(feature)
      last_inner = inner_lateral + inner_top_down
      results.insert(0, layer_block(last_inner))
    last_results = self.top_block(results[-1])
    results.extend(last_results)

    return tuple(results)

class ResNetFPN:
  def __init__(self, resnet, out_channels=256):
    self.out_channels = out_channels
    self.body = resnet
    in_channels_stage2 = 256
    in_channels_list = [
      in_channels_stage2,
      in_channels_stage2 * 2,
      in_channels_stage2 * 4,
      in_channels_stage2 * 8,
    ]
    self.fpn = FPN(in_channels_list, out_channels)

  def __call__(self, x):
    x = self.body(x)
    return self.fpn(x)




class AnchorGenerator:
  def __init__(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2), strides=(4, 8, 16, 32, 64)):
    anchors = [generate_anchors(stride, (size,), aspect_ratios) for stride, size in zip(strides, sizes)]
    self.cell_anchors = [Tensor(a) for a in anchors]

  def num_anchors_per_location(self):
    return [cell_anchors.shape[0] for cell_anchors in self.cell_anchors]


def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
  return _generate_anchors(stride, np.array(sizes, dtype=np.float32) / stride, np.array(aspect_ratios, dtype=np.float32))


def _generate_anchors(base_size, scales, aspect_ratios):
  anchor = np.array([1, 1, base_size, base_size], dtype=np.float32) - 1
  anchors = _ratio_enum(anchor, aspect_ratios)
  anchors = np.vstack(
    [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
  )
  return anchors


def _whctrs(anchor):
  w = anchor[2] - anchor[0] + 1
  h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
  ws = ws[:, np.newaxis]
  hs = hs[:, np.newaxis]
  anchors = np.hstack((
    x_ctr - 0.5 * (ws - 1),
    y_ctr - 0.5 * (hs - 1),
    x_ctr + 0.5 * (ws - 1),
    y_ctr + 0.5 * (hs - 1),
  ))
  return anchors


def _ratio_enum(anchor, ratios):
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  size = w * h
  size_ratios = size / ratios
  ws = np.round(np.sqrt(size_ratios))
  hs = np.round(ws * ratios)
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


def _scale_enum(anchor, scales):
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  ws = w * scales
  hs = h * scales
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


class RPNHead:
  def __init__(self, in_channels, num_anchors):
    self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
    self.cls_logits = nn.Conv2d(256, num_anchors, kernel_size=1)
    self.bbox_pred = nn.Conv2d(256, num_anchors * 4, kernel_size=1)


class RPN:
  def __init__(self, in_channels):
    self.anchor_generator = AnchorGenerator()
    self.head = RPNHead(in_channels, self.anchor_generator.num_anchors_per_location()[0])


def make_conv3x3(
  in_channels,
  out_channels,
  dilation=1,
  stride=1,
  use_gn=False,
):
  conv = nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=stride,
    padding=dilation,
    dilation=dilation,
    bias=False if use_gn else True
  )
  return conv


class MaskRCNNFPNFeatureExtractor:
  def __init__(self):
    resolution = 14
    scales = (1.0 / 16,)
    sampling_ratio = 0
    pooler = Pooler(
      output_size=(resolution, resolution),
      scales=scales,
      sampling_ratio=sampling_ratio,
    )
    input_size = 256
    self.pooler = pooler

    use_gn = False
    layers = (256, 256, 256, 256)
    dilation = 1

    next_feature = input_size
    self.blocks = []
    for layer_idx, layer_features in enumerate(layers, 1):
      layer_name = "mask_fcn{}".format(layer_idx)
      module = make_conv3x3(next_feature, layer_features,
                            dilation=dilation, stride=1, use_gn=use_gn
                            )
      exec(f"self.{layer_name} = module")
      next_feature = layer_features
      self.blocks.append(layer_name)


class MaskRCNNC4Predictor:
  def __init__(self):
    num_classes = 81
    dim_reduced = 256
    num_inputs = dim_reduced
    self.conv5_mask = nn.ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
    self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)


class RoIBoxFeatureExtractor:
  def __init__(self, in_channels):
    self.fc6 = nn.Linear(12544, 1024)
    self.fc7 = nn.Linear(1024, 1024)
    self.pooler = Pooler(0, 0, 0)


class Pooler:
  def __init__(self, output_size, scales, sampling_ratio):
    self.output_size = output_size
    self.scales = scales
    self.sampling_ratio = sampling_ratio


class Predictor:
  def __init__(self, ):
    self.cls_score = nn.Linear(1024, 81)
    self.bbox_pred = nn.Linear(1024, 324)


class RoIBoxHead:
  def __init__(self, in_channels):
    self.feature_extractor = RoIBoxFeatureExtractor(in_channels)
    self.predictor = Predictor()


class Mask:
  def __init__(self):
    self.feature_extractor = MaskRCNNFPNFeatureExtractor()
    self.predictor = MaskRCNNC4Predictor()


class RoIHeads:
  def __init__(self, in_channels, num_classes):
    self.box = RoIBoxHead(in_channels)
    self.mask = Mask()


class MaskRCNN:
  def __init__(self, backbone: ResNet):
    self.backbone = ResNetFPN(backbone, out_channels=256)
    self.rpn = RPN(self.backbone.out_channels)
    self.roi_heads = RoIHeads(self.backbone.out_channels, 91)

  def load_from_pretrained(self):
    fn = Path('./') / "weights/maskrcnn.pt"
    download_file("https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth", fn)

    with open(fn, "rb") as f:
      state_dict = fake_torch_load(f.read())['model']
    loaded_keys = []
    for k, v in state_dict.items():
      if "module." in k:
        k = k.replace("module.", "")
      if "stem." in k:
        k = k.replace("stem.", "")
      if "fpn_inner" in k:
        block_index = int(re.search(r"fpn_inner(\d+)", k).group(1))
        k = re.sub(r"fpn_inner\d+", f"inner_blocks.{block_index - 1}", k)
      if "fpn_layer" in k:
        block_index = int(re.search(r"fpn_layer(\d+)", k).group(1))
        k = re.sub(r"fpn_layer\d+", f"layer_blocks.{block_index - 1}", k)
      print(k)
      loaded_keys.append(k)
      get_child(self, k).assign(v.numpy()).realize()
    return loaded_keys

  def __call__(self, x):
    # TODO: add __call__ for all children
    features = self.backbone(x)
    proposals = self.rpn(features)
    detections = self.roi_heads(features, proposals)
    return detections


if __name__ == '__main__':
  resnet = resnet = ResNet(50, num_classes=None)
  model = MaskRCNN(backbone=resnet)
  model.load_from_pretrained()
