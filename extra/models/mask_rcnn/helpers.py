import extra.models.mask_rcnn as mask_rcnn
import numpy as np
import torch

from extra.models.retinanet import nms as _box_nms
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, dtypes
from typing import Tuple

USE_NP_GATHER = getenv('FULL_TINYGRAD', default='0') == '0'

# This is very slow for large arrays, or indices
def _gather(array, indices):
  indices = indices.float().to(array.device)
  reshape_arg = [1]*array.ndim + [array.shape[-1]]
  return Tensor.where(
    indices.unsqueeze(indices.ndim).expand(*indices.shape, array.shape[-1]) == Tensor.arange(array.shape[-1]).reshape(*reshape_arg).expand(*indices.shape, array.shape[-1]),
    array, 0,
  ).sum(indices.ndim)

# TODO: replace npgather with a faster gather using tinygrad only
# NOTE: this blocks the gradient
def _np_gather(array,indices):
  if isinstance(array, Tensor): array = array.numpy()
  if isinstance(indices, Tensor): indices = indices.numpy()
  if isinstance(indices, list): indices = np.asarray(indices)
  return Tensor(array[indices.astype(int)])

def _get_strides(shape):
  prod = [1]
  for idx in range(len(shape)-1, -1, -1): prod.append(prod[-1] * shape[idx])
  # something about ints is broken with gpu, cuda
  return Tensor(prod[::-1][1:], dtype=dtypes.int32).unsqueeze(0).cpu()

def _generate_anchors(base_size, scales, aspect_ratios):
  anchor = Tensor([1, 1, base_size, base_size]) - 1
  anchors = _ratio_enum(anchor, aspect_ratios)
  anchors = Tensor.cat(
    *[_scale_enum(anchors[i, :], scales).reshape(-1, 4) for i in range(anchors.shape[0])]
  )
  return anchors

def _whctrs(anchor):
  w = anchor[2] - anchor[0] + 1
  h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
  ws = ws[:, None]
  hs = hs[:, None]
  anchors = Tensor.cat(*(
    x_ctr - 0.5 * (ws - 1),
    y_ctr - 0.5 * (hs - 1),
    x_ctr + 0.5 * (ws - 1),
    y_ctr + 0.5 * (hs - 1),
  ), dim=1)
  return anchors

def _ratio_enum(anchor, ratios):
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  size = w * h
  size_ratios = size / ratios
  ws = mask_rcnn.rint(Tensor.sqrt(size_ratios))
  hs = mask_rcnn.rint(ws * ratios)
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors

def _scale_enum(anchor, scales):
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  ws = w * scales
  hs = h * scales
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors

# for gather with indicies only on axis=0
def tensor_gather(tensor, indices):
  if not isinstance(indices, Tensor):
    indices = Tensor(indices, requires_grad=False)
  if len(tensor.shape) > 2:
    rem_shape = list(tensor.shape)[1:]
    tensor = tensor.reshape(tensor.shape[0], -1)
  else:
    rem_shape = None
  if len(tensor.shape) > 1:
    tensor = tensor.T
    repeat_arg = [1]*(tensor.ndim-1) + [tensor.shape[-2]]
    indices = indices.unsqueeze(indices.ndim).repeat(repeat_arg)
    ret = _gather(tensor, indices)
    if rem_shape:
      ret = ret.reshape([indices.shape[0]] + rem_shape)
  else:
    ret = _gather(tensor, indices)
  del indices
  return ret

# with keys as integer array for all axes
def tensor_getitem(tensor, *keys):
  # something about ints is broken with gpu, cuda
  flat_keys = Tensor.stack([key.expand((sum(keys)).shape).reshape(-1) for key in keys], dim=1).cpu().cast(dtypes.int32)
  strides = _get_strides(tensor.shape)
  idxs = (flat_keys * strides).sum(1)
  gatherer = _np_gather if USE_NP_GATHER else _gather
  return gatherer(tensor.reshape(-1), idxs).reshape(sum(keys).shape)

def cat_boxlist(bboxes):
  size = bboxes[0].size
  mode = bboxes[0].mode
  fields = set(bboxes[0].fields())
  cat_box_list = [bbox.bbox for bbox in bboxes if bbox.bbox.shape[0] > 0]

  if len(cat_box_list) > 0:
    cat_boxes = mask_rcnn.BoxList(Tensor.cat(*cat_box_list, dim=0), size, mode)
  else:
    cat_boxes = mask_rcnn.BoxList(bboxes[0].bbox, size, mode)
  for field in fields:
    cat_field_list = [bbox.get_field(field) for bbox in bboxes if bbox.get_field(field).shape[0] > 0]

    if len(cat_box_list) > 0:
      data = Tensor.cat(*cat_field_list, dim=0)
    else:
      data = bboxes[0].get_field(field)

    cat_boxes.add_field(field, data)

  return cat_boxes

def rint(tensor):
  x = (tensor*2).cast(dtypes.int32).contiguous().cast(dtypes.float32)/2
  return (x<0).where(x.floor(), x.ceil())

def nearest_interpolate(tensor, scale_factor):
  bs, c, py, px = tensor.shape
  return tensor.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, scale_factor, px, scale_factor).reshape(bs, c, py * scale_factor, px * scale_factor)

def meshgrid(x, y):
  grid_x = Tensor.cat(*[x[idx:idx+1].expand(y.shape).unsqueeze(0) for idx in range(x.shape[0])])
  grid_y = Tensor.cat(*[y.unsqueeze(0)]*x.shape[0])
  return grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)

def topk(input_, k, dim=-1, largest=True, sorted=False):
  k = min(k, input_.shape[dim]-1)
  input_ = input_.numpy()
  if largest: input_ *= -1
  ind = np.argpartition(input_, k, axis=dim)
  if largest: input_ *= -1
  ind = np.take(ind, np.arange(k), axis=dim) # k non-sorted indices
  input_ = np.take_along_axis(input_, ind, axis=dim) # k non-sorted values
  if not sorted: return Tensor(input_), ind
  if largest: input_ *= -1
  ind_part = np.argsort(input_, axis=dim)
  ind = np.take_along_axis(ind, ind_part, axis=dim)
  if largest: input_ *= -1
  val = np.take_along_axis(input_, ind_part, axis=dim)
  return Tensor(val), ind

def permute_and_flatten(layer:Tensor, N, A, C, H, W):
  layer = layer.reshape(N, -1, C, H, W)
  layer = layer.permute(0, 3, 4, 1, 2)
  layer = layer.reshape(N, -1, C)
  return layer

def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
  return _generate_anchors(stride, Tensor(list(sizes)) / stride, Tensor(list(aspect_ratios)))

def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
  if nms_thresh <= 0:
    return boxlist
  mode = boxlist.mode
  boxlist = boxlist.convert("xyxy")
  boxes = boxlist.bbox
  score = boxlist.get_field(score_field)
  keep = _box_nms(boxes.numpy(), score.numpy(), nms_thresh)
  if max_proposals > 0:
    keep = keep[:max_proposals]
  boxlist = boxlist[keep]
  return boxlist.convert(mode)

def remove_small_boxes(boxlist, min_size):
  xywh_boxes = boxlist.convert("xywh").bbox
  _, _, ws, hs = xywh_boxes.chunk(4, dim=1)
  keep = ((
          (ws >= min_size) * (hs >= min_size)
  ) > 0).reshape(-1)
  if keep.sum().numpy() == len(boxlist):
    return boxlist
  else:
    keep = keep.numpy().nonzero()[0]
  return boxlist[keep]

# NOTE: implement this in tinygrad
def nonzero(self:Tensor) -> Tensor: return Tensor(torch.from_numpy(self.numpy()).nonzero().squeeze(1).numpy())

def tilde(x: Tensor) -> Tensor:
  if x.dtype == dtypes.bool: return (1 - x).cast(dtypes.bool)
  return (x + 1) * -1  # this seems to be what the ~ operator does in pytorch for non bool

def concat_box_prediction_layers(box_cls, box_regression) -> Tuple[Tensor, ...]:
  box_cls_flattened, box_regression_flattened = [], []
  # for each feature level, permute the outputs to make them be in the
  # same format as the labels. Note that the labels are computed for
  # all feature levels concatenated, so we keep the same representation
  # for the objectness and the box_regression
  for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
    N, AxC, H, W = box_cls_per_level.shape
    Ax4 = box_regression_per_level.shape[1]
    A = Ax4 // 4
    C = AxC // A
    box_cls_per_level = mask_rcnn.permute_and_flatten(box_cls_per_level, N, A, C, H, W)
    box_cls_flattened.append(box_cls_per_level)

    box_regression_per_level = mask_rcnn.permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
    box_regression_flattened.append(box_regression_per_level)
  # concatenate on the first dimension (representing the feature levels), to
  # take into account the way the labels were generated (with all feature maps
  # being concatenated as well)
  box_cls = Tensor.cat(*box_cls_flattened, dim=1).reshape(-1, C)
  box_regression = Tensor.cat(*box_regression_flattened, dim=1).reshape(-1, 4)
  return box_cls, box_regression

def boxlist_iou(boxlist1:mask_rcnn.BoxList, boxlist2:mask_rcnn.BoxList) -> Tensor:
  assert boxlist1.size == boxlist2.size, "boxlists should have the same size"
  area1, area2 = boxlist1.area(), boxlist2.area()
  box1, box2 = boxlist1.bbox, boxlist2.bbox
  lt = Tensor.maximum(box1[:, None, :2], box2[:, :2])
  rb = Tensor.minimum(box1[:, None, 2:], box2[:, 2:])
  TO_REMOVE = 1
  wh = (rb - lt + TO_REMOVE).maximum(0)
  inter = wh[:, :, 0] * wh[:, :, 1]
  return inter / (area1[:, None] + area2 - inter)
