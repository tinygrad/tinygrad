import torch
import pycocotools.mask as mask_utils

import extra.models.mask_rcnn as mask_rcnn
from tinygrad.tensor import Tensor

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class BoxList:
  def __init__(self, bbox, image_size, mode="xyxy"):
    if not isinstance(bbox, Tensor):
      bbox = Tensor(bbox)
    if bbox.ndim != 2:
      raise ValueError(
        "bbox should have 2 dimensions, got {}".format(bbox.ndim)
      )
    if bbox.shape[-1] != 4:
      raise ValueError(
        "last dimenion of bbox should have a "
        "size of 4, got {}".format(bbox.shape[-1])
      )
    if mode not in ("xyxy", "xywh"):
      raise ValueError("mode should be 'xyxy' or 'xywh'")

    self.bbox = bbox
    self.size = image_size  # (image_width, image_height)
    self.mode = mode
    self.extra_fields = {}

  def __repr__(self):
    s = self.__class__.__name__ + "("
    s += "num_boxes={}, ".format(len(self))
    s += "image_width={}, ".format(self.size[0])
    s += "image_height={}, ".format(self.size[1])
    s += "mode={})".format(self.mode)
    return s

  def area(self):
    box = self.bbox
    if self.mode == "xyxy":
      TO_REMOVE = 1
      area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
    elif self.mode == "xywh":
      area = box[:, 2] * box[:, 3]
    return area

  def add_field(self, field, field_data):
    self.extra_fields[field] = field_data

  def get_field(self, field):
    return self.extra_fields[field]

  def has_field(self, field):
    return field in self.extra_fields

  def fields(self):
    return list(self.extra_fields.keys())

  def _copy_extra_fields(self, bbox):
    for k, v in bbox.extra_fields.items():
      self.extra_fields[k] = v

  def convert(self, mode):
    if mode == self.mode:
      return self
    xmin, ymin, xmax, ymax = self._split_into_xyxy()
    if mode == "xyxy":
      bbox = Tensor.cat(*(xmin, ymin, xmax, ymax), dim=-1)
      bbox = BoxList(bbox, self.size, mode=mode)
    else:
      TO_REMOVE = 1
      bbox = Tensor.cat(
        *(xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
      )
      bbox = BoxList(bbox, self.size, mode=mode)
    bbox._copy_extra_fields(self)
    return bbox

  def _split_into_xyxy(self):
    if self.mode == "xyxy":
      xmin, ymin, xmax, ymax = self.bbox.chunk(4, dim=-1)
      return xmin, ymin, xmax, ymax
    if self.mode == "xywh":
      TO_REMOVE = 1
      xmin, ymin, w, h = self.bbox.chunk(4, dim=-1)
      return (
        xmin,
        ymin,
        xmin + (w - TO_REMOVE).maximum(0),
        ymin + (h - TO_REMOVE).maximum(0),
      )

  def resize(self, size, *args, **kwargs):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
    if ratios[0] == ratios[1]:
      ratio = ratios[0]
      scaled_box = self.bbox * ratio
      bbox = BoxList(scaled_box, size, mode=self.mode)
      for k, v in self.extra_fields.items():
        if not isinstance(v, Tensor):
          v = v.resize(size, *args, **kwargs)
        bbox.add_field(k, v)
      return bbox

    ratio_width, ratio_height = ratios
    xmin, ymin, xmax, ymax = self._split_into_xyxy()
    scaled_xmin = xmin * ratio_width
    scaled_xmax = xmax * ratio_width
    scaled_ymin = ymin * ratio_height
    scaled_ymax = ymax * ratio_height
    scaled_box = Tensor.cat(
      *(scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
    )
    bbox = BoxList(scaled_box, size, mode="xyxy")
    for k, v in self.extra_fields.items():
      if not isinstance(v, Tensor):
        v = v.resize(size, *args, **kwargs)
      bbox.add_field(k, v)

    return bbox.convert(self.mode)

  def transpose(self, method):
    image_width, image_height = self.size
    xmin, ymin, xmax, ymax = self._split_into_xyxy()
    if method == FLIP_LEFT_RIGHT:
      TO_REMOVE = 1
      transposed_xmin = image_width - xmax - TO_REMOVE
      transposed_xmax = image_width - xmin - TO_REMOVE
      transposed_ymin = ymin
      transposed_ymax = ymax
    elif method == FLIP_TOP_BOTTOM:
      transposed_xmin = xmin
      transposed_xmax = xmax
      transposed_ymin = image_height - ymax
      transposed_ymax = image_height - ymin

    transposed_boxes = Tensor.cat(
      *(transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
    )
    bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
    for k, v in self.extra_fields.items():
      if not isinstance(v, Tensor):
        v = v.transpose(method)
      bbox.add_field(k, v)
    return bbox.convert(self.mode)

  def clip_to_image(self, remove_empty=True):
    TO_REMOVE = 1
    bb1 = self.bbox.clip(min_=0, max_=self.size[0] - TO_REMOVE)[:, 0]
    bb2 = self.bbox.clip(min_=0, max_=self.size[1] - TO_REMOVE)[:, 1]
    bb3 = self.bbox.clip(min_=0, max_=self.size[0] - TO_REMOVE)[:, 2]
    bb4 = self.bbox.clip(min_=0, max_=self.size[1] - TO_REMOVE)[:, 3]
    self.bbox = Tensor.stack((bb1, bb2, bb3, bb4), dim=1)
    # TODO: avoid using np
    if remove_empty:
      box = self.bbox.numpy()
      keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
      return self[keep.tolist()]
    return self

  def __getitem__(self, item):
    if isinstance(item, list):
      if len(item) == 0:
        return []
      if sum(item) == len(item) and isinstance(item[0], bool):
        return self
    bbox = BoxList(mask_rcnn.tensor_gather(self.bbox, item), self.size, self.mode)
    try:
      for k, v in self.extra_fields.items():
        bbox.add_field(k, mask_rcnn.tensor_gather(v, item))
    except:
      return bbox
    return bbox

  def __len__(self):
    return self.bbox.shape[0]
  
  def copy_with_fields(self, fields, skip_missing=False):
    bbox = BoxList(self.bbox, self.size, self.mode)
    if not isinstance(fields, (list, tuple)):
      fields = [fields]
    for field in fields:
      if self.has_field(field):
        bbox.add_field(field, self.get_field(field))
      elif not skip_missing:
        raise KeyError("Field '{}' not found in {}".format(field, self))
    return bbox
  
  # TODO: migrate to tinygrad
  def crop(self, box):
    """
    Cropss a rectangular region from this bounding box. The box is a
    4-tuple defining the left, upper, right, and lower pixel
    coordinate.
    """
    xmin, ymin, xmax, ymax = self._split_into_xyxy()
    w, h = box[2] - box[0], box[3] - box[1]
    cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
    cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
    cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
    cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

    # TODO should I filter empty boxes here?
    if False:
        is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

    cropped_box = torch.cat(
        (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
    )
    bbox = BoxList(cropped_box, (w, h), mode="xyxy")
    # bbox._copy_extra_fields(self)
    for k, v in self.extra_fields.items():
        if not isinstance(v, torch.Tensor):
            v = v.crop(box)
        bbox.add_field(k, v)
    return bbox.convert(self.mode)

class Polygons:
  """
  This class holds a set of polygons that represents a single instance
  of an object mask. The object can be represented as a set of
  polygons
  """

  def __init__(self, polygons, size, mode):
    # assert isinstance(polygons, list), '{}'.format(polygons)
    if isinstance(polygons, list):
      polygons = [torch.as_tensor(p, dtype=torch.float32) for p in polygons]
    elif isinstance(polygons, Polygons):
      polygons = polygons.polygons

    self.polygons = polygons
    self.size = size
    self.mode = mode

  def transpose(self, method):
    if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
      raise NotImplementedError(
          "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
      )

    flipped_polygons = []
    width, height = self.size
    if method == FLIP_LEFT_RIGHT:
      dim = width
      idx = 0
    elif method == FLIP_TOP_BOTTOM:
      dim = height
      idx = 1

    for poly in self.polygons:
      p = poly.clone()
      TO_REMOVE = 1
      p[idx::2] = dim - poly[idx::2] - TO_REMOVE
      flipped_polygons.append(p)

    return Polygons(flipped_polygons, size=self.size, mode=self.mode)

  def crop(self, box):
    w, h = box[2] - box[0], box[3] - box[1]

    # TODO chck if necessary
    w = max(w, 1)
    h = max(h, 1)

    cropped_polygons = []
    for poly in self.polygons:
      p = poly.clone()
      p[0::2] = p[0::2] - box[0]  # .clamp(min=0, max=w)
      p[1::2] = p[1::2] - box[1]  # .clamp(min=0, max=h)
      cropped_polygons.append(p)

    return Polygons(cropped_polygons, size=(w, h), mode=self.mode)

  def resize(self, size, *args, **kwargs):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
    if ratios[0] == ratios[1]:
        ratio = ratios[0]
        scaled_polys = [p * ratio for p in self.polygons]
        return Polygons(scaled_polys, size, mode=self.mode)

    ratio_w, ratio_h = ratios
    scaled_polygons = []
    for poly in self.polygons:
        p = poly.clone()
        p[0::2] *= ratio_w
        p[1::2] *= ratio_h
        scaled_polygons.append(p)

    return Polygons(scaled_polygons, size=size, mode=self.mode)

  def convert(self, mode):
    width, height = self.size
    if mode == "mask":
      rles = mask_utils.frPyObjects(
          [p.numpy() for p in self.polygons], height, width
      )
      rle = mask_utils.merge(rles)
      mask = mask_utils.decode(rle)
      mask = torch.from_numpy(mask)
      # TODO add squeeze?
      return mask

  def __repr__(self):
    s = self.__class__.__name__ + "("
    s += "num_polygons={}, ".format(len(self.polygons))
    s += "image_width={}, ".format(self.size[0])
    s += "image_height={}, ".format(self.size[1])
    s += "mode={})".format(self.mode)
    return s

class SegmentationMask:
  """
  This class stores the segmentations for all objects in the image
  """

  def __init__(self, polygons, size, mode=None):
    """
    Arguments:
      polygons: a list of list of lists of numbers. The first
          level of the list correspond to individual instances,
          the second level to all the polygons that compose the
          object, and the third level to the polygon coordinates.
    """
    assert isinstance(polygons, list)

    self.polygons = [Polygons(p, size, mode) for p in polygons]
    self.size = size
    self.mode = mode

  def transpose(self, method):
    if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
        raise NotImplementedError(
            "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
        )

    flipped = []
    for polygon in self.polygons:
        flipped.append(polygon.transpose(method))
    return SegmentationMask(flipped, size=self.size, mode=self.mode)

  def crop(self, box):
    w, h = box[2] - box[0], box[3] - box[1]
    cropped = []
    for polygon in self.polygons:
      cropped.append(polygon.crop(box))
    return SegmentationMask(cropped, size=(w, h), mode=self.mode)

  def resize(self, size, *args, **kwargs):
    scaled = []
    for polygon in self.polygons:
      scaled.append(polygon.resize(size, *args, **kwargs))
    return SegmentationMask(scaled, size=size, mode=self.mode)

  def to(self, *args, **kwargs):
    return self

  def __getitem__(self, item):
    if isinstance(item, (int, slice)):
      selected_polygons = [self.polygons[item]]
    else:
      # advanced indexing on a single dimension
      selected_polygons = []
      if isinstance(item, torch.Tensor) and \
              (item.dtype == torch.uint8 or item.dtype == torch.bool):
        item = item.nonzero()
        item = item.squeeze(1) if item.numel() > 0 else item
        item = item.tolist()
      for i in item:
        selected_polygons.append(self.polygons[i])
    return SegmentationMask(selected_polygons, size=self.size, mode=self.mode)

  def __iter__(self):
    return iter(self.polygons)

  def __repr__(self):
    s = self.__class__.__name__ + "("
    s += "num_instances={}, ".format(len(self.polygons))
    s += "image_width={}, ".format(self.size[0])
    s += "image_height={})".format(self.size[1])
    return s
