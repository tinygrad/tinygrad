from tinygrad.tensor import Tensor
import pycocotools.mask as mask_utils

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class Mask(object):
  """
  This class is unfinished and not meant for use yet
  It is supposed to contain the mask for an object as
  a 2d tensor
  """
  def __init__(self, masks, size, mode):
    self.masks = masks
    self.size = size
    self.mode = mode

  def transpose(self, method):
    if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
      raise NotImplementedError("Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented")

    width, height = self.size
    if method == FLIP_LEFT_RIGHT:
      dim = width
      idx = 2
    elif method == FLIP_TOP_BOTTOM:
      dim = height
      idx = 1

    flip_idx = list(range(dim)[::-1])
    flipped_masks = self.masks.index_select(dim, flip_idx)
    return Mask(flipped_masks, self.size, self.mode)

  def crop(self, box):
    w, h = box[2] - box[0], box[3] - box[1]

    cropped_masks = self.masks[:, box[1] : box[3], box[0] : box[2]]
    return Mask(cropped_masks, size=(w, h), mode=self.mode)

  def resize(self, size, *args, **kwargs):
    pass

class Polygons(object):
  """
  This class holds a set of polygons that represents a single instance
  of an object mask. The object can be represented as a set of
  polygons
  """
  def __init__(self, polygons, size, mode):
    # assert isinstance(polygons, list), '{}'.format(polygons)
    if isinstance(polygons, list):
      polygons = [Tensor(p) for p in polygons]
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
      p = poly.numpy().copy()
      TO_REMOVE = 1
      p[idx::2] = dim - poly.numpy()[idx::2] - TO_REMOVE
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
      p = poly.numpy().copy()
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
      mask = Tensor(mask)
      # TODO add squeeze?
      return mask

  def __repr__(self):
    s = self.__class__.__name__ + "("
    s += "num_polygons={}, ".format(len(self.polygons))
    s += "image_width={}, ".format(self.size[0])
    s += "image_height={}, ".format(self.size[1])
    s += "mode={})".format(self.mode)
    return s

class SegmentationMask(object):
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
      raise NotImplementedError("Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented")

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
      if isinstance(item, Tensor) and \
        (item.dtype == dtypes.uint8 or item.dtype == dtypes.bool):
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