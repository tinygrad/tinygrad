from tinygrad.tensor import Tensor

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