from tinygrad.tensor import Tensor

# numpy is still used for easy indexing

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class BoxList:
  def __init__(self, bbox, image_size, mode="xyxy"):
    if not isinstance(bbox, Tensor):
      bbox = Tensor(bbox, requires_grad=True)
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
        xmin + (w - TO_REMOVE).clip(0, w - TO_REMOVE), #TO-DO: clip make min, max optional
        ymin + (h - TO_REMOVE).clip(0, h - TO_REMOVE),
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
    # print(self.bbox)
    TO_REMOVE = 1
    # TODO find solutions for 
    # 'Tensor' object does not support item assignment
    # One day we will make Tensors to support all these operations
    bb0 = self.bbox[:,0].clip(min_=0, max_=self.size[0] - TO_REMOVE)
    bb1 = self.bbox[:,1].clip(min_=0, max_=self.size[1] - TO_REMOVE)
    bb2 = self.bbox[:,2].clip(min_=0, max_=self.size[0] - TO_REMOVE)
    bb3 = self.bbox[:,3].clip(min_=0, max_=self.size[1] - TO_REMOVE)
    if remove_empty:
      # keep = (bb3 > bb1) * (bb2 > bb0)
      keep = ((bb3 > bb1) * (bb2 > bb0)).numpy().astype(dtype=bool)
      return self[keep]
    return self

  def __getitem__(self, item):
    if isinstance(item, list):
      if len(item) == 0:
        return []
      if sum(item) == len(item) and isinstance(item[0], bool):
        return self
    bbox = BoxList(Tensor(self.bbox.numpy()[item], requires_grad=True), self.size, self.mode)
    # bbox = BoxList(tensor_gather(self.bbox, item), self.size, self.mode
    # bbox = BoxList(self.bbox[item], self.size, self.mode)
    for k, v in self.extra_fields.items():
      if not isinstance(v, Tensor):
        bbox.add_field(k, v[item])
      else:
        bbox.add_field(k, Tensor(v.numpy()[item]))
    return bbox

  def __len__(self):
    return self.bbox.shape[0]