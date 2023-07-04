import itertools, numpy as np
from tensorboard.compat.proto.histogram_pb2 import HistogramProto
from tensorboard.compat.proto.summary_pb2 import Summary
from tinygrad.tensor import Tensor

def to_np(x):
  if isinstance(x, np.ndarray):return x
  if np.isscalar(x): return np.array([x])
  if isinstance(x, Tensor): return x.realize().detach().numpy()
  raise NotImplementedError("Got {}, but numpy array or tinygrad tensor are expected.".format(type(x)))

def scalar(name, tensor):
  return Summary(value=[Summary.Value(tag=name, simple_value=float(to_np(tensor).squeeze()))])

def histogram(name, values, bins, max_bins=None):
  return Summary(value=[Summary.Value(tag=name, histo=make_histogram(to_np(values).astype(float), bins, max_bins))])
def make_histogram(values, bins, max_bins=None):
  if not values.size: raise ValueError("The input has no element.")
  counts, limits = np.histogram(values.flatten(), bins=bins)
  if max_bins and len(counts) > max_bins:
    subsampling = len(counts) // max_bins
    subsampling_remainder = len(counts) % subsampling
    if subsampling_remainder:
      counts = np.pad(counts, [[0, subsampling - subsampling_remainder]], mode="constant")
    counts = counts.reshape(-1, subsampling).sum(axis=-1)
    limits = np.concatenate([limits[:-1:subsampling], limits[-1:]])
  cum_counts = np.cumsum(counts > 0)
  start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
  start, end = int(start), int(end) + 1
  counts = np.concatenate([[0], counts[:end]]) if start == 0 else counts[start - 1 : end]
  limits = limits[start : end + 1]
  if not counts.size or not limits.size: raise ValueError("The histogram is empty, please file a bug report.")
  return HistogramProto(min=values.min(),max=values.max(),num=len(values),sum=values.sum(),sum_squares=values.dot(values),bucket_limit=limits.tolist(),bucket=counts.tolist())

def image(tag, tensor, rescale=1, dataformats="NCHW"):
  tensor = convert_to_HWC(to_np(tensor), dataformats)
  scale_factor = 1 if (tensor if isinstance(tensor, np.ndarray) else tensor.numpy()).dtype == np.uint8 else 255
  tensor = (tensor.astype(np.float32) * scale_factor).clip(0, 255).astype(np.uint8)
  return Summary(value=[Summary.Value(tag=tag, image=make_image(tensor, rescale))])
def make_image(tensor, rescale=1):
  height, width, channel = tensor.shape
  from PIL import Image
  import io
  with io.BytesIO() as output:
    Image.fromarray(tensor).resize(
      (int(width * rescale), int(height * rescale)),
      getattr(Image.Resampling, 'LANCZOS', Image.ANTIALIAS)
    ).save(output, format="PNG")
    image_string = output.getvalue()
  return Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)
def make_grid(I, n_col=8):
  assert isinstance(I, np.ndarray), "plugin error, should pass numpy array here"
  if I.shape[1] == 1: I = np.repeat(I, 3, axis=1)
  assert I.ndim == 4 and I.shape[1] == 3
  n_img, _, H, W = I.shape
  n_col = min(n_img, n_col)
  n_row = int(np.ceil(float(n_img) / n_col))
  canvas, positions = np.zeros((3, H * n_row, W * n_col), dtype=I.dtype), list(itertools.product(range(n_row), range(n_col)))
  for i, (y, x) in enumerate(positions[:n_img]): canvas[:, y * H : (y + 1) * H, x * W : (x + 1) * W] = I[i]
  return canvas
def convert_to_HWC(tensor, input_format):  # tensor: numpy array
  input_format = input_format.upper()
  assert len(set(input_format)) == len(input_format), f"Duplicated dimension shorthand in input_format: {input_format}"
  assert len(tensor.shape) == len(input_format), f"Size of input tensor and input_format are different. tensor shape: {tensor.shape}, input_format: {input_format}"
  format_len_to_func = {
    4: lambda t, f: make_grid(t.transpose([f.find(c) for c in "NCHW"])).transpose(1, 0),
    3: lambda t, f: np.concatenate([t] * 3, 2) if (t := t.transpose([f.find(c) for c in "HWC"])).shape[2] == 1 else t,
    2: lambda t, f: np.stack([t] * 3, 2) if (t := t.transpose([f.find(c) for c in "HW"])).shape[2] == 1 else t
  }
  return format_len_to_func[len(input_format)](tensor, input_format)
