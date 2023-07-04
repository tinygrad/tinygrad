import itertools
import numpy as np
from typing import Dict, List, Any

from tensorboard.compat.proto import struct_pb2
from tensorboard.compat.proto.summary_pb2 import HistogramProto
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.summary_pb2 import SummaryMetadata

from tinygrad.tensor import Tensor


def to_np(x):
  if isinstance(x, np.ndarray):return x
  if np.isscalar(x): return np.array([x])
  if isinstance(x, Tensor): return x.realize().detach().numpy()
  raise NotImplementedError("Got {}, but numpy array or tinygrad tensor are expected.".format(type(x)))

def validate_dict(d, name):
  if not isinstance(d, dict): raise TypeError(f"parameter: {name} should be a dictionary, nothing logged.")

def hparams(hparam_dict=None, metric_dict=None, hparam_domain_discrete: Dict[str, List[Any]] = None):
  from tensorboard.plugins.hparams.api_pb2 import (Experiment, HParamInfo, MetricInfo, MetricName, Status, DataType)
  from tensorboard.plugins.hparams.metadata import (PLUGIN_NAME, PLUGIN_DATA_VERSION, EXPERIMENT_TAG, SESSION_START_INFO_TAG, SESSION_END_INFO_TAG)
  from tensorboard.plugins.hparams.plugin_data_pb2 import (HParamsPluginData,SessionEndInfo, SessionStartInfo)
  validate_dict(hparam_dict, "hparam_dict")
  validate_dict(metric_dict, "metric_dict")
  hparam_domain_discrete = hparam_domain_discrete or {}
  validate_dict(hparam_domain_discrete, "hparam_domain_discrete")
  for k, v in hparam_domain_discrete.items():
    if k not in hparam_dict or not isinstance(v, list) or not all(isinstance(d, type(hparam_dict[k])) for d in v):
      raise TypeError(f"parameter: hparam_domain_discrete[{k}] should be a list of same type as hparam_dict[{k}].")
  hps, ssi = [], SessionStartInfo()
  for k, v in hparam_dict.items():
    if v is None or isinstance(v, Tensor):
      continue
    type_map = {int: "DATA_TYPE_FLOAT64", float: "DATA_TYPE_FLOAT64", str: "DATA_TYPE_STRING", bool: "DATA_TYPE_BOOL"}
    if isinstance(v, (int, float)): ssi.hparams[k].number_value = v
    elif isinstance(v, str): ssi.hparams[k].string_value = v
    elif isinstance(v, bool): ssi.hparams[k].bool_value = v
    if k in hparam_domain_discrete:
      domain_discrete = []
      for d in hparam_domain_discrete[k]:
        if isinstance(d, (int, float)):
          domain_discrete.append(struct_pb2.Value(number_value=d))
        elif isinstance(d, str):
          domain_discrete.append(struct_pb2.Value(string_value=d))
        else:
          domain_discrete.append(struct_pb2.Value(bool_value=d))
      return struct_pb2.ListValue(values=domain_discrete)
    else:
      domain_discrete = None
      hps.append(HParamInfo(name=k, type=DataType.Value(type_map[type(v)]), domain_discrete=domain_discrete))
  def get_summary(content, tag):
    smd = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name=PLUGIN_NAME, content=content.SerializeToString()))
    return Summary(value=[Summary.Value(tag=tag, metadata=smd)])
  ssi = get_summary(HParamsPluginData(session_start_info=ssi, version=PLUGIN_DATA_VERSION),SESSION_START_INFO_TAG)
  mts = [MetricInfo(name=MetricName(tag=k)) for k in metric_dict.keys()]
  exp = get_summary(HParamsPluginData(experiment=Experiment(hparam_infos=hps, metric_infos=mts),version=PLUGIN_DATA_VERSION), EXPERIMENT_TAG)
  sei = get_summary(HParamsPluginData(session_end_info=SessionEndInfo(status=Status.Value("STATUS_SUCCESS")),version=PLUGIN_DATA_VERSION), SESSION_END_INFO_TAG)
  return exp, ssi, sei

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
    4: lambda t, f: make_grid(t.transpose([f.find(c) for c in "NCHW"])).transpose((1, 2, 0)),
    3: lambda t, f: t if (t := t.transpose([f.find(c) for c in "HWC"])).shape[2] != 1 else np.concatenate([t, t, t], 2),
    2: lambda t, f: np.stack([t.transpose([f.find(c) for c in "HW"])]*3, 2)
  }
  return format_len_to_func[len(input_format)](tensor, input_format)
