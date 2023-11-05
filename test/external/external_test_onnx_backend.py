import unittest
from onnx.backend.base import Backend, BackendRep
import onnx.backend.test
import numpy as np
from tinygrad.tensor import Tensor
from typing import Any, Tuple
from tinygrad.helpers import getenv, CI

# pip3 install tabulate
pytest_plugins = 'onnx.backend.test.report',

from extra.onnx import get_run_onnx

class TinygradModel(BackendRep):
  def __init__(self, run_onnx, input_names):
    super().__init__()
    self.fxn = run_onnx
    self.input_names = input_names

  def run(self, inputs: Any, **kwargs: Any) -> Tuple[Any, ...]:
    real_inputs = {k:v for k,v in zip(self.input_names, inputs)}
    ret = self.fxn(real_inputs, debug=True)
    return tuple(x.numpy() if isinstance(x, Tensor) else [i.numpy() for i in x] if isinstance(x, list) else np.array(x) for x in ret.values())

class TinygradBackend(Backend):
  @classmethod
  def prepare(cls, model, device):
    input_all = [x.name for x in model.graph.input]
    input_initializer = [x.name for x in model.graph.initializer]
    net_feed_input = [x for x in input_all if x not in input_initializer]
    print("prepare", cls, device, net_feed_input)
    run_onnx = get_run_onnx(model)
    return TinygradModel(run_onnx, net_feed_input)

  @classmethod
  def supports_device(cls, device: str) -> bool:
    return device == "CPU"

backend_test = onnx.backend.test.BackendTest(TinygradBackend, __name__)

# no support for reduce with multiply (needs llop)
backend_test.exclude('test_reduce_prod_*')

# TODO figure out why it's returning wrong values, geohotstan's uneducated guess is it's due to imprecision from float64 (double) -> float32
# see Type Constraints: https://onnx.ai/onnx/operators/onnx_aionnxpreviewtraining_Adam.html#type-constraints
backend_test.exclude('test_adam_multiple_cpu')
backend_test.exclude('test_nesterov_momentum_cpu')

# we only support float32
backend_test.exclude('uint8')
backend_test.exclude('uint16')
backend_test.exclude('uint32')
backend_test.exclude('uint64')
backend_test.exclude('int8')
backend_test.exclude('int16')
backend_test.exclude('float64')
backend_test.exclude('string')

backend_test.exclude('test_pow_types_int*')
backend_test.exclude('test_cast_*')
backend_test.exclude('test_castlike_*')
backend_test.exclude('test_convinteger_*')
backend_test.exclude('test_matmulinteger_*')

backend_test.exclude('test_reduce_log_sum_exp*') # dependent on actual float64 implementation for backends
backend_test.exclude('test_operator_add*') # dependent on float64 math. Without it values default to 0 or inf

# we don't support indexes
# backend_test.exclude('test_argmax_*') # Needs more work: select_last_index
# backend_test.exclude('test_argmin_*') # Needs more work: select_last_index
backend_test.exclude('test_nonzero_*')

# no support for mod
backend_test.exclude('test_mod_*')

# no boolean ops (2d, 3d, 4d)
backend_test.exclude('test_bitshift_*')

# no scatternd gathernd
backend_test.exclude('test_gathernd_*')
backend_test.exclude('test_scatternd_*')

# no quantize
backend_test.exclude('test_dynamicquantizelinear_*')
backend_test.exclude('test_qlinearmatmul_*')
backend_test.exclude('test_qlinearconv_*')
backend_test.exclude('test_quantizelinear_*')

# no rnn
backend_test.exclude('test_gru_*')
backend_test.exclude('test_rnn_*')
backend_test.exclude('test_lstm_*')
backend_test.exclude('test_simple_rnn_*')

# no control flow
backend_test.exclude('test_if_*')
backend_test.exclude('test_loop*')
backend_test.exclude('test_range_float_type_positive_delta_expanded_cpu') # requires loop

# unsupported (strange) ops
backend_test.exclude('test_bitwise_*')
backend_test.exclude('test_blackmanwindow_*')
backend_test.exclude('test_bernoulli_*')
backend_test.exclude('test_cumsum_*')
backend_test.exclude('test_det_*')

backend_test.exclude('test_tril_zero_cpu') # TODO: zero array support
backend_test.exclude('test_triu_zero_cpu') # TODO: zero array support

backend_test.exclude('test_col2im_*')
backend_test.exclude('test_hammingwindow_*')
backend_test.exclude('test_hannwindow_*')
backend_test.exclude('test_hardmax_*')
backend_test.exclude('test_gridsample_*')
backend_test.exclude('test_dft_*')
backend_test.exclude('test_einsum_*')
backend_test.exclude('test_strnorm_*')
backend_test.exclude('test_unique_*')
backend_test.exclude('test_sequence_*')
backend_test.exclude('test_nonmaxsuppression_*')
backend_test.exclude('test_reversesequence_*')
backend_test.exclude('test_roialign_*')
backend_test.exclude('test_top_k_*')
backend_test.exclude('test_tfidfvectorizer_*')
backend_test.exclude('test_stft_*')
backend_test.exclude('test_melweightmatrix_*')

# more strange ops
backend_test.exclude('test_basic_deform_conv_*')
backend_test.exclude('test_deform_conv_*')
backend_test.exclude('test_lppool_*')
backend_test.exclude('test_depthtospace_*')
backend_test.exclude('test_spacetodepth_*')
backend_test.exclude('test_scan*')
backend_test.exclude('test_split_to_sequence_*')
backend_test.exclude('test_resize_downsample_scales_cubic_*') # unsure how to implement cubic
backend_test.exclude('test_resize_downsample_sizes_cubic_*') # unsure how to implement cubic
backend_test.exclude('test_resize_upsample_scales_cubic_*') # unsure how to implement cubic
backend_test.exclude('test_resize_upsample_sizes_cubic_*') # unsure how to implement cubic

# rest of the failing tests
backend_test.exclude('test_averagepool_2d_dilations_cpu') # dilations != 1 not supported for avgpool
backend_test.exclude('test_convtranspose_autopad_same_cpu') # TODO geohotstan has no idea how this is done, autopad requires output_shape but output_shape requires pads from autopad
backend_test.exclude('test_optional_has_element_empty_optional_input_cpu') # Attempts to create Tensor from None
backend_test.exclude('test_range_int32_type_negative_delta_expanded_cpu') # AttributeProto.GRAPH not implemented
backend_test.exclude('test_reshape_allowzero_reordered_cpu') # reshaping to 0 shape
backend_test.exclude('test_resize_downsample_scales_linear_antialias_cpu') # antialias not implemented
backend_test.exclude('test_resize_downsample_sizes_linear_antialias_cpu') # antialias not implemented
backend_test.exclude('test_resize_tf_crop_and_resize_cpu') # unsure about fill value after clip
backend_test.exclude('test_operator_addconstant_cpu') # bad data type

# issue 1556 https://github.com/tinygrad/tinygrad/issues/1556
backend_test.exclude('test_isinf_cpu')
backend_test.exclude('test_isinf_negative_cpu')
backend_test.exclude('test_isinf_positive_cpu')
backend_test.exclude('test_isnan_cpu')

# issue 1791 fast math messes with these https://github.com/tinygrad/tinygrad/issues/1791
backend_test.exclude('test_resize_upsample_sizes_nearest_axes_2_3_cpu')
backend_test.exclude('test_resize_upsample_sizes_nearest_axes_3_2_cpu')
backend_test.exclude('test_resize_upsample_sizes_nearest_cpu')

# issue 2067 potentially also a fastmath issue https://github.com/tinygrad/tinygrad/issues/2067
if getenv('METAL'):
  backend_test.exclude('test_maxpool_2d_pads_cpu')
  backend_test.exclude('test_maxpool_2d_same_lower_cpu')

# Don't know how to treat special TensorProto like TensorProto.FLOAT8E4M3FN
if getenv("CPU") or getenv("TORCH"):
  backend_test.exclude('test_dequantizelinear_axis_cpu')
  backend_test.exclude('test_dequantizelinear_cpu')

# compiled backends cannot reshape to and from 0
if getenv('LLVM') or getenv('GPU') or getenv('CLANG') or getenv('METAL') or getenv('CUDA'):
  backend_test.exclude('test_slice_start_out_of_bounds_cpu')
  backend_test.exclude('test_constantofshape_int_shape_zero_cpu')

if getenv('GPU') or getenv('METAL'):
  backend_test.exclude('test_mish_cpu') # weird inaccuracy
  backend_test.exclude('test_mish_expanded_cpu') # weird inaccuracy
  backend_test.exclude('test_eyelike_with_dtype_cpu') # backend does not support dtype: Double

# Segfaults in CI
if (getenv('LLVM') or getenv('CUDA')) and CI:
  backend_test.exclude('test_max_float16_cpu')
  backend_test.exclude('test_min_float16_cpu')

# disable model tests for now since they are slow
if not getenv("MODELTESTS"):
  for x in backend_test.test_suite:
    if 'OnnxBackendRealModelTest' in str(type(x)):
      backend_test.exclude(str(x).split(" ")[0])
else:
  # model tests all pass!
  backend_test.include('test_resnet50')
  backend_test.include('test_inception_v1')
  backend_test.include('test_inception_v2')
  backend_test.include('test_densenet121')
  backend_test.include('test_shufflenet')
  backend_test.include('test_squeezenet')
  backend_test.include('test_bvlc_alexnet')
  backend_test.include('test_zfnet512')
  backend_test.include('test_vgg19')

globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
  unittest.main()
