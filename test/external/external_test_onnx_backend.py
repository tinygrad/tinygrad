import unittest
from typing import Any, Tuple
from onnx.backend.base import Backend, BackendRep
import onnx.backend.test
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, CI
from tinygrad.device import Device, Compiled

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

# about different dtypes
backend_test.exclude('int8')  #  OverflowError: cannot convert float infinity to integer

if Device.DEFAULT in ["TORCH"]:
  backend_test.exclude('uint16')
  backend_test.exclude('uint32')
  backend_test.exclude('uint64')
if Device.DEFAULT in ["METAL"]:
  backend_test.exclude('float64')

# string Tensors
backend_test.exclude('string')
backend_test.exclude('test_regex_*')
backend_test.exclude('test_cast_FLOAT_to_STRING_cpu')
backend_test.exclude('test_cast_STRING_to_FLOAT_cpu')
backend_test.exclude('FLOAT_to_STRING_cpu')
backend_test.exclude('FLOAT_to_STRING_expanded_cpu')
backend_test.exclude('STRING_to_FLOAT_cpu')
backend_test.exclude('STRING_to_FLOAT_expanded_cpu')

# TODO support for float8
backend_test.exclude('test_cast_FLOAT16_to_FLOAT8E4M3FNUZ_cpu')
backend_test.exclude('test_cast_FLOAT16_to_FLOAT8E4M3FN_cpu')
backend_test.exclude('test_cast_FLOAT16_to_FLOAT8E5M2FNUZ_cpu')
backend_test.exclude('test_cast_FLOAT16_to_FLOAT8E5M2_cpu')
backend_test.exclude('test_cast_FLOAT_to_FLOAT8E4M3FNUZ_cpu')
backend_test.exclude('test_cast_FLOAT_to_FLOAT8E4M3FN_cpu')
backend_test.exclude('test_cast_FLOAT_to_FLOAT8E5M2FNUZ_cpu')
backend_test.exclude('test_cast_FLOAT_to_FLOAT8E5M2_cpu')
backend_test.exclude('test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_cpu')
backend_test.exclude('test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN_cpu')
backend_test.exclude('test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_cpu')
backend_test.exclude('test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2_cpu')
backend_test.exclude('test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_cpu')
backend_test.exclude('test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN_cpu')
backend_test.exclude('test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_cpu')
backend_test.exclude('test_cast_no_saturate_FLOAT_to_FLOAT8E5M2_cpu')
backend_test.exclude('test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_cpu')
backend_test.exclude('test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded_cpu')
backend_test.exclude('test_castlike_FLOAT_to_FLOAT8E4M3FN_cpu')
backend_test.exclude('test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded_cpu')
backend_test.exclude('test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_cpu')
backend_test.exclude('test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded_cpu')
backend_test.exclude('test_castlike_FLOAT_to_FLOAT8E5M2_cpu')
backend_test.exclude('test_castlike_FLOAT_to_FLOAT8E5M2_expanded_cpu')

# TODO support for bfloat16 casting
backend_test.exclude('test_cast_FLOAT_to_BFLOAT16_cpu')
backend_test.exclude('test_cast_BFLOAT16_to_FLOAT_cpu')
backend_test.exclude('test_castlike_FLOAT_to_BFLOAT16_cpu')
backend_test.exclude('test_castlike_FLOAT_to_BFLOAT16_expanded_cpu')
backend_test.exclude('test_castlike_BFLOAT16_to_FLOAT_cpu')
backend_test.exclude('test_castlike_BFLOAT16_to_FLOAT_expanded_cpu')

backend_test.exclude('test_pow_types_int*')
backend_test.exclude('test_convinteger_*')
backend_test.exclude('test_matmulinteger_*')

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
# control flow uses AttributeProto.GRAPH
backend_test.exclude('test_if_*')
backend_test.exclude('test_loop*')
backend_test.exclude('test_range_float_type_positive_delta_expanded_cpu') # requires loop
backend_test.exclude('test_affine_grid_2d_align_corners_expanded_cpu')
backend_test.exclude('test_affine_grid_2d_expanded_cpu')
backend_test.exclude('test_affine_grid_3d_align_corners_expanded_cpu')
backend_test.exclude('test_affine_grid_3d_expanded_cpu')
backend_test.exclude('test_range_int32_type_negative_delta_expanded_cpu')

# unsupported (strange) ops
backend_test.exclude('test_bitwise_*')
backend_test.exclude('test_blackmanwindow_*')
backend_test.exclude('test_bernoulli_*')
backend_test.exclude('test_cumsum_*')
backend_test.exclude('test_det_*')

backend_test.exclude('test_tril_zero_cpu') # TODO: zero array tril support
backend_test.exclude('test_triu_zero_cpu') # TODO: zero array triu support

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
backend_test.exclude('test_resize_downsample_scales_linear_antialias_cpu') # antialias not implemented
backend_test.exclude('test_resize_downsample_sizes_linear_antialias_cpu') # antialias not implemented
backend_test.exclude('test_resize_tf_crop_and_resize_cpu') # unsure about fill value after clip
backend_test.exclude('test_ai_onnx_ml_label_encoder_tensor_value_only_mapping_cpu') # bad data type string
backend_test.exclude('test_ai_onnx_ml_label_encoder_tensor_mapping_cpu') # bad data type string

# issue 1556 https://github.com/tinygrad/tinygrad/issues/1556
backend_test.exclude('test_isinf_cpu')
backend_test.exclude('test_isinf_negative_cpu')
backend_test.exclude('test_isinf_positive_cpu')
backend_test.exclude('test_isinf_float16_cpu')
backend_test.exclude('test_isnan_float16_cpu')
backend_test.exclude('test_isnan_cpu')

# issue 1791 fast math messes with these https://github.com/tinygrad/tinygrad/issues/1791
backend_test.exclude('test_resize_upsample_sizes_nearest_axes_2_3_cpu')
backend_test.exclude('test_resize_upsample_sizes_nearest_axes_3_2_cpu')
backend_test.exclude('test_resize_upsample_sizes_nearest_cpu')

# issue 2067 potentially also a fastmath issue https://github.com/tinygrad/tinygrad/issues/2067
if Device.DEFAULT in ['METAL']:
  backend_test.exclude('test_maxpool_2d_pads_cpu')
  backend_test.exclude('test_maxpool_2d_same_lower_cpu')

if Device.DEFAULT in ['GPU', 'METAL']:
  # backend does not support dtype: Double
  backend_test.exclude('test_eyelike_with_dtype_cpu')
  backend_test.exclude('test_max_float64_cpu')
  backend_test.exclude('test_min_float64_cpu')
  backend_test.exclude('test_cast_FLOAT16_to_DOUBLE_cpu')
  backend_test.exclude('test_cast_FLOAT_to_DOUBLE_cpu')
  backend_test.exclude('test_castlike_FLOAT16_to_DOUBLE_cpu')
  backend_test.exclude('test_castlike_FLOAT16_to_DOUBLE_expanded_cpu')
  backend_test.exclude('test_castlike_FLOAT_to_DOUBLE_cpu')
  backend_test.exclude('test_castlike_FLOAT_to_DOUBLE_expanded_cpu')
  backend_test.exclude('test_cast_DOUBLE_*')
  backend_test.exclude('test_castlike_DOUBLE_*')
  backend_test.exclude('test_reduce_log_sum_exp_*')
  backend_test.exclude('test_operator_add_*')
  # weird inaccuracy
  backend_test.exclude('test_mish_cpu')
  backend_test.exclude('test_mish_expanded_cpu')

# Segfaults in CI, GPU requires cl_khr_fp16
if Device.DEFAULT in ['LLVM', 'GPU'] and CI:
  backend_test.exclude('test_dequantizelinear_e4m3fn_float16_cpu')
  backend_test.exclude('test_max_float16_cpu')
  backend_test.exclude('test_min_float16_cpu')
  backend_test.exclude('test_cast_DOUBLE_to_FLOAT16_cpu')
  backend_test.exclude('test_cast_FLOAT8E4M3FNUZ_to_FLOAT16_cpu')
  backend_test.exclude('test_cast_FLOAT8E4M3FN_to_FLOAT16_cpu')
  backend_test.exclude('test_cast_FLOAT8E5M2FNUZ_to_FLOAT16_cpu')
  backend_test.exclude('test_cast_FLOAT8E5M2_to_FLOAT16_cpu')
  backend_test.exclude('test_cast_FLOAT_to_FLOAT16_cpu')
  backend_test.exclude('test_cast_FLOAT16_*')
  backend_test.exclude('test_castlike_FLOAT_to_FLOAT16_cpu')
  backend_test.exclude('test_castlike_FLOAT_to_FLOAT16_expanded_cpu')
  backend_test.exclude('test_castlike_DOUBLE_to_FLOAT16_cpu')
  backend_test.exclude('test_castlike_DOUBLE_to_FLOAT16_expanded_cpu')
  backend_test.exclude('test_castlike_FLOAT16_*')

# undefined symbol: __truncdfhf2
if Device.DEFAULT == 'CLANG' and CI:
  backend_test.exclude('test_cast_DOUBLE_to_FLOAT16_cpu')
  backend_test.exclude('test_castlike_DOUBLE_to_FLOAT16_cpu')
  backend_test.exclude('test_castlike_DOUBLE_to_FLOAT16_expanded_cpu')

# TODO: this somehow passes in CI but does not pass if run locally
if isinstance(Device[Device.DEFAULT], Compiled):
  backend_test.exclude('test_MaxPool3d_stride_padding_cpu')

# TODO: this somehow passes in CI but does not pass if run locally
if Device.DEFAULT == 'METAL':
  backend_test.exclude('test_maxpool_2d_same_upper_cpu')

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
