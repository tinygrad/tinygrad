import unittest
from onnx.backend.base import Backend, BackendRep
import onnx.backend.test
import numpy as np
from tinygrad.tensor import Tensor
from typing import Any, Tuple

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
    return tuple(x.numpy() if isinstance(x, Tensor) else np.array(x) for x in ret.values())

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

# no binaryops min or max (needs llop, should add and replace relu)
backend_test.exclude('test_min_*')
backend_test.exclude('test_max_*')

# add support for SoftmaxCrossEntropyLoss and NegativeLogLikelihoodLoss
backend_test.exclude('test_sce_*')

# no optimizers (add them)
backend_test.exclude('test_adagrad_*')
backend_test.exclude('test_adam_*')
backend_test.exclude('test_nesterov_momentum_*')

# we only support float32
backend_test.exclude('test_add_uint8_*')
backend_test.exclude('test_sub_uint8_*')
backend_test.exclude('test_div_uint8_*')
backend_test.exclude('test_mul_uint8_*')
backend_test.exclude('test_cast_*')
backend_test.exclude('test_castlike_*')

# no support for nan or inf
backend_test.exclude('test_isinf_*')
backend_test.exclude('test_isnan_*')

# no support for mod
backend_test.exclude('test_mod_*')

# no trig ops
backend_test.exclude('test_acos_*')
backend_test.exclude('test_acosh_*')
backend_test.exclude('test_asin_*')
backend_test.exclude('test_asinh_*')
backend_test.exclude('test_atan_*')
backend_test.exclude('test_atanh_*')
backend_test.exclude('test_cos_*')
backend_test.exclude('test_cosh_*')
backend_test.exclude('test_sin_*')
backend_test.exclude('test_sinh_*')
backend_test.exclude('test_tan_*')

# no boolean ops (2d, 3d, 4d)
backend_test.exclude('test_and*')
backend_test.exclude('test_xor*')
backend_test.exclude('test_or*')
backend_test.exclude('test_bitshift_*')
backend_test.exclude('test_not_*')

# no scatter gather
backend_test.exclude('test_gather_*')
backend_test.exclude('test_gathernd_*')
backend_test.exclude('test_scatter_*')
backend_test.exclude('test_scatternd_*')

# no quantize
backend_test.exclude('test_dequantizelinear_*')
backend_test.exclude('test_dynamicquantizelinear_*')
backend_test.exclude('test_qlinearmatmul_*')
backend_test.exclude('test_quantizelinear_*')

# unsupported (strange) ops
backend_test.exclude('test_argmax_*')
backend_test.exclude('test_argmin_*')
backend_test.exclude('test_bitwise_*')
backend_test.exclude('test_blackmanwindow_*')
backend_test.exclude('test_bernoulli_*')
backend_test.exclude('test_cumsum_*')
backend_test.exclude('test_tril_*')
backend_test.exclude('test_triu_*')
backend_test.exclude('test_convinteger_*')
backend_test.exclude('test_col2im_*')
backend_test.exclude('test_hammingwindow_*')
backend_test.exclude('test_hannwindow_*')
backend_test.exclude('test_hardmax_*')
backend_test.exclude('test_gru_*')
backend_test.exclude('test_gridsample_*')
backend_test.exclude('test_if_*')
backend_test.exclude('test_compress_*')
backend_test.exclude('test_det_*')
backend_test.exclude('test_dft_*')
backend_test.exclude('test_einsum_*')
backend_test.exclude('test_erf_*')
backend_test.exclude('test_strnorm_*')
backend_test.exclude('test_unique_*')
backend_test.exclude('test_sequence_*')
backend_test.exclude('test_nonmaxsuppression_*')
backend_test.exclude('test_reversesequence_*')
backend_test.exclude('test_roialign_*')
backend_test.exclude('test_rnn_*')
backend_test.exclude('test_top_k_*')

# disable model tests for now since they are slow
for x in backend_test.test_suite:
  if 'OnnxBackendRealModelTest' in str(type(x)):
    backend_test.exclude(str(x).split(" ")[0])

#backend_test.include('test_tile_*')

# passing node tests
"""
backend_test.include('test_unsqueeze_*')
backend_test.include('test_gemm_*')
backend_test.include('test_batchnorm_*')
backend_test.include('test_transpose_*')
backend_test.include('test_shape_*')
backend_test.include('test_flatten_*')
backend_test.include('test_sum_*')
backend_test.include('test_global*')
backend_test.include('test_log_softmax*')
backend_test.include('test_softplus*')
"""

# requires Less, which would be a new llop
#backend_test.include('test_clip_*')

# broken empty tensor
#backend_test.include('test_reduce_sum_*')
#backend_test.include('test_reduce_l1_')

# requires cast
#backend_test.include('test_reduce_log_sum*')
#backend_test.include('test_pow_*')

# almost passing node tests
#backend_test.include('test_PReLU*')
#backend_test.include('test_expand_*')
#backend_test.include('test_conv_.*')
#backend_test.include('test_dropout_*')
#backend_test.include('test_reshape_*')

# good to investigate
#backend_test.include('test_slice_*')

# failing for real reasons
#backend_test.include('test_averagepool_2d_*')
#backend_test.include('test_maxpool_2d_*')

"""
backend_test.include('test_tanh_*')

# should be passing (good place to start!)
"""

# requires CastLike?
#backend_test.include('test_relu_*')
#backend_test.include('test_elu_*')
#backend_test.include('test_leakyrelu_*')
#backend_test.include('test_hardsigmoid_*')

# failing for lack of type support
#backend_test.include('test_add_*')
#backend_test.include('test_sub_*')
#backend_test.include('test_div_*')


# the node tests, slowly
#backend_test.include('test_softmax_*')
#backend_test.include('test_lrn_*')

# working big model tests
#backend_test.include('test_resnet50')
#backend_test.include('test_densenet121')
#backend_test.include('test_vgg19')

"""
# wrong big model tests
backend_test.include('test_shufflenet')
backend_test.include('test_inception_v2')
backend_test.include('test_squeezenet')
"""

"""
# unsupported big model tests : LRN
backend_test.include('test_bvlc_alexnet')
backend_test.include('test_inception_v1')
backend_test.include('test_zfnet512')
"""

globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
  unittest.main()
