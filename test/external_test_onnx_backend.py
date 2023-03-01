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

# add support for SoftmaxCrossEntropyLoss and NegativeLogLikelihoodLoss
backend_test.exclude('test_sce_*')

# no support for reduce with multiply (needs llop)
backend_test.exclude('test_reduce_prod_*')

# no optimizers (add them?)
backend_test.exclude('test_adagrad_*')
backend_test.exclude('test_adam_*')
backend_test.exclude('test_nesterov_momentum_*')
backend_test.exclude('test_momentum_*')

# disable some creation ops
backend_test.exclude('test_eyelike_*')

# we only support float32
backend_test.exclude('test_add_uint8_*')
backend_test.exclude('test_sub_uint8_*')
backend_test.exclude('test_div_uint8_*')
backend_test.exclude('test_mul_uint8_*')
backend_test.exclude('test_pow_types_int*')
backend_test.exclude('test_cast_*')
backend_test.exclude('test_castlike_*')
backend_test.exclude('test_convinteger_*')
backend_test.exclude('test_matmulinteger_*')

# we don't support rounding
backend_test.exclude('test_round_*')
backend_test.exclude('test_ceil_*')
backend_test.exclude('test_floor_*')

# we don't support indexes
backend_test.exclude('test_argmax_*')
backend_test.exclude('test_argmin_*')
backend_test.exclude('test_nonzero_*')

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

# unsupported (strange) ops
backend_test.exclude('test_bitwise_*')
backend_test.exclude('test_blackmanwindow_*')
backend_test.exclude('test_bernoulli_*')
backend_test.exclude('test_cumsum_*')
backend_test.exclude('test_tril_*')
backend_test.exclude('test_triu_*')
backend_test.exclude('test_col2im_*')
backend_test.exclude('test_hammingwindow_*')
backend_test.exclude('test_hannwindow_*')
backend_test.exclude('test_hardmax_*')
backend_test.exclude('test_gridsample_*')
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
backend_test.exclude('test_top_k_*')
backend_test.exclude('test_tfidfvectorizer_*')
backend_test.exclude('test_stft_*')
backend_test.exclude('test_melweightmatrix_*')

# disable model tests for now since they are slow
if True:
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
