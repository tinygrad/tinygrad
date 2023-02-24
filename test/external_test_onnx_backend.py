import unittest
from onnx.backend.base import Backend, BackendRep
import onnx.backend.test
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
    return tuple(x.numpy() for x in ret.values())

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

# the node tests
#for x in backend_test.test_suite:
#  if 'OnnxBackendNodeModelTest' in str(type(x)):
#    backend_test.include(str(x).split(" ")[0])

# passing node tests
backend_test.include('test_unsqueeze_*')
backend_test.include('test_gemm_*')
backend_test.include('test_batchnorm_*')

"""
backend_test.include('test_sum_*')
backend_test.include('test_transpose_*')
backend_test.include('test_tanh_*')

# should be passing (good place to start!)
backend_test.include('test_conv_.*')
backend_test.include('test_reshape_*')
backend_test.include('test_flatten_*')
backend_test.include('test_expand_*')
backend_test.include('test_clip_*')
"""

# requires CastLike?
#backend_test.include('test_relu_*')
#backend_test.include('test_elu_*')

# failing for lack of type support
#backend_test.include('test_add_*')
#backend_test.include('test_sub_*')
#backend_test.include('test_div_*')

# the node tests, slowly
#backend_test.include('test_reduce_sum_*')
#backend_test.include('test_shape_*')
#backend_test.include('test_softmax_*')
#backend_test.include('test_slice_*')
#backend_test.include('test_lrn_*')
#backend_test.include('test_batchnorm_*')
#backend_test.include('test_maxpool_*')
#backend_test.include('test_averagepool_*')

"""
# working big model tests
backend_test.include('test_resnet50')
backend_test.include('test_densenet121')

# wrong big model tests
backend_test.include('test_shufflenet')
backend_test.include('test_inception_v2')
"""

"""
# unsupported big model tests : LRN
backend_test.include('test_bvlc_alexnet')
backend_test.include('test_inception_v1')
backend_test.include('test_zfnet512')

# unsupported big model tests : Dropout
backend_test.include('test_squeezenet')
backend_test.include('test_vgg19')
"""

globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
  unittest.main()
