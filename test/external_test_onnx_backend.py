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
    ret = next(iter(ret.values())).numpy()
    return (ret,)

class TinygradBackend(Backend):
  @classmethod
  def prepare(cls, onnx_model, device):
    input_names = [inp.name for inp in onnx_model.graph.input]
    print("prepare", cls, device, input_names)
    run_onnx = get_run_onnx(onnx_model)
    return TinygradModel(run_onnx, input_names)
  
  @classmethod
  def supports_device(cls, device: str) -> bool:
    return device == "CPU"

backend_test = onnx.backend.test.BackendTest(TinygradBackend, __name__) 

# only the node tests for now
for x in backend_test.test_suite:
  if 'OnnxBackendNodeModelTest' in str(type(x)):
    backend_test.include(str(x).split(" ")[0])

globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
  unittest.main()
