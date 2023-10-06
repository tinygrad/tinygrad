import unittest
from extra.export_model import export_model, EXPORT_SUPPORTED_DEVICE
from tinygrad.tensor import Tensor, Device
import json

class MockMultiInputModel:
  def forward(self, x1, x2, x3):
    return x1 + x2 + x3

# TODO: move compile_efficientnet tests here
@unittest.skipUnless(Device.DEFAULT in EXPORT_SUPPORTED_DEVICE, f"Model export is not supported on {Device.DEFAULT}")
class TextModelExport(unittest.TestCase):
  def test_multi_input_model_export(self):
    model = MockMultiInputModel()
    inputs = [Tensor.rand(2,2), Tensor.rand(2,2), Tensor.rand(2,2)]
    prg, inp_sizes, _, _ = export_model(model, "", *inputs)
    prg = json.loads(prg)

    assert len(inputs) == len(prg["inputs"]) == len(inp_sizes), f"Model and exported inputs don't match: mdl={len(inputs)}, prg={len(prg['inputs'])}, inp_sizes={len(inp_sizes)}"

    for i in range(len(inputs)): 
      assert f"input{i}" in inp_sizes, f"input{i} not captured in inp_sizes"
      assert f"input{i}" in prg["buffers"], f"input{i} not captured in exported buffers"

    for i, exported_input in enumerate(prg["inputs"]): 
      assert inputs[i].dtype.name == exported_input["dtype"], f"Model and exported input dtype don't match: mdl={inputs[i].dtype.name}, prg={exported_input['dtype']}"

if __name__ == '__main__':
  unittest.main()
