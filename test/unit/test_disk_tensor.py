import pathlib
import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.state import safe_load, safe_save, get_state_dict

class TestSafetensors(unittest.TestCase):
  def test_real_safetensors(self):
    import torch
    from safetensors.torch import save_file
    torch.manual_seed(1337)
    tensors = {
      "weight1": torch.randn((16, 16)),
      "weight2": torch.arange(0, 17, dtype=torch.uint8),
      "weight3": torch.arange(0, 17, dtype=torch.int32).reshape(17,1,1),
      "weight4": torch.arange(0, 2, dtype=torch.uint8),
    }
    save_file(tensors, "/tmp/model.safetensors")

    ret = safe_load("/tmp/model.safetensors")
    for k,v in tensors.items(): np.testing.assert_array_equal(ret[k].numpy(), v.numpy())
    safe_save(ret, "/tmp/model.safetensors_alt")
    with open("/tmp/model.safetensors", "rb") as f:
      with open("/tmp/model.safetensors_alt", "rb") as g:
        assert f.read() == g.read()
    ret2 = safe_load("/tmp/model.safetensors_alt")
    for k,v in tensors.items(): np.testing.assert_array_equal(ret2[k].numpy(), v.numpy())

  def test_efficientnet_safetensors(self):
    from models.efficientnet import EfficientNet
    model = EfficientNet(0)
    state_dict = get_state_dict(model)
    safe_save(state_dict, "/tmp/eff0")
    state_dict_loaded = safe_load("/tmp/eff0")
    for k,v in state_dict.items():
      np.testing.assert_array_equal(v.numpy(), state_dict_loaded[k].numpy())

class TestDiskTensor(unittest.TestCase):
  def test_empty(self):
    pathlib.Path("/tmp/dt1").unlink(missing_ok=True)
    Tensor.empty(100, 100, device="disk:/tmp/dt1")

  def test_write_ones(self):
    pathlib.Path("/tmp/dt2").unlink(missing_ok=True)

    out = Tensor.ones(10, 10, device="CPU")
    outdisk = out.to("disk:/tmp/dt2")
    print(outdisk)
    outdisk.realize()
    del out, outdisk

    # test file
    with open("/tmp/dt2", "rb") as f:
      assert f.read() == b"\x00\x00\x80\x3F" * 100

    # test load alt
    reloaded = Tensor.empty(10, 10, device="disk:/tmp/dt2")
    out = reloaded.numpy()
    assert np.all(out == 1.)

  def test_slice(self):
    pathlib.Path("/tmp/dt3").unlink(missing_ok=True)
    Tensor.arange(10, device="disk:/tmp/dt3").realize()

    slice_me = Tensor.empty(10, device="disk:/tmp/dt3")
    print(slice_me)
    is_3 = slice_me[3:4].cpu()
    assert is_3.numpy()[0] == 3

  def test_slice_2d(self):
    pathlib.Path("/tmp/dt5").unlink(missing_ok=True)
    Tensor.arange(100, device="CPU").to("disk:/tmp/dt5").realize()
    slice_me = Tensor.empty(10, 10, device="disk:/tmp/dt5")
    tst = slice_me[1].numpy()
    print(tst)
    np.testing.assert_allclose(tst, np.arange(10, 20))

  def test_assign_slice(self):
    pathlib.Path("/tmp/dt4").unlink(missing_ok=True)
    cc = Tensor.arange(10, device="CPU").to("disk:/tmp/dt4").realize()
    #cc.assign(np.ones(10)).realize()
    print(cc[3:5].numpy())
    cc[3:5].assign([13, 12]).realize()
    print(cc.numpy())

if __name__ == "__main__":
  unittest.main()

