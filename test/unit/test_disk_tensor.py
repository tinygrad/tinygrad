import pathlib
import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.state import safe_load, safe_save, get_state_dict
from tinygrad.helpers import dtypes
from tinygrad.runtime.ops_disk import RawDiskBuffer
from extra.helpers import Timing

test_fn = pathlib.Path(__file__).parent.parent.parent / "weights/LLaMA/7B/consolidated.00.pth"
#test_size = test_fn.stat().st_size
test_size = 1024*1024*1024*2

# sudo su -c 'sync; echo 1 > /proc/sys/vm/drop_caches' && python3 test/unit/test_disk_tensor.py TestRawDiskBuffer.test_readinto_read_speed
@unittest.skipIf(not test_fn.exists(), "download LLaMA weights for read in speed tests")
class TestRawDiskBuffer(unittest.TestCase):
  def test_readinto_read_speed(self):
    tst = np.empty(test_size, np.uint8)
    with open(test_fn, "rb") as f:
      with Timing("copy in ", lambda et_ns: f" {test_size/et_ns:.2f} GB/s"):
        f.readinto(tst)

  def test_mmap_read_speed(self):
    db = RawDiskBuffer(test_size, dtype=dtypes.uint8, device=test_fn)
    tst = np.empty(test_size, np.uint8)
    with Timing("copy in ", lambda et_ns: f" {test_size/et_ns:.2f} GB/s"):
      np.copyto(tst, db.toCPU())

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
    assert sorted(list(state_dict_loaded.keys())) == sorted(list(state_dict.keys()))
    for k,v in state_dict.items():
      np.testing.assert_array_equal(v.numpy(), state_dict_loaded[k].numpy())

    # load with the real safetensors
    from safetensors import safe_open
    with safe_open("/tmp/eff0", framework="pt", device="cpu") as f:
      assert sorted(list(f.keys())) == sorted(list(state_dict.keys()))
      for k in f.keys():
        np.testing.assert_array_equal(f.get_tensor(k).numpy(), state_dict[k].numpy())

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

