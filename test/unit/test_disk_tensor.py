import pathlib
import unittest
import numpy as np
from tinygrad.tensor import Tensor, Device
from tinygrad.nn.state import safe_load, safe_save, get_state_dict, torch_load
from tinygrad.helpers import dtypes
from tinygrad.runtime.ops_disk import RawDiskBuffer
from tinygrad.helpers import Timing
from extra.utils import fetch_as_file, temp

def compare_weights_both(url):
  import torch
  fn = fetch_as_file(url)
  tg_weights = get_state_dict(torch_load(fn))
  torch_weights = get_state_dict(torch.load(fn), tensor_type=torch.Tensor)
  assert list(tg_weights.keys()) == list(torch_weights.keys())
  for k in tg_weights:
    np.testing.assert_equal(tg_weights[k].numpy(), torch_weights[k].numpy(), err_msg=f"mismatch at {k}, {tg_weights[k].shape}")
  print(f"compared {len(tg_weights)} weights")

class TestTorchLoad(unittest.TestCase):
  # pytorch pkl format
  def test_load_enet(self): compare_weights_both("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth")
  # pytorch zip format
  def test_load_enet_alt(self): compare_weights_both("https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth")
  # pytorch zip format
  def test_load_convnext(self): compare_weights_both('https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth')
  # TODO: support pytorch tar format with minimal lines
  #def test_load_resnet(self): compare_weights_both('https://download.pytorch.org/models/resnet50-19c8e357.pth')

test_fn = pathlib.Path(__file__).parents[2] / "weights/LLaMA/7B/consolidated.00.pth"
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
@unittest.skipIf(Device.DEFAULT == "WEBGPU", "webgpu doesn't support uint8 datatype")
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
    save_file(tensors, temp("model.safetensors"))

    ret = safe_load(temp("model.safetensors"))
    for k,v in tensors.items(): np.testing.assert_array_equal(ret[k].numpy(), v.numpy())
    safe_save(ret, temp("model.safetensors_alt"))
    with open(temp("model.safetensors"), "rb") as f:
      with open(temp("model.safetensors_alt"), "rb") as g:
        assert f.read() == g.read()
    ret2 = safe_load(temp("model.safetensors_alt"))
    for k,v in tensors.items(): np.testing.assert_array_equal(ret2[k].numpy(), v.numpy())

  def test_efficientnet_safetensors(self):
    from models.efficientnet import EfficientNet
    model = EfficientNet(0)
    state_dict = get_state_dict(model)
    safe_save(state_dict, temp("eff0"))
    state_dict_loaded = safe_load(temp("eff0"))
    assert sorted(list(state_dict_loaded.keys())) == sorted(list(state_dict.keys()))
    for k,v in state_dict.items():
      np.testing.assert_array_equal(v.numpy(), state_dict_loaded[k].numpy())

    # load with the real safetensors
    from safetensors import safe_open
    with safe_open(temp("eff0"), framework="pt", device="cpu") as f:
      assert sorted(list(f.keys())) == sorted(list(state_dict.keys()))
      for k in f.keys():
        np.testing.assert_array_equal(f.get_tensor(k).numpy(), state_dict[k].numpy())

  def test_huggingface_enet_safetensors(self):
    # test a real file
    fn = fetch_as_file("https://huggingface.co/timm/mobilenetv3_small_075.lamb_in1k/resolve/main/model.safetensors")
    state_dict = safe_load(fn)
    assert len(state_dict.keys()) == 244
    assert 'blocks.2.2.se.conv_reduce.weight' in state_dict
    assert state_dict['blocks.0.0.bn1.num_batches_tracked'].numpy() == 276570
    assert state_dict['blocks.2.0.bn2.num_batches_tracked'].numpy() == 276570

  def test_metadata(self):
    metadata = {"hello": "world"}
    safe_save({}, temp('metadata.safetensors'), metadata)
    import struct
    with open(temp('metadata.safetensors'), 'rb') as f:
      dat = f.read()
    sz = struct.unpack(">Q", dat[0:8])[0]
    import json
    assert json.loads(dat[8:8+sz])['__metadata__']['hello'] == 'world'

def helper_test_disk_tensor(fn, data, np_fxn, tinygrad_fxn=None):
  if tinygrad_fxn is None: tinygrad_fxn = np_fxn
  pathlib.Path(temp(fn)).unlink(missing_ok=True)
  tinygrad_tensor = Tensor(data, device="CPU").to(f"disk:{temp(fn)}")
  numpy_arr = np.array(data)
  tinygrad_fxn(tinygrad_tensor)
  np_fxn(numpy_arr)
  np.testing.assert_allclose(tinygrad_tensor.numpy(), numpy_arr)

class TestDiskTensor(unittest.TestCase):
  def test_empty(self):
    pathlib.Path(temp("dt1")).unlink(missing_ok=True)
    Tensor.empty(100, 100, device=f"disk:{temp('dt1')}")

  def test_write_ones(self):
    pathlib.Path(temp("dt2")).unlink(missing_ok=True)

    out = Tensor.ones(10, 10, device="CPU")
    outdisk = out.to(f"disk:{temp('dt2')}")
    print(outdisk)
    outdisk.realize()
    del out, outdisk

    # test file
    with open(temp("dt2"), "rb") as f:
      assert f.read() == b"\x00\x00\x80\x3F" * 100

    # test load alt
    reloaded = Tensor.empty(10, 10, device=f"disk:{temp('dt2')}")
    out = reloaded.numpy()
    assert np.all(out == 1.)

  def test_assign_slice(self):
    def assign(x,s,y): x[s] = y
    helper_test_disk_tensor("dt3", [0,1,2,3], lambda x: assign(x, slice(0,2), [13, 12]))
    helper_test_disk_tensor("dt4", [[0,1,2,3],[4,5,6,7]], lambda x: assign(x, slice(0,1), [[13, 12, 11, 10]]))

  def test_reshape(self):
    helper_test_disk_tensor("dt5", [1,2,3,4,5], lambda x: x.reshape((1,5)))
    helper_test_disk_tensor("dt6", [1,2,3,4], lambda x: x.reshape((2,2)))

if __name__ == "__main__":
  unittest.main()
