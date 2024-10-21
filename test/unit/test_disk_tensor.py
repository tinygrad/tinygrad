import os
import pathlib, tempfile, unittest
import tarfile, ggml, ctypes

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.dtype import DType
from tinygrad.nn.state import safe_load, safe_save, get_state_dict, torch_load, tar_extract, ggml_data_to_tensor, gguf_load
from tinygrad.helpers import Timing, fetch, temp, CI
from test.helpers import is_dtype_supported

def compare_weights_both(url):
  import torch
  fn = fetch(url)
  tg_weights = get_state_dict(torch_load(fn))
  torch_weights = get_state_dict(torch.load(fn, map_location=torch.device('cpu'), weights_only=True), tensor_type=torch.Tensor)
  assert list(tg_weights.keys()) == list(torch_weights.keys())
  for k in tg_weights:
    if tg_weights[k].dtype == dtypes.bfloat16: tg_weights[k] = torch_weights[k].float() # numpy doesn't support bfloat16
    if torch_weights[k].dtype == torch.bfloat16: torch_weights[k] = torch_weights[k].float() # numpy doesn't support bfloat16
    if torch_weights[k].requires_grad: torch_weights[k] = torch_weights[k].detach()
    np.testing.assert_equal(tg_weights[k].numpy(), torch_weights[k].numpy(), err_msg=f"mismatch at {k}, {tg_weights[k].shape}")
  print(f"compared {len(tg_weights)} weights")

class TestTorchLoad(unittest.TestCase):
  # pytorch pkl format
  def test_load_enet(self): compare_weights_both("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth")
  # pytorch zip format
  def test_load_enet_alt(self): compare_weights_both("https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth")
  # pytorch zip format
  def test_load_convnext(self): compare_weights_both('https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth')

  @unittest.skipUnless(is_dtype_supported(dtypes.float16), "need float16 support")
  def test_load_llama2bfloat(self): compare_weights_both("https://huggingface.co/qazalin/bf16-lightweight/resolve/main/consolidated.00.pth?download=true")

  # pytorch tar format
  def test_load_resnet(self): compare_weights_both('https://download.pytorch.org/models/resnet50-19c8e357.pth')

test_fn = pathlib.Path(__file__).parents[2] / "weights/LLaMA/7B/consolidated.00.pth"
#test_size = test_fn.stat().st_size
test_size = 1024*1024*1024*2

def _test_bitcasted(t: Tensor, dt: DType, expected):
  np.testing.assert_allclose(t.bitcast(dt).numpy(), expected)

# sudo su -c 'sync; echo 1 > /proc/sys/vm/drop_caches' && python3 test/unit/test_disk_tensor.py TestRawDiskBuffer.test_readinto_read_speed
class TestRawDiskBuffer(unittest.TestCase):
  @unittest.skipIf(not test_fn.exists(), "download LLaMA weights for read in speed tests")
  def test_readinto_read_speed(self):
    tst = np.empty(test_size, np.uint8)
    with open(test_fn, "rb") as f:
      with Timing("copy in ", lambda et_ns: f" {test_size/et_ns:.2f} GB/s"):
        f.readinto(tst)
  def test_bitcasts_on_disk(self):
    _, tmp = tempfile.mkstemp()
    # ground truth = https://evanw.github.io/float-toy/
    t = Tensor.empty((128, 128), dtype=dtypes.uint8, device=f"disk:{tmp}") # uint8
    # all zeroes
    _test_bitcasted(t, dtypes.float16, 0.0)
    _test_bitcasted(t, dtypes.uint16, 0)
    _test_bitcasted(t, dtypes.float32, 0.0)
    _test_bitcasted(t, dtypes.uint32, 0)
    # pi in float16 stored via int16
    t.bitcast(dtypes.uint16).assign(Tensor.full((128, 64), 0x4248, dtype=dtypes.uint16)).realize()
    _test_bitcasted(t, dtypes.float16, 3.140625)
    _test_bitcasted(t, dtypes.float32, 50.064727)
    _test_bitcasted(t, dtypes.uint16, 0x4248)
    _test_bitcasted(t, dtypes.uint32, 0x42484248)
    # pi in float32 stored via float32
    t.bitcast(dtypes.float32).assign(Tensor.full((128, 32), 3.1415927, dtype=dtypes.float32)).realize()
    _test_bitcasted(t, dtypes.float32, 3.1415927)
    _test_bitcasted(t, dtypes.uint32, 0x40490FDB)
    # doesn't suport normal cast
    with self.assertRaises(RuntimeError):
      Tensor.empty((4,), dtype=dtypes.int16, device=f"disk:{tmp}").cast(dtypes.float16)

    # Those two should be moved to test_dtype.py:test_shape_change_bitcast after bitcast works on non-disk
    with self.assertRaises(RuntimeError):
      # should fail because 3 int8 is 3 bytes but float16 is two and 3 isn't a multiple of 2
      Tensor.empty((3,), dtype=dtypes.int8, device=f"DISK:{tmp}").bitcast(dtypes.float16)

    with self.assertRaises(RuntimeError):
      # should fail because backprop through bitcast is undefined
      Tensor.empty((4,), dtype=dtypes.int8, requires_grad=True, device=f"DISK:{tmp}").bitcast(dtypes.float16)

    pathlib.Path(tmp).unlink()

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
    save_file(tensors, temp("real.safetensors"))

    ret = safe_load(temp("real.safetensors"))
    for k,v in tensors.items(): np.testing.assert_array_equal(ret[k].numpy(), v.numpy())
    safe_save(ret, temp("real.safetensors_alt"))
    with open(temp("real.safetensors"), "rb") as f:
      with open(temp("real.safetensors_alt"), "rb") as g:
        assert f.read() == g.read()
    ret2 = safe_load(temp("real.safetensors_alt"))
    for k,v in tensors.items(): np.testing.assert_array_equal(ret2[k].numpy(), v.numpy())

  def test_real_safetensors_open(self):
    fn = temp("real_safe")
    state_dict = {"tmp": Tensor.rand(10,10)}
    safe_save(state_dict, fn)
    import os
    assert os.path.getsize(fn) == 8+0x40+(10*10*4)
    from safetensors import safe_open
    with safe_open(fn, framework="pt", device="cpu") as f:
      assert sorted(f.keys()) == sorted(state_dict.keys())
      for k in f.keys():
        np.testing.assert_array_equal(f.get_tensor(k).numpy(), state_dict[k].numpy())

  def test_efficientnet_safetensors(self):
    from extra.models.efficientnet import EfficientNet
    model = EfficientNet(0)
    state_dict = get_state_dict(model)
    safe_save(state_dict, temp("eff0"))
    state_dict_loaded = safe_load(temp("eff0"))
    assert sorted(state_dict_loaded.keys()) == sorted(state_dict.keys())
    for k,v in state_dict.items():
      np.testing.assert_array_equal(v.numpy(), state_dict_loaded[k].numpy())

    # load with the real safetensors
    from safetensors import safe_open
    with safe_open(temp("eff0"), framework="pt", device="cpu") as f:
      assert sorted(f.keys()) == sorted(state_dict.keys())
      for k in f.keys():
        np.testing.assert_array_equal(f.get_tensor(k).numpy(), state_dict[k].numpy())

  def test_huggingface_enet_safetensors(self):
    # test a real file
    fn = fetch("https://huggingface.co/timm/mobilenetv3_small_075.lamb_in1k/resolve/main/model.safetensors")
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

  def test_save_all_dtypes(self):
    for dtype in dtypes.fields().values():
      if dtype in [dtypes.bfloat16]: continue # not supported in numpy
      path = temp(f"ones.{dtype}.safetensors")
      ones = Tensor(np.random.rand(10,10), dtype=dtype)
      safe_save(get_state_dict(ones), path)
      np.testing.assert_equal(ones.numpy(), list(safe_load(path).values())[0].numpy())

  def test_load_supported_types(self):
    import torch
    from safetensors.torch import save_file
    from safetensors.numpy import save_file as np_save_file
    torch.manual_seed(1337)
    tensors = {
      "weight_F16": torch.randn((2, 2), dtype=torch.float16),
      "weight_F32": torch.randn((2, 2), dtype=torch.float32),
      "weight_U8": torch.tensor([1, 2, 3], dtype=torch.uint8),
      "weight_I8": torch.tensor([-1, 2, 3], dtype=torch.int8),
      "weight_I32": torch.tensor([-1, 2, 3], dtype=torch.int32),
      "weight_I64": torch.tensor([-1, 2, 3], dtype=torch.int64),
      "weight_F64": torch.randn((2, 2), dtype=torch.double),
      "weight_BOOL": torch.tensor([True, False], dtype=torch.bool),
      "weight_I16": torch.tensor([127, 64], dtype=torch.short),
      "weight_BF16": torch.randn((2, 2), dtype=torch.bfloat16),
    }
    save_file(tensors, temp("dtypes.safetensors"))

    loaded = safe_load(temp("dtypes.safetensors"))
    for k,v in loaded.items():
      if v.dtype != dtypes.bfloat16:
        assert v.numpy().dtype == tensors[k].numpy().dtype
        np.testing.assert_allclose(v.numpy(), tensors[k].numpy())

    # pytorch does not support U16, U32, and U64 dtypes.
    tensors = {
      "weight_U16": np.array([1, 2, 3], dtype=np.uint16),
      "weight_U32": np.array([1, 2, 3], dtype=np.uint32),
      "weight_U64": np.array([1, 2, 3], dtype=np.uint64),
    }
    np_save_file(tensors, temp("dtypes.safetensors"))

    loaded = safe_load(temp("dtypes.safetensors"))
    for k,v in loaded.items():
      assert v.numpy().dtype == tensors[k].dtype
      np.testing.assert_allclose(v.numpy(), tensors[k])

def helper_test_disk_tensor(fn, data, np_fxn, tinygrad_fxn=None):
  if tinygrad_fxn is None: tinygrad_fxn = np_fxn
  pathlib.Path(temp(fn)).unlink(missing_ok=True)
  tinygrad_tensor = Tensor(data, device="CLANG").to(f"disk:{temp(fn)}")
  numpy_arr = np.array(data)
  tinygrad_fxn(tinygrad_tensor)
  np_fxn(numpy_arr)
  np.testing.assert_allclose(tinygrad_tensor.numpy(), numpy_arr)

class TestDiskTensor(unittest.TestCase):
  def test_empty(self):
    pathlib.Path(temp("dt_empty")).unlink(missing_ok=True)
    Tensor.empty(100, 100, device=f"disk:{temp('dt_empty')}")

  def test_simple_read(self):
    fn = pathlib.Path(temp("dt_simple_read"))
    fn.unlink(missing_ok=True)
    fn.write_bytes(bytes(range(256)))
    t = Tensor.empty(16, 16, device=f"disk:{temp('dt_simple_read')}", dtype=dtypes.uint8)
    out = t[1].to(Device.DEFAULT).tolist()
    assert out == list(range(16, 32))

  def test_simple_read_bitcast(self):
    fn = pathlib.Path(temp("dt_simple_read_bitcast"))
    fn.unlink(missing_ok=True)
    fn.write_bytes(bytes(range(256))*2)
    t = Tensor.empty(16, 16*2, device=f"disk:{temp('dt_simple_read_bitcast')}", dtype=dtypes.uint8)
    out = t[1].bitcast(dtypes.uint16).to(Device.DEFAULT).tolist()
    tout = [(x//256, x%256) for x in out]
    assert tout == list([(x+1,x) for x in range(32,64,2)])

  def test_simple_read_bitcast_alt(self):
    fn = pathlib.Path(temp("dt_simple_read_bitcast_alt"))
    fn.unlink(missing_ok=True)
    fn.write_bytes(bytes(range(256))*2)
    t = Tensor.empty(16, 16*2, device=f"disk:{temp('dt_simple_read_bitcast_alt')}", dtype=dtypes.uint8)
    out = t.bitcast(dtypes.uint16)[1].to(Device.DEFAULT).tolist()
    tout = [(x//256, x%256) for x in out]
    assert tout == list([(x+1,x) for x in range(32,64,2)])

  def test_write_ones(self):
    pathlib.Path(temp("dt_write_ones")).unlink(missing_ok=True)

    out = Tensor.ones(10, 10, device="CLANG").contiguous()
    outdisk = out.to(f"disk:{temp('dt_write_ones')}")
    print(outdisk)
    outdisk.realize()
    del out, outdisk

    import struct
    # test file
    with open(temp("dt_write_ones"), "rb") as f:
      assert f.read() == struct.pack('<f', 1.0) * 100 == b"\x00\x00\x80\x3F" * 100

    # test load alt
    reloaded = Tensor.empty(10, 10, device=f"disk:{temp('dt_write_ones')}")
    np.testing.assert_almost_equal(reloaded.numpy(), np.ones((10, 10)))

  def test_assign_slice(self):
    def assign(x,s,y): x[s] = y
    helper_test_disk_tensor("dt_assign_slice_1", [0,1,2,3], lambda x: assign(x, slice(0,2), [13, 12]))
    helper_test_disk_tensor("dt_assign_slice_2", [[0,1,2,3],[4,5,6,7]], lambda x: assign(x, slice(0,1), [[13, 12, 11, 10]]))

  def test_reshape(self):
    helper_test_disk_tensor("dt_reshape_1", [1,2,3,4,5], lambda x: x.reshape((1,5)))
    helper_test_disk_tensor("dt_reshape_2", [1,2,3,4], lambda x: x.reshape((2,2)))

  def test_assign_to_different_dtype(self):
    # NOTE: this is similar to Y_train in fetch_cifar
    t = Tensor.empty(10, device=f'disk:{temp("dt_assign_to_different_dtype")}', dtype=dtypes.int64)

    for i in range(5):
      data = np.array([3, 3])
      idx = 2 * i
      t[idx:idx+2].assign(data)

    np.testing.assert_array_equal(t.numpy(), np.array([3] * 10))

  def test_bitcast(self):
    with open(temp('dt_bitcast'), "wb") as f: f.write(bytes(range(10,20)))
    t = Tensor.empty(5, dtype=dtypes.int16, device=f"disk:{temp('dt_bitcast')}")
    ret = t.to("CLANG").bitcast(dtypes.uint16) + 1
    assert ret.tolist() == [2827, 3341, 3855, 4369, 4883]

  def test_bitcast_view(self):
    with open(temp('dt_bitcast_view'), "wb") as f: f.write(bytes(range(10, 24)))
    t = Tensor.empty(3, dtype=dtypes.uint, device=f"disk:{temp('dt_bitcast_view')}").shrink([(0, 2)])
    ret = t.bitcast(dtypes.uint16).to("CLANG") + 1
    assert ret.tolist() == [2827, 3341, 3855, 4369]

  def test_bf16_disk_write_read(self):
    t = Tensor([10000, -1, -1000, -10000, 20], dtype=dtypes.float32)
    t.to(f"disk:{temp('dt_bf16_disk_write_read_f32')}").realize()

    # hack to "cast" f32 -> bf16
    with open(temp('dt_bf16_disk_write_read_f32'), "rb") as f: dat = f.read()
    adat = b''.join([dat[i+2:i+4] for i in range(0, len(dat), 4)])
    with open(temp('dt_bf16_disk_write_read_bf16'), "wb") as f: f.write(adat)

    t = Tensor.empty(5, dtype=dtypes.bfloat16, device=f"disk:{temp('dt_bf16_disk_write_read_bf16')}")
    ct = t.llvm_bf16_cast(dtypes.float)
    assert ct.numpy().tolist() == [9984., -1, -1000, -9984, 20]

  def test_copy_from_disk(self):
    fn = pathlib.Path(temp("dt_copy_from_disk"))
    fn.unlink(missing_ok=True)
    fn.write_bytes(bytes(range(256))*1024)

    t = Tensor.empty(256*1024, device=f"disk:{temp('dt_copy_from_disk')}", dtype=dtypes.uint8)
    on_dev = t.to(Device.DEFAULT).realize()
    np.testing.assert_equal(on_dev.numpy(), t.numpy())

  def test_copy_from_disk_offset(self):
    fn = pathlib.Path(temp("dt_copy_from_disk_offset"))
    fn.unlink(missing_ok=True)
    fn.write_bytes(bytes(range(256))*1024)

    for off in [314, 991, 2048, 4096]:
      t = Tensor.empty(256*1024, device=f"disk:{temp('dt_copy_from_disk_offset')}", dtype=dtypes.uint8)[off:]
      on_dev = t.to(Device.DEFAULT).realize()
      np.testing.assert_equal(on_dev.numpy(), t.numpy())

  def test_copy_from_disk_huge(self):
    if CI and not hasattr(Device["DISK"], 'io_uring'): self.skipTest("slow on ci without iouring")

    fn = pathlib.Path(temp("dt_copy_from_disk_huge"))
    fn.unlink(missing_ok=True)
    fn.write_bytes(bytes(range(256))*1024*256)

    for off in [0, 551]:
      t = Tensor.empty(256*1024*256, device=f"disk:{temp('dt_copy_from_disk_huge')}", dtype=dtypes.uint8)[off:]
      on_dev = t.to(Device.DEFAULT).realize()
      np.testing.assert_equal(on_dev.numpy(), t.numpy())

class TestTarExtract(unittest.TestCase):
  def setUp(self):
    self.test_dir = tempfile.mkdtemp()
    self.test_files = {
      'file1.txt': b'Hello, World!',
      'file2.bin': b'\x00\x01\x02\x03\x04',
      'empty_file.txt': b''
    }
    self.tar_path = os.path.join(self.test_dir, 'test.tar')
    with tarfile.open(self.tar_path, 'w') as tar:
      for filename, content in self.test_files.items():
        file_path = os.path.join(self.test_dir, filename)
        with open(file_path, 'wb') as f:
          f.write(content)
        tar.add(file_path, arcname=filename)

    # Create invalid tar file
    self.invalid_tar_path = os.path.join(self.test_dir, 'invalid.tar')
    with open(self.invalid_tar_path, 'wb') as f:
      f.write(b'This is not a valid tar file')

  def tearDown(self):
    for filename in self.test_files:
      os.remove(os.path.join(self.test_dir, filename))
    os.remove(self.tar_path)
    os.remove(self.invalid_tar_path)
    os.rmdir(self.test_dir)

  def test_tar_extract_returns_dict(self):
    result = tar_extract(self.tar_path)
    self.assertIsInstance(result, dict)

  def test_tar_extract_correct_keys(self):
    result = tar_extract(self.tar_path)
    self.assertEqual(set(result.keys()), set(self.test_files.keys()))

  def test_tar_extract_content_size(self):
    result = tar_extract(self.tar_path)
    for filename, content in self.test_files.items():
      self.assertEqual(len(result[filename]), len(content))

  def test_tar_extract_content_values(self):
    result = tar_extract(self.tar_path)
    for filename, content in self.test_files.items():
      np.testing.assert_array_equal(result[filename].numpy(), np.frombuffer(content, dtype=np.uint8))

  def test_tar_extract_empty_file(self):
    result = tar_extract(self.tar_path)
    self.assertEqual(len(result['empty_file.txt']), 0)

  def test_tar_extract_non_existent_file(self):
    with self.assertRaises(FileNotFoundError):
      tar_extract('non_existent_file.tar')

  def test_tar_extract_invalid_file(self):
    with self.assertRaises(tarfile.ReadError):
      tar_extract(self.invalid_tar_path)

class TestPathTensor(unittest.TestCase):
  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.test_file = pathlib.Path(self.temp_dir.name) / "test_file.bin"
    self.test_data = np.arange(100, dtype=np.uint8).tobytes()
    with open(self.test_file, "wb") as f:
      f.write(self.test_data)

  def tearDown(self):
    self.temp_dir.cleanup()

  def test_path_tensor_no_device(self):
    t = Tensor(self.test_file)
    self.assertEqual(t.shape, (100,))
    self.assertEqual(t.dtype, dtypes.uint8)
    self.assertTrue(t.device.startswith("DISK:"))
    np.testing.assert_array_equal(t.numpy(), np.frombuffer(self.test_data, dtype=np.uint8))

  def test_path_tensor_with_device(self):
    t = Tensor(self.test_file, device="CPU")
    self.assertEqual(t.shape, (100,))
    self.assertEqual(t.dtype, dtypes.uint8)
    self.assertEqual(t.device, "CPU")
    np.testing.assert_array_equal(t.numpy(), np.frombuffer(self.test_data, dtype=np.uint8))

  def test_path_tensor_empty_file(self):
    empty_file = pathlib.Path(self.temp_dir.name) / "empty_file.bin"
    empty_file.touch()
    t = Tensor(empty_file)
    self.assertEqual(t.shape, (0,))
    self.assertEqual(t.dtype, dtypes.uint8)
    self.assertTrue(t.device.startswith("DISK:"))

  def test_path_tensor_non_existent_file(self):
    non_existent_file = pathlib.Path(self.temp_dir.name) / "non_existent.bin"
    with self.assertRaises(FileNotFoundError):
      Tensor(non_existent_file)

  def test_path_tensor_with_dtype(self):
    t = Tensor(self.test_file, dtype=dtypes.int16)
    self.assertEqual(t.shape, (50,))
    self.assertEqual(t.dtype, dtypes.int16)
    self.assertTrue(t.device.startswith("DISK:"))
    np.testing.assert_array_equal(t.numpy(), np.frombuffer(self.test_data, dtype=np.int16))

  def test_path_tensor_copy_to_device(self):
    t = Tensor(self.test_file)
    t_cpu = t.to("CPU")
    self.assertEqual(t_cpu.device, "CPU")
    np.testing.assert_array_equal(t_cpu.numpy(), np.frombuffer(self.test_data, dtype=np.uint8))

ggml_test_block_count = 4
ggml_type_to_np_dtype = {
  ggml.GGML_TYPE_F16: np.float16, ggml.GGML_TYPE_F32:np.float32, ggml.GGML_TYPE_F64:np.float64,
  ggml.GGML_TYPE_I8:np.int8, ggml.GGML_TYPE_I16: np.int16, ggml.GGML_TYPE_I32: np.int32, ggml.GGML_TYPE_I64: np.int64,
}
np_dtype_to_ctype = { np.float16: ctypes.c_uint16 }
gguf_val_getters = [
  ggml.gguf_get_val_u8, ggml.gguf_get_val_i8, ggml.gguf_get_val_u16, ggml.gguf_get_val_i16,
  ggml.gguf_get_val_u32, ggml.gguf_get_val_i32, ggml.gguf_get_val_f32, ggml.gguf_get_val_bool,
  lambda *args: ggml.gguf_get_val_str(*args).decode("utf-8"), None,
  ggml.gguf_get_val_u64, ggml.gguf_get_val_i64, ggml.gguf_get_val_f64,
]

def ggml_tensor_to_numpy(tensor: ggml.ggml_tensor_p):
  ctx: ggml.ggml_context_p | None = None
  ggml_type, n_dims, n_els = tensor.contents.type, ggml.ggml_n_dims(tensor), ggml.ggml_nelements(tensor)
  shape = tuple(reversed(tensor.contents.ne[:n_dims]))
  if ggml_type not in ggml_type_to_np_dtype:
    ctx = ggml.ggml_init(ggml.ggml_init_params(mem_size=n_els * 5 + 500, mem_buffer=None))
    ntensor = ggml.ggml_new_tensor(ctx, ggml.GGML_TYPE_F32, n_dims, tensor.contents.ne)
    type_traits = ggml.ggml_internal_get_type_traits(ggml_type)
    type_traits.to_float(ggml.ggml_get_data(tensor), ggml.ggml_get_data_f32(ntensor), n_els)
    tensor, ggml_type = ntensor, ggml.GGML_TYPE_F32

  np_type = ggml_type_to_np_dtype[ggml_type]
  ctypes_type = np_dtype_to_ctype.get(np_type, None) or np.ctypeslib.as_ctypes_type(np_type)
  data = ggml.ggml_get_data(tensor)
  if data is None: raise ValueError("tensor data is None")
  arr = (ctypes_type * ggml.ggml_nelements(tensor)).from_address(data)
  strides = tuple(reversed(tensor.contents.nb[:n_dims]))
  output = np.ctypeslib.as_array(arr)
  output.dtype = np_type
  return np.lib.stride_tricks.as_strided(output, shape=shape, strides=strides), ctx

@unittest.skipIf(any(not is_dtype_supported(t) for t in [ dtypes.uint8, dtypes.half ]), "Backend must support uint8 and half")
class TestGGUF(unittest.TestCase):
  def setUp(self) -> None:
    params = ggml.ggml_init_params(mem_size=0, mem_buffer=None, no_alloc=False)
    self.ctx = ctypes.cast(ggml.ggml_init(params), ctypes.POINTER(ctypes.c_void_p))
  def tearDown(self) -> None: ggml.ggml_free(self.ctx)

  def test_load_tinyllama_q8_0(self): self._test_load_gguf("https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q8_0.gguf?download=true")
  def test_load_tinyllama_q4_0(self): self._test_load_gguf("https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf?download=true")
  def test_load_gpt2_q4_1(self): self._test_load_gguf("https://huggingface.co/PrunaAI/gpt2-GGUF-smashed/resolve/main/gpt2.Q4_1.gguf?download=true")
  def test_load_sample_q6_k(self): self._test_load_gguf("https://huggingface.co/Isotr0py/test-gguf-sample/resolve/main/Quant_Q6_K_1024.gguf?download=true")

  def test_dequantization_q4_0(self): self._test_dequantization(ggml.GGML_TYPE_Q4_0)
  def test_dequantization_q4_1(self): self._test_dequantization(ggml.GGML_TYPE_Q4_1)
  def test_dequantization_q8_0(self): self._test_dequantization(ggml.GGML_TYPE_Q8_0)
  def test_dequantization_q6_k(self): self._test_dequantization(ggml.GGML_TYPE_Q6_K)

  def test_expected_failure_unknown_type(self):
    with self.assertRaises(ValueError):
      ggml_data_to_tensor(Tensor.empty(512, dtype=dtypes.uint8), 256, 1337)

  def _test_dequantization(self, ttype: int):
    type_traits = ggml.ggml_internal_get_type_traits(ttype)
    n_el, n_bytes = ggml_test_block_count * type_traits.blck_size, ggml_test_block_count * type_traits.type_size

    data_in = (np.random.random((n_el,)).astype(np.float32) * 100 - 50).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    c_q_data, c_dq_data = (ctypes.c_char * n_bytes)(0), (ctypes.c_float * n_el)(0)
    type_traits.from_float(data_in, c_q_data, n_el)
    type_traits.to_float(c_q_data, c_dq_data, n_el)

    q_tensor = Tensor(np.frombuffer(c_q_data, dtype=np.uint8, count=n_bytes))
    dq_tensor = ggml_data_to_tensor(q_tensor, n_el, ttype).reshape(n_el)

    np.testing.assert_equal(dq_tensor.numpy(), np.frombuffer(c_dq_data, dtype=np.float32))
  def _test_load_gguf(self, url: str):
    fp = fetch(url)
    model_size = os.stat(fp).st_size
    gguf_tensor = Tensor.empty(model_size, dtype=dtypes.uint8, device=f"disk:{fp}").to(Device.DEFAULT)
    kv_data, tensors = gguf_load(gguf_tensor)

    gguf_params = ggml.gguf_init_params(ctx=self.ctx, no_alloc=False)
    gguf_ctx = ggml.gguf_init_from_file(str(fp).encode("utf8"), gguf_params)
    param_ctx = gguf_params.ctx.contents.value

    for ggml_tensor_idx in range(ggml.gguf_get_n_tensors(gguf_ctx)):
      tensor_name = ggml.gguf_get_tensor_name(gguf_ctx, ggml_tensor_idx)
      ggml_tensor = ggml.ggml_get_tensor(param_ctx, tensor_name)
      ggml_tensor_numpy, temp_ctx = ggml_tensor_to_numpy(ggml_tensor)
      tensor = tensors.get(tensor_name.decode("utf-8"))
      np.testing.assert_equal(tensor.numpy(), ggml_tensor_numpy)
      if temp_ctx is not None: ggml.ggml_free(temp_ctx)

    for gguf_key_id in range(ggml.gguf_get_n_kv(gguf_ctx)):
      v = kv_data[ggml.gguf_get_key(gguf_ctx, gguf_key_id).decode("utf-8")]
      v_type = ggml.gguf_get_kv_type(gguf_ctx, gguf_key_id)
      if (get_fn := gguf_val_getters[v_type]) is not None: self.assertEqual(get_fn(gguf_ctx, gguf_key_id), v)

    ggml.gguf_free(gguf_ctx)

if __name__ == "__main__":
  unittest.main()
