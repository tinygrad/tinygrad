import unittest, math, os
from tempfile import TemporaryDirectory
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv
from examples.mlperf.lr_schedulers import LambdaLR, LambdaLinearScheduler
from examples.mlperf.model_train import train_stable_diffusion
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters

class TestLRScheduler(unittest.TestCase):
  def test_linear_lr_warmup(self):
    BS, BASE_LR = 304, 2.5e-7
    lr = BS * BASE_LR
    # Use a dummy Tensor parameter for optimizer because the lr_scheduler only needs the optimizer's device and lr, the params aren't touched.
    optimizer = AdamW(get_parameters([Tensor([1.])]))
    lambda_lr_callback = LambdaLinearScheduler(1000, 1.0, 1.0, 1e-06, 10000000000000).schedule
    lr_scheduler = LambdaLR(optimizer, Tensor(lr, dtype=dtypes.float, device=optimizer.device), lambda_lr_callback)
    lrs = {}

    # with above settings, optimizer.lr should warm up to lr over 1000 steps linearly
    for i in range(1200):
      lr_scheduler.step()
      if i in {0, 499, 998, 999, 1000, 1199}:
        lrs[i] = optimizer.lr.item()

    assert math.isclose(lr, lrs[999], rel_tol=0.0, abs_tol=1e-11)
    assert lrs[999] == lrs[1000] == lrs[1199]
    assert math.isclose(lrs[999] / lrs[0], 1000, rel_tol=0.0, abs_tol=1)
    assert math.isclose(lrs[999] / lrs[499], 2, rel_tol=0.0, abs_tol=1e-5)

class TestTrain(unittest.TestCase):
  def test_train_to_ckpt(self):
    Device.DEFAULT="NULL"
    # train for num_steps, save checkpoint, and stop training
    num_steps = 42
    os.environ.update({"MODEL": "stable_diffusion", "TOTAL_CKPTS": "1", "CKPT_STEP_INTERVAL": str(num_steps), "GPUS": "8", "BS": "304", "PARALLEL": "0"})
    # NOTE: update these based on where data/checkpoints are on your system
    if not getenv("DATADIR", ""): os.environ["DATADIR"] = "/raid/datasets/stable_diffusion"
    if not getenv("CKPTDIR", ""): os.environ["CKPTDIR"] = "/raid/weights/stable_diffusion"
    with TemporaryDirectory(prefix="test-train") as tmp:
      os.environ["UNET_CKPTDIR"] = tmp
      with Tensor.train():
        saved_ckpts = train_stable_diffusion()
      expected_ckpt = f"{tmp}/{num_steps}.safetensors"
      assert len(saved_ckpts) == 1 and saved_ckpts[0] == expected_ckpt

if __name__=="__main__":
  unittest.main()