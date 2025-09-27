import unittest, os
from tempfile import TemporaryDirectory
from tinygrad import Device
from tinygrad.helpers import getenv
from extra.models.unet import UNetModel
from examples.stable_diffusion import mlperf_params
from tinygrad.nn.state import get_state_dict, safe_save
from examples.mlperf.model_eval import eval_stable_diffusion

class TestEval(unittest.TestCase):
  def test_eval_ckpt(self):
    Device.DEFAULT="NULL"
    os.environ.update({"MODEL": "stable_diffusion", "GPUS": "8", "PARALLEL": "0", "EVAL_SAMPLES": "600"})
    os.environ.update({"CONTEXT_BS": "816", "DENOISE_BS": "600", "DECODE_BS": "384", "INCEPTION_BS": "560", "CLIP_BS": "240"})
    # NOTE: update these based on where data/checkpoints are on your system
    if not getenv("DATADIR", ""): os.environ["DATADIR"] = "/raid/datasets/stable_diffusion"
    if not getenv("CKPTDIR", ""): os.environ["CKPTDIR"] = "/raid/weights/stable_diffusion"
    with TemporaryDirectory(prefix="test-eval") as tmp:
      os.environ["EVAL_CKPT_DIR"] = tmp
      safe_save(get_state_dict(UNetModel(**mlperf_params)), f"{tmp}/0.safetensors")
      clip, fid, ckpt = eval_stable_diffusion()
    assert ckpt == 0
    assert clip == 0
    assert fid > 0 and fid < 1000

if __name__=="__main__":
  unittest.main()