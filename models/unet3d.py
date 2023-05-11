# https://github.com/wolny/pytorch-3dunet
from pathlib import Path
from extra.utils import download_file, fake_torch_load
import tinygrad.nn as nn

class SingleConv:
  def __init__(self, in_channels, out_channels):
    self.groupnorm = nn.GroupNorm(1, in_channels) # 1 group?
    self.conv = nn.Conv2d(in_channels, out_channels, (3,3,3), bias=False)
  def __call__(self, x):
    return self.conv(self.groupnorm(x)).relu()

def get_basic_module(c0, c1, c2): return {"SingleConv1": SingleConv(c0, c1), "SingleConv2": SingleConv(c1, c2)}

class UNet3D:
  def __init__(self):
    ups = [16,32,64,128,256]
    self.encoders = [get_basic_module(ups[i] if i != 0 else 1, ups[i], ups[i+1]) for i in range(4)]
    self.decoders = [get_basic_module(ups[-1-i] + ups[-2+i], ups[-2+i], ups[-2+i]) for i in range(3)]
    self.final_conv = nn.Conv2d(32, 1, (1,1,1))

  def __call__(self, x):
    # TODO: make 2D conv generic for 3D, might already work with kernel_size=(3,3,3)
    pass

  def load_from_pretrained(self):
    fn = Path(__file__).parent.parent / "weights/unet-3d.ckpt"
    download_file("https://oc.embl.de/index.php/s/61s67Mg5VQy7dh9/download?path=%2FLateral-Root-Primordia%2Funet_bce_dice_ds1x&files=best_checkpoint.pytorch", fn)
    state = fake_torch_load(open(fn, "rb").read())['model_state_dict']
    for x in state.keys():
      print(x, state[x].shape)
