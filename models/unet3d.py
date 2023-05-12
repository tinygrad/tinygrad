# https://github.com/wolny/pytorch-3dunet
from pathlib import Path
from extra.utils import download_file, fake_torch_load, get_child
import tinygrad.nn as nn

class SingleConv:
  def __init__(self, in_channels, out_channels):
    self.groupnorm = nn.GroupNorm(1, in_channels) # 1 group?
    # TODO: make 2D conv generic for 3D, might already work with kernel_size=(3,3,3)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1,1,1,1), bias=False)
  def __call__(self, x):
    return self.conv(self.groupnorm(x)).relu()

class BasicModule:
  def __init__(self, c0, c1, c2):
    self.basic_module = {"SingleConv1": SingleConv(c0, c1), "SingleConv2": SingleConv(c1, c2)}
  def __call__(self, x):
    return self.basic_module['SingleConv2'](self.basic_module['SingleConv1'](x))

class UNet3D:
  def __init__(self):
    ups = [16,32,64,128,256]
    self.encoders = [BasicModule(ups[i] if i != 0 else 1, ups[i], ups[i+1]) for i in range(4)]
    self.decoders = [BasicModule(ups[-1-i] + ups[-2-i], ups[-2-i], ups[-2-i]) for i in range(3)]
    self.final_conv = nn.Conv2d(32, 1, (1,1,1))

  def __call__(self, x):
    intermediates = [x]
    for e in self.encoders: intermediates.append(e(intermediates[-1]))
    ret = intermediates[-1]
    for d,i in zip(self.decoders, intermediates[:-1][::-1]): ret = d(ret.cat(i, dim=1))
    return ret

  def load_from_pretrained(self):
    fn = Path(__file__).parent.parent / "weights/unet-3d.ckpt"
    download_file("https://oc.embl.de/index.php/s/61s67Mg5VQy7dh9/download?path=%2FLateral-Root-Primordia%2Funet_bce_dice_ds1x&files=best_checkpoint.pytorch", fn)
    state_dict = fake_torch_load(open(fn, "rb").read())['model_state_dict']
    for k, v in state_dict.items():
      print(k, v.shape)
      obj = get_child(self, k)
      assert obj.shape == v.shape, (k, obj.shape, v.shape)
      obj.assign(v.numpy())
