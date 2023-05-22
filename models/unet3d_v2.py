from pathlib import Path
from tinygrad import nn
from tinygrad.tensor import Tensor
from extra.utils import download_file, get_child

class DownsampleBlock:
  def __init__(self, c0, c1):
    self.conv1 = [nn.Conv2d(c0, c1, kernel_size=(3, 3, 3), padding=(1, 1, 1, 1, 1, 1), bias=False), nn.GroupNorm(1, c1), Tensor.relu]
    self.conv2 = [nn.Conv2d(c1, c1, kernel_size=(3, 3, 3), padding=(1, 1, 1, 1, 1, 1), bias=False), nn.GroupNorm(1, c1), Tensor.relu]

  def __call__(self, x):
    return self.conv2(self.conv1(x))

class UpsampleBlock:
  def __init__(self, c0, c1, c2):
    self.upsample_conv = [nn.Conv2d(c1, c0, kernel_size=(2, 2, 2))]  # TODO: probably ConvTranspose
    self.conv1 = [nn.Conv2d(c1 * 2, c1, kernel_size=(3, 3, 3), bias=False), nn.GroupNorm(1, c1), Tensor.relu]
    self.conv2 = [nn.Conv2d(c1, c1, kernel_size=(3, 3, 3), bias=False), nn.GroupNorm(1, c1), Tensor.relu]

  def __call__(self, x):
    return x.sequential(self.upsample_conv + self.conv1 + self.conv2)

class UNet3D:
  def __init__(self):
    ups = [32, 32, 64, 128, 256, 320, 320]
    self.input_block = DownsampleBlock(1, ups[0])
    self.downsample = [DownsampleBlock(ups[i], ups[i+1]) for i in range(1, 5)]
    self.bottleneck = DownsampleBlock(ups[-2], ups[-1])
    self.upsample = [UpsampleBlock(ups[-1-i], ups[-2-i], ups[-3-i]) for i in range(5)] 
    self.output = {"conv": nn.Conv2d(32, 3, (1, 1, 1))}

  def __call__(self, x):
    x = self.input_block(x).sequential(downsample)
    x = self.bottleneck(x).sequential(upsample)
    x = self.output(x)
    return x
    
  def load_from_pretrained(self):
    fn = Path(__file__).parent.parent / "weights" / "unet-3d.ckpt"
    download_file("https://zenodo.org/record/5597155/files/3dunet_kits19_pytorch.ptc?download=1", fn)
    import torch
    # TODO: replace with fake_torch_load
    state_dict = torch.jit.load(fn, map_location=torch.device("cpu")).state_dict()
    passed = 0  # DEBUG
    for k, v in state_dict.items():
      try:  # DEBUG
        obj = get_child(self, k)
        assert obj.shape == v.shape, (k, obj.shape, v.shape)
        obj.assign(v.numpy())
        passed += 1  # DEBUG
      except Exception as error:
        print((k, str(error)))
    print(f"{passed}/{len(state_dict)} passed")

if __name__ == "__main__":
  mdl = UNet3D()
  mdl.load_from_pretrained()
