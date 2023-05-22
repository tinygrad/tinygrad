from pathlib import Path
from tinygrad import nn
from extra.utils import download_file, get_child

class BasicModule:
  def __init__(self, c0, c1):
    # TODO: padding=(1, 1, 1, 1, 1, 1)?
    self.conv1 = [nn.Conv2d(c0, c1, kernel_size=(3, 3, 3), bias=False), nn.GroupNorm(1, c1)]
    self.conv2 = [nn.Conv2d(c1, c1, kernel_size=(3, 3, 3), bias=False), nn.GroupNorm(1, c1)]

  def __call__(self):
    return self.conv2(self.conv1(x).relu()).relu()

class Upsample:
  def __init__(self, c0, c1, c2):
    self.upsample_conv = [nn.Conv2d(c1, c0, kernel_size=(2, 2, 2))]  # TODO: probably ConvTranspose
    self.conv1 = [nn.Conv2d(c1 * 2, c1, kernel_size=(3, 3, 3), bias=False), nn.GroupNorm(1, c1)]
    self.conv2 = [nn.Conv2d(c1, c1, kernel_size=(3, 3, 3), bias=False), nn.GroupNorm(1, c1)]

class UNet3D:
  def __init__(self):
    ups = [32, 32, 64, 128, 256, 320, 320]
    self.input_block = BasicModule(1, ups[0])
    self.downsample = [BasicModule(ups[i], ups[i+1]) for i in range(1, 5)]
    self.bottleneck = BasicModule(ups[-2], ups[-1])
    self.upsample = [Upsample(ups[-1-i], ups[-2-i], ups[-3-i]) for i in range(5)] 
    self.output = {"conv": nn.Conv2d(32, 3, (1, 1, 1))}

  def __call__(self, x):
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
