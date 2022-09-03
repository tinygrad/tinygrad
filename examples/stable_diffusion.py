# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md
# this is sd-v1-4.ckpt
FILENAME = "/Users/kafka/fun/mps/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"

from extra.utils import fake_torch_load_zipped, get_child
from tinygrad.nn import Conv2d
from tinygrad.tensor import Tensor

dat = fake_torch_load_zipped(open(FILENAME, "rb"), load_weights=False)

class Normalize:
  def __init__(self, in_channels, num_groups=32):
    self.weight = Tensor.uniform(in_channels)
    self.bias = Tensor.uniform(in_channels)

  def forward(self, x):
    # TODO: write groupnorm
    pass

class AttnBlock:
  def __init__(self, in_channels):
    self.norm = Normalize(in_channels)
    self.q = Conv2d(in_channels, in_channels, 1)
    self.k = Conv2d(in_channels, in_channels, 1)
    self.v = Conv2d(in_channels, in_channels, 1)
    self.proj_out = Conv2d(in_channels, in_channels, 1)

class ResnetBlock:
  def __init__(self, in_channels, out_channels=None):
    self.norm1 = Normalize(in_channels)
    self.conv1 = Conv2d(in_channels, out_channels, 3)
    self.norm2 = Normalize(out_channels)
    self.conv2 = Conv2d(out_channels, out_channels, 3)
    if in_channels != out_channels:
      self.nin_shortcut = Conv2d(in_channels, out_channels, 1)

class Decoder:
  def __init__(self):
    sz = [(128, 256), (256, 512), (512, 512), (512, 512)]
    self.conv_in = Conv2d(4,512,3)

    arr = []
    for i,s in enumerate(sz):
      x = {}
      x['upsample'] = {"conv": Conv2d(s[0], s[0], 3, stride=2, padding=(0,1,0,1))}
      x['block'] = [ResnetBlock(s[1], s[0]),
      ResnetBlock(s[0], s[0]),
      ResnetBlock(s[0], s[0])]
      arr.append(x)
    self.up = arr

    block_in = 512
    self.mid = {
      "block_1": ResnetBlock(block_in, block_in),
      "attn_1": AttnBlock(block_in),
      "block_2": ResnetBlock(block_in, block_in),
    }

    self.norm_out = Normalize(128)
    self.conv_out = Conv2d(128, 3, 3)

class Encoder:
  def __init__(self, decode=False):
    sz = [(128, 128), (128, 256), (256, 512), (512, 512)]
    self.conv_in = Conv2d(3,128,3)

    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":
        [ResnetBlock(s[0], s[1]),
         ResnetBlock(s[1], s[1])]})
      if i != 3: arr[-1]['downsample'] = {"conv": Conv2d(s[1], s[1], 3, stride=2, padding=(0,1,0,1))}
    self.down = arr

    block_in = 512
    self.mid = {
      "block_1": ResnetBlock(block_in, block_in),
      "attn_1": AttnBlock(block_in),
      "block_2": ResnetBlock(block_in, block_in),
    }

    self.norm_out = Normalize(block_in)
    self.conv_out = Conv2d(block_in, 8, 3)

class AutoencoderKL:
  def __init__(self):
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.quant_conv = Conv2d(8, 8, 1)
    self.post_quant_conv = Conv2d(4, 4, 1)

class StableDiffusion:
  def __init__(self):
    self.first_stage_model = AutoencoderKL()

model = StableDiffusion()

for k,v in dat['state_dict'].items():
  try:
    w = get_child(model, k)
  except (AttributeError, KeyError, IndexError):
    w = None 
  print(f"{str(v.shape):30s}", w, k)
  if w is not None:
    assert w.shape == v.shape


# ** ldm.models.autoencoder.AutoencoderKL
# 3x512x512 <--> 4x64x64 (16384)
# decode torch.Size([1, 4, 64, 64]) torch.Size([1, 3, 512, 512])
# section 4.3 of paper
# first_stage_model.encoder
# first_stage_model.decoder

# ** ldm.modules.diffusionmodules.openaimodel.UNetModel
# this is what runs each time to sample. is this the LDM?
# input:  4x64x64
# output: 4x64x64
# model.diffusion_model
# it has attention?

# ** ldm.modules.encoders.modules.FrozenCLIPEmbedder
# cond_stage_model.transformer.text_model

