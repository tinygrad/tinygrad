# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md

import os
import numpy as np
from extra.utils import fake_torch_load_zipped, get_child
from tinygrad.nn import Conv2d
from tinygrad.tensor import Tensor

class Normalize:
  def __init__(self, in_channels, num_groups=32):
    self.weight = Tensor.uniform(in_channels)
    self.bias = Tensor.uniform(in_channels)
    self.num_groups = num_groups

  def __call__(self, x):
    print("norm", x.shape)
    x = x.reshape((x.shape[0], self.num_groups, x.shape[1]//self.num_groups, x.shape[2], x.shape[3]))

    # subtract mean
    x = x - x.mean(axis=(2,3,4), keepdim=True)

    # divide stddev
    eps = 1e-5
    x = x.div((x*x).mean(axis=(2,3,4), keepdim=True).add(eps).sqrt())

    # return to old shape
    x = x.reshape((x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]))

    x *= self.weight.reshape((1, -1, 1, 1))
    x += self.bias.reshape((1, -1, 1, 1))

    return x


class AttnBlock:
  def __init__(self, in_channels):
    self.norm = Normalize(in_channels)
    self.q = Conv2d(in_channels, in_channels, 1)
    self.k = Conv2d(in_channels, in_channels, 1)
    self.v = Conv2d(in_channels, in_channels, 1)
    self.proj_out = Conv2d(in_channels, in_channels, 1)

  # copied from AttnBlock in ldm repo
  def __call__(self, x):
    print("attention:", x.shape)
    h_ = self.norm(x)
    q,k,v = self.q(h_), self.k(h_), self.v(h_)

    # compute attention
    b,c,h,w = q.shape
    q = q.reshape((b,c,h*w))
    q = q.permute((0,2,1))   # b,hw,c
    k = k.reshape((b,c,h*w)) # b,c,hw
    w_ = q @ k
    w_ = w_ * (c**(-0.5))
    w_ = w_.softmax()

    # attend to values
    v = v.reshape((b,c,h*w))
    w_ = w_.permute((0,2,1))
    h_ = v @ w_
    h_ = h_.reshape((b,c,h,w))

    return x + self.proj_out(h_)

class ResnetBlock:
  def __init__(self, in_channels, out_channels=None):
    self.norm1 = Normalize(in_channels)
    self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
    self.norm2 = Normalize(out_channels)
    self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
    self.nin_shortcut = Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x

  def __call__(self, x):
    h = self.conv1(self.norm1(x).swish())
    h = self.conv2(self.norm2(h).swish())
    return self.nin_shortcut(x) + h

class Decoder:
  def __init__(self):
    sz = [(128, 256), (256, 512), (512, 512), (512, 512)]
    self.conv_in = Conv2d(4,512,3, padding=1)

    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":
        [ResnetBlock(s[1], s[0]),
         ResnetBlock(s[0], s[0]),
         ResnetBlock(s[0], s[0])]})
      if i != 0: arr[-1]['upsample'] = {"conv": Conv2d(s[0], s[0], 3, padding=1)}
    self.up = arr

    block_in = 512
    self.mid = {
      "block_1": ResnetBlock(block_in, block_in),
      "attn_1": AttnBlock(block_in),
      "block_2": ResnetBlock(block_in, block_in),
    }

    self.norm_out = Normalize(128)
    self.conv_out = Conv2d(128, 3, 3, padding=1)

  def __call__(self, x):
    x = self.conv_in(x)

    x = self.mid['block_1'](x)
    x = self.mid['attn_1'](x)
    x = self.mid['block_2'](x)

    for l in self.up[::-1]:
      print("decode", x.shape)
      for b in l['block']: x = b(x)
      if 'upsample' in l:
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html ?
        bs,c,py,px = x.shape
        x = x.reshape((bs, c, py, 1, px, 1))
        x = x.expand((bs, c, py, 2, px, 2))
        x = x.reshape((bs, c, py*2, px*2))
        x = l['upsample']['conv'](x)
    
    return self.conv_out(self.norm_out(x).swish())


class Encoder:
  def __init__(self, decode=False):
    sz = [(128, 128), (128, 256), (256, 512), (512, 512)]
    self.conv_in = Conv2d(3,128,3, padding=1)

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
    self.conv_out = Conv2d(block_in, 8, 3, padding=1)

  def __call__(self, x):
    x = self.conv_in(x)
    for l in self.down:
      print("encode", x.shape)
      for b in l['block']: x = b(x)
      if 'downsample' in l: x = l['downsample']['conv'](x)
  
    x = self.mid['block_1'](x)
    x = self.mid['attn_1'](x)
    x = self.mid['block_2'](x)

    return self.conv_out(self.norm_out(x).swish())

class AutoencoderKL:
  def __init__(self):
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.quant_conv = Conv2d(8, 8, 1)
    self.post_quant_conv = Conv2d(4, 4, 1)

  def __call__(self, x):
    latent = self.encoder(x)
    latent = self.quant_conv(latent)
    latent = latent[:, 0:4]  # only the means
    print("latent", latent.shape)
    latent = self.post_quant_conv(latent)
    return self.decoder(latent)

class StableDiffusion:
  def __init__(self):
    self.first_stage_model = AutoencoderKL()
  
  def __call__(self, x):
    return self.first_stage_model(x)

# ** ldm.models.autoencoder.AutoencoderKL (done!)
# 3x512x512 <--> 4x64x64 (16384)
# decode torch.Size([1, 4, 64, 64]) torch.Size([1, 3, 512, 512])
# section 4.3 of paper
# first_stage_model.encoder, first_stage_model.decoder

# ** ldm.modules.diffusionmodules.openaimodel.UNetModel
# this is what runs each time to sample. is this the LDM?
# input:  4x64x64
# output: 4x64x64
# model.diffusion_model
# it has attention?

# ** ldm.modules.encoders.modules.FrozenCLIPEmbedder
# cond_stage_model.transformer.text_model

# this is sd-v1-4.ckpt
#FILENAME = "/Users/kafka/fun/mps/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"
FILENAME = "/home/kafka/model.ckpt"
REAL = int(os.getenv("REAL", 0))

if __name__ == "__main__":
  model = StableDiffusion()

  # load in weights
  dat = fake_torch_load_zipped(open(FILENAME, "rb"), load_weights=REAL)
  for k,v in dat['state_dict'].items():
    try:
      w = get_child(model, k)
    except (AttributeError, KeyError, IndexError):
      w = None 
    print(f"{str(v.shape):30s}", w, k)
    if w is not None:
      assert w.shape == v.shape
      w.assign(v.astype(np.float32))

  if not REAL: exit(0)

  # load image
  #IMG = "/tmp/apple.png"
  #from PIL import Image
  #realimg = Tensor(np.array(Image.open(IMG))).permute((2,0,1)).reshape((1,3,512,512))*(1/255)
  #print(realimg.shape)
  #x = model(realimg)

  # load latent space
  nz = np.load("datasets/stable_diffusion_apple.npy")
  x = model.first_stage_model.post_quant_conv(Tensor(nz))
  x = model.first_stage_model.decoder(x)

  x = x.reshape((3,512,512)).permute((1,2,0))
  dat = (x.detach().numpy().clip(0, 1)*255).astype(np.uint8)
  print(dat.shape)

  from PIL import Image
  im = Image.fromarray(dat)
  im.save("/tmp/rendered.png")


# torch junk

#IMG = "/Users/kafka/fun/mps/stable-diffusion/outputs/txt2img-samples/grid-0006.png"
#from PIL import Image
#realimg = Tensor(np.array(Image.open(IMG))).permute((2,0,1)).reshape((1,3,512,512))*(1/255)
#print(img.shape)
#x = model(img)

#nz = np.random.randn(*nz.shape) * 100

# PYTHONPATH="$PWD:/Users/kafka/fun/mps/stable-diffusion" 
"""
from ldm.models.autoencoder import AutoencoderKL
import torch
ckpt = torch.load(FILENAME)
dat = ckpt['state_dict']
sd = {}
for k in dat:
  if k.startswith("first_stage_model."):
    sd[k[len("first_stage_model."):]] = dat[k]
print("loading", len(sd))

tmodel = AutoencoderKL(
  ddconfig = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1,2,4,4],
    "num_res_blocks": 2,
    "attn_resolutions": []
  },
  lossconfig={"target": "torch.nn.Identity"},
  embed_dim=4)
tmodel.load_state_dict(sd, strict=True)
nz = np.load("datasets/stable_diffusion_apple.npy")
zmodel = model.first_stage_model

x_torch = torch.tensor(nz)
x_tiny = Tensor(nz)

x_torch = tmodel.post_quant_conv(x_torch)
x_tiny = zmodel.post_quant_conv(x_tiny)

x_torch = tmodel.decoder.conv_in(x_torch)
x_tiny = zmodel.decoder.conv_in(x_tiny)

x_torch = tmodel.decoder.mid.block_1(x_torch, None)
x_tiny = zmodel.decoder.mid['block_1'](x_tiny)
"""

"""
x_torch = tmodel.decoder.mid.block_1.norm1(x_torch)
x_tiny = zmodel.decoder.mid['block_1'].norm1(x_tiny)

x_torch = x_torch * torch.sigmoid(x_torch)
x_tiny = x_tiny.swish()

print(zmodel.decoder.mid['block_1'].conv1.weight.shape)
print(x_tiny.shape)

x_torch = tmodel.decoder.mid.block_1.conv1(x_torch)
x_tiny = zmodel.decoder.mid['block_1'].conv1(x_tiny)
"""

#print(tmodel.decoder.mid.block_1.conv1.weight)
#print(zmodel.decoder.mid['block_1'].conv1.weight.numpy())

#print(abs(x_torch.detach().numpy() - x_tiny.numpy()).mean())
#print(x_torch.shape, x_tiny.shape)

#exit(0)


#exit(0)


"""
posterior = tmodel.encode(torch.tensor(realimg.numpy()))
z = posterior.mode()
print(z.shape)
#exit(0)
nz = z.detach().numpy()
np.save("/tmp/apple.npy", nz)
exit(0)
"""

#x, latent = tmodel(torch.tensor(realimg.numpy()))
#x = tmodel.decode(torch.tensor(nz))
#x = x.reshape(3,512,512).permute(1,2,0)

"""
x = Tensor.randn(1,4,64,64)
x = model.first_stage_model.post_quant_conv(x)
x = model.first_stage_model.decoder(x)

print(x.shape)
x = x.reshape((3,512,512)).permute((1,2,0))
print(x.shape)
if not REAL: exit(0)
"""

"""
#dat = (x.detach().numpy()*256).astype(np.uint8)
dat = (x.detach().numpy().clip(0, 1)*255).astype(np.uint8)
print(dat.shape)

from PIL import Image
im = Image.fromarray(dat)
im.save("/tmp/rendered.png")

"""
