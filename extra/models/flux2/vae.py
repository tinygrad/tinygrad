"""FLUX.2-klein conv VAE decoder, ported to tinygrad from mflux.

Mirrors mflux's `Flux2VAE` / `Flux2Decoder` (an mlx port of the diffusers
`AutoencoderKL`-style FLUX.2 VAE) exactly, so the diffusers safetensors weights
(`decoder.*`, `bn.*`, `post_quant_conv.*`) load directly via `load_state_dict`.

Architecture (config.json verified):
  latent_channels=32, block_out_channels=[128,256,512,512], layers_per_block=2,
  out_channels=3, norm_num_groups=32, act=silu, scaling_factor=1.0, shift_factor=0.0.

Decodes on the default backend (METAL / CUDA / CPU).
"""
from __future__ import annotations

from tinygrad import Tensor
from tinygrad.helpers import fetch
from tinygrad.nn import Conv2d, GroupNorm, Linear
from tinygrad.nn.state import safe_load, load_state_dict
from extra.models.flux2 import HF_BASE


# diffusers / mflux VAE config for FLUX.2-klein
LATENT_CHANNELS = 32
BLOCK_OUT_CHANNELS = (128, 256, 512, 512)
LAYERS_PER_BLOCK = 2
OUT_CHANNELS = 3
NORM_NUM_GROUPS = 32
RESNET_EPS = 1e-6
BN_EPS = 1e-4
SCALING_FACTOR = 1.0
SHIFT_FACTOR = 0.0


class ResnetBlock2D:
  """mflux Flux2ResnetBlock2D: norm1 -> silu -> conv1 -> norm2 -> silu -> conv2 + (conv_shortcut?) residual."""
  def __init__(self, in_channels: int, out_channels: int, eps: float = RESNET_EPS, groups: int = NORM_NUM_GROUPS):
    self.norm1 = GroupNorm(groups, in_channels, eps=eps)
    self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.norm2 = GroupNorm(groups, out_channels, eps=eps)
    self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    # diffusers names the projection `conv_shortcut`; present only when channels change.
    self.conv_shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else None

  def __call__(self, x: Tensor) -> Tensor:
    h = self.conv1(self.norm1(x).silu())
    h = self.conv2(self.norm2(h).silu())
    residual = self.conv_shortcut(x) if self.conv_shortcut is not None else x
    return h + residual


class AttentionBlock:
  """mflux Flux2AttentionBlock: group_norm -> (q,k,v Linear, 1 head) SDPA -> to_out Linear + residual.

  Operates in NCHW; the channel axis is the attention feature dim, spatial is the sequence.
  Weight keys are diffusers-style: group_norm.*, to_q.*, to_k.*, to_v.*, to_out.0.*.
  """
  def __init__(self, channels: int, groups: int = NORM_NUM_GROUPS, eps: float = RESNET_EPS):
    self.group_norm = GroupNorm(groups, channels, eps=eps)
    self.to_q = Linear(channels, channels)
    self.to_k = Linear(channels, channels)
    self.to_v = Linear(channels, channels)
    # diffusers stores the output projection as `to_out.0` (a 1-element Sequential).
    self.to_out = [Linear(channels, channels)]

  def __call__(self, x: Tensor) -> Tensor:
    b, c, hgt, wdt = x.shape
    # NCHW -> NHWC so the channel axis is the feature dim for the Linear projections.
    normed = self.group_norm(x).permute(0, 2, 3, 1)
    q = self.to_q(normed).reshape(b, hgt * wdt, 1, c).permute(0, 2, 1, 3)
    k = self.to_k(normed).reshape(b, hgt * wdt, 1, c).permute(0, 2, 1, 3)
    v = self.to_v(normed).reshape(b, hgt * wdt, 1, c).permute(0, 2, 1, 3)
    attended = Tensor.scaled_dot_product_attention(q, k, v)  # scale defaults to 1/sqrt(head_dim)=1/sqrt(c)
    attended = attended.permute(0, 2, 1, 3).reshape(b, hgt, wdt, c)
    attended = self.to_out[0](attended).permute(0, 3, 1, 2)  # back to NCHW
    return x + attended


class UNetMidBlock2D:
  """mflux Flux2UNetMidBlock2D: resnet -> attention -> resnet."""
  def __init__(self, channels: int, eps: float = RESNET_EPS, groups: int = NORM_NUM_GROUPS, add_attention: bool = True):
    self.resnets = [ResnetBlock2D(channels, channels, eps=eps, groups=groups),
                    ResnetBlock2D(channels, channels, eps=eps, groups=groups)]
    self.attentions = [AttentionBlock(channels, groups=groups, eps=eps)] if add_attention else []

  def __call__(self, x: Tensor) -> Tensor:
    x = self.resnets[0](x)
    if self.attentions: x = self.attentions[0](x)
    return self.resnets[1](x)


class Upsample2D:
  """mflux Flux2Upsample2D: nearest 2x (repeat on H and W) then a 3x3 conv."""
  def __init__(self, channels: int, out_channels: int | None = None):
    out_channels = out_channels or channels
    self.conv = Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)

  def __call__(self, x: Tensor) -> Tensor:
    b, c, py, px = x.shape
    # nearest-neighbour 2x: matches mflux mx.repeat(x,2,axis=2) then axis=3.
    x = x.reshape(b, c, py, 1, px, 1).expand(b, c, py, 2, px, 2).reshape(b, c, py * 2, px * 2)
    return self.conv(x)


class UpDecoderBlock2D:
  """mflux Flux2UpDecoderBlock2D: num_layers resnets then an optional upsampler."""
  def __init__(self, in_channels: int, out_channels: int, num_layers: int, eps: float = RESNET_EPS,
               groups: int = NORM_NUM_GROUPS, add_upsample: bool = True):
    self.resnets = [ResnetBlock2D(in_channels if i == 0 else out_channels, out_channels, eps=eps, groups=groups)
                    for i in range(num_layers)]
    self.upsamplers = [Upsample2D(out_channels, out_channels)] if add_upsample else []

  def __call__(self, x: Tensor) -> Tensor:
    for resnet in self.resnets: x = resnet(x)
    for upsampler in self.upsamplers: x = upsampler(x)
    return x


class Flux2Decoder:
  """mflux Flux2Decoder: conv_in -> mid_block -> up_blocks -> conv_norm_out -> silu -> conv_out."""
  def __init__(self, in_channels: int = LATENT_CHANNELS, out_channels: int = OUT_CHANNELS,
               block_out_channels: tuple[int, ...] = BLOCK_OUT_CHANNELS, layers_per_block: int = LAYERS_PER_BLOCK,
               norm_num_groups: int = NORM_NUM_GROUPS, eps: float = RESNET_EPS):
    self.conv_in = Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
    self.mid_block = UNetMidBlock2D(block_out_channels[-1], eps=eps, groups=norm_num_groups, add_attention=True)

    self.up_blocks = []
    reversed_channels = list(reversed(block_out_channels))  # (512, 512, 256, 128)
    for i, output_channel in enumerate(reversed_channels):
      prev_output_channel = output_channel if i == 0 else reversed_channels[i - 1]
      is_final_block = i == len(reversed_channels) - 1
      self.up_blocks.append(UpDecoderBlock2D(
        in_channels=prev_output_channel, out_channels=output_channel,
        num_layers=layers_per_block + 1, eps=eps, groups=norm_num_groups,
        add_upsample=not is_final_block))

    self.conv_norm_out = GroupNorm(norm_num_groups, block_out_channels[0], eps=eps)
    self.conv_out = Conv2d(block_out_channels[0], out_channels, kernel_size=3, stride=1, padding=1)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.conv_in(x)
    x = self.mid_block(x)
    for up_block in self.up_blocks: x = up_block(x)
    x = self.conv_norm_out(x).silu()
    return self.conv_out(x)


class Flux2VAEDecoder:
  """Top-level FLUX.2 VAE decoder.

  Holds the buffers/modules whose weight keys live above `decoder.*` in the diffusers
  state dict: `bn.running_mean`, `bn.running_var`, and `post_quant_conv`. `decode_packed_latents`
  reproduces mflux's exact pipeline (bn un-normalize -> 2x2 unpatchify -> post_quant_conv -> decoder).
  """
  def __init__(self, latent_channels: int = LATENT_CHANNELS, bn_eps: float = BN_EPS):
    self.latent_channels = latent_channels
    self.bn_eps = bn_eps
    self.decoder = Flux2Decoder(in_channels=latent_channels)
    # 1x1 post-quant projection applied inside decode(), before the conv decoder.
    self.post_quant_conv = Conv2d(latent_channels, latent_channels, kernel_size=1, stride=1, padding=0)
    # BatchNorm running stats over 4*latent_channels features (the packed channel count).
    self.bn = _BatchNormStats(4 * latent_channels)

  def decode(self, latents: Tensor) -> Tensor:
    # latents/scaling_factor + shift_factor is a no-op (1.0 / 0.0) but kept for faithfulness.
    latents = latents / SCALING_FACTOR + SHIFT_FACTOR
    latents = self.post_quant_conv(latents)
    return self.decoder(latents)

  def decode_packed_latents(self, packed_latents: Tensor) -> Tensor:
    if packed_latents.ndim == 5:
      packed_latents = packed_latents[:, :, 0, :, :]
    # 1. bn un-normalize: latents = packed * std + mean, std = sqrt(var + eps).
    bn_mean = self.bn.running_mean.reshape(1, -1, 1, 1)
    bn_std = (self.bn.running_var.reshape(1, -1, 1, 1) + self.bn_eps).sqrt()
    latents = packed_latents * bn_std + bn_mean
    # 2. 2x2 unpatchify: (b, 4c, h, w) -> (b, c, 2h, 2w).
    latents = self._unpatchify_latents(latents)
    # 3. decode.
    return self.decode(latents)

  @staticmethod
  def _unpatchify_latents(latents: Tensor) -> Tensor:
    b, c, h, w = latents.shape
    # mlx: reshape (b, c//4, 2, 2, h, w) -> transpose (0,1,4,2,5,3) -> reshape (b, c//4, h*2, w*2).
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)

  def __call__(self, packed_latents: Tensor) -> Tensor:
    return self.decode_packed_latents(packed_latents)


class _BatchNormStats:
  """Holds bn.running_mean / bn.running_var so they bind from the state dict keys `bn.*`."""
  def __init__(self, num_features: int):
    self.running_mean = Tensor.zeros(num_features)
    self.running_var = Tensor.ones(num_features)


# fetch downloads on first use (symlink the local cache into the fetch dir to reuse it).
VAE_WEIGHTS_URL = HF_BASE + "vae/diffusion_pytorch_model.safetensors"


def load_vae_decoder() -> Flux2VAEDecoder:
  """Build a Flux2VAEDecoder and load the diffusers safetensors weights into it.

  The module tree (decoder.*, post_quant_conv.*, bn.running_mean/var) mirrors the diffusers
  key names exactly, so we filter the full state dict to those keys and `load_state_dict`.
  Encoder / quant_conv / bn.num_batches_tracked are dropped (unused by the decode path).
  """
  model = Flux2VAEDecoder()
  full = safe_load(fetch(VAE_WEIGHTS_URL, "diffusion_pytorch_model.safetensors", subdir="flux2-klein-4b/vae"))
  wanted = set(get_decoder_state_keys(model))
  filtered = {k: v for k, v in full.items() if k in wanted}
  load_state_dict(model, filtered, strict=True, verbose=False)
  return model


def get_decoder_state_keys(model: Flux2VAEDecoder) -> list[str]:
  from tinygrad.nn.state import get_state_dict
  return list(get_state_dict(model).keys())


def decode_packed_latents(packed_latents: Tensor, model: Flux2VAEDecoder | None = None) -> Tensor:
  """Convenience helper: decode packed latents with a (cached or freshly loaded) decoder."""
  if model is None:
    model = load_vae_decoder()
  return model.decode_packed_latents(packed_latents)


if __name__ == "__main__":
  # Numerical check vs the mflux reference if present (FLUX2_VAE_REF), else a CPU shape check.
  import os
  from tinygrad import Device, GlobalCounters
  print("default device:", Device.DEFAULT)

  model = load_vae_decoder()
  ref_path = os.environ.get("FLUX2_VAE_REF", "/tmp/flux2_vae_ref.safetensors")
  if os.path.exists(ref_path):
    ref = {k: v.to(Device.DEFAULT) for k, v in safe_load(ref_path).items()}
    packed = ref["packed_latents"].float()
    GlobalCounters.reset()
    out = model.decode_packed_latents(packed).realize()
    print(f"device {Device.DEFAULT}  decoded {out.shape}  kernels {GlobalCounters.kernel_count}")
    o, r = out.float().reshape(-1), ref["decoded_image"].float().reshape(-1)
    diff = (o - r).abs()
    cos = (o * r).sum() / (o.square().sum().sqrt() * r.square().sum().sqrt())
    print(f"vs mflux: cosine {cos.item():.6f}  max {diff.max().item():.5f}  mean {diff.mean().item():.6f}")
  else:
    # For a 128x128 image: VAE downsample 8 -> 16x16 latent -> 2x2 patchify -> packed (1,128,8,8).
    packed = Tensor.randn(1, 128, 8, 8)
    out = model.decode_packed_latents(packed).realize()
    print("output image shape:", out.shape)
    assert tuple(out.shape) == (1, 3, 128, 128), f"unexpected output shape {out.shape}"
    print("OK: output is (1, 3, 128, 128)")
