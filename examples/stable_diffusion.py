# This file (and it's imports) incorporates code from the following:
# Github Name                    | License | Link
# Stability-AI/generative-models | MIT     | https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/LICENSE-CODE
# mlfoundations/open_clip        | MIT     | https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/LICENSE

from tinygrad import Tensor, TinyJit, dtypes, GlobalCounters, Device
from tinygrad.helpers import tqdm, Timing, fetch, Context, getenv, colored, THREEFRY
from tinygrad.nn.state import safe_load, load_state_dict, torch_load, get_state_dict
from tinygrad.nn import Conv2d, GroupNorm

from extra.models.clip import Embedder, FrozenClosedClipEmbedder, FrozenOpenClipEmbedder
from extra.models.unet import UNetModel, timestep_embedding

from typing import Dict, Tuple, List, Set, Optional, Type, Union
import os, argparse, tempfile, re
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
import numpy as np


def tensor_identity(x:Tensor) -> Tensor:
  return x

def append_dims(x:Tensor, t:Tensor) -> Tensor:
  dims_to_append = len(t.shape) - len(x.shape)
  assert dims_to_append >= 0
  return x.reshape(x.shape + (1,)*dims_to_append)

def get_alphas_cumprod(beta_start=0.00085, beta_end=0.0120, n_training_steps=1000):
  betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps, dtype=np.float32) ** 2
  alphas = 1.0 - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  return alphas_cumprod


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/discretizer.py#L42
class LegacyDDPMDiscretization:
  def __init__(self, linear_start:float=0.00085, linear_end:float=0.0120, num_timesteps:int=1000):
    self.num_timesteps = num_timesteps
    self.alphas_cumprod = get_alphas_cumprod(linear_start, linear_end, num_timesteps)

  def __call__(self, n:int, flip:bool=False) -> Tensor:
    if n < self.num_timesteps:
      timesteps = np.linspace(self.num_timesteps - 1, 0, n, endpoint=False).astype(int)[::-1]
      alphas_cumprod = self.alphas_cumprod[timesteps]
    elif n == self.num_timesteps:
      alphas_cumprod = self.alphas_cumprod
    sigmas = Tensor((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    sigmas = Tensor.cat(Tensor.zeros((1,)), sigmas)
    return sigmas if flip else sigmas.flip(axis=0) # sigmas is "pre-flipped", need to do oposite of flag


class DiffusionModel:
  def __init__(self, *args, **kwargs):
    self.diffusion_model = UNetModel(*args, **kwargs)


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L913
class ConcatTimestepEmbedderND(Embedder):
  def __init__(self, input_key:str, outdim:int=256):
    self.outdim = outdim
    self.input_key = input_key

  def __call__(self, x:Union[str,Tensor]):
    assert isinstance(x, Tensor)
    assert len(x.shape) == 2
    emb = timestep_embedding(x.flatten(), self.outdim)
    emb = emb.reshape((x.shape[0],-1))
    return emb

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L71
class GeneralConditioner:
  OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
  KEY2CATDIM      = {"vector": 1, "crossattn": 2, "concat": 1}
  embedders: List[Embedder]

  def __init__(self, embedders:List[Dict]):
    self.embedders = []
    for emb in embedders:
      self.embedders.append(emb["class"](**emb["args"]))

  def get_keys(self) -> Set[str]:
    return set(e.input_key for e in self.embedders)

  def __call__(self, batch:Dict, force_zero_embeddings:List=[]) -> Dict[str,Tensor]:
    output: Dict[str,Tensor] = {}

    for embedder in self.embedders:
      emb_out = embedder(batch[embedder.input_key])

      if isinstance(emb_out, Tensor):
        emb_out = [emb_out]
      else:
        assert isinstance(emb_out, (list, tuple))

      for emb in emb_out:
        if embedder.input_key in force_zero_embeddings:
          emb = Tensor.zeros_like(emb)

        out_key = self.OUTPUT_DIM2KEYS[len(emb.shape)]
        if out_key in output:
          output[out_key] = Tensor.cat(output[out_key], emb, dim=self.KEY2CATDIM[out_key])
        else:
          output[out_key] = emb

    return output


class FirstStage:
  """
  Namespace for FirstStageModel components.
  """

  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L136
  class ResnetBlock:
    def __init__(self, in_channels, out_channels=None):
      self.norm1 = GroupNorm(32, in_channels)
      self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
      self.norm2 = GroupNorm(32, out_channels)
      self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
      self.nin_shortcut = Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x

    def __call__(self, x):
      h = self.conv1(self.norm1(x).swish())
      h = self.conv2(self.norm2(h).swish())
      return self.nin_shortcut(x) + h

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L74
  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L102
  class Downsample:
    def __init__(self, dims:int):
      self.conv = Conv2d(dims, dims, kernel_size=3, stride=2, padding=(0,1,0,1))

    def __call__(self, x:Tensor) -> Tensor:
      return self.conv(x)

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L58
  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L83
  class Upsample:
    def __init__(self, dims:int):
      self.conv = Conv2d(dims, dims, kernel_size=3, stride=1, padding=1)

    def __call__(self, x:Tensor) -> Tensor:
      B,C,Y,X = x.shape
      x = x.reshape(B, C, Y, 1, X, 1).expand(B, C, Y, 2, X, 2).reshape(B, C, Y*2, X*2)
      return self.conv(x)

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L204
  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L17
  class AttnBlock:
    def __init__(self, dim:int):
      self.norm = GroupNorm(32, dim)
      self.q = Conv2d(dim, dim, 1)
      self.k = Conv2d(dim, dim, 1)
      self.v = Conv2d(dim, dim, 1)
      self.proj_out = Conv2d(dim, dim, 1)

    # copied from AttnBlock in ldm repo
    def __call__(self, x:Tensor) -> Tensor:
      h_ = self.norm(x)
      q,k,v = self.q(h_), self.k(h_), self.v(h_)

      # compute attention
      b,c,h,w = q.shape
      q,k,v = [x.reshape(b,c,h*w).transpose(1,2) for x in (q,k,v)]
      h_ = Tensor.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,c,h,w)
      return x + self.proj_out(h_)

  class MidEntry:
    def __init__(self, block_in:int):
      self.block_1 = FirstStage.ResnetBlock(block_in, block_in)
      self.attn_1  = FirstStage.AttnBlock  (block_in)
      self.block_2 = FirstStage.ResnetBlock(block_in, block_in)

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L487
  class Encoder:
    def __init__(self, ch:int, in_ch:int, out_ch:int, z_ch:int, ch_mult:List[int], num_res_blocks:int, resolution:int):
      self.conv_in = Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
      in_ch_mult = (1,) + tuple(ch_mult)

      class BlockEntry:
        def __init__(self, block:List[FirstStage.ResnetBlock], downsample):
          self.block = block
          self.downsample = downsample
      self.down: List[BlockEntry] = []
      for i_level in range(len(ch_mult)):
        block = []
        block_in  = ch * in_ch_mult[i_level]
        block_out = ch * ch_mult   [i_level]
        for _ in range(num_res_blocks):
          block.append(FirstStage.ResnetBlock(block_in, block_out))
          block_in = block_out

        downsample = tensor_identity if (i_level == len(ch_mult)-1) else FirstStage.Downsample(block_in)
        self.down.append(BlockEntry(block, downsample))

      self.mid = FirstStage.MidEntry(block_in)

      self.norm_out = GroupNorm(32, block_in)
      self.conv_out = Conv2d(block_in, 2*z_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, x:Tensor) -> Tensor:
      h = self.conv_in(x)
      for down in self.down:
        for block in down.block:
          h = block(h)
        h = down.downsample(h)

      h = h.sequential([self.mid.block_1, self.mid.attn_1, self.mid.block_2])
      h = h.sequential([self.norm_out,    Tensor.swish,    self.conv_out   ])
      return h

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/model.py#L604
  class Decoder:
    def __init__(self, ch:int, in_ch:int, out_ch:int, z_ch:int, ch_mult:List[int], num_res_blocks:int, resolution:int):
      block_in = ch * ch_mult[-1]
      curr_res = resolution // 2 ** (len(ch_mult) - 1)
      self.z_shape = (1, z_ch, curr_res, curr_res)

      self.conv_in = Conv2d(z_ch, block_in, kernel_size=3, stride=1, padding=1)

      self.mid = FirstStage.MidEntry(block_in)

      class BlockEntry:
        def __init__(self, block:List[FirstStage.ResnetBlock], upsample):
          self.block = block
          self.upsample = upsample
      self.up: List[BlockEntry] = []
      for i_level in reversed(range(len(ch_mult))):
        block = []
        block_out = ch * ch_mult[i_level]
        for _ in range(num_res_blocks + 1):
          block.append(FirstStage.ResnetBlock(block_in, block_out))
          block_in = block_out

        upsample = tensor_identity if i_level == 0 else FirstStage.Upsample(block_in)
        self.up.insert(0, BlockEntry(block, upsample))

      self.norm_out = GroupNorm(32, block_in)
      self.conv_out = Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, z:Tensor) -> Tensor:
      h = z.sequential([self.conv_in, self.mid.block_1, self.mid.attn_1, self.mid.block_2])

      for up in self.up[::-1]:
        for block in up.block:
          h = block(h)
        h = up.upsample(h)

      h = h.sequential([self.norm_out, Tensor.swish, self.conv_out])
      return h

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L102
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L437
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/autoencoder.py#L508
class FirstStageModel:
  def __init__(self, embed_dim:int=4, **kwargs):
    self.encoder = FirstStage.Encoder(**kwargs)
    self.decoder = FirstStage.Decoder(**kwargs)
    self.quant_conv = Conv2d(2*kwargs["z_ch"], 2*embed_dim, 1)
    self.post_quant_conv = Conv2d(embed_dim, kwargs["z_ch"], 1)

  def decode(self, z:Tensor) -> Tensor:
    return z.sequential([self.post_quant_conv, self.decoder])


class VanillaCFG:
  def __init__(self, scale:float):
    self.scale = scale

  def prepare_inputs(self, x:Tensor, s:Optional[Tensor], c:Dict, uc:Dict) -> Tuple[Tensor,Tensor,Tensor]:
    c_out = {}
    for k in c:
      assert k in ["vector", "crossattn", "concat"]
      c_out[k] = Tensor.cat(uc[k], c[k], dim=0)
    return Tensor.cat(x, x), (None if s is None else Tensor.cat(s, s)), c_out

  def __call__(self, x:Tensor, sigma:float) -> Tensor:
    x_u, x_c = x.chunk(2)
    x_pred = x_u + self.scale*(x_c - x_u)
    return x_pred

class Sampler(ABC):
  @abstractmethod
  def __init__(self, cfg_scale:float, timing:bool):
    pass
  @abstractmethod
  def __call__(self, denoiser, x:Tensor, c:Dict, uc:Dict, num_steps:int) -> Tensor:
    pass

class Samplers:
  """
  Namespace for Stable Diffusion samplers.
  """

  class SDv1Sampler(Sampler):
    def __init__(self, cfg_scale:float, timing:bool):
      self.cfg_scale = cfg_scale
      self.timing = timing

      self.discretization = LegacyDDPMDiscretization()
      self.guider = VanillaCFG(cfg_scale)
      self.alphas_cumprod = get_alphas_cumprod()

    def __call__(self, denoiser, x:Tensor, c:Dict, uc:Dict, num_steps:int) -> Tensor:
      timesteps   = list(range(1, 1000, 1000//num_steps))
      alphas      = Tensor(self.alphas_cumprod[timesteps])
      alphas_prev = Tensor([1.0]).cat(alphas[:-1])

      for index, timestep in (t:=tqdm(list(enumerate(timesteps))[::-1])):
        GlobalCounters.reset()
        t.set_description(f"{index:3d} {timestep:3d}")
        with Timing("step in ", enabled=self.timing, on_exit=lambda _: f", using {GlobalCounters.mem_used/1e9:.2f} GB"):
          tid        = Tensor([index])
          alpha      = alphas     [tid]
          alpha_prev = alphas_prev[tid]

          latents, _, cond = self.guider.prepare_inputs(x, None, c, uc)
          latents = denoiser(latents, Tensor([timestep]), cond)
          uc_latent, c_latent = latents[0:1], latents[1:2]
          e_t = uc_latent + self.cfg_scale * (c_latent - uc_latent)

          sqrt_one_minus_at = (1 - alpha).sqrt()
          pred_x0 = (x - sqrt_one_minus_at * e_t) / alpha.sqrt()
          dir_xt = (1. - alpha_prev).sqrt() * e_t
          x = alpha_prev.sqrt() * pred_x0 + dir_xt

          if self.timing: Device[Device.DEFAULT].synchronize()

      return x

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/sampling.py#L21
  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/sampling.py#L287
  class DPMPP2MSampler(Sampler):
    def __init__(self, cfg_scale:float, timing:bool):
      self.timing = timing
      self.discretization = LegacyDDPMDiscretization()
      self.guider = VanillaCFG(cfg_scale)

    def sampler_step(self, old_denoised:Optional[Tensor], prev_sigma:Optional[Tensor], sigma:Tensor, next_sigma:Tensor, denoiser, x:Tensor, c:Dict, uc:Dict) -> Tuple[Tensor,Tensor]:
      denoised = denoiser(*self.guider.prepare_inputs(x, sigma, c, uc))
      denoised = self.guider(denoised, sigma)

      t, t_next = sigma.log().neg(), next_sigma.log().neg()
      h = t_next - t
      r = None if prev_sigma is None else (t - prev_sigma.log().neg()) / h

      mults = [t_next.neg().exp()/t.neg().exp(), (-h).exp().sub(1)]
      if r is not None:
        mults.extend([1 + 1/(2*r), 1/(2*r)])
      mults = [append_dims(m, x) for m in mults]

      x_standard = mults[0]*x - mults[1]*denoised
      if (old_denoised is None) or (next_sigma.sum().numpy().item() < 1e-14):
        return x_standard, denoised

      denoised_d = mults[2]*denoised - mults[3]*old_denoised
      x_advanced = mults[0]*x        - mults[1]*denoised_d
      x = Tensor.where(append_dims(next_sigma, x) > 0.0, x_advanced, x_standard)
      return x, denoised

    def __call__(self, denoiser, x:Tensor, c:Dict, uc:Dict, num_steps:int) -> Tensor:
      sigmas = self.discretization(num_steps)
      x *= Tensor.sqrt(1.0 + sigmas[0] ** 2.0)

      old_denoised = None
      for i in (t:=tqdm(range(len(sigmas) - 1))):
        GlobalCounters.reset()
        t.set_description(f"{i:3d}")
        with Timing("step in ", enabled=self.timing, on_exit=lambda _: f", using {GlobalCounters.mem_used/1e9:.2f} GB"):
          x, old_denoised = self.sampler_step(
            old_denoised=old_denoised,
            prev_sigma=(None if i==0 else sigmas[i-1].reshape(x.shape[0])),
            sigma=sigmas[i].reshape(x.shape[0]),
            next_sigma=sigmas[i+1].reshape(x.shape[0]),
            denoiser=denoiser,
            x=x,
            c=c,
            uc=uc,
          )
          x.realize()
          old_denoised.realize()

          if self.timing: Device[Device.DEFAULT].synchronize()

      return x


def prep_for_jit(*tensors:Tensor) -> Tuple[Tensor,...]:
  return tuple(t.cast(dtypes.float16).realize() for t in tensors)

class StableDiffusion(ABC):
  samplers: Dict[str,Type[Sampler]] # the first entry in the dict is considered default
  @abstractmethod
  def create_conditioning(self, pos_prompt:str, img_width:int, img_height:int, aesthetic_score:float) -> Tuple[Dict,Dict]:
    pass
  @abstractmethod
  def denoise(self, x:Tensor, sigmas_or_tms:Tensor, cond:Dict) -> Tensor:
    pass
  @abstractmethod
  def decode(self, x:Tensor) -> Tensor:
    pass
  @abstractmethod
  def delete_conditioner(self) -> None:
    pass

class StableDiffusionV1(StableDiffusion):
  samplers = {
    "basic": Samplers.SDv1Sampler,
  }

  def __init__(self, first_stage:Dict, model:Dict, num_timesteps:int):
    self.cond_stage_model = FrozenClosedClipEmbedder()
    self.first_stage_model = FirstStageModel(**first_stage)
    self.model = DiffusionModel(**model)

    disc = LegacyDDPMDiscretization()
    self.sigmas = disc(num_timesteps, flip=True)
    self.alphas_cumprod = disc.alphas_cumprod

  def create_conditioning(self, pos_prompt:str, img_width:int, img_height:int, aesthetic_score:float) -> Tuple[Dict,Dict]:
    return {"crossattn": self.cond_stage_model(pos_prompt)}, {"crossattn": self.cond_stage_model("")}

  def denoise(self, x:Tensor, tms:Tensor, cond:Dict) -> Tensor:
    @TinyJit
    def run(x, tms, ctx):
      return self.model.diffusion_model(x, tms, ctx, None).realize()

    return run(*prep_for_jit(x, tms, cond["crossattn"]))

  def decode(self, x:Tensor) -> Tensor:
    return self.first_stage_model.decode(1 / 0.18215 * x)

  def delete_conditioner(self) -> None:
    del self.cond_stage_model

# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/diffusion.py#L19
class SDXL(StableDiffusion):
  samplers = {
    "dpmpp2m": Samplers.DPMPP2MSampler,
  }

  def __init__(self, conditioner:Dict, first_stage:Dict, model:Dict, num_timesteps:int):
    self.conditioner = GeneralConditioner(**conditioner)
    self.first_stage_model = FirstStageModel(**first_stage)
    self.model = DiffusionModel(**model)

    self.sigmas = LegacyDDPMDiscretization()(num_timesteps, flip=True)

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/inference/helpers.py#L173
  def create_conditioning(self, pos_prompt:str, img_width:int, img_height:int, aesthetic_score:float) -> Tuple[Dict,Dict]:
    N = 1
    batch_c : Dict = {
      "txt": pos_prompt,
      "original_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "crop_coords_top_left": Tensor([0,0]).repeat(N,1),
      "target_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
    }
    batch_uc: Dict = {
      "txt": "",
      "original_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "crop_coords_top_left": Tensor([0,0]).repeat(N,1),
      "target_size_as_tuple": Tensor([img_height,img_width]).repeat(N,1),
      "aesthetic_score": Tensor([aesthetic_score]).repeat(N,1),
    }
    return self.conditioner(batch_c), self.conditioner(batch_uc, force_zero_embeddings=["txt"])

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/diffusionmodules/denoiser.py#L42
  def denoise(self, x:Tensor, sigma:Tensor, cond:Dict) -> Tensor:

    def sigma_to_idx(s:Tensor) -> Tensor:
      dists = s - self.sigmas.unsqueeze(1)
      return dists.abs().argmin(axis=0).view(*s.shape)

    sigma = self.sigmas[sigma_to_idx(sigma)]
    sigma_shape = sigma.shape
    sigma = append_dims(sigma, x)

    c_out   = -sigma
    c_in    = 1 / (sigma**2 + 1.0) ** 0.5
    c_noise = sigma_to_idx(sigma.reshape(sigma_shape))

    @TinyJit
    def run(x, tms, ctx, y, c_out, add):
      return (self.model.diffusion_model(x, tms, ctx, y)*c_out + add).realize()

    return run(*prep_for_jit(x*c_in, c_noise, cond["crossattn"], cond["vector"], c_out, x))

  # https://github.com/tinygrad/tinygrad/blob/64cda3c481613f4ca98eeb40ad2bce7a9d0749a3/examples/stable_diffusion.py#L543
  def decode(self, x:Tensor) -> Tensor:
    return self.first_stage_model.decode(1.0 / 0.13025 * x)

  def delete_conditioner(self) -> None:
    del self.conditioner

configs: Dict = {
  # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml
  "SDv1": {
    "default_weights_url": "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt",
    "class": StableDiffusionV1,
    "args": {
      "model": {
        "adm_in_ch": None,
        "in_ch": 4,
        "out_ch": 4,
        "model_ch": 320,
        "attention_resolutions": [4, 2, 1],
        "num_res_blocks": 2,
        "channel_mult": [1, 2, 4, 4],
        "n_heads": 8,
        "transformer_depth": [1, 1, 1, 1],
        "ctx_dim": 768,
      },
      "first_stage": {
        "ch": 128,
        "in_ch": 3,
        "out_ch": 3,
        "z_ch": 4,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "resolution": 256,
      },
      "num_timesteps": 1000,
    },
  },

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/configs/inference/sd_xl_base.yaml
  "SDXL": {
    "default_weights_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    "class": SDXL,
    "args": {
      "model": {
        "adm_in_ch": 2816,
        "in_ch": 4,
        "out_ch": 4,
        "model_ch": 320,
        "attention_resolutions": [4, 2],
        "num_res_blocks": 2,
        "channel_mult": [1, 2, 4],
        "d_head": 64,
        "transformer_depth": [1, 2, 10],
        "ctx_dim": 2048,
      },
      "first_stage": {
        "ch": 128,
        "in_ch": 3,
        "out_ch": 3,
        "z_ch": 4,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "resolution": 256,
      },
      "conditioner": {
        "embedders": [
          { "class": FrozenClosedClipEmbedder, "args": { "ret_layer_idx": 11 } },
          { "class": FrozenOpenClipEmbedder,   "args": { "dims": 1280, "n_heads": 20, "layers": 32, "return_pooled": True } },
          { "class": ConcatTimestepEmbedderND, "args": { "input_key": "original_size_as_tuple" } },
          { "class": ConcatTimestepEmbedderND, "args": { "input_key": "crop_coords_top_left"   } },
          { "class": ConcatTimestepEmbedderND, "args": { "input_key": "target_size_as_tuple"   } },
        ],
      },
      "num_timesteps": 1000,
    },
  },
}

def from_pretrained(config_key:str, weights_fn:Optional[str]=None, weights_url:Optional[str]=None, fp16:bool=False) -> StableDiffusion:
  config = configs.get(config_key, None)
  assert config is not None, f"Invalid architecture key '{args.arch}', expected value in {list(configs.keys())}"
  model = config["class"](**config["args"])

  if weights_fn is not None:
    assert weights_url is None, "Got passed both a weights_fn and weights_url, options are mutually exclusive"
  else:
    weights_url = weights_url if weights_url is not None else config["default_weights_url"]
    weights_fn  = fetch(weights_url, os.path.basename(weights_url))

  loader_map = {
    "ckpt": lambda fn: torch_load(fn)["state_dict"],
    "safetensors": safe_load,
  }
  loader = loader_map.get(ext := str(weights_fn).split(".")[-1], None)
  assert loader is not None, f"Unsupported file extension '{ext}' for weights filename, expected value in {list(loader_map.keys())}"
  state_dict = loader(weights_fn)

  for k,v in state_dict.items():
    if re.match(r'model\.diffusion_model\..+_block.+proj_[a-z]+\.weight', k):
      # SDv1 has issue where weights with this pattern are shape (3,3,1,1) when we expect (3,3)
      state_dict[k] = v.squeeze()

  load_state_dict(model, state_dict, strict=False)

  if fp16:
    for k,l in get_state_dict(model).items():
      if (k.startswith("model.")):
        l.replace(l.cast(dtypes.float16).realize())

  return model

if __name__ == "__main__":
  arch_parser = argparse.ArgumentParser(description="Run SDXL", add_help=False)
  arch_parser.add_argument('--arch', type=str, default="SDv1", choices=list(configs.keys()))
  arch_args, _ = arch_parser.parse_known_args()
  defaults = {
    "SDv1": { "width": 512,  "height": 512,  "guidance": 7.5, },
    "SDXL": { "width": 1024, "height": 1024, "guidance": 6.0, },
  }[arch_args.arch]
  print(f"Using Architecture: {arch_args.arch}")
  sampler_options = list(configs[arch_args.arch]["class"].samplers.keys())

  default_prompt = "a horse sized cat eating a bagel"
  parser = argparse.ArgumentParser(description="Run StableDiffusion. Note that changing the architecture with --arch will change some" + \
      "defaults and options, so set that option before running --help.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--arch',        type=str,   default="SDv1", choices=list(configs.keys()), help="Model architecture to use")
  parser.add_argument('--sampler',     type=str,   choices=sampler_options, default=sampler_options[0], help="Sampler for generation")
  parser.add_argument('--steps',       type=int,   default=10, help="The number of diffusion steps")
  parser.add_argument('--prompt',      type=str,   default=default_prompt, help="Description of image to generate")
  parser.add_argument('--out',         type=str,   default=Path(tempfile.gettempdir())/"rendered.png", help="Output filename")
  parser.add_argument('--seed',        type=int,   help="Set the random latent seed")
  parser.add_argument('--guidance',    type=float, default=defaults["guidance"], help="Prompt strength")
  parser.add_argument('--width',       type=int,   default=defaults["width"],  help="The output image width")
  parser.add_argument('--height',      type=int,   default=defaults["height"], help="The output image height")
  parser.add_argument('--aesthetic',   type=float, default=5.0, help="Aesthetic store for conditioning, only for SDXL_Refiner")
  parser.add_argument('--weights-fn',  type=str,   help="Filename of weights to load")
  parser.add_argument('--weights-url', type=str,   help="Url to download weights from")
  parser.add_argument('--fp16',        action='store_true', help="Loads the weights as float16")
  parser.add_argument('--noshow',      action='store_true', help="Don't show the image")
  parser.add_argument('--timing',      action='store_true', help="Print timing per step")
  args = parser.parse_args()

  N = 1
  C = 4
  F = 8
  SIZE_MULT = F * 4
  assert (r := args.width  % SIZE_MULT) == 0, f"img_width must be multiple of {SIZE_MULT}, got {args.width} (remainder {r})"
  assert (r := args.height % SIZE_MULT) == 0, f"img_height must be multiple of {SIZE_MULT}, got {args.height} (remainder {r})"

  Tensor.no_grad = True
  if args.seed is not None:
    Tensor.manual_seed(args.seed)

  model = from_pretrained(args.arch, args.weights_fn, args.weights_url, args.fp16)

  c, uc = model.create_conditioning(args.prompt, args.width, args.height, args.aesthetic)
  model.delete_conditioner()
  for v in c .values(): v.realize()
  for v in uc.values(): v.realize()
  print("created conditioning")

  # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/inference/helpers.py#L101
  randn = Tensor.randn(N, C, args.height // F, args.width // F)

  SamplerCls = model.samplers.get(args.sampler, None)
  assert SamplerCls is not None, f"Somehow failed to resolve sampler '{args.sampler}' from {model.__class__.__name__} class"

  sampler = SamplerCls(args.guidance, args.timing)
  with Context(BEAM=getenv("LATEBEAM")):
    z = sampler(model.denoise, randn, c, uc, args.steps)
  print("created samples")
  x = model.decode(z).realize()
  print("decoded samples")

  # make image correct size and scale
  x = (x + 1.0) / 2.0
  x = x.reshape(3,args.height,args.width).permute(1,2,0).clip(0,1)*255
  print(x.shape)

  im = Image.fromarray(x.numpy().astype(np.uint8, copy=False))
  print(f"saving {args.out}")
  im.save(args.out)

  if not args.noshow:
    im.show()

  # validation!
  if args.prompt == default_prompt and args.steps == 10 and args.seed == 0 and args.guidance == defaults["guidance"] \
    and args.width == defaults["width"] and args.height ==  defaults["height"] and not (args.weights_fn or args.weights_url) and THREEFRY:
    ref_image = Tensor(np.array(Image.open(Path(__file__).parent / f"{args.arch}_seed0.png")))
    distance = (((x - ref_image).cast(dtypes.float) / ref_image.max())**2).mean().item()
    assert distance < 3e-4, colored(f"validation failed with {distance=}", "red")
    print(colored(f"output validated with {distance=}", "green"))
