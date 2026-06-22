"""FLUX.2-klein sampling glue: flow-match euler schedule, latent pack/unpack, ids, post-process.

Mirrors mflux `Flux2Klein.generate_image` + `FlowMatchEulerDiscreteScheduler` (the
get_timesteps_and_sigmas path, since flux2-klein-4b has requires_sigma_shift=True) and
`Flux2LatentCreator`. Kept with the model (not in the example) so the full pipeline is
importable as `extra.models.flux2`.
"""
from __future__ import annotations
import math
from tinygrad import Tensor, dtypes

# mflux flux2-klein-4b constants.
VAE_SCALE_FACTOR = 8             # VAE downsamples 8x
LATENT_CHANNELS = 32             # latent channels before 2x2 patchify (packed = 4*32 = 128)
NUM_TRAIN_TIMESTEPS = 1000


# ---------------------------------------------------------------------------
# flow-match euler schedule (mflux FlowMatchEulerDiscreteScheduler).
# ---------------------------------------------------------------------------
def _empirical_mu(image_seq_len: int, num_steps: int) -> float:
  a1, b1 = 8.73809524e-05, 1.89833333
  a2, b2 = 0.00016927, 0.45666666
  if image_seq_len > 4300:
    return float(a2 * image_seq_len + b2)
  m_200 = a2 * image_seq_len + b2
  m_10 = a1 * image_seq_len + b1
  a = (m_200 - m_10) / 190.0
  b = m_200 - 200.0 * a
  return float(a * num_steps + b)


def make_schedule(image_seq_len: int, num_inference_steps: int) -> tuple[list[float], list[float]]:
  """Returns (timesteps, sigmas). timesteps has num_steps entries (value passed to the DiT,
  already in the 0..1000 range); sigmas has num_steps+1 entries (last 0.0) for euler
  dt = sigmas[t+1]-sigmas[t]."""
  n = num_inference_steps
  sigmas = [1.0 + i * ((1.0 / n) - 1.0) / (n - 1) for i in range(n)] if n > 1 else [1.0]  # linspace(1, 1/n, n)
  em = math.exp(_empirical_mu(image_seq_len, n))
  sigmas = [em / (em + (1.0 / s - 1.0)) for s in sigmas]  # exponential time-shift
  timesteps = [s * NUM_TRAIN_TIMESTEPS for s in sigmas]
  return timesteps, sigmas + [0.0]


# ---------------------------------------------------------------------------
# latent pack / unpack / ids (mflux Flux2LatentCreator).
# ---------------------------------------------------------------------------
def prepare_packed_latents(seed: int, height: int, width: int) -> tuple[Tensor, Tensor, int, int]:
  # height/width snapped to a multiple of 16 by the caller; latent grid is /16.
  lh, lw = height // (VAE_SCALE_FACTOR * 2), width // (VAE_SCALE_FACTOR * 2)
  Tensor.manual_seed(seed)
  latents = Tensor.randn(1, LATENT_CHANNELS * 4, lh, lw, dtype=dtypes.float32)   # (1, 128, lh, lw)
  packed = latents.reshape(1, LATENT_CHANNELS * 4, lh * lw).permute(0, 2, 1)     # (1, seq, 128)
  return packed, prepare_grid_ids(lh, lw), lh, lw


def prepare_grid_ids(lh: int, lw: int) -> Tensor:
  # mflux prepare_grid_ids (t=0, h, w, layer=0) -> (seq, 4) int32 (2D for the DiT pos_embed).
  rows = Tensor.arange(lh).reshape(lh, 1).expand(lh, lw).reshape(-1)
  cols = Tensor.arange(lw).reshape(1, lw).expand(lh, lw).reshape(-1)
  z = Tensor.zeros(lh * lw)
  return Tensor.stack(z, rows, cols, z, dim=1).cast(dtypes.int32)


def unpack_to_decoder_layout(packed: Tensor, lh: int, lw: int) -> Tensor:
  # mflux: latents.reshape(b, lh, lw, C).transpose(0,3,1,2) -> (1, 128, lh, lw) for decode_packed_latents.
  return packed.reshape(1, lh, lw, packed.shape[-1]).permute(0, 3, 1, 2)


def postprocess_image(decoded: Tensor, height: int, width: int) -> Tensor:
  # mflux ImageUtil: clip(x/2 + 0.5, 0, 1) -> CHW -> HWC uint8.
  img = (decoded.float() / 2 + 0.5).clip(0, 1)
  return img.reshape(3, height, width).permute(1, 2, 0).mul(255).round().cast(dtypes.uint8)
