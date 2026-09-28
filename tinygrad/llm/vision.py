from __future__ import annotations
import base64, functools, io, math
from typing import NamedTuple
import numpy as np
from tinygrad import Tensor, nn, Device
from tinygrad.helpers import fetch
from tinygrad.llm.gguf import gguf_load

class ImageEmbed(NamedTuple):
  start: int        # index of the first image token in the (expanded) prompt
  embeds: Tensor    # (n_tokens, dim) image embeddings in raster order over the merged grid
  grid_h: int       # merged grid height (patches_y // merge_size)
  grid_w: int       # merged grid width  (patches_x // merge_size)

def load_image(src:str):
  from PIL import Image
  if src.startswith("data:"):
    header, _, data = src.partition(",")
    if ";base64" not in header: raise ValueError("only base64 data URIs are supported")
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
  if src.startswith("http://") or src.startswith("https://"): return Image.open(fetch(src)).convert("RGB")
  return Image.open(src).convert("RGB")

def smart_resize(width:int, height:int, factor:int, min_pixels:int, max_pixels:int) -> tuple[int,int]:
  # same algorithm as llama.cpp's calc_size_preserved_ratio ("smart_resize" in transformers):
  # keep the aspect ratio, align both sides to factor, clamp the pixel count to [min_pixels, max_pixels]
  w_bar, h_bar = max(factor, int(math.floor(width / factor + 0.5)) * factor), max(factor, int(math.floor(height / factor + 0.5)) * factor)
  if h_bar * w_bar > max_pixels:
    beta = math.sqrt(height * width / max_pixels)
    h_bar = max(factor, math.floor(height / beta / factor) * factor)
    w_bar = max(factor, math.floor(width / beta / factor) * factor)
  elif h_bar * w_bar < min_pixels:
    beta = math.sqrt(min_pixels / (height * width))
    h_bar, w_bar = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
  return w_bar, h_bar

def expand_image_tokens(ids:list[int], images:list, tower:Qwen3VLTower, image_pad_id:int) -> tuple[list[int], list[ImageEmbed]]:
  """Replace each single <|image_pad|> token with the grid_h*grid_w tokens the image actually occupies,
  encoding the images with the vision tower in order of appearance."""
  out, embeds, it = [], [], iter(images)
  for tid in ids:
    if tid != image_pad_id:
      out.append(tid)
      continue
    emb, gh, gw = tower.encode(next(it))
    embeds.append(ImageEmbed(len(out), emb, gh, gw))
    out.extend([image_pad_id] * (gh * gw))
  return out, embeds

def extract_message_images(messages:list[dict]) -> list:
  """Collect image sources (urls, paths, data URIs) from OpenAI-style content parts, in message order."""
  images = []
  for msg in messages:
    content = msg.get("content")
    if not isinstance(content, list): continue
    for c in content:
      if c.get("type") == "image_url": images.append(c["image_url"]["url"])
      elif c.get("type") == "image": images.append(c["image"])
  return images

def prepare_prompt(ids:list[int], images:list, tower:Qwen3VLTower|None, image_pad_id:int|None) -> tuple[list[int], list[ImageEmbed]]:
  """Expand <|image_pad|> placeholders into one token per merged image patch and encode the images."""
  if not images: return ids, []
  if tower is None: raise RuntimeError("images require a vision tower, pass --mmproj")
  if image_pad_id is None: raise RuntimeError("model has no <|image_pad|> token")
  if ids.count(image_pad_id) != len(images):
    raise RuntimeError(f"template emitted {ids.count(image_pad_id)} image slots for {len(images)} images")
  return expand_image_tokens(ids, images, tower, image_pad_id)

def mrope_positions(seq_len:int, images:list[ImageEmbed]) -> tuple[np.ndarray, int]:
  """(t, h, w) interleaved-mrope positions for the prompt, and the position the next generated token should use.
  Image token i of a gh x gw grid at prompt offset s gets (s, s + i // gw, s + i % gw); after an image the text
  position resumes at s + max(gh, gw). Matches llama.cpp's mtmd MROPE layout."""
  pos, cur, prev = np.empty((seq_len, 3), dtype=np.int32), 0, 0
  for img in sorted(images, key=lambda x: x.start):
    n = img.grid_h * img.grid_w
    assert prev <= img.start and img.start + n <= seq_len, "image tokens overlap or exceed the prompt"
    pos[prev:img.start] = np.arange(cur, cur + img.start - prev, dtype=np.int32)[:, None]
    cur += img.start - prev
    rows, cols = np.repeat(np.arange(img.grid_h), img.grid_w), np.tile(np.arange(img.grid_w), img.grid_h)
    pos[img.start:img.start+n] = np.stack([np.full(n, cur), cur + rows, cur + cols], axis=1)
    cur, prev = cur + max(img.grid_h, img.grid_w), img.start + n
  pos[prev:] = np.arange(cur, cur + seq_len - prev, dtype=np.int32)[:, None]
  return pos, cur + seq_len - prev

def imrope_freqs_cis(positions:Tensor, rope_dim:int, theta:float, sections:tuple[int, ...]) -> Tensor:
  """Per-token rope frequencies for interleaved mrope (ggml GGML_ROPE_TYPE_IMROPE): pair p of rope_dim // 2 pairs
  reads position channel p % 3 (t, h, w), falling back to t past the section caps. Same cos||sin layout as
  precompute_freqs_cis."""
  n_pairs = rope_dim // 2
  sect_dims = sum(sections)
  def pair_chan(p:int) -> int:
    sector = p % sect_dims if sect_dims else p
    return sector % 3 if sect_dims and sector < 3 * sections[sector % 3] else 0
  chan = Tensor([pair_chan(p) for p in range(n_pairs)], dtype='int32', device=positions.device)
  freqs = 1.0 / (theta ** (Tensor.arange(0, n_pairs).to(positions.device) / n_pairs))
  angles = positions.float()[:, chan] * freqs    # (T, n_pairs)
  return angles.cos().cat(angles.sin(), dim=-1)  # (T, rope_dim)

def bilinear_resize_align_corners(grid:np.ndarray, oh:int, ow:int) -> np.ndarray:
  # grid (H, W, C) -> (oh*ow, C), matching ggml_interpolate with GGML_SCALE_MODE_BILINEAR | ALIGN_CORNERS
  H, W, C = grid.shape
  if (oh, ow) == (H, W): return grid.reshape(oh * ow, C)
  ys = np.linspace(0, H - 1, oh) if oh > 1 else np.zeros(oh)
  xs = np.linspace(0, W - 1, ow) if ow > 1 else np.zeros(ow)
  y0, x0 = np.floor(ys).astype(np.int64), np.floor(xs).astype(np.int64)
  y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
  wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
  out = (grid[y0][:, x0] * (1 - wy) * (1 - wx) + grid[y1][:, x0] * wy * (1 - wx) +
         grid[y0][:, x1] * (1 - wy) * wx + grid[y1][:, x1] * wy * wx)
  return out.reshape(oh * ow, C)

class ViTBlock:
  def __init__(self, dim:int, ffn_dim:int, n_heads:int, eps:float):
    self.ln1, self.ln2 = nn.LayerNorm(dim, eps), nn.LayerNorm(dim, eps)
    self.attn_qkv, self.attn_out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
    self.ffn_up, self.ffn_down = nn.Linear(dim, ffn_dim), nn.Linear(ffn_dim, dim)
    self.dim, self.n_heads, self.head_dim = dim, n_heads, dim // n_heads
  def __call__(self, x:Tensor, cos:Tensor, sin:Tensor) -> Tensor:
    n, rope_dims = x.shape[0], self.head_dim // 2
    def rope(t:Tensor) -> Tensor:   # 2D rope over the first head_dim // 2 dims, half-split pairs (i, i + hd//2)
      t1, t2 = t[..., :rope_dims], t[..., rope_dims:rope_dims*2]
      return (t1 * cos - t2 * sin).cat(t1 * sin + t2 * cos, dim=-1)
    q, k, v = self.attn_qkv(self.ln1(x)).split([self.dim] * 3, dim=-1)
    q, k = (rope(t.reshape(n, self.n_heads, self.head_dim)).transpose(0, 1).unsqueeze(0) for t in (q, k))
    v = v.reshape(n, self.n_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
    # chunk over queries so large images don't materialize a (n_heads, n, n) score matrix
    attn = Tensor.cat(*[qc.scaled_dot_product_attention(k, v).realize() for qc in q.split(1024, dim=2)], dim=2) if n > 1024 \
      else q.scaled_dot_product_attention(k, v)
    x = x + self.attn_out(attn.squeeze(0).transpose(0, 1).reshape(n, self.dim))
    return x + self.ffn_down(self.ffn_up(self.ln2(x)).gelu())

class Qwen3VLTower:
  """Vision encoder + merger from a qwen3vl-style mmproj GGUF (clip architecture, qwen3vl_merger projector).
  Encodes an image into LLM-sized embeddings, one per spatial_merge_size**2 patches."""
  def __init__(self, kv:dict, state_dict:dict[str, Tensor], device:str|None=None):
    self.device = device or Device.DEFAULT
    if kv.get('general.architecture') != 'clip' or kv.get('clip.projector_type') != 'qwen3vl_merger':
      raise ValueError(f"unsupported mmproj: {kv.get('general.architecture')}/{kv.get('clip.projector_type')}")
    if any(kv.get('clip.vision.is_deepstack_layers', [])): raise ValueError("deepstack vision towers are not supported")
    self.n_blocks, self.dim = kv['clip.vision.block_count'], kv['clip.vision.embedding_length']
    self.n_heads, self.eps = kv['clip.vision.attention.head_count'], kv['clip.vision.attention.layer_norm_epsilon']
    self.patch_size, self.merge_size = kv['clip.vision.patch_size'], kv.get('clip.vision.spatial_merge_size', 2)
    self.image_mean = kv.get('clip.vision.image_mean', [0.5] * 3)
    self.image_std = kv.get('clip.vision.image_std', [0.5] * 3)
    # llama.cpp's set_limit_image_tokens(8, 4096) for qwen3vl
    self.min_pixels, self.max_pixels = 8 * (self.patch_size * self.merge_size) ** 2, 4096 * (self.patch_size * self.merge_size) ** 2
    self.blk = [ViTBlock(self.dim, kv['clip.vision.feed_forward_length'], self.n_heads, self.eps) for _ in range(self.n_blocks)]
    self.post_ln = nn.LayerNorm(self.dim, self.eps)
    self.patch_embd = {"weight": Tensor.zeros(self.dim, 3, self.patch_size, self.patch_size), "bias": Tensor.zeros(self.dim)}
    self.patch_embd_1 = {"weight": Tensor.zeros(self.dim, 3, self.patch_size, self.patch_size)}
    self.position_embd = {"weight": Tensor.zeros((kv['clip.vision.image_size'] // self.patch_size) ** 2, self.dim, dtype='float32')}
    self.mm_0 = nn.Linear(self.dim * self.merge_size ** 2, self.dim * self.merge_size ** 2)
    self.mm_2 = nn.Linear(self.dim * self.merge_size ** 2, kv['clip.vision.projection_dim'])
    state_dict = {(k[2:] if k.startswith('v.') else k.replace('mm.', 'mm_')).replace('patch_embd.weight.1', 'patch_embd_1.weight'): v
                  for k, v in state_dict.items()}
    nn.state.load_state_dict(self, state_dict, verbose=False, consume=True, realize=False)
    params = nn.state.get_parameters(self)
    for s in params: s.replace(s.contiguous())
    Tensor.realize(*params)
    if self.device != Device.DEFAULT:  # offload the vision tower (e.g. the LLM fills the first GPU)
      for s in params: s.replace(s.to(self.device).realize())
    self._pos_embd_np = self.position_embd["weight"].float().numpy()  # (n_side**2, dim), freed weight was fp32

  @staticmethod
  def from_gguf(path:str, device:str|None=None) -> Qwen3VLTower: return Qwen3VLTower(*gguf_load(path), device)

  @functools.cache
  def _pos_embd(self, ph:int, pw:int) -> Tensor:
    n_side = int(math.sqrt(self._pos_embd_np.shape[0]))
    grid = self._pos_embd_np.reshape(n_side, n_side, -1)
    return Tensor(bilinear_resize_align_corners(grid, ph, pw).copy()).to(self.device)  # (ph*pw, dim) raster order

  @functools.cache
  def _rope(self, ph:int, pw:int) -> tuple[Tensor, Tensor]:
    # tokens are in merge-grouped order: block (y2, x2) raster, then (dy, dx) = (0,0), (0,1), (1,0), (1,1)
    y2, x2 = np.repeat(np.arange(ph // 2), pw // 2), np.tile(np.arange(pw // 2), ph // 2)
    rows = (2 * y2[:, None] + np.array([0, 0, 1, 1])).flatten()
    cols = (2 * x2[:, None] + np.array([0, 1, 0, 1])).flatten()             # (n_patches,)
    # pairs 0..hd//4-1 rotate with the row position, the rest with the column position (ggml ROPE_TYPE_VISION),
    # and the frequency table restarts at the column half (independent sections)
    j = np.arange(self.dim // self.n_heads // 2)
    half = len(j) // 2
    angles = np.where(j < half, rows[:, None], cols[:, None]) * (10000.0 ** (-2.0 * (j % half) / len(j)))[None, :]
    return (Tensor(np.cos(angles).astype(np.float32).copy()).to(self.device),
            Tensor(np.sin(angles).astype(np.float32).copy()).to(self.device))

  def encode(self, image) -> tuple[Tensor, int, int]:
    """image: PIL image, path, URL or data URI. Returns (embeds (gh*gw, dim), gh, gw) over the merged grid."""
    img = load_image(image) if not hasattr(image, 'size') else image
    w, h = smart_resize(img.size[0], img.size[1], self.patch_size * self.merge_size, self.min_pixels, self.max_pixels)
    arr = np.asarray(img.resize((w, h), resample=2), dtype=np.float32) / 255.0   # 2 = PIL bicubic
    arr = (arr.transpose(2, 0, 1) - np.array(self.image_mean)[:, None, None]) / np.array(self.image_std)[:, None, None]
    ph, pw = h // self.patch_size, w // self.patch_size
    ps = self.patch_size
    x = Tensor(arr.copy(), device=self.device).reshape(3, ph, ps, pw, ps).permute(1, 3, 0, 2, 4).reshape(ph * pw, -1)
    # the temporal-merge conv on a still image sums both patch embeddings
    w_embd = (self.patch_embd["weight"] + self.patch_embd_1["weight"]).reshape(self.dim, -1)
    x = (x @ w_embd.T + self.patch_embd["bias"]) + self._pos_embd(ph, pw)
    # group 2x2 patches into consecutive tokens, the order the merger (and llama.cpp) expects
    x = x.reshape(ph // 2, 2, pw // 2, 2, self.dim).permute(0, 2, 1, 3, 4).reshape(ph * pw, self.dim)
    cos, sin = self._rope(ph, pw)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)                                # (n, 1, hd//2) for the head dim
    for blk in self.blk: x = blk(x, cos, sin).realize()  # realize each block to bound peak memory (whole-graph laziness piles up)
    x = self.post_ln(x).reshape(-1, self.merge_size ** 2 * self.dim)
    embds = self.mm_2(self.mm_0(x).gelu())
    return embds.to(Device.DEFAULT).realize(), ph // self.merge_size, pw // self.merge_size
