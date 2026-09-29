from __future__ import annotations
import base64, functools, hashlib, io, math
from array import array
from typing import NamedTuple
from tinygrad import Tensor, nn, Device, TinyJit
from tinygrad.helpers import fetch
from tinygrad.llm.gguf import gguf_load

class ImageEmbed(NamedTuple):
  start: int        # index of the first image token in the (expanded) prompt
  embeds: Tensor    # (max_tokens, dim) buffer holding the image embeddings in raster order over the merged grid
  grid_h: int       # merged grid height (patches_y // merge_size)
  grid_w: int       # merged grid width  (patches_x // merge_size)
  cache_key: bytes|None = None  # digest of the pixels fed to the tower; None disables cross-request reuse
  @property
  def n_tokens(self) -> int: return self.grid_h * self.grid_w

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
    embeds.append(ImageEmbed(len(out), emb, gh, gw, getattr(tower, 'image_key', None)))
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

def mrope_positions(seq_len:int, images:list[ImageEmbed]) -> tuple[list[list[int]], int]:
  """(t, h, w) interleaved-mrope positions for the prompt, and the position the next generated token should use.
  Image token i of a gh x gw grid at prompt offset s gets (s, s + i // gw, s + i % gw); after an image the text
  position resumes at s + max(gh, gw). Matches llama.cpp's mtmd MROPE layout."""
  pos: list[list[int]] = []
  cur, prev = 0, 0
  for img in sorted(images, key=lambda x: x.start):
    n = img.n_tokens
    assert prev <= img.start and img.start + n <= seq_len, "image tokens overlap or exceed the prompt"
    pos.extend([[cur + i] * 3 for i in range(img.start - prev)])
    cur += img.start - prev
    pos.extend([[cur, cur + i // img.grid_w, cur + i % img.grid_w] for i in range(n)])
    cur, prev = cur + max(img.grid_h, img.grid_w), img.start + n
  pos.extend([[cur + i] * 3 for i in range(seq_len - prev)])
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

class ViTBlock:
  def __init__(self, dim:int, ffn_dim:int, n_heads:int, eps:float):
    self.ln1, self.ln2 = nn.LayerNorm(dim, eps), nn.LayerNorm(dim, eps)
    self.attn_qkv, self.attn_out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
    self.ffn_up, self.ffn_down = nn.Linear(dim, ffn_dim), nn.Linear(ffn_dim, dim)
    self.dim, self.n_heads, self.head_dim = dim, n_heads, dim // n_heads
  def __call__(self, x:Tensor, cos:Tensor, sin:Tensor, attn_mask:Tensor) -> Tensor:
    n, rope_dims = x.shape[0], self.head_dim // 2
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)                             # (n, 1, hd//2) for the head dim
    def rope(t:Tensor) -> Tensor:   # 2D rope over the first head_dim // 2 dims, half-split pairs (i, i + hd//2)
      t1, t2 = t[..., :rope_dims], t[..., rope_dims:rope_dims*2]
      return (t1 * cos - t2 * sin).cat(t1 * sin + t2 * cos, dim=-1)
    q, k, v = self.attn_qkv(self.ln1(x.float()).half()).split([self.dim] * 3, dim=-1)  # layernorm in f32 like ggml
    # Materialize rotated Q/K so the score matmul doesn't recompute RoPE for every query/key pair.
    # GGUF biases are f32: restore f16 operands at matmul boundaries rather than promoting the attention/FFN to f32.
    q, k = (rope(t.reshape(n, self.n_heads, self.head_dim)).transpose(0, 1).half().contiguous() for t in (q, k))
    v = v.reshape(n, self.n_heads, self.head_dim).transpose(0, 1).half().contiguous()  # (H, n, hd)
    attn = q.scaled_dot_product_attention(k, v, attn_mask=attn_mask)
    x = x + self.attn_out(attn.transpose(0, 1).reshape(n, self.dim))
    return x + self.ffn_down(self.ffn_up(self.ln2(x.float()).half()).gelu().half().contiguous())

class Qwen3VLTower:
  """Vision encoder + merger from a qwen3vl-style mmproj GGUF (clip architecture, qwen3vl_merger projector).
  Encodes an image into LLM-sized embeddings, one per spatial_merge_size**2 patches.
  The whole tower is one JIT graph with FIXED shapes (padded to max_patches): the image grid shape enters only
  as tensor data (scalar geometry + a padding key mask), so kernels are fully static, compile once at warmup,
  and never again."""
  def __init__(self, kv:dict, state_dict:dict[str, Tensor], device:str|None=None, max_tokens:int=1024):
    self.device = device or Device.DEFAULT
    if kv.get('general.architecture') != 'clip' or kv.get('clip.projector_type') != 'qwen3vl_merger':
      raise ValueError(f"unsupported mmproj: {kv.get('general.architecture')}/{kv.get('clip.projector_type')}")
    if any(kv.get('clip.vision.is_deepstack_layers', [])): raise ValueError("deepstack vision towers are not supported")
    self.n_blocks, self.dim = kv['clip.vision.block_count'], kv['clip.vision.embedding_length']
    self.n_heads, self.eps = kv['clip.vision.attention.head_count'], kv['clip.vision.attention.layer_norm_epsilon']
    self.patch_size, self.merge_size = kv['clip.vision.patch_size'], kv.get('clip.vision.spatial_merge_size', 2)
    self.image_mean = kv.get('clip.vision.image_mean', [0.5] * 3)
    self.image_std = kv.get('clip.vision.image_std', [0.5] * 3)
    self.head_dim = self.dim // self.n_heads
    # llama.cpp's set_limit_image_tokens for qwen3vl is (8, 4096), but the default here is lower: the vision jit
    # arena is planned at the max patch count and must fit next to the LLM when they share a GPU
    self.min_pixels = 8 * (self.patch_size * self.merge_size) ** 2
    self.max_tokens, self.max_patches = max_tokens, max_tokens * self.merge_size ** 2
    self.max_pixels = self.max_patches * self.patch_size ** 2
    self.grid_side = kv['clip.vision.image_size'] // self.patch_size
    self.blk = [ViTBlock(self.dim, kv['clip.vision.feed_forward_length'], self.n_heads, self.eps) for _ in range(self.n_blocks)]
    self.post_ln = nn.LayerNorm(self.dim, self.eps)
    self.patch_embd = {"weight": Tensor.zeros(self.dim, 3, self.patch_size, self.patch_size), "bias": Tensor.zeros(self.dim)}
    self.patch_embd_1 = {"weight": Tensor.zeros(self.dim, 3, self.patch_size, self.patch_size)}
    self.position_embd = {"weight": Tensor.zeros(self.grid_side ** 2, self.dim, dtype='float32')}
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
    # the temporal-merge conv on a still image sums both patch embeddings
    self.patch_w = (self.patch_embd["weight"] + self.patch_embd_1["weight"]).reshape(self.dim, -1).contiguous().realize()
    self._pos_embd_flat = self.position_embd["weight"].contiguous().realize()  # (grid_side**2, dim)
    self._mean = Tensor(self.image_mean, device=self.device).reshape(1, 3, 1)
    self._std = Tensor(self.image_std, device=self.device).reshape(1, 3, 1)
    rope_j = Tensor.arange(self.head_dim // 2)
    self._rope_j = rope_j.to(self.device).realize()
    self._rope_freqs = (10000.0 ** (-2.0 * (rope_j % (self.head_dim // 4)).float() / (self.head_dim // 2))).to(self.device).realize()
    # fixed-size staging buffers; everything written with full-buffer copies (memcpy, no kernels to compile)
    self._buf_img = Tensor.zeros(self.max_patches * self.patch_size ** 2 * 3, dtype='uint8', device=self.device).realize()
    self._buf_geom = Tensor.zeros(2, dtype='int32', device=self.device).realize()     # [ph // 2, pw // 2]
    self._buf_mask = Tensor.zeros(self.max_patches, dtype='float32', device=self.device).realize()  # 0 valid, -1e4 pad
    self._arange = Tensor.arange(self.max_patches, dtype='int32').to(self.device).realize()
    self._arange_ps = Tensor.arange(self.patch_size, dtype='int32').to(self.device).realize()
    self._arange3 = Tensor.arange(3, dtype='int32').to(self.device).realize()
    self._vit = TinyJit(self._run)

  @staticmethod
  def from_gguf(path:str, device:str|None=None, max_tokens:int=1024) -> Qwen3VLTower:
    return Qwen3VLTower(*gguf_load(path), device, max_tokens)

  def _run(self, img:Tensor, geom:Tensor, pad_mask:Tensor) -> Tensor:
    n, ps, gs = self.max_patches, self.patch_size, self.grid_side
    ph2, pw2 = geom[0], geom[1]                                # scalar int tensors: half the patch grid dims
    t = self._arange                                           # token indices in merge-grouped order
    py = (2 * (t // (4 * pw2)) + (t // 2) % 2).clip(max_=2 * ph2 - 1)   # patch row of token t (padded rows clamp in-bounds)
    px = (2 * ((t // 4) % pw2) + t % 2).clip(max_=2 * pw2 - 1)          # patch col of token t
    # patchify by gather: patch vector (c, ky, kx) per token, grouped so 2x2 patch blocks are consecutive
    ky, kx = self._arange_ps.reshape(1, 1, ps, 1), self._arange_ps.reshape(1, 1, 1, ps)
    flat = (((py.reshape(-1, 1, 1, 1) * ps + ky) * (2 * pw2 * ps) + px.reshape(-1, 1, 1, 1) * ps +
             kx) * 3 + self._arange3.reshape(1, 3, 1, 1)).flatten()
    x = img[flat].reshape(n, 3, ps * ps)
    x = (x.float() / 255.0 - self._mean) / self._std
    x = x.reshape(n, 3 * ps * ps).half() @ self.patch_w.T + self.patch_embd["bias"].half()
    # learned position embeddings, bilinearly interpolated (align corners) to the patch grid
    sy, sx = py.float() * ((gs - 1) / (2 * ph2 - 1)), px.float() * ((gs - 1) / (2 * pw2 - 1))
    y0, x0 = sy.floor(), sx.floor()
    y0i, y1i = y0.cast('int32').clip(0, gs - 1), (y0.cast('int32') + 1).clip(max_=gs - 1)
    x0i, x1i = x0.cast('int32').clip(0, gs - 1), (x0.cast('int32') + 1).clip(max_=gs - 1)
    wy, wx = (sy - y0).reshape(-1, 1), (sx - x0).reshape(-1, 1)
    pe = (self._pos_embd_flat[y0i * gs + x0i] * (1 - wy) * (1 - wx) + self._pos_embd_flat[y1i * gs + x0i] * wy * (1 - wx) +
          self._pos_embd_flat[y0i * gs + x1i] * (1 - wy) * wx + self._pos_embd_flat[y1i * gs + x1i] * wy * wx)
    x = x + pe.half()
    # 2D rope; pairs 0..hd//4-1 read the row, the rest the column (ggml ROPE_TYPE_VISION, freqs restart per half)
    pos = (self._rope_j < self.head_dim // 4).reshape(1, -1).where(py.reshape(-1, 1), px.reshape(-1, 1)).float()
    angles = pos * self._rope_freqs.reshape(1, -1)
    cos, sin = angles.cos().half(), angles.sin().half()
    attn_mask = pad_mask.half().reshape(1, 1, n)               # key bias: 0 for real patches, -1e4 for padding
    for blk in self.blk: x = blk(x, cos, sin, attn_mask)
    x = self.post_ln(x.float()).half()
    return self.mm_2(self.mm_0(x.reshape(self.max_tokens, self.merge_size ** 2 * self.dim)).gelu()).half()

  def warmup(self):
    from PIL import Image
    for i in range(2): self.encode(Image.new('RGB', (512, 512), (i, i, i)))  # distinct images also warm the JIT behind the pixel cache

  def encode(self, image) -> tuple[Tensor, int, int]:
    """image: PIL image, path, URL or data URI. Returns (embeds buffer copy (max_tokens, dim), gh, gw)."""
    img = load_image(image) if not hasattr(image, 'size') else image.convert("RGB")
    w, h = smart_resize(img.size[0], img.size[1], self.patch_size * self.merge_size, self.min_pixels, self.max_pixels)
    img = img.resize((w, h), resample=2)   # 2 = PIL bicubic
    raw = img.tobytes()
    self.image_key = hashlib.sha256(raw).digest()
    return self._encode(raw, h, w)

  @functools.lru_cache(maxsize=128)
  def _encode(self, raw:bytes, h:int, w:int) -> tuple[Tensor, int, int]:
    ph, pw = h // self.patch_size, w // self.patch_size
    # fixed-size host->device copies (padded to the buffer sizes) so nothing size-dependent ever compiles
    self._buf_img.assign(Tensor(raw + bytes(int(self._buf_img.numel()) - len(raw)), dtype='uint8', device=self.device)).realize()
    n_pad = self.max_patches - ph * pw
    self._buf_mask.assign(Tensor(array('f', [0.0] * (ph * pw) + [-1e4] * n_pad).tobytes(), dtype='float32', device=self.device)).realize()
    self._buf_geom.assign(Tensor(array('i', [ph // 2, pw // 2]).tobytes(), dtype='int32', device=self.device)).realize()
    # copy out of the jit-managed buffer: the next encode overwrites it
    return self._vit(self._buf_img, self._buf_geom, self._buf_mask).to(Device.DEFAULT).clone().realize(), \
      ph // self.merge_size, pw // self.merge_size
