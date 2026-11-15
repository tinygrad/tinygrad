"""
The vision tower of Qwen 3.8 Flash Next (the mmproj GGUF, projector qwen3vl_merger): a picture's pixels -> its rows of the LLM's input
(one row a 2 x 2 group of 16 x 16 patches = 32 x 32 pixels). The HF reference is transformers' Qwen2VLImageProcessorFast + Qwen3VLVisionModel
(see results/w-gpf/vision/DESIGN.md): 27 pre-norm blocks of width 1152 with a 2D rotary embedding, a learned 48 x 48 position table
resized bilinearly, then the 2 x 2 merger (LayerNorm, 4608 -> 4608 -> GELU -> 2560). No deepstack. Float32 throughout.
"""
from __future__ import annotations
import base64, binascii, functools, hashlib, io, math, pathlib, threading, time
from dataclasses import dataclass
from typing import Any, cast
import numpy as np
from tinygrad import Tensor, TinyJit, getenv, dtypes
from tinygrad.device import Buffer
from tinygrad.llm.gguf import gguf_load
from tinygrad.llm.kernels.amd import _amd_flash_attention, _is_rdna4, amd_custom_kernels_supported

PATCH, MERGE, TEMPORAL = 16, 2, 2
FACTOR = PATCH * MERGE
MIN_PIXELS, MAX_PIXELS = 65_536, 16_777_216  # the model's (preprocessor_config.json): 64 .. 16,384 rows a picture

# ******** a picture from a request: decoded (PNG, JPEG or WebP; PIL, header checked before the pixels are), EXIF orientation applied, RGB
# (transparency on white), resized bicubic (PIL, as Qwen's qwen-vl-utils and HF's slow processor) to its tower size ********
VISION_MAX_IMAGES = getenv("VISION_MAX_IMAGES", 8)             # pictures a request
VISION_MAX_BYTES = getenv("VISION_MAX_BYTES", 20_000_000)      # a picture's file (decoded from base64)
VISION_MAX_TOKENS = getenv("VISION_MAX_TOKENS", 1024)          # a picture's rows (the resize's max_pixels = 1,024 x it: about one
# megapixel; a larger picture is resized down to it, not refused. 4,096 (the model's 4 MP) needs a lower SEQ_ROWS for the tower's buckets)
VISION_MAX_PIXELS = getenv("VISION_MAX_PIXELS", 40_000_000)    # the pixels a file decodes to (after a JPEG's draft): a larger one isn't decoded
VISION_DECODERS = getenv("VISION_DECODERS", 2)                 # pictures decoded at once in the process (the others wait)
VISION_DECODE_WAIT = getenv("VISION_DECODE_WAIT", 30)          # seconds a request waits for a decoder at most (then 503)
FORMATS = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}

class PictureError(ValueError):
  """a picture a request can't have: the message says why (a 400 to the client)"""

class PictureBusy(PictureError):
  """the decoders were busy for VISION_DECODE_WAIT seconds (a 503: the client may retry)"""

# (decoding is the memory of a picture: a 40 MP PNG with alpha is ~320 MB at its peak (RGBA + its alpha + the RGB on white): at most
# VISION_DECODERS of them at once in the process, whatever the requests)
_decoders = threading.BoundedSemaphore(VISION_DECODERS)

# (a picture a request sends again (a conversation's every earlier screenshot, each request) isn't decoded again: the decoded pictures of the
# last VISION_URL_CACHE_MB of pixels by the sha256 of their data: URL (the same bytes: the same picture). 0: none kept)
VISION_URL_CACHE_MB = getenv("VISION_URL_CACHE_MB", 256)
_url_cache:dict[tuple[bytes, int], Picture] = {}
_url_cache_bytes = [0]
_url_lock = threading.Lock()

# (a picture sent before may come as its REFERENCE: data:image/x-sha256;base64,<the sha256 of its data: URL, 64 hex digits>: the cached picture,
# not its bytes again (a client re-sending a conversation's every screenshot each request uploads ~7.6 KB a 320x288 PNG); one the cache doesn't
# have: PictureNotCached (a 400 with error.code "picture_not_cached": the client sends the request again with the pictures' data: URLs)
REF_PREFIX = "data:image/x-sha256;base64,"

class PictureNotCached(PictureError):
  """a picture's reference the server's picture cache doesn't have (any more)"""

def decode_picture(url:Any, deadline:float, max_tokens:int=VISION_MAX_TOKENS) -> Picture:
  """a data: URL's picture (data_url_bytes, load_picture) by one of the process's decoders, waited for until deadline (time.monotonic()); a
  reference's (REF_PREFIX) from the cache"""
  if isinstance(url, str) and url.startswith(REF_PREFIX):
    try: ref = bytes.fromhex(url[len(REF_PREFIX):])
    except ValueError: ref = b""
    if len(ref) != 32: raise PictureError(f"a picture reference is {REF_PREFIX}<the sha256 of its data: URL in 64 hex digits>")
    with _url_lock:
      if (pic:=_url_cache.pop((ref, max_tokens), None)) is None: raise PictureNotCached(
        "picture_not_cached: this server no longer holds the referenced picture; send its data: URL again and please retry your request")
      _url_cache[(ref, max_tokens)] = pic
    return pic
  key = (hashlib.sha256(url.encode("utf-8", "surrogatepass")).digest(), max_tokens) if isinstance(url, str) and VISION_URL_CACHE_MB > 0 else None
  if key is not None:
    with _url_lock:
      if (pic:=_url_cache.pop(key, None)) is not None:
        _url_cache[key] = pic
        return pic
  if not _decoders.acquire(timeout=max(0.0, deadline - time.monotonic())):
    raise PictureBusy(f"the server is decoding other images: retry later (waited {VISION_DECODE_WAIT} s)")
  try: pic = load_picture(data_url_bytes(url), max_tokens)
  finally: _decoders.release()
  if key is not None:
    pic.rgb.setflags(write=False)
    with _url_lock:
      if key not in _url_cache:
        _url_cache[key] = pic
        _url_cache_bytes[0] += pic.rgb.nbytes
      while _url_cache_bytes[0] > VISION_URL_CACHE_MB * 2**20 and _url_cache:
        _url_cache_bytes[0] -= _url_cache.pop(next(iter(_url_cache))).rgb.nbytes
  return pic

@dataclass(frozen=True)
class Picture:
  rgb:np.ndarray  # (h, w, 3) uint8 at its tower size
  digest:bytes    # sha256 of its pixels and size
  factor:int = FACTOR  # (the pixels of a row each way: 32 Qwen, 28 GLM 5.3 Flash)
  @property
  def grid(self) -> tuple[int, int]: return self.rgb.shape[0] // self.factor, self.rgb.shape[1] // self.factor  # (its rows' rows and columns)
  @property
  def tokens(self) -> int: return self.grid[0] * self.grid[1]
  def ids(self) -> list[int]:
    # (made once a picture: the URL cache returns the same Picture every request (a Pokemon prompt has ~900: 7 ms); the callers don't change it)
    if (ret:=self.__dict__.get("_ids")) is None: object.__setattr__(self, "_ids", ret:=picture_ids(self.digest, *self.grid))
    return ret

def data_url_bytes(url:Any) -> bytes:
  """the file of a data: URL of an image (base64); anything else a PictureError (the server fetches no http(s) URL, reads no path)"""
  if not isinstance(url, str): raise PictureError("image_url.url must be a string")
  if not url.startswith("data:"):
    raise PictureError("image_url must be a data: URL (data:image/png;base64,...): the model server does not fetch remote images")
  head, sep, data = url[5:].partition(",")
  if not sep or not head.endswith(";base64"): raise PictureError("image_url data: URL must be base64 (data:image/png;base64,...)")
  if head[:-7].lower() not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
    raise PictureError(f"image type {head[:-7][:40]!r} not supported: image/png, image/jpeg or image/webp")
  if len(data) > VISION_MAX_BYTES * 4 // 3 + 4: raise PictureError(f"image larger than {VISION_MAX_BYTES} bytes")
  try: return base64.b64decode(data, validate=True)
  except (binascii.Error, ValueError) as e: raise PictureError(f"image_url data: URL is not valid base64 ({e})") from None

# the tower's picture sizing (the server's: cli sets it with the tower): "qwen" (smart_resize, factor 32) or "glm5v" (glm_vision.canvas_size: the
# picture at the top left of a black canvas aligned to 28, never upscaled; alpha dropped as llama.cpp's)
SIZING:list[str] = ["qwen"]

def tower_size(h:int, w:int, max_tokens:int) -> tuple[int, int, int, int, int]:
  """a h x w picture's (canvas h, canvas w, content h, content w, factor) for the tower of SIZING with at most max_tokens rows"""
  if SIZING[0] == "glm5v":
    from tinygrad.llm.glm_vision import canvas_size, FACTOR as GF, MAX_PIXELS as GMAX
    return (*canvas_size(h, w, max_pixels=min(GMAX, max_tokens * GF * GF)), GF)
  hb, wb = smart_resize(h, w, max_pixels=max_tokens * FACTOR * FACTOR)
  return hb, wb, hb, wb, FACTOR

def load_picture(data:bytes, max_tokens:int=VISION_MAX_TOKENS) -> Picture:
  """
  a picture's file -> the picture at its tower size (64 .. max_tokens rows). its memory: the decoded file (3 or 4 bytes a pixel, at most
  VISION_MAX_PIXELS) and its RGB on white, freed before it returns; a JPEG at least twice its tower size each way is decoded at a reduced
  scale (PIL's draft: 1/2 .. 1/8 in the DCT), still at least its tower size, then resized as the others
  """
  from PIL import Image, ImageOps
  if len(data) > VISION_MAX_BYTES: raise PictureError(f"image larger than {VISION_MAX_BYTES} bytes")
  try:
    with Image.open(io.BytesIO(data), formats=list(FORMATS)) as f:
      # (the header's size before any pixel is decoded)
      w, h = f.size
      if w < 1 or h < 1: raise PictureError(f"image of {w} x {h} pixels")
      if max(h, w) / min(h, w) > 200: raise PictureError(f"image aspect ratio over 200: {w} x {h}")
      if getattr(f, "n_frames", 1) > 1: raise PictureError("animated images are not supported")
      # (the tower size of the file's orientation: an EXIF rotation swaps both)
      _, _, hb, wb, _ = tower_size(h, w, max_tokens)
      if f.format == "JPEG" and w >= 2 * wb and h >= 2 * hb: f.draft("RGB", (wb, hb))
      w, h = f.size
      if w * h > VISION_MAX_PIXELS: raise PictureError(f"image of {w} x {h} pixels: at most {VISION_MAX_PIXELS} pixels (the model sees a picture "
                                                         f"resized to at most {max_tokens * tower_size(h, w, max_tokens)[4] ** 2:,} pixels)")
      # (an image object no longer needed is closed: its pixels freed at once, not when the request ends)
      im:Image.Image = f
      if f.getexif().get(0x0112, 1) in (2, 3, 4, 5, 6, 7, 8):
        im = ImageOps.exif_transpose(f)
        f.close()
      # (transparency on white, as Qwen's own qwen-vl-utils to_rgb: HF's convert_rgb and llama.cpp drop alpha and keep the stored colour)
      if SIZING[0] == "qwen" and (im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)):
        rgba = im if im.mode == "RGBA" else im.convert("RGBA")
        if rgba is not im: im.close()
        alpha = rgba.getchannel("A")
        im = Image.new("RGB", rgba.size, (255, 255, 255))
        im.paste(rgba, mask=alpha)
        rgba.close()
        alpha.close()
      elif im.mode != "RGB":
        rgb_im = im.convert("RGB")
        im.close()
        im = rgb_im
      w, h = im.size
      ch, cw, hb, wb, factor = tower_size(h, w, max_tokens)
      rgb = np.ascontiguousarray(np.asarray(im.resize((wb, hb), Image.Resampling.BICUBIC) if (wb, hb) != (w, h) else im, dtype=np.uint8))
      im.close()
      if (ch, cw) != (hb, wb):  # (glm5v: on a black canvas, at the top left)
        canvas = np.zeros((ch, cw, 3), np.uint8)
        canvas[:hb, :wb] = rgb
        rgb = canvas
  except PictureError: raise
  except Exception as e: raise PictureError(f"image could not be decoded ({type(e).__name__}: {e})") from None
  return Picture(rgb, hashlib.sha256(f"{ch}x{cw}:".encode() + rgb.tobytes()).digest(), factor)

def smart_resize(h:int, w:int, min_pixels:int=MIN_PIXELS, max_pixels:int=MAX_PIXELS, factor:int=FACTOR) -> tuple[int, int]:
  """the picture's size for the tower (HF smart_resize): sides multiples of factor, the area within the bounds, the aspect kept"""
  if max(h, w) / min(h, w) > 200: raise ValueError(f"picture aspect ratio over 200: {w} x {h}")
  hb, wb = round(h / factor) * factor, round(w / factor) * factor
  if hb * wb > max_pixels:
    beta = math.sqrt(h * w / max_pixels)
    hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
  elif hb * wb < min_pixels:
    beta = math.sqrt(min_pixels / (h * w))
    hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
  return hb, wb

def patches(rgb:np.ndarray, patch:int=PATCH, mean:tuple[float, ...]=(0.5,)*3, std:tuple[float, ...]=(0.5,)*3) -> tuple[np.ndarray, tuple[int, int]]:
  """
  rgb (h, w, 3) uint8 at its tower size -> (patches, grid): every patch x patch patch of the normalized picture as (c, t, py, px) (the 2
  frames of a still picture the same), the patches in 2 x 2 merge groups (HF's pixel_values order), grid (h/patch, w/patch). Qwen: patch 16,
  mean = std = 0.5; GLM 5.3 Flash (glm_vision): patch 14, CLIP's mean / std
  """
  h, w, _ = rgb.shape
  assert h % (patch * MERGE) == 0 and w % (patch * MERGE) == 0, f"not a tower size: {w} x {h}"
  x = (rgb.astype(np.float32) / 255.0 - np.array(mean, np.float32)) / np.array(std, np.float32)   # (h, w, c)
  gh, gw = h // patch, w // patch
  x = x.transpose(2, 0, 1)[None].repeat(TEMPORAL, 0)                    # (t, c, h, w)
  x = x.reshape(TEMPORAL, 3, gh // MERGE, MERGE, patch, gw // MERGE, MERGE, patch)
  x = x.transpose(2, 5, 3, 6, 1, 0, 4, 7)                               # (gh/2, gw/2, 2, 2, c, t, py, px)
  return np.ascontiguousarray(x.reshape(gh * gw, 3 * TEMPORAL * patch * patch)), (gh, gw)

# ******** a picture in the prompt: the ids of its rows in the host's token lists (the prefix cache, the checkpoints and the stash match them; the
# device's token buffer has the image pad id there), and the M-RoPE positions of a prompt's rows ********
# a row's id: -(1 + (H << 28 | cols << 14 | k)), H 60 bits of the picture's sha256, cols its rows' columns (the merged grid's width), k the row
# (0 .. n-1): a picture's rows are the same ids only for the same picture (and size), and the ids say where a picture starts and its grid. (the
# ids are Python ints of the host's lists only (88 bits): the device's token buffer has the image pad there, see Transformer._dev_ids)
ID_BITS, H_BITS = 14, 60

def picture_ids(digest:bytes, rows:int, cols:int) -> list[int]:
  """the ids of the rows x cols rows of a picture whose sha256 (of the pixels, and the resize bounds) is digest"""
  assert 0 < cols < 1 << ID_BITS and 0 < rows * cols <= 1 << ID_BITS, f"a picture of {rows} x {cols} rows"
  base = (int.from_bytes(digest[:8], "little") & ((1 << H_BITS) - 1)) << (2 * ID_BITS) | cols << ID_BITS
  return [-(1 + (base | k)) for k in range(rows * cols)]

def picture_runs(ids:list[int]) -> list[tuple[int, int, int, int]]:
  """the pictures of a token list: (first row, rows count n, cols, first row's id) each (a run of a picture's ids, k from 0 counting up)"""
  ret, i, L, mask = [], 0, len(ids), (1 << ID_BITS) - 1
  while i < L:
    if ids[i] >= 0:
      i += 1
      continue
    v = -ids[i] - 1
    assert v & mask == 0, f"a picture's rows from its first one: id {ids[i]} at {i}"
    j = i + 1
    while j < L and ids[j] == -(1 + (v | (j - i))): j += 1
    ret.append((i, j - i, (v >> ID_BITS) & mask, ids[i]))
    i = j
  return ret

def rope_positions(runs:list[tuple[int, int, int, int]], lo:int, hi:int) -> np.ndarray:
  """
  the M-RoPE positions (3, hi-lo) (t, h, w) int32 of the rows lo..hi-1 of a prompt with the pictures runs (HF get_rope_index, llama.cpp mtmd):
  a text row at its row minus the delta of the pictures before it (sum of n - max(rows, cols)), a picture's row (r, c) at (s, s+r, s+c), s its
  first row's text position. rows before 0 are at 0, rows past the prompt text rows
  """
  rows = np.arange(lo, hi, dtype=np.int64)
  t, hp, wp = rows.copy(), rows.copy(), rows.copy()
  d = 0
  for start, n, cols, _ in runs:
    s = start - d
    inside = (rows >= start) & (rows < start + n)
    k = rows[inside] - start
    t[inside], hp[inside], wp[inside] = s, s + k // cols, s + k % cols
    after = rows >= start + n
    d += n - max(n // cols, cols)
    t[after], hp[after], wp[after] = rows[after] - d, rows[after] - d, rows[after] - d
  out = np.stack([t, hp, wp]).clip(0, None)
  return out.astype(np.int32)

def rope_delta(runs:list[tuple[int, int, int, int]]) -> int:
  """the rope delta after the pictures runs: a later row's position is its row minus it"""
  return sum(n - max(n // cols, cols) for _, n, cols, _ in runs)

# ******** the tower ********

VISION_SLICE_MS = getenv("VISION_SLICE_MS", 300)  # a bucket whose tower takes longer runs in slices of about this long (see TowerJob)
# the tower's kernels (results/w-pf53/vision): 0 = all float (HF's numerics); 1 (AMD only) = the cards' WMMA: the attention as one fused
# kernel a block (F16 q / k / v, float sums and softmax: amd.py's flash attention, head 72 padded to 96) and every matmul on F16 activations
# against the F16 weights with float sums (as llama.cpp); the softmax, norms, rope, GELU and the residual stream stay float
VISION_KERNELS = getenv("VISION_KERNELS", 0)
ATTN_SCORES = 1 << 25  # the attention's scores of a chunk of queries at most (32M floats = 128 MB: 2,048 queries of 1,024 patches, 128 of 16,384)

def bucket(n:int) -> int:
  """the padded patch count of a picture with n patches: 256 x a power of 2 (one compiled tower a bucket; at most 2x the work)"""
  b = 256
  while b < n: b *= 2
  return b

def _block_major(a:np.ndarray, gh:int, gw:int) -> np.ndarray:
  return a.reshape(gh // MERGE, MERGE, gw // MERGE, MERGE, *a.shape[2:]).swapaxes(1, 2).reshape(gh * gw, *a.shape[2:])

def pos_interp(gh:int, gw:int, side:int) -> tuple[np.ndarray, np.ndarray]:
  """the 4 bilinear corners (indices into the side x side table) and weights of every patch, block major (HF's)"""
  hg, wg = np.linspace(0, side - 1, gh, dtype=np.float32), np.linspace(0, side - 1, gw, dtype=np.float32)
  hf, wf = hg.astype(np.int64), wg.astype(np.int64)
  hc, wc = np.minimum(hf + 1, side - 1), np.minimum(wf + 1, side - 1)
  dh, dw = hg - hf, wg - wf
  idx = [hf[:, None] * side + wf[None], hf[:, None] * side + wc[None], hc[:, None] * side + wf[None], hc[:, None] * side + wc[None]]
  wt = [(1 - dh)[:, None] * (1 - dw)[None], (1 - dh)[:, None] * dw[None], dh[:, None] * (1 - dw)[None], dh[:, None] * dw[None]]
  return np.stack([_block_major(i, gh, gw) for i in idx]), np.stack([_block_major(x.astype(np.float32), gh, gw) for x in wt])

def rope_cos_sin(gh:int, gw:int, head_dim:int, theta:float=10000.0) -> tuple[np.ndarray, np.ndarray]:
  """the 2D rotary embedding of the patches (block major): half the head's frequencies of the row, half of the column, as HF"""
  rd = head_dim // 2
  inv = 1.0 / (theta ** (np.arange(0, rd, 2, dtype=np.float32) / rd))
  hp, wp = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
  hp, wp = _block_major(hp[..., None], gh, gw)[:, 0], _block_major(wp[..., None], gh, gw)[:, 0]
  f = np.concatenate([hp[:, None] * inv[None], wp[:, None] * inv[None]], axis=-1).astype(np.float32)   # (n, head_dim/2)
  emb = np.concatenate([f, f], axis=-1)
  return np.cos(emb), np.sin(emb)

def attn_pack(q:Tensor, k:Tensor, v:Tensor, km:Tensor) -> tuple[Tensor, Tensor]:
  """the fused attention's inputs (see VisionTower._attn_fused): q (heads, n, D) F16, kv (2, 1, heads, n, D) F16, D = hd + 1 up to 32's"""
  H, n, hd = q.shape
  D = -(-(int(hd) + 1) // 32) * 32
  kb = (km.reshape(1, n, 1) < -1).where(-60000.0, 0.0).expand(H, n, 1)
  qh = Tensor.cat(q * math.sqrt(D / hd), q.ones_like()[..., :1], q.zeros_like()[..., :D - hd - 1], dim=-1).half()
  kh = Tensor.cat(k, kb, k.zeros_like()[..., :D - hd - 1], dim=-1).half()
  vh = v.pad(((0, 0), (0, 0), (0, D - hd))).half()
  return qh.contiguous(), Tensor.stack(kh, vh).reshape(2, 1, H, n, D).contiguous()

class VisionTower:
  patch = PATCH
  def __init__(self, state:dict[str, Tensor], kv:dict, devices:tuple[str, ...]|None=None, scratch:dict|None=None):
    """
    the weights stay F16 (0.90 GB), cast to float in the kernels that read them; the blocks spread over devices in contiguous runs (the
    picture's rows go device to device: 9.4 MB a hop at 4,096 patches), the patch embedding on the first, the merger on the last. scratch:
    the memory of the jits' intermediates, shared with other jits that never run at the same time (a model's that runs the tower between
    its own: TinyJit's scratch), else every bucket's jit its own
    """
    self.scratch = scratch
    self._jobs:dict[int, TowerJob] = {}  # (every bucket's last job: one unfinished at most, see start)
    self.depth, self.width = int(kv["clip.vision.block_count"]), int(kv["clip.vision.embedding_length"])
    self.heads, self.eps = int(kv["clip.vision.attention.head_count"]), float(kv["clip.vision.attention.layer_norm_epsilon"])
    from tinygrad import Device
    self.devices = devices or (Device.DEFAULT,)
    self.block_dev = [self.devices[i * len(self.devices) // self.depth] for i in range(self.depth)]
    self._jits:dict[tuple[int, int, int], Any] = {}
    self._inputs:dict[int, list[Tensor]] = {}
    self._carries:dict[tuple[int, str], Tensor] = {}
    self._plans:dict[int, list[tuple[int, int]]] = {}
    self.kernels = bool(VISION_KERNELS) and amd_custom_kernels_supported(self.devices[0])
    self._weights(state)

  @property
  def patch_dim(self) -> int: return 3 * TEMPORAL * self.patch * self.patch

  def patches(self, rgb:np.ndarray) -> tuple[np.ndarray, tuple[int, int]]: return patches(rgb)

  def _weights(self, state:dict[str, Tensor]):
    def g(k:str, d:str) -> Tensor: return state[k].to(d).realize()
    d0, dl = self.devices[0], self.devices[-1]
    # (the two temporal slices of the patch kernel: a still picture's two frames are the same)
    self.patch_w = [g("v.patch_embd.weight", d0).reshape(self.width, -1), g("v.patch_embd.weight.1", d0).reshape(self.width, -1)]
    self.patch_b, self.pos = g("v.patch_embd.bias", d0), g("v.position_embd.weight", d0)
    self.side = int(round(math.sqrt(self.pos.shape[0])))
    self.blocks = [{f"{n}.{w}": g(f"v.blk.{i}.{n}.{w}", self.block_dev[i]) for n in ("ln1", "ln2", "attn_qkv", "attn_out", "ffn_up", "ffn_down")
                    for w in ("weight", "bias")} for i in range(self.depth)]
    self.post_w, self.post_b = g("v.post_ln.weight", dl), g("v.post_ln.bias", dl)
    self.mm = [(g("mm.0.weight", dl), g("mm.0.bias", dl)), (g("mm.2.weight", dl), g("mm.2.bias", dl))]
    self._pos_np:np.ndarray|None = None

  @staticmethod
  def from_gguf(path:str|pathlib.Path, devices:tuple[str, ...]|None=None, scratch:dict|None=None, **kw) -> VisionTower:
    kv, state = gguf_load(str(path))
    assert kv.get("clip.projector_type") == "qwen3vl_merger", f"not a qwen3vl_merger mmproj: {kv.get('clip.projector_type')}"
    return VisionTower(state, kv, devices, scratch, **kw)

  def _ln(self, x:Tensor, w:Tensor, b:Tensor) -> Tensor: return x.layernorm(eps=self.eps) * w.float() + b.float()

  def _lin(self, x:Tensor, w:Tensor, b:Tensor, wmma:bool=True) -> Tensor:
    if self.kernels and wmma: return x.half().matmul(w.half().T, dtype=dtypes.float) + b.float()  # (an F32 mmproj's weights: F16 here)
    return x @ w.float().T + b.float()

  def _attn_fused(self, q:Tensor, k:Tensor, v:Tensor, km:Tensor) -> Tensor:
    """
    q, k, v (heads, n, hd) float, km (1, 1, n) 0 / -1e30: one WMMA flash kernel (no scores in memory). The kernel wants a head of a
    multiple of 32: hd 72 -> 96 with zeros, and the padded keys masked by a spare column (q 1, k -60000 for a padded key: its score
    -60000 / sqrt(96), exp 0), not causal (the queries placed after every key: q_start = n). q prescaled: the kernel divides by sqrt(96)
    """
    H, n, hd = q.shape
    assert n % 32 == 0, f"the kernel loads key tiles of 32 unmasked: n {n} must be a multiple (a bucket is)"
    qh, kv = attn_pack(q, k, v, km)
    out = Tensor.empty(H, n, int(qh.shape[-1]), dtype=dtypes.float32, device=q.device)
    fxn = functools.partial(_amd_flash_attention, valid_kv_len=n, q_start=n, rdna4=_is_rdna4(q.device))
    return Tensor.custom_kernel(out, qh, kv, fxn=fxn)[0][..., :hd]

  def _attn(self, q:Tensor, k:Tensor, v:Tensor, km:Tensor) -> Tensor:
    return q.scaled_dot_product_attention(k, v, attn_mask=km)

  def _block(self, i:int, x:Tensor, cos:Tensor, sin:Tensor, km:Tensor) -> Tensor:
    """block i of x (N, width) on its device, cos / sin (N, hd), km (1, 1, N) 0 for a real patch, -1e30 for padding"""
    b, dev, n, hd = self.blocks[i], self.block_dev[i], int(x.shape[0]), self.width // self.heads
    if x.device != dev: x = x.to(dev)
    cos, sin, km = cos.to(dev), sin.to(dev), km.to(dev)
    def rope(t:Tensor) -> Tensor:  # (heads, n, hd)
      t1, t2 = t[..., :hd // 2], t[..., hd // 2:]
      return t * cos + (-t2).cat(t1, dim=-1) * sin
    h = self._ln(x, b["ln1.weight"], b["ln1.bias"])
    qkv = self._lin(h, b["attn_qkv.weight"], b["attn_qkv.bias"])
    if self.kernels: qkv = qkv.contiguous()  # (one WMMA matmul, not three narrow ones each fused with its rope)
    qkv = qkv.reshape(n, 3, self.heads, hd).permute(1, 2, 0, 3)
    q, k, v = rope(qkv[0]).contiguous().realize(), rope(qkv[1]).contiguous().realize(), qkv[2].contiguous().realize()
    # (queries in chunks: the scores of a 16,384-patch picture would be 17 GB at once; padded keys masked, padded queries' rows unused)
    # (each chunk realized: the jit's memory plan reuses one chunk's scores for the next, not all of them at once)
    if self.kernels: a = self._attn_fused(q, k, v, km).realize()
    else:
      qc = min(n, max(64, 1 << int(math.log2(ATTN_SCORES // (self.heads * n)))))
      a = Tensor.cat(*[self._attn(q[:, i:i+qc], k, v, km).realize() for i in range(0, n, qc)], dim=1)
    a = a.transpose(0, 1).reshape(n, self.width)
    if self.kernels: a = a.half().contiguous()  # (the heads' rows side by side in F16: attn_out's matmul on the WMMA path)
    x = x + self._lin(a, b["attn_out.weight"], b["attn_out.bias"])
    h = self._ln(x, b["ln2.weight"], b["ln2.bias"])
    return (x + self._lin(self._lin(h, b["ffn_up.weight"], b["ffn_up.bias"]).gelu(), b["ffn_down.weight"], b["ffn_down.bias"])).realize()

  def _stage(self, a:int, b:int, x:Tensor, posx:Tensor, cos:Tensor, sin:Tensor, kmask:Tensor) -> Tensor:
    """
    the blocks a..b-1 of a picture: x its patches (a = 0: the patch embedding and posx first) or the rows after block a-1 (a carry buffer);
    the rows after block b-1 into the carry buffer of its device, or (b = depth) the merger's (N/4, 2560)
    """
    n = int(x.shape[0])
    if a == 0:
      # (Conv3d over the 2 frames = the two slices' sums, as llama.cpp: each frame's patch times its slice)
      xs = x.reshape(n, 3, TEMPORAL, PATCH * PATCH)
      if self.kernels:  # (each frame's patches F16 contiguous first: a strided operand keeps the matmul off the WMMA path)
        f0, f1 = (xs[:, :, t].reshape(n, -1).half().contiguous() for t in range(TEMPORAL))
        x = self._lin(f0, self.patch_w[0], self.patch_b) + f1.matmul(self.patch_w[1].half().T, dtype=dtypes.float) + posx
      else: x = xs[:, :, 0].reshape(n, -1) @ self.patch_w[0].float().T + xs[:, :, 1].reshape(n, -1) @ self.patch_w[1].float().T + \
          self.patch_b.float() + posx
    km = kmask.reshape(1, 1, n)
    for i in range(a, b): x = self._block(i, x, cos, sin, km)
    if b < self.depth: return self._carry(n, self.block_dev[b - 1]).assign(x).realize()
    x = self._ln(x.to(self.devices[-1]), self.post_w, self.post_b).reshape(n // (MERGE * MERGE), -1)
    # (the merger in float: its WMMA matmul at the small buckets spills 632 B a thread: a scratch on the last card the model doesn't have)
    h = self._lin(x, *self.mm[0], wmma=False)
    h = 0.5 * h * (1 + (h / math.sqrt(2)).erf())   # (HF nn.GELU(): the erf form)
    return self._lin(h, *self.mm[1], wmma=False).realize()

  def _carry(self, n:int, dev:str) -> Tensor:
    """the buffer of the rows between two slices of a bucket of n patches on a device (one a bucket and device, made once)"""
    if (key:=(n, dev)) not in self._carries: self._carries[key] = Tensor.empty(n, self.width, dtype=dtypes.float32, device=dev).realize()
    return self._carries[key]

  def inputs(self, px:np.ndarray, grid:tuple[int, int], npad:int) -> list[np.ndarray]:
    """the tower's inputs of one picture padded to npad patches: the picture's size is DATA (one compiled graph a bucket)"""
    gh, gw = grid
    n = gh * gw
    assert px.shape[0] == n and npad >= n and npad % (MERGE * MERGE) == 0, (px.shape, grid, npad)
    idx, wt = pos_interp(gh, gw, self.side)
    if self._pos_np is None: self._pos_np = self.pos.float().numpy()
    posx = sum((self._pos_np[idx[k]] * wt[k][:, None] for k in range(4)), np.zeros((n, self.width), np.float32))
    cos, sin = rope_cos_sin(gh, gw, self.width // self.heads)
    def pad(a:np.ndarray, v:float=0.0) -> np.ndarray: return np.concatenate([a, np.full((npad - n, *a.shape[1:]), v, a.dtype)])
    kmask = np.concatenate([np.zeros(n, np.float32), np.full(npad - n, -1e30, np.float32)])
    return [pad(px.astype(np.float32)), pad(posx.astype(np.float32)), pad(cos, 1.0), pad(sin), kmask]

  def warmup(self, max_tokens:int):
    """
    every bucket's jits of pictures up to max_tokens rows compiled and captured (calls of a blank picture of its size): a server's first
    picture of a size isn't the compile's wait of everyone's prefill. a bucket whose tower takes longer than VISION_SLICE_MS runs in slices
    of k blocks (k from its measured time): a server runs one slice between the other sessions' decoding steps (see TowerJob)
    """
    b = bucket(1)
    while b <= bucket(max_tokens * MERGE * MERGE):
      blank = (np.zeros((b, self.patch_dim), np.float32), (16, b // 16))
      for _ in range(2): self(*blank)
      st = time.perf_counter()
      self(*blank)
      ms = (time.perf_counter() - st) * 1e3
      if ms > VISION_SLICE_MS:
        k = max(1, int(self.depth * VISION_SLICE_MS / ms))
        self._plans[b] = [(a, min(a + k, self.depth)) for a in range(0, self.depth, k)]
        self._jits.pop((b, 0, self.depth), None)
        for _ in range(2): self(*blank)
      b *= 2

  def run_again(self, max_tokens:int):
    """
    every bucket's tower once more with a blank picture, after the model's warm-up: its captures grow the scratch the tower's jits share and
    rebase them (their links dropped): a server's first picture of a size would link them again
    """
    b = bucket(1)
    while b <= bucket(max_tokens * MERGE * MERGE):
      self(np.zeros((b, self.patch_dim), np.float32), (16, b // 16))
      b *= 2

  def start(self, px:np.ndarray, grid:tuple[int, int]) -> TowerJob:
    """a picture's tower as a job of slices (TowerJob.step): its patches (n, 1536) (see patches), grid (gh, gw). (one job at a time: a bucket's
    inputs and carries are its own)"""
    n = grid[0] * grid[1]
    npad = bucket(n)
    assert (j:=self._jobs.get(npad)) is None or j.done, f"a tower job of the {npad}-patch bucket is in progress: finish it first"
    if npad not in self._inputs:
      self._inputs[npad] = [Tensor.empty(*a.shape, dtype=dtypes.float32, device=self.devices[0]).realize() for a in self.inputs(px, grid, npad)]
    for t, a in zip(self._inputs[npad], self.inputs(px, grid, npad)):
      dst = cast(Buffer, t.uop.buffer).ensure_allocated()
      dst.copy_from(Buffer("NPY", dst.nbytes, opaque=memoryview(bytearray(np.ascontiguousarray(a, np.float32).tobytes()))))
    self._jobs[npad] = job = TowerJob(self, npad, n // (MERGE * MERGE), self._plans.get(npad, [(0, self.depth)]))
    return job

  def __call__(self, px:np.ndarray, grid:tuple[int, int]) -> np.ndarray:
    """
    px (n, 1536) normalized patches (see patches), grid (gh, gw) -> (n/4, 2560) float32 the picture's rows, in host memory, all its slices at
    once. (no card memory a call: every bucket's inputs are buffers of its own, written from the host; its output read to the host)
    """
    job = self.start(px, grid)
    while (rows:=job.step()) is None: pass
    return rows

class TowerJob:
  """a picture's tower in slices of blocks: step() runs the next one (its rows (n, 2560) in host memory after the last, else None)"""
  def __init__(self, tower:VisionTower, npad:int, rows:int, plan:list[tuple[int, int]]):
    self.tower, self.npad, self.rows, self.plan, self.k = tower, npad, rows, plan, 0
  @property
  def done(self) -> bool: return self.k == len(self.plan)
  def step(self) -> np.ndarray|None:
    t, (a, b) = self.tower, self.plan[self.k]
    if (jit:=t._jits.get(key:=(self.npad, a, b))) is None: jit = t._jits[key] = TinyJit(functools.partial(t._stage, a, b), scratch=t.scratch)
    px, posx, cos, sin, kmask = t._inputs[self.npad]
    x = px if a == 0 else t._carry(self.npad, t.block_dev[a - 1])
    out = jit(x, posx, cos, sin, kmask)
    self.k += 1
    return out.numpy()[:self.rows] if self.done else None
