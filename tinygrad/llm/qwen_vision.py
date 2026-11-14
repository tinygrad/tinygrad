"""
The vision tower of Qwen 3.8 Flash Next (the mmproj GGUF, projector qwen3vl_merger): a picture's pixels -> its rows of the LLM's input
(one row a 2 x 2 group of 16 x 16 patches = 32 x 32 pixels). The HF reference is transformers' Qwen2VLImageProcessorFast + Qwen3VLVisionModel
(see results/w-gpf/vision/DESIGN.md): 27 pre-norm blocks of width 1152 with a 2D rotary embedding, a learned 48 x 48 position table
resized bilinearly, then the 2 x 2 merger (LayerNorm, 4608 -> 4608 -> GELU -> 2560). No deepstack. Float32 throughout.
"""
from __future__ import annotations
import base64, binascii, hashlib, io, math, threading, time
from dataclasses import dataclass
from typing import Any
import numpy as np
from tinygrad import getenv

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

def decode_picture(url:Any, deadline:float, max_tokens:int=VISION_MAX_TOKENS) -> Picture:
  """a data: URL's picture (data_url_bytes, load_picture) by one of the process's decoders, waited for until deadline (time.monotonic())"""
  if not _decoders.acquire(timeout=max(0.0, deadline - time.monotonic())):
    raise PictureBusy(f"the server is decoding other images: retry later (waited {VISION_DECODE_WAIT} s)")
  try: return load_picture(data_url_bytes(url), max_tokens)
  finally: _decoders.release()

@dataclass(frozen=True)
class Picture:
  rgb:np.ndarray  # (h, w, 3) uint8 at its tower size
  digest:bytes    # sha256 of its pixels and size
  @property
  def grid(self) -> tuple[int, int]: return self.rgb.shape[0] // FACTOR, self.rgb.shape[1] // FACTOR  # (its rows' rows and columns)
  @property
  def tokens(self) -> int: return self.grid[0] * self.grid[1]

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
      hb, wb = smart_resize(h, w, max_pixels=max_tokens * FACTOR * FACTOR)
      if f.format == "JPEG" and w >= 2 * wb and h >= 2 * hb: f.draft("RGB", (wb, hb))
      w, h = f.size
      if w * h > VISION_MAX_PIXELS: raise PictureError(f"image of {w} x {h} pixels: at most {VISION_MAX_PIXELS} pixels (the model sees a picture "
                                                         f"resized to at most {max_tokens * FACTOR * FACTOR:,} pixels)")
      # (an image object no longer needed is closed: its pixels freed at once, not when the request ends)
      im:Image.Image = f
      if f.getexif().get(0x0112, 1) in (2, 3, 4, 5, 6, 7, 8):
        im = ImageOps.exif_transpose(f)
        f.close()
      # (transparency on white, as Qwen's own qwen-vl-utils to_rgb: HF's convert_rgb and llama.cpp drop alpha and keep the stored colour)
      if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
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
      hb, wb = smart_resize(h, w, max_pixels=max_tokens * FACTOR * FACTOR)
      rgb = np.ascontiguousarray(np.asarray(im.resize((wb, hb), Image.Resampling.BICUBIC) if (wb, hb) != (w, h) else im, dtype=np.uint8))
      im.close()
  except PictureError: raise
  except Exception as e: raise PictureError(f"image could not be decoded ({type(e).__name__}: {e})") from None
  return Picture(rgb, hashlib.sha256(f"{hb}x{wb}:".encode() + rgb.tobytes()).digest())

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

def patches(rgb:np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
  """
  rgb (h, w, 3) uint8 at its tower size -> (patches, grid): every 16 x 16 patch of the normalized picture as (c, t, py, px) (the 2 frames
  of a still picture the same), the patches in 2 x 2 merge groups (HF's pixel_values order), grid (h/16, w/16)
  """
  h, w, _ = rgb.shape
  assert h % FACTOR == 0 and w % FACTOR == 0, f"not a tower size: {w} x {h}"
  x = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5                     # (h, w, c), mean = std = 0.5
  gh, gw = h // PATCH, w // PATCH
  x = x.transpose(2, 0, 1)[None].repeat(TEMPORAL, 0)                    # (t, c, h, w)
  x = x.reshape(TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
  x = x.transpose(2, 5, 3, 6, 1, 0, 4, 7)                               # (gh/2, gw/2, 2, 2, c, t, py, px)
  return np.ascontiguousarray(x.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH)), (gh, gw)
