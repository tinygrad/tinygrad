"""
The vision tower of GLM 5.3 Flash (the mmproj GGUF, projector glm5v): a picture's pixels -> its rows of the LLM's input (one row a 2 x 2 group
of 14 x 14 patches = 28 x 28 pixels). The reference is llama.cpp's tools/mtmd (models/glm4v.cpp, PROJECTOR_TYPE_GLM5V; see
results/w-gpf/glm-vision/DESIGN.md): 24 pre-norm blocks of width 1024 (RMSNorm, q / k RMSNorm a head, the same 2D rotary embedding as Qwen's
tower, SwiGLU with biases clamped at clip.vision.swiglu_clamp), no position table; then the 2 x 2 merger (a 2 x 2 conv 1024 -> 4096), fc,
LayerNorm, GELU (erf) and a clamped SwiGLU 4096 -> 10240 -> 4096. The serving side (buckets, slices, jits in the model's scratch) is
qwen_vision.VisionTower's. Float32 throughout (VISION_KERNELS=1: its F16 WMMA matmuls and fused attention, as Qwen's).
"""
from __future__ import annotations
import math, pathlib
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import gguf_load
from tinygrad.llm.qwen_vision import VisionTower, MERGE, TEMPORAL, patches

PATCH = 14
FACTOR = PATCH * MERGE
MIN_PIXELS, MAX_PIXELS = 12_544, 6_272_000  # the mmproj's clip.vision.image_min_pixels / image_max_pixels: 16 .. 8,000 rows a picture
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)  # CLIP's (the mmproj's image_mean / image_std)

def canvas_size(h:int, w:int, min_pixels:int=MIN_PIXELS, max_pixels:int=MAX_PIXELS, factor:int=FACTOR) -> tuple[int, int, int, int]:
  """
  a h x w picture's (canvas_h, canvas_w, content_h, content_w) for the tower (llama.cpp mtmd_image_preprocessor_glm5v): the canvas its sides
  ceil-aligned to factor, scaled up to min_pixels, or the largest that fits max_pixels; the picture resized into it (never up, unless below
  min_pixels) at its top left, the rest black
  """
  def align(v:int) -> int: return (v + factor - 1) // factor * factor
  ch, cw = align(h), align(w)
  if ch * cw < min_pixels:
    scale = math.sqrt(min_pixels / (h * w))
    ch, cw = align(max(1, math.ceil(h * scale))), align(max(1, math.ceil(w * scale)))
  if ch * cw > max_pixels:
    lo, hi, ch, cw = 1, h, factor, factor
    while lo <= hi:
      mid = (lo + hi) // 2
      ah, aw = align(mid), align(max(1, w * mid // h))
      if ah * aw <= max_pixels: ch, cw, lo = ah, aw, mid + 1
      else: hi = mid - 1
  scale = min(ch / h, cw / w)
  if h * w >= min_pixels: scale = min(1.0, scale)
  return ch, cw, max(1, min(ch, math.floor(h * scale))), max(1, min(cw, math.floor(w * scale)))

def swiglu_clamp(gate:Tensor, up:Tensor, limit:float) -> Tensor:
  """ggml_swiglu_clamp: silu(min(gate, limit)) * clamp(up, -limit, limit)"""
  if limit <= 0: return gate.silu() * up
  return gate.minimum(limit).silu() * up.clip(-limit, limit)

class GlmVisionTower(VisionTower):
  patch = PATCH

  def patches(self, rgb:np.ndarray) -> tuple[np.ndarray, tuple[int, int]]: return patches(rgb, PATCH, self.mean, self.std)

  def _weights(self, state:dict[str, Tensor]):
    def g(k:str, d:str) -> Tensor: return state[k].to(d).realize()
    d0, dl = self.devices[0], self.devices[-1]
    # (the projector (~0.35 GB F16) on the last card, the blocks on the others, the first (the inputs' buffers) the fewest: the card memory
    # the tower takes, at most ~0.44 GB a card instead of 0.67 on the last)
    if (m:=len(self.devices) - 1) >= 2:
      cnt = [self.depth // m + (j >= m - self.depth % m) for j in range(m)]
      self.block_dev = [self.devices[j] for j in range(m) for _ in range(cnt[j])]
    kv = self._kv
    self.mean, self.std = tuple(kv.get("clip.vision.image_mean", MEAN)), tuple(kv.get("clip.vision.image_std", STD))
    self.clamp = float(kv.get("clip.vision.swiglu_clamp", 0.0))
    self.patch_w = [g("v.patch_embd.weight", d0).reshape(self.width, -1), g("v.patch_embd.weight.1", d0).reshape(self.width, -1)]
    self.patch_b = g("v.patch_embd.bias", d0)
    names = ("ln1.weight", "ln2.weight", "attn_q_norm.weight", "attn_k_norm.weight") + \
      tuple(f"{n}.{w}" for n in ("attn_qkv", "attn_out", "ffn_up", "ffn_gate", "ffn_down") for w in ("weight", "bias"))
    self.blocks = [{n: g(f"v.blk.{i}.{n}", self.block_dev[i]) for n in names} for i in range(self.depth)]
    self.post_w = g("v.post_ln.weight", dl)
    # (the merger's 2 x 2 conv (out, in, dy, dx) as one matmul on a group's 4 rows side by side, block major: (dy, dx) = (0,0) (0,1) (1,0) (1,1))
    pm = state["mm.patch_merger.weight"]
    self.merge_w = pm.permute(0, 2, 3, 1).reshape(pm.shape[0], -1).to(dl).contiguous().realize()
    self.merge_b, self.fc_w = g("mm.patch_merger.bias", dl), g("mm.model.fc.weight", dl)
    self.pn_w, self.pn_b = g("mm.post_norm.weight", dl), g("mm.post_norm.bias", dl)
    self.up_w, self.gate_w, self.down_w = g("mm.up.weight", dl), g("mm.gate.weight", dl), g("mm.down.weight", dl)

  @staticmethod
  def from_gguf(path:str|pathlib.Path, devices:tuple[str, ...]|None=None, scratch:dict|None=None, **kw) -> GlmVisionTower:
    kv, state = gguf_load(str(path))
    assert kv.get("clip.projector_type") == "glm5v", f"not a glm5v mmproj: {kv.get('clip.projector_type')}"
    return GlmVisionTower(state, kv, devices, scratch, **kw)

  def __init__(self, state:dict[str, Tensor], kv:dict, devices:tuple[str, ...]|None=None, scratch:dict|None=None):
    self._kv = kv
    super().__init__(state, kv, devices, scratch)

  def _rms(self, x:Tensor, w:Tensor) -> Tensor: return x * (x.square().mean(-1, keepdim=True) + self.eps).rsqrt() * w.float()

  def _mm(self, x:Tensor, w:Tensor) -> Tensor: return x @ w.float().T  # (the projector's matmuls without bias, float: see the merger's note)

  def _block(self, i:int, x:Tensor, cos:Tensor, sin:Tensor, km:Tensor) -> Tensor:
    """block i of x (N, width) on its device, cos / sin (N, hd), km (1, 1, N) 0 for a real patch, -1e30 for padding"""
    b, dev, n, hd = self.blocks[i], self.block_dev[i], int(x.shape[0]), self.width // self.heads
    if x.device != dev: x = x.to(dev)
    cos, sin, km = cos.to(dev), sin.to(dev), km.to(dev)
    def rope(t:Tensor) -> Tensor:  # (heads, n, hd)
      t1, t2 = t[..., :hd // 2], t[..., hd // 2:]
      return t * cos + (-t2).cat(t1, dim=-1) * sin
    h = self._rms(x, b["ln1.weight"])
    qkv = self._lin(h, b["attn_qkv.weight"], b["attn_qkv.bias"])
    if self.kernels: qkv = qkv.contiguous()
    qkv = qkv.reshape(n, 3, self.heads, hd).permute(1, 2, 0, 3)
    q, k = self._rms(qkv[0], b["attn_q_norm.weight"]), self._rms(qkv[1], b["attn_k_norm.weight"])
    q, k, v = rope(q).contiguous().realize(), rope(k).contiguous().realize(), qkv[2].contiguous().realize()
    if self.kernels: a = self._attn_fused(q, k, v, km).realize()
    else:
      from tinygrad.llm.qwen_vision import ATTN_SCORES
      qc = min(n, max(64, 1 << int(math.log2(ATTN_SCORES // (self.heads * n)))))
      a = Tensor.cat(*[self._attn(q[:, i:i+qc], k, v, km).realize() for i in range(0, n, qc)], dim=1)
    a = a.transpose(0, 1).reshape(n, self.width)
    if self.kernels: a = a.half().contiguous()
    x = x + self._lin(a, b["attn_out.weight"], b["attn_out.bias"])
    h = self._rms(x, b["ln2.weight"])
    if self.kernels: h = h.half().contiguous()
    f = swiglu_clamp(self._lin(h, b["ffn_gate.weight"], b["ffn_gate.bias"]), self._lin(h, b["ffn_up.weight"], b["ffn_up.bias"]), self.clamp)
    return (x + self._lin(f, b["ffn_down.weight"], b["ffn_down.bias"])).realize()

  def _stage(self, a:int, b:int, x:Tensor, posx:Tensor, cos:Tensor, sin:Tensor, kmask:Tensor) -> Tensor:
    """
    the blocks a..b-1 of a picture: x its patches (a = 0: the patch embedding first) or the rows after block a-1 (a carry buffer); the rows
    after block b-1 into the carry buffer of its device, or (b = depth) the projector's (N/4, 4096). posx: unused (no position table)
    """
    n = int(x.shape[0])
    if a == 0:
      # (the two temporal kernels on the same still picture: llama.cpp's conv(W0) + conv(W1))
      xs = x.reshape(n, 3, TEMPORAL, self.patch * self.patch)
      if self.kernels:
        f0, f1 = (xs[:, :, t].reshape(n, -1).half().contiguous() for t in range(TEMPORAL))
        x = self._lin(f0, self.patch_w[0], self.patch_b) + f1.matmul(self.patch_w[1].half().T, dtype=dtypes.float)
      else: x = xs[:, :, 0].reshape(n, -1) @ self.patch_w[0].float().T + xs[:, :, 1].reshape(n, -1) @ self.patch_w[1].float().T + \
          self.patch_b.float()
    km = kmask.reshape(1, 1, n)
    for i in range(a, b): x = self._block(i, x, cos, sin, km)
    if b < self.depth: return self._carry(n, self.block_dev[b - 1]).assign(x).realize()
    x = self._rms(x.to(self.devices[-1]), self.post_w).reshape(n // (MERGE * MERGE), -1)
    # (the projector in float, as Qwen's merger)
    h = self._lin(x, self.merge_w, self.merge_b, wmma=False)
    h = self._mm(h, self.fc_w).layernorm(eps=1e-5) * self.pn_w.float() + self.pn_b.float()
    h = 0.5 * h * (1 + (h / math.sqrt(2)).erf())
    return self._mm(swiglu_clamp(self._mm(h, self.gate_w), self._mm(h, self.up_w), self.clamp), self.down_w).realize()

  def inputs(self, px:np.ndarray, grid:tuple[int, int], npad:int) -> list[np.ndarray]:
    """the tower's inputs of one picture padded to npad patches (posx a placeholder: no position table)"""
    from tinygrad.llm.qwen_vision import rope_cos_sin
    gh, gw = grid
    n = gh * gw
    assert px.shape[0] == n and npad >= n and npad % (MERGE * MERGE) == 0, (px.shape, grid, npad)
    cos, sin = rope_cos_sin(gh, gw, self.width // self.heads)
    def pad(a:np.ndarray, v:float=0.0) -> np.ndarray: return np.concatenate([a, np.full((npad - n, *a.shape[1:]), v, a.dtype)])
    kmask = np.concatenate([np.zeros(n, np.float32), np.full(npad - n, -1e30, np.float32)])
    return [pad(px.astype(np.float32)), np.zeros((1,), np.float32), pad(cos, 1.0), pad(sin), kmask]
