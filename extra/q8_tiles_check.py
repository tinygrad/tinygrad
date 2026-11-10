# Q8_TILES check: q8_quantize_tiles == q8_quantize on the rows of tiles with an expert
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.kernels.amd_qwen import q8_quantize, q8_quantize_tiles
rng = np.random.default_rng(1)
for R, F, tile in ((7056, 640, 16), (8448, 640, 32)):
  x = Tensor((rng.normal(size=(R, F))*3).astype(np.float16)).realize()
  til = np.where(rng.random(R // tile) < 0.3, rng.integers(0, 86, R // tile), -1).astype(np.int32)
  a = [t.numpy() for t in q8_quantize(x, R, F)[:3]]
  b = [t.numpy() for t in q8_quantize_tiles(x, R, F, Tensor(til))]
  act = np.repeat(til >= 0, tile)
  ok = all(np.array_equal(u[act].view(np.uint8), v[act].view(np.uint8)) for u, v in zip(a, b))
  print(R, F, tile, "active rows", act.sum(), "EQUAL" if ok else "DIFF")
# the down (IQ4_NL i8 grouped) on the sorted rows: Q8_TILES off vs on, every selection's row
from tinygrad.llm.kernels import amd_qwen
from tinygrad.llm.kernels.amd_qwen import moe_grouped_linear, QUANT_SIZES, IQ4_NL
E, NE, K, D, F = 86, 512, 10, 2560, 640
wb = rng.integers(0, 256, size=E*D*F//256*QUANT_SIZES[IQ4_NL], dtype=np.uint8)
wb.reshape(-1, 18)[:, :2] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)
w = Tensor(wb.view(np.uint16)).realize()
for T, TILE in ((576, 16), (576, 32)):
  sn = np.argsort(rng.random((T, NE)), axis=-1)[:, :K].reshape(-1)
  R = -(-(len(sn) + (TILE-1)*E) // TILE) * TILE
  ent, til, at = np.full(R, -1, np.int32), np.full(R // TILE, -1, np.int32), 0
  for e in range(E):
    idx = np.nonzero(sn == e)[0]
    ent[at:at+len(idx)] = idx
    nt = -(-len(idx) // TILE)
    til[at // TILE:at // TILE + nt] = e
    at += nt * TILE
  act = Tensor(rng.normal(size=(R, F)).astype(np.float16)).realize()
  outs = []
  for q in (0, 1):
    amd_qwen.Q8_TILES = q
    outs.append(moe_grouped_linear(w, IQ4_NL, E, act, Tensor(ent), Tensor(til), selections=T*K, sorted_out=True).numpy()[ent >= 0])
  same = outs[0].tobytes() == outs[1].tobytes()
  print("down", T, TILE, "EQUAL" if same else f"DIFF rows {np.nonzero((outs[0] != outs[1]).any(-1))[0][:10]}", "nan", np.isnan(outs[0]).sum())
  # (a float reference of a few rows: IQ4_NL dequantized in numpy)
  kv = np.array([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113], np.float32)
  blk = wb.reshape(E, D, F//32, 18)
  d = blk[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)
  qs = blk[..., 2:]
  W = np.concatenate([kv[qs & 15], kv[qs >> 4]], axis=-1) * d[..., None]  # (E, D, F//32, 32)
  W = W.reshape(E, D, F)
  rows = np.nonzero(ent >= 0)[0][:4]
  a = act.numpy().astype(np.float32)
  full = moe_grouped_linear(w, IQ4_NL, E, act, Tensor(ent), Tensor(til), selections=T*K, sorted_out=True).numpy()
  for r in rows:
    ref = W[til[r // TILE]] @ a[r]
    ok = ~np.isnan(full[r])
    print("  row", r, "rel err", float(np.abs(full[r] - ref)[ok].max() / np.abs(ref).max()), "nan", int((~ok).sum()),
          "ref nan", int(np.isnan(ref).sum()), "nan cols", np.nonzero(~ok)[0][:8])
