import ast
import pathlib
import tempfile
import unittest

import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import DEV
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.uop.ops import Ops
from tinygrad.uop.uir import parse_uir

import slopkernels
from slopkernels.bench import _desc, _run_op


def _run_uir(uir:str, *srcs:Tensor) -> list[Tensor]:
  call = parse_uir(uir).call(*(s.uop for s in srcs))
  return [Tensor(s.uop.after(call)) for s in srcs]


def _call_op(name:str, target:tuple[int, ...], device:str|tuple[str, ...], *srcs:Tensor, **attrs:object) -> Tensor:
  uir = slopkernels.get(name, target=target, device=device, srcs=tuple(_desc(s) for s in srcs), **attrs)
  assert uir is not None, f"slopkernels.get({name!r}) returned None"
  return _run_op(uir, *srcs)


def _is_pure_spec(v:object) -> bool:
  if v is None or isinstance(v, (bool, int, float, str)): return True
  if isinstance(v, tuple): return all(_is_pure_spec(x) for x in v)
  return False


def _make_q8_0(rng:np.random.Generator, rows:int, cols:int) -> tuple[Tensor, Tensor, Tensor]:
  raw_np = np.zeros((rows * cols // 32, 34), dtype=np.uint8)
  raw_np[:, :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
  raw_np[:, 2:] = rng.integers(-60, 60, size=(rows * cols // 32, 32), dtype=np.int8).view(np.uint8)
  raw_u8 = Tensor(raw_np.reshape(-1)).realize()
  w_ref = ggml_data_to_tensor(raw_u8, rows * cols, 8).reshape(rows, cols).realize()
  w_u16 = raw_u8.bitcast("uint16").contiguous().realize()
  return raw_u8, w_u16, w_ref


def _amd_target() -> tuple[int, ...]|None:
  dev = Tensor.empty(1).device
  if not isinstance(dev, str) or dev.split(":")[0] != "AMD": return None
  t = getattr(Device[dev], "target", None)
  return t if t is not None and t[0] in (11, 12, 9) and isinstance(Device[dev].renderer, HIPRenderer) else None


class TestSlopKernelsUIR(unittest.TestCase):
  def test_narrow_api_and_purity_boundary(self):
    """slopkernels exposes only {get, list_kernels, compile_all}, uses pure primitive specs only,
    contains pure UOp builders only (zero Tensor/Buffer/Device/getenv), and tinygrad only imports top-level slopkernels."""
    self.assertEqual(set(slopkernels.__all__), {"get", "list_kernels", "compile_all"})
    for name, (_, sample) in slopkernels._REGISTRY.items():
      for k, v in sample.items():
        self.assertTrue(_is_pure_spec(v), f"{name}.{k} must be pure primitive/tuple, got {type(v).__name__}")
    # 1. Verify slopkernels/__init__.py and slopkernels/src/*.py are 100% pure UOp (no Tensor/Buffer/Device/getenv/realize/numpy)
    pkg_dir = pathlib.Path(slopkernels.__file__).resolve().parent
    forbidden_names = {"Tensor", "Buffer", "Device", "getenv", "nn"}
    forbidden_attrs = {"realize", "numpy", "item"}
    for p in [pkg_dir / "__init__.py", *(pkg_dir / "src").glob("*.py")]:
      tree = ast.parse(p.read_text())
      for node in ast.walk(tree):
        if isinstance(node, ast.Name):
          self.assertNotIn(node.id, forbidden_names, f"{p.name} must be pure UOp (found {node.id})")
        elif isinstance(node, ast.ImportFrom):
          for alias in node.names:
            self.assertNotIn(alias.name, forbidden_names, f"{p.name} must not import {alias.name}")
        elif isinstance(node, ast.Attribute):
          self.assertNotIn(node.attr, forbidden_attrs, f"{p.name} must not call .{node.attr}()")
    # 2. Verify tinygrad only imports from top-level slopkernels (__all__) and has no llm/slop.py
    import tinygrad
    root = pathlib.Path(tinygrad.__file__).resolve().parent
    self.assertFalse((root / "llm" / "slop.py").exists(), "tinygrad/llm/slop.py should not exist")
    found: list[str] = []
    for p in root.rglob("*.py"):
      txt = p.read_text()
      if "slopkernels" not in txt: continue
      tree = ast.parse(txt)
      for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "slopkernels":
          self.assertEqual(node.module, "slopkernels", f"{p} must only import from top-level slopkernels")
          for alias in node.names:
            self.assertIn(alias.name, slopkernels.__all__, f"{p} imported {alias.name!r} outside narrow API")
            found.append(alias.name)
        elif isinstance(node, ast.Import):
          for alias in node.names:
            self.assertNotEqual(alias.name.split(".")[0], "slopkernels", f"{p} must import from slopkernels narrow API")
    if not (root / "llm" / "kernels").exists(): self.assertGreater(len(found), 0)

  def test_compile_all_offline_to_uir(self):
    """Every registered universal slopkernel compiles offline to a .uir file and round-trips to an Ops.SINK with KernelInfo."""
    with tempfile.TemporaryDirectory() as td:
      uirs = slopkernels.compile_all(td, target=_amd_target())
      self.assertEqual(tuple(sorted(uirs)), slopkernels.list_kernels())
      self.assertGreaterEqual(len(uirs), 40)
      for name, txt in uirs.items():
        self.assertIsInstance(txt, str)
        self.assertTrue((pathlib.Path(td) / f"{name}.uir").is_file())
        u = parse_uir(txt)
        self.assertIn(u.op, (Ops.SINK, Ops.AFTER, Ops.RESHAPE, Ops.REDUCE, Ops.PERMUTE))
        if u.op is Ops.SINK: self.assertIsNotNone(u.arg)
    # Verify compile_all(target=...) actually emits RDNA4 vs RDNA3 low-level WMMA settings
    uirs_12 = slopkernels.compile_all(target="gfx1201")
    uirs_11 = slopkernels.compile_all(target="gfx1100")
    self.assertEqual(uirs_12["f16_wmma"], slopkernels.get("f16_wmma", out_features=64, in_features=128, tokens=16, kgroups=2, lanes=32))
    self.assertEqual(uirs_11["f16_wmma"], slopkernels.get("f16_wmma", out_features=64, in_features=128, tokens=16, kgroups=1, lanes=32))
    self.assertNotEqual(uirs_12["f16_wmma"], uirs_11["f16_wmma"])
    self.assertEqual(uirs_12["quant_linear_wmma"],
                     slopkernels.get("quant_linear_wmma", out_features=64, in_features=256, ggml_type=8, tokens=16, kgroups=2, lanes=32, rdna4=True))
    self.assertEqual(uirs_11["quant_linear_wmma"],
                     slopkernels.get("quant_linear_wmma", out_features=64, in_features=256, ggml_type=8, tokens=16, kgroups=1, lanes=32, rdna4=False))
    self.assertNotEqual(uirs_12["quant_linear_wmma"], uirs_11["quant_linear_wmma"])
    self.assertNotEqual(uirs_12["flash_attention"], uirs_11["flash_attention"])

  def test_individual_kernel_uirs_execute_exact(self):
    """Execute universal UIR kernels from slopkernels.get(...) on MOCK+AMD and verify numerical exactness."""
    if (tgt := _amd_target()) is None: self.skipTest("AMD required")
    rng = np.random.default_rng(7)
    # 1. f16_gemv
    x = Tensor(rng.standard_normal((2, 128)).astype(np.float32)).realize()
    w = Tensor(rng.standard_normal((64, 128)).astype(np.float16)).realize()
    got = _run_uir(slopkernels.get("f16_gemv", in_features=128, out_features=64, tokens=2), Tensor.empty(2, 64, dtype="float32"), w, x)[0].numpy()
    np.testing.assert_allclose(got, x.numpy() @ w.numpy().astype(np.float32).T, rtol=2e-3, atol=2e-3)
    # 2. rmsnorm
    xr = Tensor(rng.standard_normal((1, 2, 256)).astype(np.float32)).realize()
    wr = Tensor(rng.standard_normal((256,)).astype(np.float16)).realize()
    got_rn = _run_uir(slopkernels.get("rmsnorm", b=1, t=2, d=256, eps=1e-6), Tensor.empty(1, 2, 256, dtype="float32"), xr, wr)[0].numpy()
    np.testing.assert_allclose(got_rn, (xr.rmsnorm(eps=1e-6) * wr.float()).numpy(), rtol=1e-4, atol=1e-4)
    # 3. situ
    g = Tensor(rng.standard_normal((1, 2, 256)).astype(np.float32)).realize()
    u = Tensor(rng.standard_normal((1, 2, 256)).astype(np.float32)).realize()
    got_s = _run_uir(slopkernels.get("situ", b=1, t=2, d=256, b0=4.0, b1=4.0), Tensor.empty(1, 2, 256, dtype="float32"), g, u)[0].numpy()
    np.testing.assert_allclose(got_s, ((g / 4.0).tanh() * 4.0 * g.sigmoid() * (u / 4.0).tanh() * 4.0).numpy(), rtol=1e-5, atol=1e-5)
    # 4. q8_quantize + quant_decode + quant_linear_wmma (Q8_0)
    raw_np = np.zeros((64 * 256 // 32, 34), dtype=np.uint8)
    raw_np[:, :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw_np[:, 2:] = rng.integers(-60, 60, size=(64 * 256 // 32, 32), dtype=np.int8).view(np.uint8)
    raw_u8 = Tensor(raw_np.reshape(-1)).realize()
    w_ref = ggml_data_to_tensor(raw_u8, 64 * 256, 8).reshape(64, 256).realize()
    w_u16 = raw_u8.bitcast("uint16").contiguous().realize()
    x1 = Tensor(rng.standard_normal((1, 256)).astype(np.float32)).realize()
    xq, xd, xs = _run_uir(slopkernels.get("q8_quantize", tokens=1, in_features=256), Tensor.empty(1, 8, 8, dtype="uint32"),
                          Tensor.empty(1, 8, dtype="float32"), Tensor.empty(1, 8, 2, dtype="float32"), x1)[:3]
    got_q = _run_uir(slopkernels.get("quant_decode", out_features=64, in_features=256, ggml_type=8, tokens=1),
                     Tensor.empty(1, 64, dtype="float32"), w_u16, xq, xd, xs)[0].numpy()
    np.testing.assert_allclose(got_q, (x1 @ w_ref.T).numpy(), rtol=3e-2, atol=5e-2)
    x16 = Tensor(rng.standard_normal((16, 256)).astype(np.float16)).realize()
    cdna, rdna4 = tgt[0] == 9, tgt[0] == 12
    got_w = _run_uir(slopkernels.get("quant_linear_wmma", out_features=64, in_features=256, ggml_type=8, tokens=16,
                                     kgroups=4 if cdna else 2 if rdna4 else 1, lanes=64 if cdna else 32, rdna4=rdna4),
                     Tensor.empty(16, 64, dtype="float32"), w_u16, x16)[0].numpy()
    np.testing.assert_allclose(got_w, (x16.float() @ w_ref.T).numpy(), rtol=2e-2, atol=2e-2)
    # 5. flash_attention_decode_partial + flash_decode_combine
    q_fa = Tensor(rng.standard_normal((1, 4, 1, 64)).astype(np.float16)).realize()
    kv_fa = Tensor(rng.standard_normal((2, 1, 2, 128, 64)).astype(np.float16)).realize()
    vkl = 37
    p_buf, s_buf = _run_uir(slopkernels.get("flash_attention_decode_partial", b=1, h=4, m=1, d=64, h_kv=2,
                                            physical_n=128, chunks=2, max_kv_len=128, valid_kv_len=vkl),
                            Tensor.empty(1, 4, 2, 64, dtype="float32"), Tensor.empty(1, 4, 2, 2, dtype="float32"), q_fa, kv_fa)[:2]
    got_fa = _run_uir(slopkernels.get("flash_decode_combine", b=1, h=4, m=1, d=64, chunks=2, valid_kv_len=vkl),
                      Tensor.empty(1, 4, 1, 64, dtype="float32"), p_buf, s_buf)[0].numpy()
    ref_fa = q_fa.float().scaled_dot_product_attention(kv_fa[0, :, :, :vkl].float(), kv_fa[1, :, :, :vkl].float(), enable_gqa=True).numpy()
    np.testing.assert_allclose(got_fa, ref_fa, rtol=5e-3, atol=5e-3)
    # 6. fwht (orthogonality: pre then post == identity)
    xf = Tensor(rng.standard_normal((4, 128)).astype(np.float32)).realize()
    sgn = Tensor(rng.choice([-1.0, 1.0], size=(128,)).astype(np.float32)).realize()
    y_pre = _run_uir(slopkernels.get("fwht", n=4, d=128, pre=True), Tensor.empty(4, 128, dtype="float32"), xf, sgn)[0]
    y_back = _run_uir(slopkernels.get("fwht", n=4, d=128, pre=False), Tensor.empty(4, 128, dtype="float32"), y_pre, sgn)[0].numpy()
    np.testing.assert_allclose(y_back, xf.numpy(), rtol=1e-4, atol=1e-4)
    # 7. q8_embed, rows_gather, rows_scatter
    ids = Tensor(np.array([0, 3, 7, 1], dtype=np.int32)).realize()
    got_emb = _run_uir(slopkernels.get("q8_embed", r=4, vocab=64, d=256), Tensor.empty(4, 256, dtype="float32"), raw_u8, ids)[0].numpy()
    np.testing.assert_allclose(got_emb, w_ref.half().float().numpy()[[0, 3, 7, 1]], rtol=1e-5, atol=1e-5)
    tbl = Tensor(rng.standard_normal((16, 256)).astype(np.float16)).realize()
    got_g = _run_uir(slopkernels.get("rows_gather", n=4, p=16, f=256), Tensor.empty(4, 256, dtype="float32"), tbl, ids)[0].numpy()
    np.testing.assert_allclose(got_g, tbl.numpy()[[0, 3, 7, 1]].astype(np.float32), rtol=0, atol=0)
    dst = Tensor.zeros(16, 256, dtype="float16").contiguous().realize()
    src_rows = Tensor(rng.standard_normal((4, 256)).astype(np.float32)).realize()
    got_sc = _run_uir(slopkernels.get("rows_scatter", n=4, p=16, f=256), dst, src_rows, ids)[0].numpy()
    np.testing.assert_allclose(got_sc[[0, 3, 7, 1]], src_rows.numpy().astype(np.float16), rtol=0, atol=0)
    # 8. q8_pack + q8_unpack + q8_sum
    xp = Tensor(rng.standard_normal((2, 256)).astype(np.float32)).realize()
    pk = _run_uir(slopkernels.get("q8_pack", t=2, d=256), Tensor.empty(2, 8, 9, dtype="uint32"), xp)[0]
    unpk = _run_uir(slopkernels.get("q8_unpack", t=2, d=256, out_dtype="f32"), Tensor.empty(2, 256, dtype="float32"), pk)[0].numpy()
    np.testing.assert_allclose(unpk, xp.numpy(), rtol=3e-2, atol=3e-2)
    xm = Tensor(rng.standard_normal((2, 256)).astype(np.float32)).realize()
    qsum = _run_uir(slopkernels.get("q8_sum", t=2, d=256, n_packed=1), Tensor.empty(2, 256, dtype="float32"), xm, pk)[0].numpy()
    np.testing.assert_allclose(qsum, xm.numpy() + unpk, rtol=1e-4, atol=1e-4)
    # 9. local_argmax & moe_softmax_router
    la_x = Tensor(rng.standard_normal((2, 512)).astype(np.float32)).realize()
    la_n = Tensor(np.array([300], dtype=np.int32)).realize()
    la_out = _run_uir(slopkernels.get("local_argmax", b=2, v=512), Tensor.empty(2, 2, dtype="float32"), la_x, la_n)[0].numpy()
    np.testing.assert_equal(la_out[:, 1].view(np.int32), np.argmax(la_x.numpy()[:, :300], axis=-1))
    r_logits = Tensor(rng.standard_normal((2, 64)).astype(np.float32)).realize()
    r_sel, r_probs = _run_uir(slopkernels.get("moe_softmax_router", tokens=2, e=64, k=4),
                              Tensor.empty(2, 4, dtype="int32"), Tensor.empty(2, 4, dtype="float32"), r_logits)[:2]
    sel_np, probs_np = r_sel.numpy(), r_probs.numpy()
    for row in range(2):
      top4 = np.argsort(r_logits.numpy()[row])[-4:]
      self.assertEqual(set(sel_np[row].tolist()), set(top4.tolist()))
      np.testing.assert_allclose(probs_np[row].sum(), 1.0, rtol=1e-5, atol=1e-5)

  def test_f16_wmma_lds_and_small_tile(self):
    """Verify both the 4-wave software-pipelined LDS f16_wmma kernel (128x128x96) and small-tile fallback (16x64x64)."""
    if (tgt := _amd_target()) is None: self.skipTest("AMD required")
    rng = np.random.default_rng(17)
    cdna, rdna4 = tgt[0] == 9, tgt[0] == 12
    kgroups, lanes = (4, 64) if cdna else (2, 32) if rdna4 else (1, 32)
    for m, n, k in ((128, 128, 96), (16, 64, 64)):
      with self.subTest(m=m, n=n, k=k):
        x = Tensor(rng.standard_normal((m, k)).astype(np.float16)).realize()
        w = Tensor(rng.standard_normal((n, k)).astype(np.float16)).realize()
        uir = slopkernels.get("f16_wmma", out_features=n, in_features=k, tokens=m, kgroups=kgroups, lanes=lanes)
        assert uir is not None
        got = _run_uir(uir, Tensor.empty(m, n, dtype="float32"), w, x)[0].numpy()
        ref = x.numpy().astype(np.float32) @ w.numpy().astype(np.float32).T
        np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3)

  def test_logical_op_uirs_execute_exact(self):
    """Directly invoke all 6 logical op UIRs via slopkernels.get + call_with_output (no silent tinygrad fallback)."""
    if (tgt := _amd_target()) is None: self.skipTest("AMD required")
    dev = Tensor.empty(1).device
    rng = np.random.default_rng(23)
    # 1. dense_linear (with bias)
    w_d = Tensor(rng.standard_normal((64, 128)).astype(np.float16)).realize()
    x_d = Tensor(rng.standard_normal((2, 128)).astype(np.float32)).realize()
    b_d = Tensor(rng.standard_normal((64,)).astype(np.float16)).realize()
    got_d = _call_op("dense_linear", tgt, dev, w_d, x_d, b_d).numpy()
    np.testing.assert_allclose(got_d, x_d.numpy() @ w_d.numpy().astype(np.float32).T + b_d.numpy().astype(np.float32), rtol=2e-3, atol=2e-3)
    # 2. quant_linear: decode + split-K (in_features=1280) + WMMA (M=16)
    _, w_u16, w_ref = _make_q8_0(rng, 64, 1280)
    x_q1 = Tensor(rng.standard_normal((2, 1280)).astype(np.float32)).realize()
    ref_q1 = (x_q1 @ w_ref.T).numpy()
    got_q1 = _call_op("quant_linear", tgt, dev, w_u16, x_q1, ggml_type=8, in_features=1280, out_features=64, groups=1).numpy()
    np.testing.assert_allclose(got_q1, ref_q1, rtol=3e-2, atol=2e-2 * np.abs(ref_q1).max())
    x_q16 = Tensor(rng.standard_normal((16, 1280)).astype(np.float16)).realize()
    ref_q16 = (x_q16.float() @ w_ref.T).numpy()
    got_q16 = _call_op("quant_linear", tgt, dev, w_u16, x_q16, ggml_type=8, in_features=1280, out_features=64, groups=1).numpy()
    np.testing.assert_allclose(got_q16, ref_q16, rtol=2e-2, atol=2e-2 * np.abs(ref_q16).max())
    # 3. moe_linear
    _, mw_u16, mw_ref = _make_q8_0(rng, 4 * 16, 256)
    mw_ref_np = mw_ref.numpy().reshape(4, 16, 256)
    sel_m = Tensor(np.array([2, 0], dtype=np.int32)).realize()
    x_m = Tensor(rng.standard_normal((1, 1, 1, 256)).astype(np.float32)).realize()
    got_m = _call_op("moe_linear", tgt, dev, mw_u16, sel_m, x_m, ggml_type=8, sel_shape=(1, 1, 2),
                     in_features=256, local_out=16, global_out=16).numpy()[0, 0]
    for j, eid in enumerate((2, 0)):
      np.testing.assert_allclose(got_m[j], mw_ref_np[eid] @ x_m.numpy()[0, 0, 0], rtol=3e-2, atol=5e-2)
    # 4. flash_attention_op (decode + prefill)
    kv_fa = Tensor(rng.standard_normal((2, 1, 2, 128, 64)).astype(np.float16)).realize()
    q_dec = Tensor(rng.standard_normal((1, 4, 1, 64)).astype(np.float16)).realize()
    ref_fd = q_dec.float().scaled_dot_product_attention(kv_fa[0, :, :, :53].float(), kv_fa[1, :, :, :53].float(), enable_gqa=True).numpy()
    np.testing.assert_allclose(_call_op("flash_attention_op", tgt, dev, q_dec, kv_fa, valid_kv_len=53, decode=True).numpy(),
                               ref_fd, rtol=5e-3, atol=5e-3)
    if tgt[0] != 9:
      q_pre = Tensor(rng.standard_normal((1, 4, 32, 64)).astype(np.float16)).realize()
      mask = Tensor.full((32, 64), float("-inf"), dtype=dtypes.float32).triu(64 - 32 + 1)
      ref_fp = q_pre.float().scaled_dot_product_attention(kv_fa[0, :, :, :64].float(), kv_fa[1, :, :, :64].float(),
                                                          attn_mask=mask, enable_gqa=True).numpy()
      np.testing.assert_allclose(_call_op("flash_attention_op", tgt, dev, q_pre, kv_fa, valid_kv_len=64, decode=False).numpy(),
                                 ref_fp, rtol=5e-3, atol=5e-3)
    # 5. gated_delta_prefill_op
    gd_q = Tensor(rng.normal(0, 0.1, (1, 2, 4, 32)).astype(np.float32)).realize()
    gd_k = Tensor(rng.normal(0, 0.1, (1, 2, 4, 32)).astype(np.float32)).realize()
    gd_v = Tensor(rng.standard_normal((1, 2, 4, 8)).astype(np.float32)).realize()
    gd_b = Tensor(rng.uniform(0, 1, (1, 2, 4)).astype(np.float32)).realize()
    gd_a = Tensor(rng.uniform(0.8, 1.0, (1, 2, 4)).astype(np.float32)).realize()
    gd_s = Tensor(rng.normal(0, 0.1, (1, 2, 8, 32)).astype(np.float32)).contiguous().realize()
    st_ref = gd_s.numpy().copy()
    outs_ref = []
    for t in range(4):
      st_ref *= gd_a.numpy()[:, :, t, None, None]
      delta = (gd_v.numpy()[:, :, t] - (st_ref * gd_k.numpy()[:, :, t, None]).sum(-1)) * gd_b.numpy()[:, :, t, None]
      st_ref += delta[..., None] * gd_k.numpy()[:, :, t, None]
      outs_ref.append((st_ref * gd_q.numpy()[:, :, t, None]).sum(-1))
    got_gd = _call_op("gated_delta_prefill_op", tgt, dev, gd_q, gd_k, gd_v, gd_b, gd_a, gd_s).numpy()
    np.testing.assert_allclose(got_gd, np.stack(outs_ref, axis=2), rtol=2e-4, atol=2e-6)
    np.testing.assert_allclose(gd_s.numpy(), st_ref, rtol=2e-4, atol=2e-6)
    # 6. hc_coeffs_op
    hc_mix = Tensor(rng.normal(0, 0.2, (2, 24)).astype(np.float32)).realize()
    hc_flat = Tensor(rng.standard_normal((2, 128)).astype(np.float32)).realize()
    hc_base = Tensor(rng.normal(0, 0.2, (24,)).astype(np.float16)).realize()
    hc_scale = Tensor(rng.uniform(0.5, 1.5, (3,)).astype(np.float16)).realize()
    got_hc = _call_op("hc_coeffs_op", tgt, dev, hc_mix, hc_flat, hc_base, hc_scale, count=4, iters=2, eps=1e-6, norm_eps=1e-6).numpy()
    self.assertEqual(got_hc.shape, (2, 24))
    self.assertTrue(np.all(np.isfinite(got_hc)))
    # Verify doubly-stochastic Sinkhorn 4x4 block rows sum ~1
    np.testing.assert_allclose(got_hc[:, 8:].reshape(2, 4, 4).sum(-1), 1.0, rtol=2e-2, atol=2e-2)

  def test_multi_gpu_tp_op_uirs(self):
    """Verify 2-GPU Tensor Parallel sharding in slopkernels/src/ops.py on MOCK AMD (quant_linear, moe_linear, flash_attention_op)."""
    if (tgt := _amd_target()) is None or not DEV.interface.startswith("MOCK"): self.skipTest("MOCK AMD required for 2-GPU TP")
    devs = ("AMD:0", "AMD:1")
    rng = np.random.default_rng(31)
    # 1. quant_linear column-sharded (w_axis=0, ndev=2)
    _, w_u16, w_ref = _make_q8_0(rng, 64, 256)
    w_col = w_u16.reshape(64, -1).shard(devs, axis=0).flatten().realize()
    x_rep = Tensor(rng.standard_normal((2, 256)).astype(np.float32)).shard(devs, axis=None).realize()
    got_col = _call_op("quant_linear", tgt, devs, w_col, x_rep, ggml_type=8, in_features=256, out_features=64, groups=1, w_axis=0).numpy()
    np.testing.assert_allclose(got_col, (x_rep.to("AMD:0") @ w_ref.T).numpy(), rtol=3e-2, atol=5e-2)
    # 2. quant_linear row-sharded (w_axis=1, x_axis=1, ndev=2 -> split-K all-reduce)
    _, w_u16_r, w_ref_r = _make_q8_0(rng, 32, 512)
    w_row = w_u16_r.reshape(32, 2, -1).permute(1, 0, 2).contiguous().shard(devs, axis=0).flatten().realize()
    x_sh = Tensor(rng.standard_normal((2, 512)).astype(np.float32)).shard(devs, axis=1).realize()
    ref_row = (x_sh.to("AMD:0") @ w_ref_r.T).numpy()
    got_row = _call_op("quant_linear", tgt, devs, w_row, x_sh, ggml_type=8, in_features=512, out_features=32, groups=1, w_axis=1).numpy()
    np.testing.assert_allclose(got_row, ref_row, rtol=3e-2, atol=2e-2 * np.abs(ref_row).max())
    # 3. moe_linear expert-sharded (w_axis=0, ndev=2)
    _, mw_u16, mw_ref = _make_q8_0(rng, 4 * 16, 256)
    mw_ref_np = mw_ref.numpy().reshape(4, 16, 256)
    mw_sh = mw_u16.reshape(4, -1).shard(devs, axis=0).flatten().realize()
    sel_sh = Tensor(np.array([3, 1], dtype=np.int32)).shard(devs, axis=None).realize()
    xm_sh = Tensor(rng.standard_normal((1, 1, 1, 256)).astype(np.float32)).shard(devs, axis=None).realize()
    got_ms = _call_op("moe_linear", tgt, devs, mw_sh, sel_sh, xm_sh, ggml_type=8, sel_shape=(1, 1, 2),
                      in_features=256, local_out=16, global_out=16, w_axis=0, ndev=2).sum(0).numpy()[0, 0]
    for j, eid in enumerate((3, 1)):
      np.testing.assert_allclose(got_ms[j], mw_ref_np[eid] @ xm_sh.to("AMD:0").numpy()[0, 0, 0], rtol=3e-2, atol=5e-2)
    # 4. flash_attention_op head-sharded (q_axis=1, ndev=2, decode)
    q_fa = Tensor(rng.standard_normal((1, 4, 1, 64)).astype(np.float16)).realize()
    kv_fa = Tensor(rng.standard_normal((2, 1, 2, 128, 64)).astype(np.float16)).realize()
    q_sh, kv_sh = q_fa.shard(devs, axis=1).realize(), kv_fa.shard(devs, axis=2).realize()
    ref_fs = q_fa.float().scaled_dot_product_attention(kv_fa[0, :, :, :41].float(), kv_fa[1, :, :, :41].float(), enable_gqa=True).numpy()
    np.testing.assert_allclose(_call_op("flash_attention_op", tgt, devs, q_sh, kv_sh, valid_kv_len=41, decode=True).numpy(),
                               ref_fs, rtol=5e-3, atol=5e-3)

  def test_extra_kernels_execute_exact(self):
    """Numerically verify previously compile-only extra kernels (q8k_quantize, iq4xs_embed, rmsnorm_q8, argmax_chunks/final, moe_sum_slots)."""
    if _amd_target() is None: self.skipTest("AMD required")
    rng = np.random.default_rng(43)
    # 1. q8k_quantize
    xq = Tensor(rng.standard_normal((2, 256)).astype(np.float32)).realize()
    q_out, sc_out = _run_uir(slopkernels.get("q8k_quantize", tokens=2, in_features=256),
                             Tensor.empty(2, 8, 8, dtype="uint32"), Tensor.empty(2, 1, dtype="float32"), xq)[:2]
    sc_ref = np.maximum(np.abs(xq.numpy()).max(-1, keepdims=True) / 127.0, 1e-8)
    np.testing.assert_allclose(sc_out.numpy(), sc_ref, rtol=1e-6)
    np.testing.assert_array_equal(q_out.bitcast(dtypes.int8).reshape(2, 256).numpy(),
                                  np.clip(np.rint(xq.numpy() / sc_ref), -127, 127).astype(np.int8))
    # 2. iq4xs_embed
    raw_iq = rng.integers(0, 256, (16, 136), dtype=np.uint8)
    raw_iq[:, :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw_iq_t = Tensor(raw_iq.reshape(-1)).realize()
    w_iq_ref = ggml_data_to_tensor(raw_iq_t, 16 * 256, 23).reshape(16, 256).half().float().numpy()
    ids = Tensor(np.array([5, 0, 15, 2], dtype=np.int32)).realize()
    got_iq = _run_uir(slopkernels.get("iq4xs_embed", r=4, vocab=16, d=256), Tensor.empty(4, 256, dtype="float32"), raw_iq_t, ids)[0].numpy()
    np.testing.assert_allclose(got_iq, w_iq_ref[[5, 0, 15, 2]], rtol=1e-5, atol=1e-5)
    # 3. rmsnorm with fused q8=True
    xr = Tensor(rng.standard_normal((1, 2, 256)).astype(np.float32)).realize()
    wr = Tensor(rng.standard_normal((256,)).astype(np.float16)).realize()
    rn_out, rn_q, rn_d, rn_s = _run_uir(slopkernels.get("rmsnorm", b=1, t=2, d=256, eps=1e-6, q8=True),
                                        Tensor.empty(1, 2, 256, dtype="float32"), Tensor.empty(2, 8, 8, dtype="uint32"),
                                        Tensor.empty(2, 8, dtype="float32"), Tensor.empty(2, 8, 2, dtype="float32"), xr, wr)[:4]
    rn_ref = (xr.rmsnorm(eps=1e-6) * wr.float()).numpy()
    np.testing.assert_allclose(rn_out.numpy(), rn_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(rn_q.bitcast(dtypes.int8).reshape(1, 2, 8, 32).numpy().astype(np.float32) * rn_d.numpy().reshape(1, 2, 8, 1),
                               rn_ref.reshape(1, 2, 8, 32), rtol=3e-2, atol=3e-2)
    self.assertEqual(rn_s.shape, (2, 8, 2))
    # 4. argmax_chunks + argmax_final
    ax = Tensor(rng.standard_normal((2, 1024)).astype(np.float32)).realize()
    an = Tensor(np.array([777], dtype=np.int32)).realize()
    parts = _run_uir(slopkernels.get("argmax_chunks", b=2, v=1024, chunks=4), Tensor.empty(2, 4, 2, dtype="float32"), ax, an)[0]
    final = _run_uir(slopkernels.get("argmax_final", b=2, chunks=4), Tensor.empty(2, 2, dtype="float32"), parts)[0].numpy()
    np.testing.assert_equal(final[:, 1].view(np.int32), np.argmax(ax.numpy()[:, :777], axis=-1))
    # 5. moe_sum_slots (with has_add=True)
    sl_in = Tensor(rng.standard_normal((4, 256)).astype(np.float32)).realize()
    sl_add = Tensor(rng.standard_normal((1, 1, 256)).astype(np.float32)).realize()
    got_ss = _run_uir(slopkernels.get("moe_sum_slots", b=1, t=1, slots=4, d=256, has_add=True),
                      Tensor.empty(1, 1, 256, dtype="float32"), sl_in, sl_add)[0].numpy()
    np.testing.assert_allclose(got_ss, sl_in.numpy().sum(0)[None, None] + sl_add.numpy(), rtol=1e-5, atol=1e-5)

  def test_review_regressions(self):
    """Regressions: f16 Linear dtype parity, non-multiple-of-4 multi_proj tail, and int start_pos in gated_delta_prefill."""
    if (tgt := _amd_target()) is None: self.skipTest("AMD required")
    from tinygrad import Context
    from tinygrad.llm.model import Linear
    dev = Tensor.empty(1).device
    rng = np.random.default_rng(51)
    # 1. Mixed-dtype Linear parity across SLOP=0 and SLOP=1 (f16@f16, f32@f64, and f16@f16 + f32 bias overflow)
    lin = Linear(32, 16, bias=False)
    lin.weight = Tensor(rng.standard_normal((16, 32)).astype(np.float16)).realize()
    x16 = Tensor(rng.standard_normal((1, 32)).astype(np.float16)).realize()
    with Context(SLOP=0): y0 = lin(x16).realize()
    with Context(SLOP=1): y1 = lin(x16).realize()
    self.assertEqual(y1.dtype, y0.dtype)
    self.assertEqual(y1.dtype, dtypes.float16)
    np.testing.assert_allclose(y1.numpy(), y0.numpy(), rtol=1e-3, atol=1e-3)
    lin64 = Linear(32, 16, bias=False)
    lin64.weight = Tensor(rng.standard_normal((16, 32)).astype(np.float32)).realize()
    x64 = Tensor(rng.standard_normal((1, 32)).astype(np.float64)).realize()
    with Context(SLOP=0): y64_0 = lin64(x64).realize()
    with Context(SLOP=1): y64_1 = lin64(x64).realize()
    self.assertEqual(y64_1.dtype, dtypes.float64)
    np.testing.assert_allclose(y64_1.numpy(), y64_0.numpy(), rtol=1e-6, atol=1e-6)
    lin_ovf = Linear(32, 16, bias=True)
    lin_ovf.weight = Tensor.full((16, 32), 64.0, dtype="float16").contiguous().realize()
    lin_ovf.bias = Tensor.full((16,), -131072.0, dtype="float32").contiguous().realize()
    x_ovf = Tensor.full((1, 32), 64.0, dtype="float16").contiguous().realize()
    with Context(SLOP=0): yovf_0 = lin_ovf(x_ovf).realize()
    with Context(SLOP=1): yovf_1 = lin_ovf(x_ovf).realize()
    self.assertEqual(yovf_1.dtype, yovf_0.dtype)
    np.testing.assert_array_equal(yovf_1.numpy(), yovf_0.numpy())
    # 2. Decode attention when G > WAVES*WARP_SIZE: logical flash_attention_op rejects over-wide G=64, D=256 cleanly,
    #    and low-level flash_attention_decode_partial emits workgroup-stride stores covering all G heads
    self.assertIsNone(slopkernels.get("flash_attention_op", target=tgt, device=dev,
                                      srcs=(((1, 64, 1, 256), (1, 64, 1, 256), None, "f16"),
                                            ((2, 1, 1, 128, 256), (2, 1, 1, 128, 256), None, "f16")),
                                      valid_kv_len=96, decode=True))
    p40_uir = slopkernels.get("flash_attention_decode_partial", b=1, h=40, m=1, d=64, h_kv=1,
                              physical_n=64, chunks=1, max_kv_len=64, valid_kv_len=48, waves=1)
    assert p40_uir is not None
    # stats is param 1: with G=40 and WAVES*WARP_SIZE=32, 2 stride passes emit 4 stores to stats (M & L for heads 0..31 and 32..39)
    stats_stores = [u for u in parse_uir(p40_uir).toposort()
                    if u.op is Ops.STORE and any(x.op is Ops.PARAM and x.arg.slot == 1 for x in u.src[0].toposort())]
    self.assertEqual(len(stats_stores), 4)
    # 3. nucleus_round handles non-multiple-of-256 element counts (e.g. 257 unit weights)
    z257 = Tensor.ones(1, 1, 1, 257, dtype="float32").contiguous().realize()
    e257 = Tensor.ones(1, 1, 1, 257, dtype="float32").contiguous().realize()
    t257 = Tensor.full((1, 4), 0.5, dtype="float32").contiguous().realize()
    nr_uir = slopkernels.get("nucleus_round", b=1, c_blocks=1, c=257, g=4)
    assert nr_uir is not None
    got_nr = _run_uir(nr_uir, Tensor.empty(1, 1, 4, dtype="float32"), z257, e257, t257)[0].numpy()
    np.testing.assert_allclose(got_nr, np.full((1, 1, 4), 257.0, dtype=np.float32), rtol=0, atol=0)
    # 2. quant_decode_multi_proj handles non-multiple-of-4 row counts like (5, 8)
    _, w5_u16, w5_ref = _make_q8_0(rng, 5, 256)
    _, w8_u16, w8_ref = _make_q8_0(rng, 8, 256)
    xm = Tensor(rng.standard_normal((1, 256)).astype(np.float32)).realize()
    xq, xd, xs = _run_uir(slopkernels.get("q8_quantize", tokens=1, in_features=256), Tensor.empty(1, 8, 8, dtype="uint32"),
                          Tensor.empty(1, 8, dtype="float32"), Tensor.empty(1, 8, 2, dtype="float32"), xm)[:3]
    out5_init = Tensor.full((1, 5), -999.0, dtype="float32").contiguous().realize()
    out8_init = Tensor.full((1, 8), -999.0, dtype="float32").contiguous().realize()
    mp_uir = slopkernels.get("quant_decode_multi_proj", out_features_tuple=(5, 8), in_features=256, ggml_type=8, tokens=1)
    assert mp_uir is not None
    got5, got8 = _run_uir(mp_uir, out5_init, out8_init, w5_u16, w8_u16, xq, xd, xs)[:2]
    ref5, ref8 = (xm @ w5_ref.T).numpy(), (xm @ w8_ref.T).numpy()
    np.testing.assert_allclose(got5.numpy(), ref5, rtol=3e-2, atol=max(5e-2, 2e-2 * float(np.abs(ref5).max())))
    np.testing.assert_allclose(got8.numpy(), ref8, rtol=3e-2, atol=max(5e-2, 2e-2 * float(np.abs(ref8).max())))
    # 3. gated_delta_prefill_op accepts integer start_pos=0 (zeros initial state) and start_pos=3 (keeps state)
    gd_q = Tensor(rng.normal(0, 0.1, (1, 2, 4, 32)).astype(np.float32)).realize()
    gd_k = Tensor(rng.normal(0, 0.1, (1, 2, 4, 32)).astype(np.float32)).realize()
    gd_v = Tensor(rng.standard_normal((1, 2, 4, 8)).astype(np.float32)).realize()
    gd_b = Tensor(rng.uniform(0, 1, (1, 2, 4)).astype(np.float32)).realize()
    gd_a = Tensor(rng.uniform(0.8, 1.0, (1, 2, 4)).astype(np.float32)).realize()
    init_np = rng.normal(0, 0.2, (1, 2, 8, 32)).astype(np.float32)
    for sp in (0, 3):
      gd_s = Tensor(init_np.copy()).contiguous().realize()
      st_ref = np.zeros_like(init_np) if sp == 0 else init_np.copy()
      outs_ref = []
      for t in range(4):
        st_ref *= gd_a.numpy()[:, :, t, None, None]
        delta = (gd_v.numpy()[:, :, t] - (st_ref * gd_k.numpy()[:, :, t, None]).sum(-1)) * gd_b.numpy()[:, :, t, None]
        st_ref += delta[..., None] * gd_k.numpy()[:, :, t, None]
        outs_ref.append((st_ref * gd_q.numpy()[:, :, t, None]).sum(-1))
      got_gd = _call_op("gated_delta_prefill_op", tgt, dev, gd_q, gd_k, gd_v, gd_b, gd_a, gd_s, start_pos=sp).numpy()
      np.testing.assert_allclose(got_gd, np.stack(outs_ref, axis=2), rtol=2e-4, atol=2e-6)
      np.testing.assert_allclose(gd_s.numpy(), st_ref, rtol=2e-4, atol=2e-6)
    # 6. Public kernel builder slot binding: IQ4_XS quant_linear_wmma with symbolic valid_tokens & flash_attention with const valid_kv_len
    from tinygrad import UOp
    iq_uir = slopkernels.get("quant_linear_wmma", out_features=64, in_features=256, ggml_type=23, tokens=32, valid_tokens=(1, 32))
    assert iq_uir is not None
    iq_params = {u.arg.slot: u.arg.name for u in parse_uir(iq_uir).toposort() if u.op is Ops.PARAM}
    self.assertEqual(set(iq_params), {0, 1, 2, 3, 4})
    self.assertEqual(iq_params[4], "valid_tokens")
    if tgt[0] != 9:
      fa_uir = slopkernels.get("flash_attention", bh=4, m=32, d=64, b=1, h_kv=2, physical_n=128,
                               valid_kv_len=64, q_start=(0, 32), rdna4=tgt[0] == 12)
      assert fa_uir is not None
      fa_params = {u.arg.slot: u.arg.name for u in parse_uir(fa_uir).toposort() if u.op is Ops.PARAM}
      self.assertEqual(set(fa_params), {0, 1, 2, 3})
      self.assertEqual(fa_params[3], "q_start")
      q_fa = Tensor(rng.normal(0, 0.2, (1, 4, 32, 64)).astype(np.float16)).realize()
      kv_fa = Tensor(rng.normal(0, 0.2, (2, 1, 2, 128, 64)).astype(np.float16)).realize()
      out_fa = Tensor.empty(4, 32, 64, dtype="float32")
      call_fa = parse_uir(fa_uir).call(out_fa.uop, q_fa.reshape(4, 32, 64).uop, kv_fa.uop, UOp.variable("q_start", 0, 32).bind(32))
      got_fa = Tensor(out_fa.uop.after(call_fa)).reshape(1, 4, 32, 64).numpy()
      mask = Tensor.full((32, 64), float("-inf"), dtype=dtypes.float32).triu(64 - 32 + 1)
      ref_fa = q_fa.float().scaled_dot_product_attention(kv_fa[0, :, :, :64].float(), kv_fa[1, :, :, :64].float(),
                                                         attn_mask=mask, enable_gqa=True).numpy()
      np.testing.assert_allclose(got_fa, ref_fa, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
  unittest.main()
