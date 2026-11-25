import ast, pathlib, subprocess, unittest

import numpy as np
from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.helpers import DEV, ceildiv, unwrap
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.uop.ops import Ops

import slopkernels
from slopkernels.bench import _krun, _q8


def _amd_target() -> tuple[int, ...]|None:
  dev = Tensor.empty(1).device
  if not isinstance(dev, str) or dev.split(":")[0] != "AMD": return None
  t = getattr(Device[dev], "target", None)
  return t if t is not None and t[0] in (9, 11, 12) and isinstance(Device[dev].renderer, HIPRenderer) else None


def _make_q8_0(rng:np.random.Generator, rows:int, cols:int) -> tuple[Tensor, Tensor, Tensor]:
  raw_np = np.zeros((rows * cols // 32, 34), dtype=np.uint8)
  raw_np[:, :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
  raw_np[:, 2:] = rng.integers(-60, 60, size=(rows * cols // 32, 32), dtype=np.int8).view(np.uint8)
  raw_u8 = Tensor(raw_np.reshape(-1)).realize()
  w_ref = ggml_data_to_tensor(raw_u8, rows * cols, 8).reshape(rows, cols).realize()
  return raw_u8, raw_u8.bitcast("uint16").contiguous().realize(), w_ref


class TestSlopKernelsUIR(unittest.TestCase):
  def test_narrow_api_and_purity_boundary(self):
    """slopkernels exposes only {get, generate_all}, uses pure UOp builders only, and tinygrad only calls get(name) in llm/model.py."""
    self.assertEqual(set(slopkernels.__all__), {"generate_all", "get"})
    pkg_dir = pathlib.Path(slopkernels.__file__).resolve().parent
    forbidden_names, forbidden_attrs = {"Tensor", "Buffer", "Device", "getenv", "nn"}, {"realize", "numpy", "item"}
    for p in [pkg_dir / "__init__.py", *(pkg_dir / "src").glob("*.py")]:
      tree = ast.parse(p.read_text())
      for node in ast.walk(tree):
        if isinstance(node, ast.Name): self.assertNotIn(node.id, forbidden_names, f"{p.name} must be pure UOp (found {node.id})")
        elif isinstance(node, ast.ImportFrom):
          for alias in node.names: self.assertNotIn(alias.name, forbidden_names, f"{p.name} must not import {alias.name}")
        elif isinstance(node, ast.Attribute): self.assertNotIn(node.attr, forbidden_attrs, f"{p.name} must not call .{node.attr}()")
    import tinygrad
    root = pathlib.Path(tinygrad.__file__).resolve().parent
    model_py = (root / "llm" / "model.py").resolve()
    for p in root.rglob("*.py"):
      if "slopkernels" not in (txt := p.read_text()): continue
      tree = ast.parse(txt)
      slop_imports = [n for n in ast.walk(tree)
                      if (isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("slopkernels"))
                      or (isinstance(n, ast.Import) and any(a.name.startswith("slopkernels") for a in n.names))]
      if p.resolve() == model_py:
        self.assertEqual(len(slop_imports), 1)
        imp = slop_imports[0]
        assert isinstance(imp, ast.ImportFrom)
        self.assertEqual((imp.module, [a.name for a in imp.names]), ("slopkernels", ["get"]))
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_load_kernel"):
          calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "get"]
          self.assertEqual(len(calls), 1)
          self.assertEqual((len(calls[0].args), len(calls[0].keywords)), (1, 0))
      else:
        self.assertEqual(slop_imports, [], f"{p} must not import slopkernels")
    tracked_uir = subprocess.run(["git", "ls-files", "*.uir"], cwd=str(root.parent), capture_output=True, text=True, check=False)
    if tracked_uir.returncode == 0:
      self.assertEqual(tracked_uir.stdout.strip(), "", f".uir files must not be tracked in git:\n{tracked_uir.stdout}")

  def test_generate_all_and_get_disk_roundtrip(self):
    """slopkernels.get(name) writes fresh color-free .uir to slopkernels/uir/ and round-trips through disk; generate_all() writes all kernels."""
    uir_dir = slopkernels.UIR_DIR
    sample = uir_dir / "cdna_gemm_iq4_xs_f16.uir"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("stale garbage\n", encoding="utf-8")
    slopkernels.get.cache_clear()
    u = slopkernels.get("cdna_gemm_iq4_xs_f16")
    self.assertIsNotNone(u)
    assert u is not None
    self.assertIs(u.op, Ops.SINK)
    self.assertTrue(any(x.is_template for x in u.toposort()))
    self.assertNotIn("stale garbage", sample.read_text(encoding="utf-8"))
    self.assertNotIn("\x1b", sample.read_text(encoding="utf-8"))
    self.assertIsNone(slopkernels.get("nonexistent_llm_kernel"))
    all_uirs = slopkernels.generate_all()
    self.assertEqual(len(all_uirs), len(slopkernels._BUILDERS))
    self.assertEqual(len(list(uir_dir.glob("*.uir"))), len(all_uirs))
    for name, uir in all_uirs.items():
      self.assertNotIn("\x1b", uir)
      self.assertIn("addrspace=TEMPLATE", uir)
      self.assertEqual((uir_dir / f"{name}.uir").read_text(encoding="utf-8").rstrip("\n"), uir)

  def test_templated_llm_kernels_execute_exact(self):
    if (tgt := _amd_target()) is None: self.skipTest("AMD required")
    dev, rng = Tensor.empty(1).device, np.random.default_rng(23)
    arch_gemv, arch_gemm = "cdna" if tgt[0] == 9 else "rdna", "cdna" if tgt[0] == 9 else "rdna4" if tgt[0] == 12 else "rdna3"
    # 1. Dense GEMV
    w_d = Tensor(rng.standard_normal((64, 128)).astype(np.float16)).realize()
    x_d = Tensor(rng.standard_normal((2, 128)).astype(np.float32)).realize()
    got_d = _krun("amd_gemv_f16_f32", (2, 64), dev, w_d, x_d, tokens=2, out_features=64, in_blks=1, val_chunk=4, k_unroll=1).numpy()
    np.testing.assert_allclose(got_d, x_d.numpy() @ w_d.numpy().astype(np.float32).T, rtol=2e-3, atol=2e-3)
    # 2. Quantized GEMV (multi-wave) & WMMA (M=16)
    _, w_u16, w_ref = _make_q8_0(rng, 64, 1280)
    w_len, wv = int(w_u16.numel()), min(ceildiv(1280, 1024), 32)
    x_q1 = Tensor(rng.standard_normal((2, 1280)).astype(np.float32)).realize()
    ref_q1 = (x_q1 @ w_ref.T).numpy()
    got_q1 = _krun(f"{arch_gemv}_gemv_q8_0", (2, 64), dev, w_u16, *_q8(x_q1, 2, 1280, dev),
                   tokens=2, out_features=64, in_features=1280, waves=wv, raw_len=w_len, group_rows=0).numpy()
    np.testing.assert_allclose(got_q1, ref_q1, rtol=3e-2, atol=2e-2 * np.abs(ref_q1).max())
    x_q16 = Tensor(rng.standard_normal((16, 1280)).astype(np.float16)).realize()
    ref_q16 = (x_q16.float() @ w_ref.T).numpy()
    got_q16 = _krun(f"{arch_gemm}_gemm_q8_0_f16", (16, 64), dev, w_u16, x_q16,
                    m_blks=1, m_tiles=1, out_features=64, in_features=1280, raw_len=w_len).numpy()
    np.testing.assert_allclose(got_q16, ref_q16, rtol=2e-2, atol=2e-2 * np.abs(ref_q16).max())
    if tgt[0] != 9:
      x_q64 = Tensor(rng.standard_normal((64, 1280)).astype(np.float16)).realize()
      ref_q64 = (x_q64.float() @ w_ref.T).numpy()
      got_q64 = _krun(f"{arch_gemm}_gemm_q8_0_2d", (64, 64), dev, w_u16, x_q64,
                      m_blks=1, n_blks=1, wm=2, wn=2, tm=2, tn=2, in_features=1280, split_k=1, raw_len=w_len).numpy()
      np.testing.assert_allclose(got_q64, ref_q64, rtol=2e-2, atol=2e-2 * np.abs(ref_q64).max())
    # 3. MoE decode
    _, mw_u16, mw_ref = _make_q8_0(rng, 4 * 16, 256)
    mw_ref_np = mw_ref.numpy().reshape(4, 16, 256)
    sel_m = Tensor(np.array([2, 0], dtype=np.int32)).realize()
    x_m = Tensor(rng.standard_normal((1, 256)).astype(np.float32)).realize()
    got_m = _krun(f"{arch_gemv}_moe_q8_0", (2, 16), dev, mw_u16, *_q8(x_m, 1, 256, dev), sel_m,
                  rows=2, x_rows=1, out_features=16, in_features=256, waves=1, raw_len=int(mw_u16.numel()), x_share=2, ndev=1).numpy()
    for j, eid in enumerate((2, 0)):
      np.testing.assert_allclose(got_m[j], mw_ref_np[eid] @ x_m.numpy()[0], rtol=3e-2, atol=5e-2)
    # 4. Flash attention decode + prefill
    kv_fa = Tensor(rng.standard_normal((2, 1, 2, 128, 64)).astype(np.float16)).realize()
    q_dec = Tensor(rng.standard_normal((1, 4, 1, 64)).astype(np.float16)).realize()
    kp = unwrap(slopkernels.get("amd_flash_decode_partial")).specialize(
      B=1, H_KV=2, G=2, d_blks=1, n_blks=2, C=2, valid_kv_offset=0, waves=16)
    partial, stats = UOp.alloc((1, 4, 2, 64), dtypes.half, device=dev), UOp.alloc((1, 4, 2, 2), dtypes.float32, device=dev)
    cp = kp.call(partial, stats, q_dec.uop, kv_fa.uop, UOp.const(53))
    got_fd = _krun("amd_flash_decode_combine", (1, 4, 1, 64), dev, partial.after(cp), stats.after(cp), UOp.const(53),
                   B=1, H=4, C=2, d_blks=1, valid_kv_offset=0, step_keys=16).numpy()
    ref_fd = q_dec.float().scaled_dot_product_attention(kv_fa[0, :, :, :53].float(), kv_fa[1, :, :, :53].float(), enable_gqa=True).numpy()
    np.testing.assert_allclose(got_fd, ref_fd, rtol=5e-3, atol=5e-3)
    if tgt[0] != 9:
      q_pre = Tensor(rng.standard_normal((1, 4, 32, 64)).astype(np.float16)).realize()
      got_fp = _krun(f"{arch_gemm}_flash_attn", (4, 32, 64), dev, q_pre.reshape(4, 32, 64), kv_fa, UOp.const(64), UOp.const(32),
                     BH=4, m_blks=1, B=1, H_KV=2, d_blks=1, n_blks=4, waves_m=2, bn_tiles=2, gqa_w=1, valid_kv_offset=0,
                     q_start_from_end=1).reshape(1, 4, 32, 64)
      mask = Tensor.full((32, 64), float("-inf"), dtype=dtypes.float32).triu(64 - 32 + 1)
      ref_fp = q_pre.float().scaled_dot_product_attention(kv_fa[0, :, :, :64].float(), kv_fa[1, :, :, :64].float(), attn_mask=mask, enable_gqa=True)
      np.testing.assert_allclose(got_fp.numpy(), ref_fp.numpy(), rtol=5e-3, atol=5e-3)
    # 5. Gated delta prefill (start_pos=0 resets, start_pos=3 preserves)
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
      got_gd = _krun("amd_gated_delta", (1, 2, 4, 8), dev, gd_q, gd_k, gd_v, gd_b, gd_a.reshape(1, 2, 4, 1), gd_s,
                     (gd_q * gd_k).sum(-1).contiguous(), UOp.const(sp), batch=1, heads=2, tokens=4, v_tiles=2, k_blks=1, alpha_k=1).numpy()
      np.testing.assert_allclose(got_gd, np.stack(outs_ref, axis=2), rtol=2e-4, atol=2e-6)
      np.testing.assert_allclose(gd_s.numpy(), st_ref, rtol=2e-4, atol=2e-6)
    # 6. mHC coeffs
    hc_mix = Tensor(rng.normal(0, 0.2, (2, 24)).astype(np.float32)).realize()
    hc_flat = Tensor(rng.standard_normal((2, 128)).astype(np.float32)).contiguous().realize()
    hc_base = Tensor(rng.normal(0, 0.2, (24,)).astype(np.float16)).realize()
    hc_scale = Tensor(rng.uniform(0.5, 1.5, (3,)).astype(np.float16)).realize()
    got_hc = _krun("amd_hc_coeffs_f16_f16", (2, 24), dev, hc_mix, hc_flat, hc_base, hc_scale,
                   tokens=2, features=128, waves=1, iters=2, eps=1e-6, norm_eps=1e-6).numpy()
    np.testing.assert_allclose(got_hc[:, 8:].reshape(2, 4, 4).sum(-1), 1.0, rtol=2e-2, atol=2e-2)

  def test_multi_gpu_tp(self):
    if (tgt := _amd_target()) is None or not DEV.interface.startswith("MOCK"): self.skipTest("MOCK AMD required for 2-GPU TP")
    devs, rng, arch = ("AMD:0", "AMD:1"), np.random.default_rng(31), "cdna" if tgt[0] == 9 else "rdna"
    # 1. Column-sharded GEMV
    _, w_u16, w_ref = _make_q8_0(rng, 64, 256)
    w_col = w_u16.reshape(64, -1).shard(devs, axis=0).flatten().realize()
    x_rep = Tensor(rng.standard_normal((2, 256)).astype(np.float32)).shard(devs, axis=None).realize()
    got_col = _krun(f"{arch}_gemv_q8_0", (2, 32), devs, w_col, *_q8(x_rep, 2, 256, devs), axis=1,
                    tokens=2, out_features=32, in_features=256, waves=1, raw_len=int(w_col.uop.max_shard_shape[0]), group_rows=0)
    np.testing.assert_allclose(got_col.numpy(), (x_rep.to("AMD:0") @ w_ref.T).numpy(), rtol=3e-2, atol=5e-2)
    # 2. Row-sharded GEMV (split-K all-reduce)
    _, w_u16_r, w_ref_r = _make_q8_0(rng, 32, 512)
    w_row = w_u16_r.reshape(32, 2, -1).permute(1, 0, 2).contiguous().shard(devs, axis=0).flatten().realize()
    x_sh = Tensor(rng.standard_normal((2, 512)).astype(np.float32)).shard(devs, axis=1).realize()
    ref_row = (x_sh.to("AMD:0") @ w_ref_r.T).numpy()
    got_row = _krun(f"{arch}_gemv_q8_0", (2, 32), devs, w_row, *_q8(x_sh, 2, 256, devs), axis=0,
                    tokens=2, out_features=32, in_features=256, waves=1, raw_len=int(w_row.uop.max_shard_shape[0]),
                    group_rows=0).reshape(2, 2, 32).sum(0)
    np.testing.assert_allclose(got_row.numpy(), ref_row, rtol=3e-2, atol=2e-2 * np.abs(ref_row).max())
    # 3. Expert-sharded MoE
    _, mw_u16, mw_ref = _make_q8_0(rng, 4 * 16, 256)
    mw_sh = mw_u16.reshape(4, -1).shard(devs, axis=0).flatten().realize()
    sel_sh = Tensor(np.array([3, 1], dtype=np.int32)).shard(devs, axis=None).realize()
    xm_sh = Tensor(rng.standard_normal((1, 256)).astype(np.float32)).shard(devs, axis=None).realize()
    got_ms = _krun(f"{arch}_moe_q8_0", (2, 16), devs, mw_sh, *_q8(xm_sh, 1, 256, devs), sel_sh, axis=0, rows=2, x_rows=1,
                   out_features=16, in_features=256, waves=1, raw_len=int(mw_sh.uop.max_shard_shape[0]), x_share=2,
                   ndev=2).reshape(2, 2, 16).sum(0).numpy()
    for j, eid in enumerate((3, 1)):
      np.testing.assert_allclose(got_ms[j], mw_ref.numpy().reshape(4, 16, 256)[eid] @ xm_sh.to("AMD:0").numpy()[0], rtol=3e-2, atol=5e-2)
    # 4. Head-sharded flash decode
    q_fa = Tensor(rng.standard_normal((1, 4, 1, 64)).astype(np.float16)).realize()
    kv_fa = Tensor(rng.standard_normal((2, 1, 2, 128, 64)).astype(np.float16)).realize()
    q_sh, kv_sh = q_fa.shard(devs, axis=1).realize(), kv_fa.shard(devs, axis=2).realize()
    kp = unwrap(slopkernels.get("amd_flash_decode_partial")).specialize(
      B=1, H_KV=1, G=2, d_blks=1, n_blks=2, C=2, valid_kv_offset=0, waves=16)
    partial, stats = UOp.alloc((1, 2, 2, 64), dtypes.half, device=devs, axis=1), UOp.alloc((1, 2, 2, 2), dtypes.float32, device=devs, axis=1)
    cp = kp.call(partial, stats, q_sh.uop, kv_sh.uop, UOp.const(41))
    got_fs = _krun("amd_flash_decode_combine", (1, 2, 1, 64), devs, partial.after(cp), stats.after(cp), UOp.const(41), axis=1,
                   B=1, H=2, C=2, d_blks=1, valid_kv_offset=0, step_keys=16).numpy()
    ref_fs = q_fa.float().scaled_dot_product_attention(kv_fa[0, :, :, :41].float(), kv_fa[1, :, :, :41].float(), enable_gqa=True).numpy()
    np.testing.assert_allclose(got_fs, ref_fs, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
  unittest.main()
