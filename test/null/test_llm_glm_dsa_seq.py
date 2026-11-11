import os
import functools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tinygrad import Tensor, UOp, nn, dtypes, Context
from tinygrad.uop.ops import Ops
from tinygrad.llm import glm_dsa
from tinygrad.llm.glm_dsa import Indexer, MLATransformerBlock, Transformer


class TestGLMDSASequence(unittest.TestCase):
  def setUp(self):
    self.enterContext(Context(DEV="CPU"))
    self.devs = ("CPU",)

  def test_region_alignment_drives_move_chunks(self):
    self.assertEqual(Transformer.SEQ_MOVE, Transformer.REGION_ALIGN)
    self.assertEqual(Transformer.REGION_ALIGN % (64*14), 0)

  def test_direct_key_prefill_region(self):
    idx = Indexer.__new__(Indexer)
    idx.idx = SimpleNamespace(kpool=1, n_heads=16, head_dim=4, top_k=2, rope_dim=2, rope_interleave=False)
    idx.max_context, idx.shard, idx.cache_dk = 16, 0, Tensor.zeros(32, 4, dtype="half")
    idx.k_norm = idx.attn_k = lambda x: x
    idx.attn_q_b = lambda x: x.unsqueeze(1).expand(x.shape[0], 16, x.shape[1], x.shape[2]).reshape(*x.shape[:2], 16*x.shape[2])
    idx.proj = lambda x: Tensor.ones(*x.shape[:2], 16)
    x = Tensor(np.arange(1, 9, dtype=np.float16).reshape(1, 2, 4))
    q = Tensor.zeros(1, 2, 4)
    old_base = glm_dsa._seq_base[0]
    glm_dsa._seq_base[0] = 8
    try:
      with patch.object(glm_dsa, "indexer_decode_index_rows", return_value=Tensor.zeros(2, 4, dtype="int32")) as score:
        idx.dsa_picks(x, q, 2, Tensor([[1, 0], [1, 0]]))
      score.assert_called_once()
      self.assertEqual(score.call_args.kwargs["pbase"], 8)
      self.assertEqual(score.call_args.kwargs["width"], 16)
      score.call_args.args[1].realize()  # the mocked scorer does not carry pk's store into its return graph
      # start=2, base=8: the two direct-key rows land at 10..12, not 2..4.
      np.testing.assert_array_equal(idx.cache_dk[10:12].numpy(), x[0].half().numpy())
      np.testing.assert_array_equal(idx.cache_dk[2:4].numpy(), np.zeros((2, 4), dtype=np.float16))
    finally: glm_dsa._seq_base[0] = old_base

  def test_token_after_prefill_uses_region_base(self):
    model = Transformer.__new__(Transformer)
    model._token_buf = Tensor.full((1, 32), -1, dtype="int32")
    model.output_norm = lambda x: x
    model._sample = lambda h, temp, device: Tensor([9], dtype="int32")
    old_base = glm_dsa._seq_base[0]
    glm_dsa._seq_base[0] = 8
    try:
      tokens = model._token_buf[:, 8:10]
      model._tail(Tensor.zeros(1, 2, 3), tokens, 0, None).realize()
      self.assertEqual(model._token_buf[0, 10].item(), 9)
      self.assertEqual(model._token_buf[0, 2].item(), -1)
    finally: glm_dsa._seq_base[0] = old_base

  def test_seq_embedding_is_a_gather(self):
    # (a view of the token buffer at seq_base+start_pos (two variables): its embedding a lookup of the rows, not a reduce over the vocabulary)
    model = Transformer.__new__(Transformer)
    model.token_embd = nn.Embedding(1024, 64)
    buf = Tensor.zeros(1, 4096, dtype="int32").contiguous().realize()
    vb, sp = UOp.variable("seq_base", 0, 2048), UOp.variable("start_pos", 0, 1023)
    nt = UOp.variable("toks", 1, 256)
    old_base = glm_dsa._seq_base[0]
    glm_dsa._seq_base[0] = vb
    try: x = model._embed(buf[:, vb+sp:vb+sp+nt])
    finally: glm_dsa._seq_base[0] = old_base
    kernels = [si for si in x.schedule_linear().src if si.src[0].op is Ops.SINK]
    # (the reduce of a one-hot: the one-hot (tokens, vocabulary) of bools stored by a kernel of its own)
    stores = [u for k in kernels for u in k.src[0].toposort() if u.op is Ops.STORE]
    self.assertFalse(any(u.src[1].dtype == dtypes.bool for u in stores), "the embedding is a reduce of a one-hot of the vocabulary")

  def test_main_path_token_buffer_has_the_regions(self):
    # (the main path making the token buffer first (generate, warmup): GLM_SEQ's regions in it too, not max_context+1)
    model = Transformer.__new__(Transformer)
    model.max_context, model.config = 65536, SimpleNamespace(cache_rows=107520)
    self.assertEqual(model._main_token_rows(), model._token_rows())
    self.assertGreaterEqual(model._main_token_rows(), model.seq_rows() + Transformer.REGION_ALIGN + 1)
    model.config = SimpleNamespace(cache_rows=0)
    self.assertEqual(model._main_token_rows(), 65537)

  def test_q8_pack_needs_whole_warps_of_groups(self):
    # (MLA's 576 dims (18 groups of 32): the kernels pack 8 groups a warp, the last 2 would be left out: refused, mla_sparse_seq sends half)
    from tinygrad.llm.kernels.amd_glm_dsa import _q8_pack_kernel, _q8_unpack_kernel
    self.assertRaises(AssertionError, _q8_pack_kernel, Tensor.empty(4, 18, 9, dtype="uint32").uop, Tensor.empty(4, 576, dtype="half").uop)
    self.assertRaises(AssertionError, _q8_unpack_kernel, Tensor.empty(4, 576, dtype="half").uop, Tensor.empty(4, 18, 9, dtype="uint32").uop)

  def test_seq_p2p_pack_kernels(self):
    # (SEQ_P2P_PACK: mla_sparse_seq's one shot gather and reduce scatter of packed words: default off, the kernels build)
    from tinygrad.llm.kernels.amd_glm_dsa import _p2p_allreduce_kernel, _seq_p2p_pack, P2P_AR_CLASSES, P2P_AR_MAXG, P2P_AR_MAXW
    with patch.dict(os.environ, {"SEQ_P2P_PACK": "0"}): self.assertIsNone(_seq_p2p_pack())
    with patch.dict(os.environ, {"SEQ_P2P_PACK": "3x8"}): self.assertEqual(_seq_p2p_pack(), (3, 8))
    N, n = 4, 256
    mb = [Tensor.empty(P2P_AR_CLASSES*P2P_AR_MAXG*2*(N+1)*P2P_AR_MAXW, dtype="uint64") for _ in range(N)]
    counter = Tensor.empty(P2P_AR_CLASSES*P2P_AR_MAXG+1, dtype="int32")
    for kw in (dict(gather=True), dict(scatter=True)):
      out, x = Tensor.empty(N*n*3, dtype="float32"), Tensor.empty(1, N*n*3, dtype="float32")
      fxn = functools.partial(_p2p_allreduce_kernel, N=N, n=n, T=256, EP=1, pack=(3, 8), **kw)
      call = Tensor.custom_kernel(out, x, counter, *mb, fxn=fxn)[0].uop.src[1]
      self.assertIs(call.src[0].op, Ops.SINK)
    with self.assertRaises(AssertionError):
      Tensor.custom_kernel(out, x, counter, *mb, fxn=functools.partial(_p2p_allreduce_kernel, N=N, n=n, T=256, EP=1, pack=(3, 8), half=True,
                                                                       gather=True))
    # (P2P_FLAG: four halves a word, a flag a workgroup: the half gather and reduce scatter build)
    for kw in (dict(gather=True), dict(scatter=True)):
      out, x = Tensor.empty(N*n*4, dtype="float32"), Tensor.empty(1, N*n*4, dtype="float32")
      fxn = functools.partial(_p2p_allreduce_kernel, N=N, n=n, T=256, EP=1, half=True, flag=True, **kw)
      self.assertIs(Tensor.custom_kernel(out, x, counter, *mb, fxn=fxn)[0].uop.src[1].src[0].op, Ops.SINK)
    # (and two floats a word: the float gathers and all to all)
    for kw in (dict(gather=True), dict(scatter=True, gather=True)):
      out, x = Tensor.empty(N*N*n*2, dtype="float32"), Tensor.empty(1, N*n*2, dtype="float32")
      fxn = functools.partial(_p2p_allreduce_kernel, N=N, n=n, T=256, EP=1, flag=True, **kw)
      self.assertIs(Tensor.custom_kernel(out, x, counter, *mb, fxn=fxn)[0].uop.src[1].src[0].op, Ops.SINK)
    # Eight-bit epochs accepted call 1's stale word at call 257. Refuse the old SEQ formats before model construction.
    for unsafe in ("6x8", "4x12", "3x7"):
      with patch.dict(os.environ, {"SEQ_P2P_PACK": unsafe}), self.assertRaises(AssertionError): _seq_p2p_pack()

  def test_sample_greedy_rows_exact(self):
    # (a row of temperature 0 is the argmax of its logits (the first of a tie), whatever the noise: it was logits / 1e-12 plus Gumbel noise,
    # a tie broken by the noise and a u the device's log takes to 0 an infinite noise (batched greedy decoding differed run to run))
    model = Transformer.__new__(Transformer)
    model.config = SimpleNamespace(vocab_size=512)
    lg = np.zeros((16, 512), dtype=np.float32)
    lg[:, 7] = lg[:, 300] = 5.0  # (a tie: 7 the first)
    lg[8:, 42] = 9.0
    model.output = lambda h: h
    temps = Tensor([[0.0]]*12 + [[0.7]]*4, device="CPU")
    for seed in range(3):
      noise = (Tensor([seed], dtype=dtypes.uint32, device="CPU"), Tensor(np.arange(16, dtype=np.int32), device="CPU"))
      tok = model._sample(Tensor(lg, device="CPU"), temps, "CPU", noise).numpy().reshape(-1)
      self.assertEqual(list(tok[:12]), [7]*8 + [42]*4)
    # (top_p rows too: (temperature, top_p) pairs)
    temps2 = Tensor([[0.0, 0.9]]*16, device="CPU")
    tok = model._sample(Tensor(lg, device="CPU"), temps2, "CPU", noise).numpy().reshape(-1)
    self.assertEqual(list(tok), [7]*8 + [42]*8)
    # (a u the device's log takes to 0: -log(u) 0, its noise an infinity. as a u of 1 here: column 100's)
    un = np.full((16, 512), 0.5, dtype=np.float32)
    un[:, 100] = 1.0
    with patch.object(glm_dsa, "_hash_uniform", lambda seed, keys, like: Tensor(un, device="CPU")):
      tok = model._sample(Tensor(lg, device="CPU"), temps, "CPU", noise).numpy().reshape(-1)
    self.assertEqual(list(tok[:12]), [7]*8 + [42]*4)

  def test_p2p_late_raises(self):
    # (a p2p poll that ran out: its sums wrong. a long wait reads the devices' notes and raises, never returns the tokens)
    import time
    model = Transformer.__new__(Transformer)
    model.token_embd = SimpleNamespace(weight=SimpleNamespace(device=("NULL:0", "NULL:1")))
    with patch.object(glm_dsa, "p2p_check", side_effect=RuntimeError("a p2p poll ran out")) as chk:
      model._p2p_mark = time.perf_counter()
      self.assertEqual(model._wait_host(lambda: 5, "x"), 5)
      chk.assert_not_called()
      model._p2p_mark = time.perf_counter() - 2*glm_dsa.P2P_CHECK_S
      with self.assertRaises(RuntimeError): model._wait_host(lambda: 5, "x")
    from tinygrad.llm.kernels import amd_glm_dsa
    with patch.object(amd_glm_dsa, "p2p_late", return_value=[0, 7]), self.assertRaises(RuntimeError): amd_glm_dsa.p2p_check(("NULL:0", "NULL:1"))
    with patch.object(amd_glm_dsa, "p2p_late", return_value=[0, 0]): amd_glm_dsa.p2p_check(("NULL:0", "NULL:1"))
    # (the notes: the counter after the device's index, cleared)
    from tinygrad.device import Buffer
    from tinygrad.llm.kernels.amd_glm_dsa import P2P_AR_NCTR as NC
    bufs = [Buffer("CPU", (NC+2)*4, initial_value=np.array([5]*NC + [d, 7*d], dtype=np.int32).tobytes()) for d in range(2)]
    with patch.object(amd_glm_dsa, "p2p_allreduce", return_value=SimpleNamespace(counter_bufs=bufs)):
      self.assertEqual(amd_glm_dsa.p2p_late(("CPU", "CPU"), clear=True), [0, 7])
      self.assertEqual(amd_glm_dsa.p2p_late(("CPU", "CPU")), [0, 0])
    self.assertEqual(list(np.frombuffer(bufs[1].as_memoryview(), dtype=np.int32)[-3:]), [5, 1, 0])
    # (the packed allreduce's kernel builds with the ranges' counters)
    from tinygrad.llm.kernels.amd_glm_dsa import _p2p_allreduce_kernel, P2P_AR_CLASSES, P2P_AR_MAXG, P2P_AR_MAXW, P2P_AR_NCTR
    N = 4
    mb = [Tensor.empty(P2P_AR_CLASSES*P2P_AR_MAXG*2*(N+1)*P2P_AR_MAXW, dtype="uint64") for _ in range(N)]
    counter = Tensor.empty(P2P_AR_NCTR+2, dtype="int32")
    for T, EP in ((256, 1), (512, 8)):
      n = 2*T*EP
      out, x = Tensor.empty(n*4, dtype="float32"), Tensor.empty(N, n*4, dtype="float32")
      fxn = functools.partial(_p2p_allreduce_kernel, N=N, n=n, T=T, EP=EP, two_shot=True, pack=(4, 12))
      self.assertIs(Tensor.custom_kernel(out, x, counter, *mb, fxn=fxn)[0].uop.src[1].src[0].op, Ops.SINK)
    with self.assertRaises(AssertionError):
      Tensor.custom_kernel(out, x, counter, *mb, fxn=functools.partial(_p2p_allreduce_kernel, N=N, n=192, T=64, EP=1, two_shot=True, pack=(4, 12)))

  def test_seq_i8_reaches_dsa_kernels(self):
    # (glm5next's _forward_seqs_glm sets amd_glm's flag: the DSA kernels' (amd_glm_dsa's) is set by the DSA model's, restored after)
    from tinygrad.helpers import getenv
    from tinygrad.llm.kernels import amd_glm_dsa
    seen = []
    def fwd(self, vs): seen.append(amd_glm_dsa.Q8_ROWS_I8[0])
    for env, want in (("1", True), ("0", False)):
      with patch.dict(os.environ, {"SEQ_I8": env}), patch.object(glm_dsa.GLMSeqDriver, "_forward_seqs_glm", fwd):
        getenv.cache_clear()
        Transformer._forward_seqs_glm(Transformer.__new__(Transformer), ())
      self.assertEqual(seen[-1], want)
      self.assertFalse(amd_glm_dsa.Q8_ROWS_I8[0])
    getenv.cache_clear()

  def test_rows_f16_kernels(self):
    # (ROWS_F16: the f16 WMMA rows of batched decoding on _rows_coop_f16_wmma_kernel (default) or _rows_f16_wmma_kernel; the kernels build)
    from tinygrad.helpers import getenv
    from tinygrad.llm.kernels import amd_glm_dsa as K
    getenv.cache_clear()
    with patch.object(K, "_wmma_rdna4", lambda d: True):
      self.assertEqual(K._rows_f16_config(16, 2048, 6144, K.Q8_0, "NULL"), dict(coop=2, chunk=16))  # (the default)
      self.assertEqual(K._rows_f16_config(48, 2048, 6144, K.Q8_0, "NULL")["unroll"], 8)  # (3 token tiles: past ROWS_COOP=2's waves)
      self.assertIsNone(K._rows_f16_config(16, 11072, 6144, K.Q6_K, "NULL"))
      with patch.dict(os.environ, {"ROWS_F16": "0"}):
        getenv.cache_clear()
        self.assertIsNone(K._rows_f16_config(16, 2048, 6144, K.Q8_0, "NULL"))
      with patch.dict(os.environ, {"ROWS_F16": "1", "ROWS_COOP": "0", "ROWS_CHUNK": "8"}):
        getenv.cache_clear()
        self.assertEqual(K._rows_f16_config(16, 2048, 6144, K.Q8_0, "NULL")["chunk"], 8)
        self.assertEqual(K._rows_f16_config(16, 6144, 160, K.Q8_0, "NULL")["chunk"], 0)  # (5 groups)
        self.assertIsNone(K._rows_f16_config(128, 2048, 6144, K.Q8_0, "NULL"))
      getenv.cache_clear()
    for typ, raw, chunk, pkh in ((K.Q8_0, Tensor.empty(64*256*17//32, dtype="half"), 8, False), (K.Q8_0, Tensor.empty(64*256*17//64), 8, True),
                                 (K.Q6_K, Tensor.empty(64*105, dtype="uint16"), 0, False)):
      out, x = Tensor.empty(32, 64, dtype="float32"), Tensor.empty(32, 256, dtype="half")
      fxn = functools.partial(K._rows_f16_wmma_kernel, out_features=64, in_features=256, ggml_type=typ, chunk=chunk, pkh=pkh)
      self.assertIs(Tensor.custom_kernel(out, raw, x, fxn=fxn)[0].uop.src[1].src[0].op, Ops.SINK)
    out, x, raw = Tensor.empty(32, 64, dtype="float32"), Tensor.empty(32, 512, dtype="half"), Tensor.empty(64*512*17//32, dtype="uint16")
    fxn = functools.partial(K._rows_coop_f16_wmma_kernel, out_features=64, in_features=512, ggml_type=K.Q8_0, coop=2, chunk=16)
    self.assertIs(Tensor.custom_kernel(out, raw, x, fxn=fxn)[0].uop.src[1].src[0].op, Ops.SINK)

  def test_half_rows_local_scales(self):
    # (a row parallel layer's input in half: every device's part its own scale, no cross device max; the same halves below HALF_ROWS_MAX)
    from tinygrad.llm.kernels.amd_glm_dsa import _half_rows, HALF_ROWS_MAX
    devs = ("CPU:0", "CPU:1", "CPU:2", "CPU:3")
    a = np.random.default_rng(0).standard_normal((16, 64)).astype(np.float32) * 100
    a[3, 40] = HALF_ROWS_MAX * 5  # (a row past half's range on device 2 only)
    x = Tensor(a).shard(devs, axis=1).realize()
    xh, sc = _half_rows(x, 16, 4)
    self.assertEqual(sc.shape, (4, 16, 1))
    self.assertEqual(sc.uop.axis, 0)
    s = sc.numpy()
    self.assertEqual(s[2, 3, 0], 8.0)
    self.assertEqual(float(np.delete(s.reshape(-1), 2*16+3).max()), 1.0)
    want = (a.reshape(16, 4, 16) / s.transpose(1, 0, 2)).astype(np.float16).reshape(16, 64)
    np.testing.assert_array_equal(xh.numpy(), want)
    # below the max: the global path's halves exactly
    b = a.copy()
    b[3, 40] = 1.0
    np.testing.assert_array_equal(_half_rows(Tensor(b).shard(devs, axis=1), 16, 4)[0].numpy(), _half_rows(Tensor(b), 16)[0].numpy())

  def test_prefill_exchange_kernels(self):
    # (SEQ_GATHER_HALF: the gather stores halves, no bitcast copy; IDX_MERGE_SPLIT: a device's rows' merge, its lists into the block's rows;
    # SEQ_ATTN_TOK: the wmma attention's lists of hg head groups a token)
    from tinygrad.llm.kernels.amd_glm_dsa import _seq_gather_kernel, _rows_into_kernel, _mla_sparse_wmma_kernel, _seq_need_kernel
    def sink(*ts, fxn): return Tensor.custom_kernel(*ts, fxn=fxn)[0].uop.src[1].src[0]
    sp = UOp.const(100, dtypes.weakint)
    part, dev = Tensor.empty(64, 144, dtype="uint64"), Tensor.empty(1, dtype="int32")
    for dt, W in (("half", 576), ("uint64", 144)):
      k = sink(Tensor.empty(256, W, dtype=dt), part, dev, fxn=functools.partial(_seq_gather_kernel, start_pos=sp, k=1, n=4))
      self.assertIs(k.op, Ops.SINK)
    # (SEQ_GATHER_FUSED: every part in one kernel)
    from tinygrad.llm.kernels.amd_glm_dsa import _seq_gather_fused_kernel
    k = sink(Tensor.empty(256, 576, dtype="half"), dev, *[Tensor.empty(64, 144, dtype="uint64") for _ in range(4)],
             fxn=functools.partial(_seq_gather_fused_kernel, start_pos=sp, n=4))
    self.assertIs(k.op, Ops.SINK)
    # (SEQ_GATHER_UNION: the positions of the lists marked, the gather of only those)
    need = Tensor.zeros(256, dtype="int32").contiguous()
    self.assertIs(sink(need, Tensor.empty(8, 34, dtype="int32"), fxn=_seq_need_kernel).op, Ops.SINK)
    self.assertIs(sink(Tensor.empty(256, 576, dtype="half"), part, dev, need, fxn=functools.partial(_seq_gather_kernel, start_pos=sp, k=1, n=4)).op,
                  Ops.SINK)
    self.assertIs(sink(Tensor.empty(64, 66, dtype="int32"), Tensor.empty(40, 66, dtype="int32"),
                       fxn=functools.partial(_rows_into_kernel, row0=8, rows=30)).op, Ops.SINK)
    q, cache, index = Tensor.empty(5*16*4, 576, dtype="half"), Tensor.empty(256, 576, dtype="half"), Tensor.empty(4, 66, dtype="int32")
    a, b = (sink(Tensor.empty(5*16*4, 576), q, cache, index, fxn=functools.partial(_mla_sparse_wmma_kernel, scale=0.1, valid_len=sp, hg=hg))
            for hg in (0, 5))
    self.assertIsNot(a, b)
    # (Q8_GEMM_SC: the gemm scales x's rows as it loads them)
    from tinygrad.llm.kernels.amd_glm_dsa import _q8_gemm_kernel
    M, N, K = 96, 128, 256
    raw, xg, scg = Tensor.empty(N*(K//32)*17, dtype="uint16"), Tensor.empty(M, K, dtype="half"), Tensor.empty(M, 1)
    for extra in ((), (scg,)):
      fxn = functools.partial(_q8_gemm_kernel, token_tile=96, out_features=N, in_features=K)
      self.assertIs(sink(Tensor.empty(M, N), raw, xg, *extra, fxn=fxn).op, Ops.SINK)
    # (GEMM_PAD: the token tiles of a chunk's tokens only)
    rows = UOp.variable("toks", 1, M)
    fxn = functools.partial(_q8_gemm_kernel, token_tile=96, out_features=N, in_features=K, rows=rows)
    k = sink(Tensor.empty(M, N), raw, xg, fxn=fxn)
    self.assertIn(rows, k.toposort())
    from tinygrad.llm.kernels.amd_glm_dsa import _iq4_linear_f16_wmma_kernel, _q8k_quantize_kernel
    fxn = functools.partial(_iq4_linear_f16_wmma_kernel, out_features=N, in_features=K, rdna4=True, rows=rows)
    self.assertIn(rows, sink(Tensor.empty(M, N), Tensor.empty(N*K//2, dtype="uint32"), xg, Tensor.empty(256, dtype="uint32"), fxn=fxn).toposort())
    qk = Tensor.custom_kernel(Tensor.empty(M, K//32, 8, dtype="uint32"), Tensor.empty(M, K//256), xg,
                              fxn=functools.partial(_q8k_quantize_kernel, tokens=M, in_features=K, rows=rows))[0].uop.src[1].src[0]
    self.assertIn(rows, qk.toposort())

  def test_lists_pad(self):
    # (LISTS_PAD: the counts of a chunk's padding rows zero, the tokens' rows as they were)
    from tinygrad.llm.kernels.amd_glm_dsa import lists_pad, _lists_pad_kernel
    T = UOp.variable("toks", 1, 8).bind(5)
    k = Tensor.custom_kernel(Tensor.empty(8, 5, dtype="int32"), fxn=functools.partial(_lists_pad_kernel, T=T.unbind_all()[0]))[0].uop.src[1].src[0]
    self.assertIs(k.op, Ops.SINK)
    stores = [u for u in k.toposort() if u.op is Ops.STORE]
    self.assertEqual(len(stores), 1)
    self.assertIs(lists_pad(t:=Tensor.empty(8, 5, dtype="int32"), 8), t)
    # (IDX_PAD: the indexer's scores and candidates of the first rows only)
    from tinygrad.llm.kernels.amd_glm_dsa import _indexer_scores_kernel, _pool_cand_kernel
    rv, sp = UOp.variable("rv", 0, 32), UOp.const(100, dtypes.weakint)
    sc = Tensor.custom_kernel(Tensor.empty(32, 256), Tensor.empty(32, 16, 128, dtype="half"), Tensor.empty(256, 128, dtype="half"),
                              Tensor.empty(32, 16), fxn=functools.partial(_indexer_scores_kernel, pools=sp, kpool=1, rows=rv))[0]
    self.assertIn(rv, sc.uop.src[1].src[0].toposort())
    cd = Tensor.custom_kernel(Tensor.empty(32, 64, 2), Tensor.empty(32, 256), Tensor.empty(1, dtype="int32"),
                              fxn=functools.partial(_pool_cand_kernel, q_start=sp, pools=sp, k=64, kpool=1, n=4, valid=rv))[0]
    self.assertIn(rv, cd.uop.src[1].src[0].toposort())

  def test_dsa_stash_roundtrip(self):
    def t(shape, axis=None):
      x = Tensor(np.arange(np.prod(shape), dtype=np.float16).reshape(shape))
      return x.shard(self.devs, axis) if axis is not None else x.to(self.devs)

    model = Transformer.__new__(Transformer)
    model.config = SimpleNamespace(spec_drafts=2, cache_rows=128, kv_lora_rank=512)
    model.token_embd = SimpleNamespace(weight=SimpleNamespace(device=self.devs))
    model.SEQ_MOVE, model.seq_rows = 32, lambda: 128
    idx = Indexer.__new__(Indexer)
    idx.idx, idx.shard, idx.cache_dk = SimpleNamespace(kpool=1), 0, t((128, 2))
    block = MLATransformerBlock.__new__(MLATransformerBlock)
    block.indexer, block.cache_s = idx, t((128, 2), axis=0)
    model.blk, model.mtp = [block], None
    model.seq_spec_h, model.seq_ckpt_h = t((12, 3)), t((2, 3))
    model.seq_ckpt_rings = []
    sources = [block.cache_s, idx.cache_dk, model.seq_spec_h, model.seq_ckpt_h]
    values = [x.numpy().copy() for x in sources]
    with patch.dict(os.environ, {"SEQ_STASH_GB": "0.000002", "SEQ_STASH_ENTRIES": "1"}): model._seq_stash_init()
    self.assertEqual(model.stash_entries, 1)
    self.assertGreater(model.stash_chunks, 0)
    self.assertEqual(model._stash_rings, [])

    model._stash_kv_copy(True, 1, 0, 0)
    model._stash_state_copy(True, 0, 0, 0, 0, 0)
    for x in sources: x.assign(Tensor.zeros_like(x)).realize()
    model._stash_kv_copy(False, 1, 32, 0)  # region row 32 restore from host chunk 0
    model._stash_state_copy(False, 1, 0, 0, 0, 0)
    for actual, saved in zip((block.cache_s, idx.cache_dk), values):
      expected = np.zeros_like(actual.numpy())
      expected[32:64] = saved[:32]
      np.testing.assert_array_equal(actual.numpy(), expected)
    spec, ckpt = np.zeros((12, 3), dtype=np.float16), np.zeros((2, 3), dtype=np.float16)
    spec[6], ckpt[1] = values[2][0], values[3][0]
    np.testing.assert_array_equal(model.seq_spec_h.numpy(), spec)
    np.testing.assert_array_equal(model.seq_ckpt_h.numpy(), ckpt)


if __name__ == '__main__': unittest.main()
