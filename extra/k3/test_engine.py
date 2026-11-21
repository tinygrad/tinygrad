#!/usr/bin/env python3
# the engine's forward against master's model with the routed experts on a tiny random kimi-k3 on two CPU "gpus": every expert in the pools
# (a slot a code, each gpu its units), the host's part stubbed by exact routing (TAU=0: a token's top k in the store's two parts)
# DEV=CPU PYTHONPATH=. python3 extra/k3/test_engine.py (DEV=AMD: the jitted steps with the host's step_io too)
import ctypes, itertools, json, pathlib, re, tempfile, unittest
from unittest.mock import patch
from typing import Callable
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType, quants
from tinygrad import Tensor, Device, UOp, nn, dtypes
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import get_call_arg_uops, get_call_outs_ins
from tinygrad.helpers import round_up
from tinygrad.llm.kernels.amd import amd_custom_kernels_supported
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, ExpertGating, MLATransformerBlock, Indexer, IndexerConfig
from extra.k3.engine import Engine
from extra.k3.host import IoT, host_c
from extra.k3.draft import Eagle3, DSpark, MTP
from extra.k3.eval import window_evaluate, window_dump

DIM, LAT, FF, E, K, G, VOCAB = 64, 32, 64, 8, 4, 2, 32
CONFIG = TransformerConfig(num_blocks=3, dim=DIM, hidden_dim=FF, n_heads=2, n_kv_heads=1, norm_eps=1e-5, vocab_size=VOCAB, head_dim=24,
  rope_theta=10000.0, rope_dim=8, v_head_dim=16, max_context=16, num_experts=E, num_experts_per_tok=K, expert_gating_func=ExpertGating.SIGMOID,
  norm_topk_prob=True, q_lora_rank=16, kv_lora_rank=16, shared_expert_dim=FF, ssm_layers=(True, True, False), attn_output_gate=True,
  leading_dense_blocks=1, dense_hidden_dim=64, expert_bias=True, shared_expert_gate=False, expert_latent_dim=LAT, attn_res_block_size=1,
  situ=(4.0, 25.0), ssm=SSMConfig(conv_kernel=2, state_size=16, group_count=2, time_step_rank=2, inner_size=32, kda=True, gate_lower_bound=-5.0,
                                    full_rank=(True, False)))

def write(path:pathlib.Path, weights:dict[str, np.ndarray], experts:bool):
  w = GGUFWriter(path, "kimi-k3")
  for key, value in {'context_length': 16, 'embedding_length': DIM, 'feed_forward_length': 64, 'expert_feed_forward_length': FF, 'block_count': 3,
                     'attention.head_count': 2, 'ssm.conv_kernel': 2, 'kda.head_dim': 16, 'attention.q_lora_rank': 16, 'attention.kv_lora_rank': 16,
                     'rope.dimension_count': 8, 'attention.key_length_mla': 24, 'attention.value_length_mla': 16, 'expert_count': E,
                     'expert_used_count': K, 'expert_shared_count': 1, 'leading_dense_block_count': 1, 'expert_gating_func': 2,
                     'expert_latent_length': LAT, 'attn_res.block_size': 1}.items():
    w.add_uint32(f'kimi-k3.{key}', value)
  w.add_array('kimi-k3.attention.head_count_kv', [0, 0, 1])
  w.add_float32('kimi-k3.attention.layer_norm_rms_epsilon', 1e-5)
  w.add_float32('kimi-k3.kda.gate_lower_bound', -5.0)
  w.add_float32('kimi-k3.activation.situ_beta', 4.0)
  w.add_float32('kimi-k3.activation.situ_linear_beta', 25.0)
  w.add_bool('kimi-k3.expert_weights_norm', True)
  w.add_array('tokenizer.ggml.tokens', [str(i) for i in range(VOCAB)])
  for name, x in weights.items():
    if '_exps.' not in name: w.add_tensor(name.replace('ssm_out', 'attn_output'), x)
    elif experts: w.add_tensor(name, quants.quantize(x, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)
  w.write_header_to_file()
  w.write_kv_data_to_file()
  w.write_tensors_to_file()
  w.close()

def parts(weights:dict[str, np.ndarray], li:int, e:int, part:int) -> list[bytes]:
  """each gpu's part of expert e of moe block li as the store has it: its units' gate rows, up rows, then down rows (q8_0), padded"""
  gate, up, down = (quants.quantize(weights[f'blk.{li}.ffn_{p}_exps.weight'][e], GGMLQuantizationType.Q8_0) for p in ('gate', 'up', 'down'))
  u = FF // G
  return [(gate[g*u:(g+1)*u].tobytes() + up[g*u:(g+1)*u].tobytes() + down[:, g*u//32*34:(g+1)*u//32*34].tobytes()).ljust(part, b'\0')
          for g in range(G)]

def eagle(folder:str, rng:np.random.Generator, lora:int=16) -> str:
  """a tiny random EAGLE-3 head (kimi-k3-eagle3.1-mla's layout) on the hidden states after blocks 0, 1, 2"""
  H, nope, rope, vd, ql, ff, p, a = 2, 8, 8, 8, 16, 32, "layers.0.", "layers.0.self_attn."
  json.dump({"num_attention_heads": H, "qk_nope_head_dim": nope, "qk_rope_head_dim": rope, "v_head_dim": vd, "kv_lora_rank": lora,
             "rms_norm_eps": 1e-5, "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]}, "rope_theta": 10000.0, "rope_scaling":
             {"factor": 4.0, "original_max_position_embeddings": 16, "beta_fast": 32, "beta_slow": 1, "mscale_all_dim": 1.0}},
            open(f"{folder}/config.json", "w"))
  shapes = {**{f"fc_norm.{i}.weight": (DIM,) for i in range(3)}, "fc.weight": (DIM, 3*DIM), p+"input_layernorm.weight": (DIM,),
            p+"hidden_norm.weight": (DIM,), a+"q_a_proj.weight": (ql, 2*DIM), a+"q_a_layernorm.weight": (ql,),
            a+"q_b_proj.weight": (H*(nope+rope), ql),
            a+"kv_a_proj_with_mqa.weight": (lora+rope, 2*DIM), a+"kv_a_layernorm.weight": (lora,), a+"kv_b_proj.weight": (H*(nope+vd), lora),
            a+"o_proj.weight": (DIM, H*vd), p+"post_attention_layernorm.weight": (DIM,), p+"mlp.gate_proj.weight": (ff, DIM),
            p+"mlp.up_proj.weight": (ff, DIM), p+"mlp.down_proj.weight": (DIM, ff), "norm.weight": (DIM,), "lm_head.weight": (VOCAB, DIM)}
  nn.state.safe_save({k: Tensor(rng.normal(0, 0.3, s).astype(np.float32)) for k, s in shapes.items()}, f"{folder}/model.safetensors")
  return folder

class TestEngine(unittest.TestCase):
  def test_invalid_window_counts(self):
    live, ok = 2, np.ones(4, dtype=np.uint32)
    tags = [np.array([1], np.uint32) for _ in range(2)]
    for a, b in [(ok, ok[:]*0), (ok[:]*0, ok[:]*0), (np.array([4, 0, 0, 0]), np.array([4, 0, 0, 0]))]:
      buf = [(None, tags), None, None, (None, [np.r_[0, a], np.r_[0, b]])]
      e = SimpleNamespace(m={"n_experts":4}, io=None, bufs=[buf], model=SimpleNamespace(config=SimpleNamespace(num_experts_per_tok=2)))
      with self.assertRaisesRegex(RuntimeError, "expert counts"): Engine.counted(e, 0, live)

  def test_window_tag(self):  # the tag the host waits on written by a kernel that reads the counts it gives the host (and the forecast's): a
    # gpu orders a graph's kernels by the buffers they touch, not by the graph's order
    def base(u:UOp) -> UOp: return u if u.op in (Ops.PARAM, Ops.BUFFER) else base(u.src[0])
    eng = self.engine(2 * E)
    for forecast in (False, True):
      jit, tag = (eng.forecast_jit, 2) if forecast else (eng.routes[min(eng.routes)][0], 1)  # (its args: the biases, the tag, the counts)
      writers = 0
      for c in [c for c in jit.captured._linear.src if c.op is Ops.CALL]:
        args, (outs, ins) = get_call_arg_uops(c), get_call_outs_ins(c)
        if not any(base(args[i]).op is Ops.PARAM and base(args[i]).arg.slot == tag for i in outs): continue
        writers, read = writers + 1, [base(args[i]) for i in ins]
        self.assertIn(tag + 1, [u.arg.slot for u in read if u.op is Ops.PARAM])
        if forecast: self.assertIn(eng.forecast_counts[0].uop.base, read)
      self.assertEqual(writers, 1)

  def setUp(self):
    Engine.bank = dtypes.float  # (a window's streams in float: as steps' to their rounding; the half bank: test_window_pieces)
    rng = np.random.default_rng(0)
    weights = {n: (1 + rng.normal(0, 0.1, t.shape) if 'norm' in n or 'res_score' in n else rng.normal(0, 0.3, t.shape)).astype(np.float32)
               for n, t in nn.state.get_state_dict(Transformer(CONFIG)).items()}
    folder = tempfile.mkdtemp()
    write(full:=pathlib.Path(folder)/'full.gguf', weights, True)
    write(trunk:=pathlib.Path(folder)/'trunk.gguf', weights, False)
    self.ref, self.trunk, self.weights = Transformer.from_gguf(full)[0], trunk, weights
    self.eng = self.engine(2 * E)

  def engine(self, slots:int, draft:Callable[[Transformer], DSpark]|None=None, caches:bytes=b'\0', tokens:int=2, widths:tuple[int, ...]=(),
             source:str="", short_window:int=0) -> Engine:
    # (the engine on the trunk (and draft's on it), every expert in its slot past the context's caches and the window's scratch (the pool's
    # first slots, of caches' bytes))
    part, layers = 4096, [1, 2]
    m = {"n_experts": E, "lat": LAT, "units": [FF // G] * G, "part_bytes": [part] * G, "expert_type": "Q8_0", "block": [32, 34], "layers": layers}
    model = Transformer.from_gguf(self.trunk, shard=G)[0]
    eng = Engine(model, m, slots=slots + 96, tokens=tokens, chunk=4, window=8, draft=draft(model) if draft else None,
                 widths=widths, source=source, short_window=short_window)  # (96: the caches', the scratch's)
    if short_window: eng.capture_short_window()
    at = eng.arena[1]
    assert at <= 96
    pools = [(caches * (at * part) + b''.join(parts(self.weights, li, e, part)[g] for li in layers for e in range(E))).ljust(eng.slots * part,
                                                                                                                       b'\0') for g in range(G)]
    eng.pool.assign(Tensor(np.frombuffer(b''.join(pools), np.dtype(eng.word.fmt))).shard(eng.devs, 0)).realize()
    eng.state["slot_of"][:] = at + np.arange(len(layers) * E)
    eng.state["expert_in"][at:at + len(layers) * E] = np.arange(len(layers) * E)
    return eng

  def want(self, tok:int, pos:int) -> np.ndarray:  # master's model's logits (the engine's forward of it, with the routed experts)
    return Engine.forward(SimpleNamespace(model=self.ref, draft=None), Tensor([[tok]], dtype=dtypes.int32), pos)[0].numpy()[0]

  def stub(self, copy:Callable[[int, int], int]=lambda li, j: 0, slot_of:list[np.ndarray]|None=None):
    # the host's part stubbed: a token's top k by sigmoid + bias, the weights the sigmoids normalized, in the two parts (an entry j*T+t is token
    # t's j-th expert and its weight, in copy(li, j) of layer li's: its slot in slot_of (default: the engine's), in that copy's section of an
    # engine of copies)
    eng = self.eng
    tables:dict[int, list[Tensor]] = {}
    n = len(eng.types) * E
    slots_of = slot_of or [eng.state["slot_of"][c * n:(c + 1) * n] // w for c, w in enumerate(eng.spans)]
    def host(li:int, i:int, dep:Tensor, *_) -> Tensor:  # (part 1's dep: the scores)
      if i == 0:
        sig = dep[:, 0].contiguous().to("CPU").sigmoid()
        sel = (sig + Tensor(eng.bias[li], device="CPU")).topk(K)[1]
        w = sig.gather(-1, sel) / sig.gather(-1, sel).sum(-1, keepdim=True)
        def table(ks:slice) -> Tensor:  # (an entry a token's expert's slot and weight, j*T+t)
          js = range(K)[ks]
          slots = Tensor.stack(*[Tensor(slots_of[copy(li, j)], device="CPU")[sel[:, j] + li * E] for j in js])
          out = [Tensor([1], dtype=dtypes.uint32, device="CPU")]
          for c in range(len(eng.ms)):
            on = Tensor([[len(eng.ms) == 1 or copy(li, j) == c] for j in js], device="CPU")
            out += [on.where(slots, 0).flatten().bitcast(dtypes.uint32),
                    on.where(w[:, ks].T, 0).flatten().cast(dtypes.float32).bitcast(dtypes.uint32)]
          return out[0].cat(*out[1:]).shard(eng.devs)
        tables[li] = [table(slice(0, eng.ks[0])), table(slice(eng.ks[0], K))]
      return tables[li][i]
    eng.host = host  # type: ignore[method-assign]

  def test_forward(self):
    self.stub()
    tok = 3
    for pos in range(6):
      ours, want = self.eng.forward(Tensor([[tok]], dtype=dtypes.int32), pos)[0].numpy()[0], self.want(tok, pos)
      np.testing.assert_allclose(ours, want, rtol=1e-2, atol=1e-2 * np.abs(want).max(), err_msg=f"position {pos}")
      self.assertEqual(ours.argmax(), want.argmax())
      tok = int(want.argmax())

  def test_verify(self):  # a step of [the next, a wrong guess], rollback(1): the step after as after the next alone
    self.stub()
    def logits(toks:list[int], pos:int) -> np.ndarray: return self.eng.forward(Tensor([toks], dtype=dtypes.int32), pos)[0].numpy()
    want = [logits([t], pos)[0] for pos, t in enumerate([3, 5, 7])]
    logits([3], 0)
    np.testing.assert_allclose(logits([5, 1], 1)[0], want[1], rtol=1e-4, atol=1e-4)
    self.eng.model.rollback(1)
    np.testing.assert_allclose(logits([7], 2)[0], want[2], rtol=1e-4, atol=1e-4)

  def test_sampled(self):  # at a temperature a step's token drawn from the softmax of its logits over it (gumbel-max), at 0 the greedy one
    self.stub()
    eng, want, n = self.eng, self.want(3, 0), 256
    eng.tok_np[:1], eng.given_np[:1], eng.rng = [3], [-1], np.random.default_rng(0)
    eng.model._copy(0, save=True)
    def draw() -> int:
      eng.model._copy(0, save=False)
      eng.key_np[:] = eng.rng.integers(1 << 32, size=2)
      eng.jits[1](eng.v_pos.bind(0))
      return int(eng.res_np[1])
    self.assertEqual(draw(), want.argmax())
    eng.temperature_np[:] = 2.0
    p = np.exp((want - want.max()) / 2.0)
    self.assertLess(np.abs(np.bincount([draw() for _ in range(n)], minlength=VOCAB) / n - p / p.sum()).sum() / 2, 0.2)

  def test_window(self):  # a prompt of 9: a window of its first 8 (two chunks), then a token: the logits after it as a token a step's
    self.stub()
    toks = [3, 5, 7, 1, 2, 9, 4, 6, 8]
    for pos, t in enumerate(toks): want = self.eng.forward(Tensor([[t]], dtype=dtypes.int32), pos)[0].numpy()[0]
    self.eng.window(toks[:8], 0, toks[1:9])
    ours = self.eng.forward(Tensor([toks[8:]], dtype=dtypes.int32), 8)[0].numpy()[0]
    np.testing.assert_allclose(ours, want, rtol=1e-3, atol=1e-3 * np.abs(want).max())

  def test_short_chunk(self):
    toks = [3, 5, 7, 1, 2, 9, 4]
    outs = []
    for cap in (4, 2):
      with patch("extra.k3.engine.SHORT_CHUNK", cap): self.eng = self.engine(2*E)
      self.stub()
      self.eng.window(toks[:-1], 0, toks[1:])
      outs.append(self.eng.forward(Tensor([[toks[-1]]]), 6)[0].numpy())
    np.testing.assert_allclose(outs[1], outs[0], rtol=1e-3, atol=1e-3 * np.abs(outs[0]).max())

  def test_adaptive_capture_without_recurrent_state(self):
    self.eng.short_window = 4
    with patch.object(self.eng.model, "_states", return_value=[]):
      self.eng.capture_short_window()
    self.assertIsNotNone(self.eng.short_engine)

  def test_adaptive_short_transition(self):
    toks = [3, 5, 7, 1, 2, 9, 4, 6, 8, 2, 7, 3, 5, 4, 9, 1]
    out, state, keys = [], [], []
    for short_window in (0, 4):
      self.eng = eng = self.engine(2 * E, short_window=short_window)
      self.stub()
      keys.append(eng.graph_key)
      large, small = eng.window, eng.short_engine
      observed = []
      def record(name, fn):
        def window(toks, pos, given):
          observed.append((name, pos, len(toks)))
          return fn(toks, pos, given)
        return window
      eng.window = record("large", large)  # type: ignore[method-assign]
      if small is not None:
        self.assertIs(small.pool, eng.pool)
        self.assertIs(small.state, eng.state)
        small.window = record("small", small.window)  # type: ignore[method-assign]
      with patch.object(eng, "steps_faster", return_value=False):
        self.assertEqual(eng.prefill(toks[:6], 0), 5)
        self.assertEqual(eng.prefill(toks[:9], 5), 8)
        self.assertEqual(eng.prefill(toks, 8), 15)
      self.assertEqual(observed, [("large", 0, 5), ("small" if short_window else "large", 5, 3), ("large", 8, 7)])
      self.assertEqual(eng.grown, 16)
      out.append(eng.forward(Tensor([[toks[-1]]]), 15)[0].numpy())
      state.append([t.numpy() for t in eng.model._states()])
    self.assertEqual(*keys)
    np.testing.assert_allclose(out[1], out[0], rtol=1e-3, atol=1e-3)
    for a, b in zip(*state): np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)

  def test_window_forecast(self):
    sequence = [3, 4, 5, 6]
    residuals = []
    for forecast in (False, True):
      eng = self.eng = self.engine(2 * E)
      if not forecast: eng.forecast_jit = None
      self.stub()
      counted = eng.counted
      predictions = []
      def capture(li:int, live:int) -> np.ndarray:
        counts = counted(li, live)
        if forecast and li == 0:
          assert eng.forecast_counts is not None
          scores = eng.win["guess_logits"].numpy()[:live]
          keys = 1 / (1 + np.exp(-scores)) + eng.bias[li + 1]
          top = np.argsort(-keys, axis=-1, kind="stable")[:, :K]
          predictions.append(np.bincount(top.ravel(), minlength=E))
          for a in eng.forecast_counts[1]: np.testing.assert_array_equal(a, predictions[-1])
        return counts
      with patch.object(eng, "counted", capture):
        eng.window(sequence[:-1], 0, sequence[1:])
      self.assertEqual(len(predictions), int(forecast))
      residuals.append(eng.in_order(Tensor(eng.win["bank"].uop), 0, len(sequence) - 1).numpy())
    np.testing.assert_array_equal(*residuals)

  def test_window_evaluate(self):  # a completion teacher forced through windows (two) scored against its token steps' top 5: as them
    self.stub()
    ids, outs, tops = [3, 5, 7], [1, 2, 9, 4, 6, 8, 2, 7, 3, 5], []
    for pos, t in enumerate(ids + outs[:-1]):
      lp = Tensor(self.eng.forward(Tensor([[t]], dtype=dtypes.int32), pos)[0].numpy()[0]).log_softmax().numpy()
      if pos >= len(ids) - 1: tops.append([[float(lp[i]), int(i)] for i in np.argsort(-lp)[:5]])
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as f:
      f.write(json.dumps({"task_id": "toy", "prompt_ids": ids, "output_ids": outs, "output_top": tops}) + "\n")
      f.flush()
      stats = window_evaluate(self.eng, f"{f.name}:1:{len(outs)}", lambda: False)
    self.assertEqual(len(stats), len(outs))
    # (a window's rows symbolic: its kernels fused apart from a step's, a half's rounding apart, the toy's random weights amplify it; on AMD the
    # steps' experts take q8 activations. without its routed experts a window is 0.15 apart)
    self.assertLess(max(k for _, k in stats), 1e-2)

  def test_grow(self):  # the caches' slots of a context's positions (in steps of a chunk) pinned as it grows, their experts dropped
    eng, st = self.eng, self.eng.state
    st["expert_in"][0], st["slot_of"][len(st["slot_of"]) - 1] = len(st["slot_of"]) - 1, 0  # (an expert in the caches' first slot)
    self.assertEqual(st["mask"].sum(), 0)
    eng.grow(1)
    self.assertEqual((eng.grown, st["mask"][0], st["expert_in"][0], st["slot_of"][-1]), (eng.chunk, 1, -1, -1))
    eng.grow(eng.max_context)
    self.assertEqual((eng.grown, st["mask"].sum()), (eng.max_context, eng.slots_of(eng.cache_bytes)))

  def test_prefill_few_free_slots(self):
    ids, eng, layers = [2, 0, 27, 24, 26, 17, 26, 10, 14], self.eng, [1, 2]
    eng.keep_logits = True
    self.stub()
    def run(t:int, pos:int) -> np.ndarray:
      eng.tok_np[:1], eng.given_np[:1] = [t], ids[pos + 1:pos + 2] or [-1]
      return eng.run(1, eng.v_pos.bind(pos))[1].numpy()
    for pos, t in enumerate(ids[:-1]): run(t, pos)
    states, want = [s.numpy() for s in eng.model._states()], run(ids[-1], len(ids) - 1)
    st, at, part = eng.state, eng.arena[1], eng.part
    eng.grow(eng.max_context)
    mask, slot_of, expert_in = st["mask"].copy(), st["slot_of"].copy(), st["expert_in"].copy()
    def put(code:int, slot:int):
      for g, data in enumerate(parts(self.weights, layers[code // E], code % E, part)):
        eng.pool.uop.buffer.bufs[g].as_memoryview(allow_zero_copy=True)[slot * part:(slot + 1) * part] = data
    def codes(ptr:int, n:int) -> list[int]: return np.ctypeslib.as_array((ctypes.c_int64 * n).from_address(ptr)).tolist() if n else []
    def read_codes(io, n:int, cs:int, slots:int) -> int:
      for c, s in zip(codes(cs, n), codes(slots, n)): put(c, s)
      return 0
    def read_ahead(io, n:int, cs:int):
      for c in codes(cs, n): put(c, int(st["slot_of"][c]))
    fake = SimpleNamespace(flags=host_c().flags, pump=lambda io: 0, read_wait=lambda io, n, cs: 0, read_codes=read_codes, read_ahead=read_ahead)
    eng.step_s[max(eng.jits)].append(10.0)
    for free in (E + 2, E - 2):
      with self.subTest(free=free), patch("extra.k3.engine.host_c", lambda: fake):
        st["slot_of"][:], st["expert_in"][:], st["mask"][:eng.tail], st["mask"][slice(*eng.arena)] = -1, -1, 1, 0
        st["mask"][at:at + free] = 0
        self.assertEqual(eng.prefill(ids, 0), len(ids) - 1)
        st["mask"][:], st["slot_of"][:], st["expert_in"][:] = mask, slot_of, expert_in
        for code in range(len(layers) * E): put(code, at + code)
        for got, w in zip([s.numpy() for s in eng.model._states()] + [run(ids[-1], len(ids) - 1)], states + [want]):
          np.testing.assert_allclose(got, w, rtol=3e-3, atol=3e-3 * np.abs(w).max())
        self.assertEqual(got.argmax(), want.argmax())

  def test_failed_window(self):
    ids, eng = [3, 5, 7, 1, 2, 9, 4, 6, 8, 3, 1, 7], self.engine(2 * E + 4)
    bank, counted = eng.win["bank"], eng.counted
    def fail(*args) -> np.ndarray:
      if eng.win["bank"] is not bank: raise RuntimeError("window reads failed")
      return counted(*args)
    eng.steps_faster, eng.counted, eng.model._cached_tokens = lambda n: False, fail, ids[:2]  # type: ignore[method-assign]
    with self.assertRaises(RuntimeError): next(eng.generate(ids))
    self.assertEqual((eng.win["bank"] is bank, eng.lent_from, eng.model._cached_tokens, eng.state["mask"][slice(*eng.arena)].any()),
                     (True, eng.tail, [], False))

  def test_window_pieces(self):  # a read window of 2 pieces (its pool lends a second piece's bank): states, logits as after steps
    for bank, tol in ((dtypes.float, 3e-3), (dtypes.half, 1e-2)):  # (the streams in half: as close as their rounding)
      with self.subTest(bank=bank):
        Engine.bank = bank
        self.pieces(tol)

  def pieces(self, tol:float):
    ids, eng = [3, 5, 7, 1, 2, 9, 4, 6, 8, 3, 1, 7], self.engine(2 * E + 4)
    self.eng, eng.keep_logits = eng, True
    self.stub()
    self.assertGreaterEqual(eng.pieces(), 2)
    def run(t:int, pos:int) -> np.ndarray:
      eng.tok_np[:1], eng.given_np[:1] = [t], ids[pos + 1:pos + 2] or [-1]
      return eng.run(1, eng.v_pos.bind(pos))[1].numpy()
    for pos, t in enumerate(ids[:11]): run(t, pos)
    states, want = [s.numpy() for s in eng.model._states()], run(ids[11], 11)
    eng.window(ids[:11], 0, ids[1:12])
    for got, w in zip([s.numpy() for s in eng.model._states()], states): np.testing.assert_allclose(got, w, rtol=tol, atol=tol * np.abs(w).max())
    np.testing.assert_allclose(run(ids[11], 11), want, rtol=tol, atol=tol * np.abs(want).max())

  def test_window_dump(self):  # the taps (a block's last stream, mixed ones: the last block's by the output's) and the target's next tokens of a
    # dump through windows (two) as a token a step's
    self.stub()
    eng, taps, seq = self.eng, [(0, False), (1, True), (2, True), (1, False)], [3, 5, 7, 1, 2, 9, 4, 6, 8, 2, 7]
    eng.dump_taps = taps
    eng.window_init(8)
    eng.window_lend()
    eng.draft = SimpleNamespace(taps=taps)  # (forward's taps)
    want = []
    for pos, t in enumerate(seq):
      lg, h = eng.forward(Tensor([[t]], dtype=dtypes.int32), pos)
      Tensor.realize(lg, h)  # (a step's graph once: it moves the states)
      want.append((lg.numpy()[0], h.numpy()[0, 0]))
    eng.draft = None
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as f, tempfile.TemporaryDirectory() as out:
      f.write(json.dumps({"task_id": "toy", "prompt_ids": seq[:3], "output_ids": seq[3:]}) + "\n")
      f.flush()
      window_dump(eng, f"{f.name}:1:{len(seq)}", lambda: False, out)
      self.assertEqual(json.load(open(f"{out}/meta.json")), {"taps": ["0", "1m", "2m", "1"], "seqs": [{"task_id": "toy", "start": 0,
                                                                                                         "len": len(seq), "prompt": 3}]})
      np.testing.assert_equal(np.load(f"{out}/tokens.npy"), seq)
      np.testing.assert_equal(np.load(f"{out}/next.npy"), [lg.argmax() for lg, _ in want])
      tol = 3e-2 if amd_custom_kernels_supported(eng.devs) else 1e-3  # (as test_window_states: a window's kernels apart from a step's)
      for j, n in enumerate(["0", "1m", "2m", "1"]):
        w = np.stack([h[j] for _, h in want])
        np.testing.assert_allclose(np.load(f"{out}/tap{n}.npy"), w, rtol=tol, atol=tol * np.abs(w).max(), err_msg=n)

  def test_proposal(self):  # a scored draft's verify step: the width whose expected tokens (by the scores) a measured second are most, one not
    # measured first; unscored: every guess
    eng = SimpleNamespace(tokens=8, widths=[2, 4, 8], step_s={2: [0.05], 4: [0.08], 8: []}, chain=None)
    g, score = list(range(7)), np.full(7, 0.9)
    self.assertEqual(Engine.proposal(eng, g, score, 0, 0), g)  # type: ignore[arg-type]
    eng.step_s[8] = [0.14]  # (1.9 / .05 = 38, 3.44 / .08 = 43, 5.2 / .14 = 37 tokens a second)
    self.assertEqual(Engine.proposal(eng, g, score, 0, 0), g[:3])  # type: ignore[arg-type]
    self.assertEqual(Engine.proposal(eng, g, np.full(7, 0.3), 0, 0), g[:1])  # type: ignore[arg-type]
    self.assertEqual(Engine.proposal(eng, g, None, 0, 0), g)  # type: ignore[arg-type]

  def test_generate_kept(self):  # a verify step's kept guesses are tokens, the step's guesses the next step's proposal
    steps, seen = iter([([5], [6, 7], None), ([6, 9], [8, 3], None), ([8, 3], [1, 2], None)]), []
    model = SimpleNamespace(fork=lambda *a: 0, save=lambda t: None, rollback=lambda n: None)
    eng = SimpleNamespace(get_start_pos=lambda t: 0, model=model, prefill=lambda t, p: p, step=lambda *a: next(steps), tokens=2, max_context=8,
                          draft=True, proposal=lambda g, *a: seen.append(g) or g[:1], verified=np.zeros(3, int), W=0, chunk=0,
                          counters=lambda: np.zeros(15), jits={1: None, 2: None}, io=SimpleNamespace(tail=1), temperature_np=np.zeros(1))
    eng.steps = lambda *a, **kw: Engine.steps(eng, *a, **kw)  # type: ignore[arg-type]
    self.assertEqual([t for _, t in zip(range(4), Engine.generate(eng, [0]))], [5, 6, 9, 8])  # type: ignore[arg-type]
    self.assertEqual(seen, [[6, 7], [8, 3]])

  def test_steps_faster(self):  # steps or a window of n tokens: a window the store's experts not cached read (5 s here) and a token's (measured:
    # their median), a step measured (of the most tokens there's a graph of)
    eng = SimpleNamespace(step_s={1: [], 2: []}, window_s=[], jits={1: None}, bias=[0], m={"n_experts": 1000},
                          slots=25, part=10**9, store_part=10**9, devs=("A",))
    self.assertEqual([Engine.steps_faster(eng, n) for n in (64, 526)], [True, False])  # type: ignore[arg-type]  # (before any measured)
    eng.slots *= 2
    eng.part //= 2  # same cache capacity and bytes per stored expert
    self.assertTrue(Engine.steps_faster(eng, 80))  # type: ignore[arg-type]
    eng.window_s = [(9182, 60.0), (9595, 17.0), (9182, 16.5)]  # (a cold first one, sizes alike: a line through them would be anything)
    eng.step_s[1] = [0.035, 0.04]
    self.assertEqual([Engine.steps_faster(eng, n) for n in (64, 120, 526, 9000)], [True, True, False, False])  # type: ignore[arg-type]
    eng.step_s[1] = [0.008]
    self.assertTrue(Engine.steps_faster(eng, 526))  # type: ignore[arg-type]
    eng.step_s[1], eng.step_s[2], eng.jits[2] = [0.06], [0.06], None
    self.assertTrue(Engine.steps_faster(eng, 150))  # type: ignore[arg-type]

  def test_session(self):  # a long session a short request diverges from early kept: its caches and states back when a request extends it
    self.stub()
    eng, m = self.eng, self.eng.model
    def state() -> list[np.ndarray]: return [getattr(o, k).numpy() for o, k in eng.caches] + [s.numpy() for s in m._states()]
    A, S = [3, 5, 7, 1, 2, 9, 4, 6, 8, 2, 7], [3, 5, 1, 1, 6]
    for pos, t in enumerate(A): eng.forward(Tensor([[t]], dtype=dtypes.int32), pos)[0].realize()
    m._cached_tokens, want = A[:], state()
    eng.resume(S)
    self.assertEqual(eng.session, (A, 2, m._saved))
    for pos, t in enumerate(S[2:], 2): eng.forward(Tensor([[t]], dtype=dtypes.int32), pos)[0].realize()  # (the short request's)
    m._cached_tokens = S[:]
    eng.resume(A + [6])
    self.assertEqual((eng.session, m._cached_tokens, eng.get_start_pos(A + [6])), (None, A, len(A)))
    for got, w in zip(state(), want): np.testing.assert_equal(got, w)

  def test_session_fork(self):  # a server's requests: an earlier one, a conversation's turns (its replies rendered again: the prompt's states),
    # short requests between them (one past the states saved before, two in a row): each turn fed from where it would have been without them,
    # its states, caches, logits as without them
    P, G, S = [3, 5, 7, 1, 2, 9, 4, 6], [8, 2], [[3, 5, 1, 1, 6], [3, 5, 2, 6]]
    turns = [(P[:4] + [6], []), (P, G), (P + [7, 1, 5], [3]), (P + [7, 1, 5, 3, 6], [])]
    sides = [[], [], [P[:5] + [1]], S]
    def serve(tokens:list[int], gen:list[int]) -> tuple[int, list[np.ndarray]]:  # (LLMServer's get_start_pos, then generate's, its steps)
      eng, m = self.eng, self.eng.model
      eng.get_start_pos(tokens)
      start = eng.get_start_pos(tokens)
      fork = m.fork(tokens, start, 1)
      for pos, t in enumerate(tokens[start:] + gen, start):
        logits = eng.forward(Tensor([[t]], dtype=dtypes.int32), pos)[0].numpy()
        if pos + 1 in (fork, len(tokens)): m.save(tokens[:pos + 1])
      m._cached_tokens = tokens + gen
      return start, [getattr(o, k).numpy() for o, k in eng.caches] + [s.numpy() for s in m._states()] + [logits]
    self.stub()
    want = [serve(*turn) for turn in turns]
    self.eng = self.engine(2 * E)
    self.stub()
    for i, turn in enumerate(turns):
      for side in sides[i]: serve(side, [4])
      start, got = serve(*turn)
      self.assertEqual(start, want[i][0])
      for g, w in zip(got, want[i][1]): np.testing.assert_equal(g, w)
    for side in (P[:5] + [1], S[0]): serve(side, [4])
    for g, w in zip(serve(*turns[3])[1], want[3][1]): np.testing.assert_equal(g, w)
    serve(S[0], [4] * 6)  # (a reply past the kept rows: the session gone, the next short request's kept)
    serve(S[1], [4])
    self.assertEqual(self.eng.session[:2] if self.eng.session else None, (S[0] + [4] * 6, 2))

  def test_prefill(self):  # windows of a prompt but its last token (each with the tokens after it given) of up to W, split evenly, none without
    calls:list = []
    self.eng.window = lambda toks, pos, given: calls.append((pos, toks, given))  # type: ignore[method-assign,assignment,misc]
    self.eng.step_s[max(self.eng.jits)].append(10.0)  # (steps slow: windows)
    self.eng.pieces = lambda: 1  # type: ignore[method-assign]  # (a read window of a piece)
    self.assertEqual(self.eng.prefill(list(range(10)), 2), 9)
    self.assertEqual(calls, [(2, [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9])])
    self.eng.W = 4
    self.assertEqual(self.eng.prefill(list(range(12)), 0), 11)
    self.assertEqual([(p, len(t)) for p, t, _ in calls[1:]], [(0, 4), (4, 4), (8, 3)])
    self.eng.W = 8
    self.assertEqual(self.eng.prefill(list(range(16)), 10), 10)  # the live rows fit, but their whole chunk would cross the context
    self.assertEqual(len(calls), 4)
    self.eng.W = 0
    self.assertEqual(self.eng.prefill(list(range(9)), 3), 3)
    self.assertEqual(len(calls), 4)

  def test_window_states(self): # recurrent states, caches (the indexer's, the draft's) after a window as after its tokens a step each, then decode
    # (on AMD a step's experts take q8 activations, a window's not: as close as those)
    self.stub()
    eng, folder, rng = self.eng, tempfile.mkdtemp(), np.random.default_rng(4)
    eng.keep_logits = True
    eng.draft = Eagle3(eagle(folder, rng), lambda t: eng.model.token_embd(t).float(), eng.devs, eng.model.max_context)
    for b in eng.model.blk:
      if isinstance(b, MLATransformerBlock):
        b.indexer = Indexer(replace(b.config, indexer=IndexerConfig(2, 16, 4)))
        for p in nn.state.get_parameters(b.indexer): p.replace(Tensor(rng.normal(0, 0.3, p.shape).astype(np.float32)).shard(eng.devs, p.uop.axis))
        b._init_state(Tensor.empty(1, 1, DIM, device=eng.devs))
        b.indexer.cache_kg.realize()
    eng.window_init(8)  # (the window's graphs on the draft and the indexers)
    eng.window_lend()
    ids = [3, 5, 7, 1, 2, 9, 4, 6, 8, 3, 1, 7, 2, 5, 9, 4]
    def run(t:int, pos:int) -> np.ndarray:
      eng.tok_np[:1], eng.given_np[:1] = [t], ids[pos + 1:pos + 2] or [-1]
      return eng.run(1, eng.v_pos.bind(pos))[1].numpy()
    def snapshot(n:int) -> list[np.ndarray]:
      caches = [eng.draft.cache, *(b.cache_k for b in eng.model.blk if isinstance(b, MLATransformerBlock)),
                *(b.indexer.cache_kg for b in eng.model.blk if isinstance(b, MLATransformerBlock) and b.indexer)]
      return [s.numpy() for s in eng.model._states()] + [c.numpy()[..., :n, :] for c in caches]
    tol = 3e-2 if amd_custom_kernels_supported(eng.devs) else 3e-3  # (a window's rows symbolic: its kernels fused apart from a step's)
    def close(got:np.ndarray, want:np.ndarray): np.testing.assert_allclose(got, want, rtol=tol, atol=tol * np.abs(want).max())
    def equal(a:list[np.ndarray], b:list[np.ndarray]):
      for got, want in zip(a, b): close(got, want)
    for start, live in ((0, 1), (0, 3), (0, 6), (4, 3), (8, 5), (12, 3)):
      with self.subTest(start=start, live=live):
        for pos, t in enumerate(ids[:start + live]): run(t, pos)
        states, after = snapshot(start + live), run(ids[start + live], start + live)
        states_after = snapshot(start + live + 1)
        for pos, t in enumerate(ids[:start]): run(t, pos)
        eng.window(ids[start:start + live], start, ids[start + 1:start + 1 + live])
        equal(snapshot(start + live), states)
        close(run(ids[start + live], start + live), after)
        equal(snapshot(start + live + 1), states_after)

  def test_states_allocated(self):  # the blocks' caches and states and the draft's allocated by the engine, before its pool takes what's free
    model, folder = Transformer.from_gguf(self.trunk, shard=G)[0], tempfile.mkdtemp()
    draft = Eagle3(eagle(folder, np.random.default_rng(1)), lambda t: model.token_embd(t).float(), self.eng.devs, model.max_context)
    Engine(model, self.eng.m, slots=1, tokens=2, draft=draft)
    for t in (*(b.cache_k for b in model.blk if isinstance(b, MLATransformerBlock)), draft.cache, model.blk[0].kept[0][0]):
      self.assertTrue(all(b.is_allocated() for b in t.uop.base.buffer.bufs))

  def test_draft_step(self):  # a jitted verify step with the draft: its tokens and guesses as an unjitted one's
    self.stub()
    eng, folder = self.eng, tempfile.mkdtemp()
    eng.draft = Eagle3(eagle(folder, np.random.default_rng(1)), lambda t: eng.model.token_embd(t).float(), eng.devs, eng.model.max_context)
    eng.tok_np[:2], eng.given_np[:2] = [3, 5], [-1, -1]
    eng.run(2, UOp.variable("start_pos", 0, 15).bind(0))
    want = eng.res_np.copy()
    for _ in range(3): eng.jits[2](eng.v_pos.bind(0))
    np.testing.assert_equal(eng.res_np, want)

  def test_dspark_step(self):  # a step's block (after the token after its kept rows, on the rows up to it, the keys of the draft's window of
    # positions) as the draft's own on all its cache's positions: jitted steps of a token (a symbolic window), then verify steps
    # (its cache the pool's, sharded by heads, the pool's bytes there NaNs till the context grows: its keys past the context masked, zero)
    from extra.k3.test_draft import dspark
    path, toks = dspark(tempfile.mkdtemp(), np.random.default_rng(5), window=6), [3, 5, 7, 1, 2, 9, 4, 6, 8, 2]
    self.eng = eng = self.engine(2 * E, lambda m: DSpark(path, lambda t: m.token_embd(t).float(), m.output, m.token_embd.weight.device,
                                                         m.max_context, width=2), caches=b'\xff')
    self.stub()
    d = eng.draft
    assert isinstance(d, DSpark) and d.cache.uop.axis == 3
    eng.grow(eng.max_context)
    def want(anchors:list[int], pos:int) -> np.ndarray:
      g, _, _ = d(Tensor(anchors, dtype=dtypes.int32).shard(eng.devs), Tensor(list(range(pos + 1, pos + 1 + len(anchors))), dtype=dtypes.int32)
                  .shard(eng.devs), 2, (0, eng.model.max_context))
      return g.numpy().flatten()
    for pos, t in enumerate(toks[:8]):
      eng.tok_np[:1], eng.given_np[:1] = [t], [toks[pos + 1]]
      eng.jits[1](eng.v_pos.bind(pos))
      np.testing.assert_equal(eng.res_np[2:4], want([toks[pos + 1]], pos), err_msg=f"position {pos}")
    eng.tok_np[:2], eng.given_np[:2] = toks[8:], [-1, -1]
    eng.run(2, eng.v_pos.bind(8))
    kept = int(toks[9] == eng.res_np[1])  # (the step's second token right: the block after the target's token after it)
    np.testing.assert_equal(eng.res_np[3:5], want([int(eng.res_np[1 + kept])], 8 + kept))
    eng.model.rollback(1)  # (the step's first token kept: the next at 9, its second token the given one after its first, kept)
    eng.tok_np[:2], eng.given_np[:2] = [int(eng.res_np[1]), 7], [7, -1]
    eng.run(2, eng.v_pos.bind(9))
    np.testing.assert_equal(eng.res_np[3:5], want([int(eng.res_np[2])], 10))

  def test_window_dspark(self):  # the draft's cache after a window (its taps split on their rows over the gpus, a chunk's gathered) as after steps
    from extra.k3.test_draft import dspark
    path, ids = dspark(tempfile.mkdtemp(), np.random.default_rng(5), window=6), [3, 5, 7, 1, 2, 9, 4, 6, 8, 2, 1]
    self.eng = eng = self.engine(2 * E, lambda m: DSpark(path, lambda t: m.token_embd(t).float(), m.output, m.token_embd.weight.device,
                                                         m.max_context, width=2))
    self.stub()
    eng.grow(16)
    assert eng.fill_jit is not None and len(eng.taps) > 1
    for pos, t in enumerate(ids[:10]):
      eng.tok_np[:1], eng.given_np[:1] = [t], [ids[pos + 1]]
      eng.run(1, eng.v_pos.bind(pos))
    want = eng.draft.cache.numpy()[..., :10, :].astype(np.float32)
    eng.window(ids[:10], 0, ids[1:11])
    tol = 3e-2 if amd_custom_kernels_supported(eng.devs) else 3e-3
    np.testing.assert_allclose(eng.draft.cache.numpy()[..., :10, :].astype(np.float32), want, rtol=tol, atol=tol * np.abs(want).max())

  @unittest.skipUnless(amd_custom_kernels_supported(Device.DEFAULT), "the graph's host syncs and the expert kernel: AMD")
  def test_generate_dspark(self):  # a scored draft's verify steps of widths 2 and 4 (a server's): the target's greedy tokens, its guesses kept
    from extra.k3.test_draft import dspark
    path = dspark(tempfile.mkdtemp(), np.random.default_rng(5), window=6)
    self.eng = eng = self.engine(2 * E, lambda m: DSpark(path, lambda t: m.token_embd(t).float(), m.output, m.token_embd.weight.device,
                                                         m.max_context, width=3), tokens=4, widths=(2, 4))
    eng.warm()
    want, tok = [], 3
    for pos in range(10): want.append(tok:=int(self.want(tok, pos).argmax()))
    self.assertEqual([t for _, t in zip(range(10), eng.generate([3]))], want)

  @unittest.skipUnless(amd_custom_kernels_supported(Device.DEFAULT), "the graph's host syncs and the expert kernel: AMD")
  def test_generate(self):  # the jitted steps with the host's step_io (every expert cached: no reads), captured on dry steps
    self.eng.warm()
    want, tok = [], 3
    for pos in range(6): want.append(tok:=int(self.want(tok, pos).argmax()))
    tokens = [3]
    self.assertEqual([t for _, t in zip(range(3), self.eng.generate(tokens))], want[:3])
    self.assertEqual(self.eng.get_start_pos(tokens + [want[3]]), 3)  # (a prompt that extends the tokens fed: only the rest)
    self.assertEqual([t for _, t in zip(range(2), self.eng.generate(tokens + [want[3]]))], want[4:])
    self.assertEqual((self.eng.io.refs, self.eng.io.misses), (self.eng.io.hits, 0))

  @unittest.skipUnless(amd_custom_kernels_supported(Device.DEFAULT), "the graph's host syncs and the expert kernel: AMD")
  def test_generate_fork(self):  # prompts sharing X: the third fed from where the second diverged from the first (a chunk's end, its states saved)
    self.eng.warm()
    X = [3, 5, 7, 9]
    def take(prompt:list[int]) -> list[int]: return [t for _, t in zip(range(3), self.eng.generate(prompt))]
    want = take(X + [2, 4])
    take(X + [6, 8])
    self.assertEqual(self.eng.get_start_pos(X + [2, 4]), len(X))
    self.assertEqual(take(X + [2, 4]), want)

class TestPrefillPrior(unittest.TestCase):
  def engine(self) -> Engine:
    eng = Engine.__new__(Engine)
    eng.io = IoT(copies=2, layers=2, decay=.5)
    eng.state = {"freq": np.zeros(12, dtype=np.float32), "mask": np.zeros(8, np.uint8), "flying": np.zeros(12, np.int32)}
    eng.m, eng.tail, eng.arena, eng.max_primary_span, eng.grow = {"n_experts": 3}, 8, (0, 0), 1, lambda n: None
    eng.W, eng.chunk, eng.short_window, eng.short_engine = 8, 4, 4, None
    eng.session, eng.B, eng.v_at = None, 8, UOp.variable("at", 0, 24)
    eng.tokens, eng.draft, eng.verified = 1, None, np.zeros(3, int)
    eng.jits, eng.temperature_np = {1: None}, np.zeros(1)
    eng.model = SimpleNamespace(max_context=32, _cached_tokens=[], _saved=[], fork=lambda tokens, start, step: start, save=lambda tokens: None,
                                config=SimpleNamespace(num_experts_per_tok=1))
    eng.model.get_start_pos = lambda tokens: sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens, eng.model._cached_tokens)))
    return eng

  def test_caps_joint_layer_frequency(self):
    eng = self.engine()
    freq = eng.state["freq"].reshape(2, 2, 3)
    freq[:] = [[[20, 30, 50], [0, 1, 0]], [[80, 10, 10], [0, 0, 2]]]
    original = freq.copy()
    eng.cap_prefill_prior()
    np.testing.assert_array_equal(freq[:, 0], original[:, 0] * np.float32(.2))
    np.testing.assert_array_equal(freq[:, 1], original[:, 1])
    eng.io.decay = 1
    freq[:] = original
    eng.cap_prefill_prior()
    np.testing.assert_array_equal(freq, original)

  def test_caps_once_after_fork_and_compact_window(self):
    eng = self.engine()
    seen = []
    def window(toks, at, given, label="full"):
      seen.append((label, at, len(toks)))
      eng.state["freq"][:] += 20
    def step(toks, at, given):
      seen.append(("step", at, float(eng.state["freq"].reshape(2, 2, 3)[:, 0].sum())))
      return [7], [], None
    eng.model.fork = lambda *args: 4
    with patch.multiple(eng, window=window, step=step, pieces=lambda: 1, steps_faster=lambda n: False, counters=lambda: np.zeros(15)), \
         patch.object(eng, "cap_prefill_prior", wraps=eng.cap_prefill_prior) as cap:
      eng.short_engine = SimpleNamespace(window=lambda *args: window(*args, label="compact"), chunk=2)
      output = eng.generate(list(range(8)))
      self.assertEqual(next(output), 7)
      output.close()
      self.assertEqual([event[:2] for event in seen], [("full", 0), ("compact", 4), ("step", 7)])
      self.assertEqual([event[2] for event in seen[:2]], [4, 3])
      self.assertAlmostEqual(seen[2][2], 40, places=4)
      cap.assert_called_once()

  def test_caps_at_first_output_batch_after_step_only_tail(self):
    eng = self.engine()
    eng.state["freq"][:] = 20
    eng.model.fork = lambda *args: 4
    eng.jits = {1: None, 2: None}
    steps = []
    def step(toks, at, given):
      steps.append((at, len(toks), float(eng.state["freq"].reshape(2, 2, 3)[:, 0].sum())))
      return [7] * len(toks), [], None
    with patch.multiple(eng, window=lambda *a: None, step=step, pieces=lambda: 1, steps_faster=lambda n: n < 4,
                        counters=lambda: np.zeros(15)):
      output = eng.generate(list(range(7)))
      self.assertEqual(next(output), 7)
      output.close()
    self.assertEqual(steps[:1], [(4, 2, 120)])
    self.assertEqual(steps[1][:2], (6, 1))
    self.assertAlmostEqual(steps[1][2], 40, places=4)

  def test_skips_steps_only_warm_reuse_and_restore(self):
    eng = self.engine()
    eng.state["freq"][:] = 100
    kept = list(range(8))
    eng.model._cached_tokens = [99, 98]
    eng.session = (kept, 0, [])
    eng.keep_jits = [None, lambda at: None]
    with patch.multiple(eng, step=lambda *a: ([7], [], None), pieces=lambda: 1, steps_faster=lambda n: True, counters=lambda: np.zeros(15)), \
         patch.object(eng, "cap_prefill_prior", wraps=eng.cap_prefill_prior) as cap:
      for prompt in (kept + [8], kept + [8, 9, 10]):
        output = eng.generate(prompt)
        self.assertEqual(next(output), 7)
        output.close()
        self.assertIsNone(eng.session)
      cap.assert_not_called()
      np.testing.assert_array_equal(eng.state["freq"], np.full(12, 100, dtype=np.float32))


GLM = TransformerConfig(num_blocks=2, dim=DIM, hidden_dim=FF, n_heads=2, n_kv_heads=1, norm_eps=1e-5, vocab_size=VOCAB, head_dim=24,
  rope_theta=10000.0, rope_dim=8, v_head_dim=16, max_context=16, num_experts=E, num_experts_per_tok=K, expert_gating_func=ExpertGating.SIGMOID,
  norm_topk_prob=True, q_lora_rank=16, kv_lora_rank=16, shared_expert_dim=FF, leading_dense_blocks=1, dense_hidden_dim=64, expert_bias=True,
  shared_expert_gate=False, routed_scaling_factor=2.5, rope_interleave=True, indexer=IndexerConfig(2, 16, 4, rope_dim=8), mtp=True)

def write_glm(path:pathlib.Path, weights:dict[str, np.ndarray], experts:bool):
  w = GGUFWriter(path, "glm-dsa")
  for key, value in {'context_length': 16, 'embedding_length': DIM, 'feed_forward_length': 64, 'expert_feed_forward_length': FF, 'block_count': 3,
                     'nextn_predict_layers': 1, 'attention.head_count': 2, 'attention.head_count_kv': 1, 'attention.q_lora_rank': 16,
                     'attention.kv_lora_rank': 16, 'rope.dimension_count': 8, 'attention.key_length_mla': 24, 'attention.value_length_mla': 16,
                     'expert_count': E, 'expert_used_count': K, 'expert_shared_count': 1, 'leading_dense_block_count': 1, 'expert_gating_func': 2,
                     'attention.indexer.head_count': 2, 'attention.indexer.key_length': 16, 'attention.indexer.top_k': 4}.items():
    w.add_uint32(f'glm-dsa.{key}', value)
  w.add_float32('glm-dsa.attention.layer_norm_rms_epsilon', 1e-5)
  w.add_float32('glm-dsa.expert_weights_scale', 2.5)
  w.add_bool('glm-dsa.expert_weights_norm', True)
  w.add_array('tokenizer.ggml.tokens', [str(i) for i in range(VOCAB)])
  for name, x in weights.items():
    if '_exps.' not in name: w.add_tensor(name, x)
    elif experts: w.add_tensor(name, quants.quantize(x, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)
  w.write_header_to_file()
  w.write_kv_data_to_file()
  w.write_tensors_to_file()
  w.close()

class TestMTP(unittest.TestCase):
  stub = TestEngine.stub
  def setUp(self):
    Engine.bank = dtypes.half
    rng = np.random.default_rng(0)
    def gguf_name(n:str) -> str: return re.sub(r'^mtp\.', 'blk.2.', re.sub(r'^mtp\.(eh_proj|enorm|hnorm|shared_head_norm)', r'blk.2.nextn.\1', n))
    self.weights = {gguf_name(n): (1 + rng.normal(0, 0.1, t.shape) if 'norm' in n else rng.normal(0, 0.3, t.shape)).astype(np.float32)
                    for n, t in nn.state.get_state_dict(Transformer(GLM)).items()}
    folder = tempfile.mkdtemp()
    write_glm(full:=pathlib.Path(folder)/'full.gguf', self.weights, True)
    write_glm(trunk:=pathlib.Path(folder)/'trunk.gguf', self.weights, False)
    self.ref, self.trunk = Transformer.from_gguf(full)[0], trunk

  def engine(self, tokens:int, copies:list[tuple[int, dict[str, np.ndarray], list[int]]]|None=None, own:bool=True) -> Engine:
    """an engine of the store's copies (part, weights, layers) of the experts, every expert in its slots past the caches' and the scratch (own:
    the copies after the first the engine's, else only in the pool: self.slot_of the slots of each)"""
    copies = copies or [(8192, self.weights, [1, 2])]
    ms = [{"n_experts": E, "lat": DIM, "units": [FF // G] * G, "part_bytes": [part] * G, "expert_type": "Q8_0", "block": [32, 34], "layers": layers}
          for part, _, layers in copies]
    model = Transformer.from_gguf(self.trunk, shard=G)[0]
    self.eng = eng = Engine(model, ms[0], slots=sum(-(-p // 8192) * len(ls) for p, _, ls in copies) * E + 160, tokens=tokens, chunk=4, window=8,
                            draft=MTP(model), copies=tuple(ms[1:]) if own else ())
    pools, at, self.slot_of = [bytearray(eng.slots * eng.part) for _ in range(G)], round_up(eng.arena[1], 2), []
    for c, (_, weights, layers) in enumerate(copies):
      self.slot_of = [*self.slot_of, np.zeros(2 * E, np.int32)]
      for li, e in itertools.product(range(2), range(E)):
        if ms[0]["layers"][li] not in layers: continue
        w = eng.layer_spans[c if own else 0][li]
        for g, b in enumerate(parts(weights, ms[0]["layers"][li], e, w * eng.part)): pools[g][at * eng.part:(at + w) * eng.part] = b
        self.slot_of[c][li * E + e], code = at // w, (c * 2 + li) * E + e
        if own or c == 0: eng.state["slot_of"][code], eng.state["expert_in"][at:at + w] = at, code
        at += w
    assert at <= eng.tail
    eng.pool.assign(Tensor(np.frombuffer(b''.join(pools), np.dtype(eng.word.fmt))).shard(eng.devs, 0)).realize()
    self.stub()
    return eng

  def greedy(self, n:int) -> tuple[list[int], list[int]]:
    ref, draft, toks, guesses = self.ref, MTP(self.ref), [3], []
    for pos in range(n):
      logits, hidden = Engine.forward(SimpleNamespace(model=ref, draft=draft), Tensor([[toks[-1]]], dtype=dtypes.int32), pos)
      toks.append(int(logits.numpy()[0].argmax()))
      guesses.append(int(draft.guess(Tensor([[toks[-1]]], dtype=dtypes.int32), hidden, pos, 0)[0].numpy()[0]))
    return toks, guesses

  def step(self, toks:list[int], pos:int) -> tuple[list[int], int]:
    eng, T = self.eng, len(toks)
    eng.tok_np[:T], eng.given_np[:T] = toks, [-1] * T
    eng.run(T, eng.v_pos.bind(pos))
    return eng.res_np[1:1 + T].tolist(), int(eng.res_np[1 + T])

  def chain(self, g:int, pos:int, row:int) -> list[int]:
    self.eng.given_np[0], self.eng.cres_np[0] = g, 0
    self.eng.run_chain(self.eng.v_pos.bind(pos), self.eng.v_row.bind(row))
    return self.eng.cres_np[1:1 + self.eng.links].tolist()

  def test_verify(self):
    for tokens in (2, 3):
      with self.subTest(tokens=tokens):
        self.engine(tokens)
        toks, guesses = self.greedy(13)
        (t,), g = self.step([3], 0)
        self.assertEqual((t, g), (toks[1], guesses[0]))
        out, pos, row = [3, t], 1, 0
        for i in itertools.takewhile(lambda _: pos + tokens <= len(guesses), itertools.count()):
          gs = toks[pos + 1:pos + tokens] if i % 2 else [g, *(self.chain(g, pos, row) if tokens > 2 else [])]
          targets, g = self.step([out[-1], *gs], pos)
          row = next((j for j, x in enumerate(gs) if targets[j] != x), len(gs))
          self.assertEqual(g, guesses[pos + row])
          out += gs[:row] + [targets[row]]
          pos += row + 1
        self.assertEqual(out, toks[:len(out)])

  def test_window(self):
    eng, (toks, guesses) = self.engine(2), self.greedy(9)
    b = eng.draft.blk[0]
    def caches() -> list[np.ndarray]: return [b.cache_k.numpy()[..., :8, :], b.indexer.cache_kg.numpy()[:, :8]]
    for pos in range(8):
      eng.tok_np[:1], eng.given_np[:1] = [toks[pos]], [toks[pos + 1]]
      eng.run(1, eng.v_pos.bind(pos))
    want = caches()
    eng.window(toks[:8], 0, toks[1:9])
    for got, w in zip(caches(), want): np.testing.assert_allclose(got, w, rtol=1e-2, atol=1e-2 * np.abs(w).max())
    self.assertEqual(self.step([toks[8]], 8), ([toks[9]], guesses[8]))

  def test_copies(self):  # a second copy of block 1's experts (its part half the first's): the same weights in it, a step's experts in either
    # copy: the first copy's logits; others in it: where they're in it, the weights of an engine of a copy holding both copies' (its slots)
    toks, rng = [3, 5, 7, 1, 2, 9, 4, 6, 8], np.random.default_rng(3)
    other = {n: rng.normal(0, 0.3, w.shape).astype(np.float32) if '_exps.' in n else w for n, w in self.weights.items()}
    def copy(li:int, j:int) -> int: return int(li == 0 and j >= 2)
    def run(copies:list|None, own:bool=True, policy:Callable[[int, int], int]=copy) -> list[np.ndarray]:
      eng, out = self.engine(2, copies, own), []
      eng.keep_logits = True
      self.stub(policy, self.slot_of)
      for pos, t in enumerate(toks[:4]):
        eng.tok_np[:1], eng.given_np[:1] = [t], [-1]
        out += [eng.run(1, eng.v_pos.bind(pos))[1].numpy(), eng.res_np[1:3].copy()]
      eng.window(toks[:8], 0, toks[1:9])
      eng.tok_np[:1], eng.given_np[:1] = [toks[8]], [-1]
      return out + [eng.run(1, eng.v_pos.bind(8))[1].numpy(), eng.res_np[1:3].copy()]
    def same(a:list[np.ndarray], b:list[np.ndarray]):  # (the tokens, the logits to a sum's rounding: the copies' outputs added)
      for x, y in zip(a, b): np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6 * np.abs(y).max()) if x.dtype == np.float32 else \
        np.testing.assert_equal(x, y)
    want = run(None, policy=lambda li, j: 0)
    same(run([(16384, self.weights, [1, 2]), (8192, self.weights, [1])]), want)
    self.assertEqual(self.eng.spans, [1])
    same(got:=run([(16384, self.weights, [1, 2]), (8192, other, [1])]), run([(8192, self.weights, [1, 2]), (8192, other, [1])], own=False))
    self.assertFalse(np.allclose(got[0], want[0], rtol=1e-2))

  def test_chain(self):
    eng, (toks, _) = self.engine(4), self.greedy(4)
    for pos in range(3): self.step(toks[pos:pos + 1], pos)
    _, g = self.step([toks[3]], 3)
    hid, out = Tensor(eng.dx.numpy()[:1].reshape(1, 1, -1)).to(eng.devs), []
    for k in range(2): out += self.chain(out[-1] if out else g, 4 + k, 0)
    want, tok = [], g
    for k in range(2):
      logits, hid, _ = eng.draft(Tensor([[tok]], dtype=dtypes.int32).to(eng.devs), None, 4 + k, hid=hid)
      want.append(tok:=int(logits.numpy()[0, 0].argmax()))
    self.assertEqual(out, want)

class TestSharedWindowIndices(unittest.TestCase):
  def test_layer_major_indices(self):
    class Block:
      def __init__(self, offset): self.offset = offset
      def _indexer_mask(self, h, pos, prev):
        if self.offset is None: return prev
        last = h[0, :, 0].int() - self.offset
        return (last-2).stack(last, dim=-1).unsqueeze(0)
    blocks = [Block(x) for x in (0, None, None, 1, None)]
    eng = Engine.__new__(Engine)
    eng.chunk, eng.index_roles = 4, dict(zip(blocks, ("write", "read", "read", "write", "read")))
    # Two 8-row pieces with different contents, including two chunks and an irregular final chunk.
    pieces = [Tensor.empty(8, 2, dtype=dtypes.int32).realize() for _ in range(2)]
    values = Tensor.arange(32).float().square()
    for i, b in enumerate(blocks):
      for start, live in [(0, 4), (4, 4), (8, 3)]:
        eng.win = {"indices": pieces[start//8]}
        n = UOp.variable("test_live", 1, 4).bind(live)
        c0 = UOp.variable("test_c0", 0, 4).bind(start%8)
        h = Tensor.arange(5+start, 9+start)[:n].reshape(1, n, 1)
        got = eng.window_mask(b, h, 5+start, c0)
        want = None
        for ref in blocks[:i+1]: want = ref._indexer_mask(h, (5+start,), want)
        def attention(ix): return (values[ix] * ix.float().softmax(-1)).sum(-1)
        np.testing.assert_equal(got.pad_to((1, 4, 2)).numpy(), want.pad_to((1, 4, 2)).numpy())
        np.testing.assert_allclose(attention(got).pad_to((1, 4)).numpy(), attention(want).pad_to((1, 4)).numpy(), rtol=1e-6)
    np.testing.assert_equal(pieces[1].numpy()[3], [-1, -1])

if __name__ == "__main__":
  unittest.main()
