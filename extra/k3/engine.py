#!/usr/bin/env python3
# K3 decode with the routed experts streamed from the nvme drives: master's model (a trunk gguf without the routed experts, sharded over the
# gpus) in a jitted step graph of the engine's own forward. at a moe block every gpu writes its router scores and a tag to host memory, the
# host routes, reads the missing experts from the drives into each gpu's pool of slots and gives the gpus the slots and weights in two parts
# (host.py's step_io, in c alongside the graph), the gpus wait for each (host_wait) and run their units of the experts over the pool
# DEV=AMD:HIP PYTHONPATH=. python3 extra/k3/engine.py trunk.gguf "prompt"
import collections, copy, ctypes, functools, gc, itertools, math, os, signal, sys, time
import numpy as np
from typing import Callable, Generator, Iterator, cast
from tinygrad import Tensor, TinyJit, UOp, Device, dtypes, nn
from tinygrad.dtype import DType
from tinygrad.helpers import DEBUG, getenv, prod, round_up, unwrap
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.engine.jit import _TinyJit
from tinygrad.uop.ops import Ops, AxisType
from tinygrad.llm.model import Transformer, FFNBlock, GatedDeltaNetBlock, MLATransformerBlock, AttentionResidual, gated_activation
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_QUANT
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported, partial_sum, allgather_last, all_to_all, PARTIALS, \
  ordered_expert_sum, ordered_expert_sum_supported, scatter, ring_sum
from tinygrad.llm.kernels.amd_gguf import find_packed, copies_linear, expert_tiles, moe_route, moe_counts, q8_0_linear, QUANT_NAMES, \
  HALFWORD_QUANTS, TILE_QUANTS, tiles_dtype
from extra.k3.host import IoT, host_c, wait, reads, free_bytes, host_tensor, VICTIMS
from extra.k3.draft import Eagle3, DSpark, MTP, load
from extra.k3 import persist

# TAU: a token's experts under this weight left out unless cached, TAUC: those under it left out cached or not, BETA: a cached expert's
# routing bonus, FILL: the experts tau left out read anyway (a layer), PREFETCH: next-layer guesses per row (rows 2 onward 2 when all experts
# use their primary copy). SLOTS: on a gpu (default: what's free but SPARE bytes).
# BIG: a token's experts of a rank under it read in the store's first copy (of copies: host.py).
TAU, TAUC, BETA, FILL, PREFETCH = getenv("TAU", 0.065), getenv("TAUC", 0.0), getenv("BETA", 0.07), getenv("FILL", 2), getenv("PREFETCH", 5)
BIG = getenv("BIG", 2)
SLOTS, SPARE = getenv("SLOTS", 0), getenv("SPARE", 7 << 28)
# (the drives' bytes a second, a window's seconds a token, a step's before any are measured: tinys3's, 195 GB/s (k3's store not cached in 4.8 s),
# 1.26 ms, 50 ms)
READ_BW, WINDOW_TOKEN_S, STEP_S = 195e9, 1.3e-3, 0.05
SHORT_CHUNK = 256
SESSION = 4096  # (a kept session's rows a short request that diverges from it early may write)
COPY = 8  # (the context's copy: the tokens after an earlier place of its last COPY. K3.1's logged steps replayed: 8 and 12 within 0.2%, < 4 worse)
TILES = (16, 64)  # (a window's tiles of an expert's rows: the first that holds its rows an expert on average)

def _after(t:Tensor, dep:list[Tensor]) -> Tensor: return Tensor(t.uop.after(*(d.uop for d in dep)))

def _stored(t:Tensor, dep:list[Tensor]) -> Tensor:  # (t into a buffer of its own written after dep)
  return _after(Tensor.empty(*t.shape, dtype=t.dtype, device=t.device, axis=t.uop.axis), dep).assign(t)

class _WindowArenas(dict):
  def __init__(self, pool_key:tuple, room:UOp):
    super().__init__({pool_key: room})
    self.pool_key = pool_key

  def __setitem__(self, key, value):
    if key == self.pool_key: raise MemoryError("short window needs more than the shared scratch")
    super().__setitem__(key, value)


class Copies:
  """a request's context: where each run of COPY tokens before its last token last ended"""
  def __init__(self):
    self.ends:dict[tuple[int, ...], int] = {}
    self.upto = COPY - 1

  def __call__(self, tokens:list[int], n:int) -> list[int]:
    """the n tokens after the latest earlier place of the context's last COPY tokens (none if fewer follow)"""
    for e in range(self.upto, len(tokens) - 1): self.ends[tuple(tokens[e - COPY + 1:e + 1])] = e + 1
    self.upto = max(self.upto, len(tokens) - 1)
    at = self.ends.get(tuple(tokens[-COPY:]), len(tokens))
    return tokens[at:at + n] if at + n <= len(tokens) else []

class Engine:
  bank:DType = dtypes.half  # (a window's residual streams: in half, the attends read half the bytes)
  bank_streams:int|None = None  # (the bank's streams: the model's blocks' if None)
  def __init__(self, model:Transformer, m:dict, slots:int=SLOTS, tokens:int=1, guess:str="", draft:Eagle3|DSpark|MTP|None=None, logits:bool=False,
               chunk:int=0, window:int=0, taps:list[tuple[int, bool]]|None=None, widths:tuple[int, ...]=(), source:str="",
               copies:tuple[dict, ...]=(), short_window:int=0):
    """the moe blocks of model with their routed experts streamed from the store of manifest m into a pool of slots on each gpu, a part of
    every expert each (its units), in steps of a token or (a draft's verify steps) of tokens, a prompt in windows of up to window tokens (their
    attention in chunks), keeping their logits or not. taps: without a draft, the (block, mixed) streams a window keeps (draft.taps). widths:
    a scored draft's verify steps of fewer tokens too (their graphs captured), each step's the one its guesses' scores expect most tokens a
    second of. source: the model's gguf (a start's graphs are kept for the next start of the same code, model and config: persist.py). copies:
    the manifests of the store's other copies of the experts (of some of its layers: a step's expert read in one of them, BIG). short_window:
    an alternate window capacity for continuations, sharing the same pool and caches"""
    self.model, self.m, self.devs, self.draft, self.keep_logits = model, m, model.token_embd.weight.device, draft, logits
    self.source, self.saved, self.names = source, cast(dict, {}), {}
    self.t_stage = time.perf_counter()
    self.dump_taps = taps or []
    stack = [*model.blk, *(draft.blk if draft is not None else [])]
    blocks = [stack[b] for b in m["layers"] if b < len(stack)]
    # Layer-major windows must keep the most recent indexer's selection for each query until its last consumer.
    self.index_roles:dict[FFNBlock, str] = {}
    idx = model.config.indexer
    if window and idx is not None and model.max_context > idx.top_k and any(model.config.indexer_shared):
      owner = None
      for b in model.blk:
        if getattr(b, "indexer", None) is not None: owner = b
        elif owner is not None: self.index_roles[owner], self.index_roles[b] = "write", "read"
      # Other indexers/backends return dense floating masks. Token-major steps already carry those exactly.
      if self.index_roles and not (idx.kpool == 1 and not (idx.msa or idx.qsa) and idx.head_dim % 16 == 0 and
                                   amd_custom_kernels_supported(self.devs)):
        window = short_window = 0
    self.short_window, self.short_engine = short_window, cast(Engine|None, None)
    if short_window:
      short_chunk = min(SHORT_CHUNK, short_window // 2)
      if not (0 < short_window < window and 0 < short_chunk < chunk and short_window % short_chunk == short_chunk % len(self.devs) == 0):
        raise ValueError("a short window needs a smaller window and an evenly sharded smaller chunk")
    self.forecast_counts = host_tensor(self.devs, m["n_experts"], dtypes.uint32) if window else None
    # (a window's experts graph waits on the gpus for the host's part 1 (a graph's host_wait: amd), launched with its post before the counts
    # when none of the layer's reads are in flight: a wait polls host memory, the drives' reads slower)
    self.waits = amd_custom_kernels_supported(self.devs)
    self.ms = [m, *copies]
    L, E, K, G, self.lat, self.units = len(blocks), m["n_experts"], model.config.num_experts_per_tok, len(self.devs), m["lat"], m["units"][0]
    assert all(len(set(c["part_bytes"])) == 1 and c["units"] == [self.units] * G and c["lat"] == self.lat for c in self.ms), \
      "a part of every expert on each gpu"
    self.store_part = min(c["part_bytes"][0] for c in self.ms)
    assert E <= 4096 and K <= 64 and tokens <= 8 and tokens * K <= 512 and 32 * tokens <= 1024, "past the host's fixed arrays"
    self.tokens, self.chunk, n = tokens, chunk, max(tokens, chunk, window)  # (n: a step's or a window's most tokens)
    qtype, layers = {n.upper(): t for t, n in QUANT_NAMES.items()}, [b for b in m["layers"] if b < len(stack)]
    self.copies = [[(qtype[t[0]], qtype[t[2]]) if t else None for t in (c["types"].get(str(b)) if "types" in c else [c["expert_type"]] * 3
                                                                      if b in c["layers"] else None for b in layers)] for c in self.ms]
    self.types, self.layer_copies = self.copies[0], [tuple(c[li] for c in self.copies) for li in range(len(layers))]
    self.tiles_dtype = tiles_dtype(self.devs, [t for c in self.copies for ty in c if ty is not None for t in ty])
    self.expert_sizes = [[0 if ty is None else round_up(self.nbytes(ty[0], 2 * self.units * self.lat) +
                          self.nbytes(ty[1], self.units * self.lat), 4096) for ty in types] for types in self.copies]
    # (a pool's slot: of an expert's size in 1-4 parts, the one that pads the experts least (the largest of those), an expert its span of them)
    every = [size for sizes in self.expert_sizes for size in sizes if size]
    self.part = min({round_up(-(-size // k), 4096) for size in every for k in range(1, 5)},
                    key=lambda p: (sum(-(-size // p) * p - size for size in every), -p))
    self.layer_spans = [[self.slots_of(size) if size else 0 for size in sizes] for sizes in self.expert_sizes]
    self.spans = sorted({w for spans in self.layer_spans for w in spans if w}, reverse=True)
    self.span_groups = np.array([[self.spans.index(w) if w else 0 for w in spans] for spans in self.layer_spans], np.int64)
    self.max_primary_span = max(self.layer_spans[0])
    # each layer's host buffers on every gpu: its tag, router scores (a token's rows: the router's, the next layer's guess; a gpu its experts')
    # and the two parts of a step's slots and weights, [ready | slots | weights]: entry j*T+t is token t's j-th expert and its weight (a
    # window's: [ready | an expert's slot], [ready | an expert's count])
    self.ks, self.act = (K // 2, K - K // 2), functools.partial(gated_activation, config=model.config)
    part = [1 + max(2 * k * tokens * len(self.ms), E) for k in self.ks]  # (a part's words)
    bufs = [[host_tensor(self.devs, sz, dt) for sz, dt in ((1, dtypes.uint32), (n * 2 * E // G, dtypes.float32), *((p, dtypes.uint32) for p in part))]
            for _ in blocks]
    self.bufs, self.tags, self.parts = bufs, [b[0][0].uop.reshape(1, 1).unshard(0) for b in bufs], [[b[2 + i][0] for b in bufs] for i in range(2)]
    self.scores = [b[1][0].uop.reshape(n, 2, E // G).unshard(2) for b in bufs]
    # the scorers: a layer's router and the guess of the next layer's experts on its input (GUESS=dir: guess-<block>.npy maps fitted on traces,
    # else the next layer's router), a gpu its experts' rows of each (in q8_0 where there's a kernel). the routers' memory freed: the scorers route
    # (the routers' float copies not held: the scorers' graphs only, freed once they're realized)
    def router(li:int) -> Tensor: return blocks[li].ffn_gate_inp.weight.to(self.devs[0]).float()
    def guessed(li:int) -> Tensor:
      return Tensor(np.load(f)).to(self.devs[0]) if os.path.exists(f:=f"{guess}/guess-{m['layers'][li]}.npy") else router(li)
    def scorer(w:Tensor) -> Tensor|Linear:  # (rows a gpu's: its experts' of the router, then of the guess)
      w = w.reshape(2, G, E // G, -1).permute(1, 0, 2, 3).reshape(2 * E, -1).half()
      return q8_0_linear(w, self.devs, 0) if amd_custom_kernels_supported(self.devs) else w.shard(self.devs, 0)
    # (the scorers kept for the next start: of the gguf and the guess maps)
    loaded = persist.load_scorers(self, guess)
    self.scorers = loaded or [scorer(Tensor.stack(router(li), guessed(min(li + 1, L - 1)))) for li in range(L)]
    Tensor.realize(*(biases:=[b.exp_probs_b["bias"] for b in blocks]), *[s.weight if isinstance(s, Linear) else s for s in self.scorers])
    if loaded is None: persist.save_scorers(self, guess, self.scorers)
    self.bias = list(Tensor.stack(*[t.float() for t in biases]).numpy())
    for b in model.blk:  # (a verify step's recurrent states after its first token kept: a rollback)
      if isinstance(b, GatedDeltaNetBlock): b.keep = tokens - 1
    for li, b in enumerate(blocks):
      b._feed_forward = functools.partial(self.feed_forward, li, b)  # type: ignore[method-assign]
      b.ffn_gate_inp.weight = Tensor.empty(0)
    # (the weights a quantized kernel reads packed: not realized here, the step's graph decodes them for its kernels. the others decoded once)
    Tensor.realize(*[t.replace(t.contiguous()) for t in nn.state.get_parameters([model, draft]) if t.numel() and find_packed(t, t.shape[-1]) is None])
    weights = {id(t) for t in nn.state.get_parameters(stack)}
    # (a draft holding the trunk's weights would realize them as its state: decoded, their quantized kernels lost)
    assert not {id(t) for t in nn.state.get_parameters(draft)} - weights & {id(t) for t in nn.state.get_parameters(model)}, \
      "a draft with the trunk's weights"
    for b in stack: b._init_state(Tensor.empty(1, 1, model.config.dim, device=self.devs))
    # the context's caches (the mla blocks' latents, their indexers' keys, the draft's): lent from the pool (never allocated here)
    owners = [(o, k) for b in stack for o, k in ((b, "cache_k"), (getattr(b, "indexer", None), "cache_kg")) if hasattr(o, k)]
    self.caches = owners + ([(draft, "cache")] if hasattr(draft, "cache") else [])
    weights |= {id(getattr(o, k)) for o, k in self.caches}
    Tensor.realize(*(states:=[t for t in nn.state.get_parameters([model.blk, draft]) if id(t) not in weights and t.numel()]))
    # (the empty ones allocated too, else at their first use: the ones with a buffer, not the folded constants)
    for buf in (t.uop.base.buffer for t in states if t.uop.base.op in (Ops.BUFFER, Ops.UNSHARD)):
      for x in getattr(buf, "bufs", [buf]): x.ensure_allocated()
    del states
    model.save([])  # (and the two sets of saved recurrent states)
    # a step's graph for each count of tokens, their tokens and the draft's given tokens in host memory
    self.v_pos, self.v_live = UOp.variable("start_pos", 0, model.max_context - 1), UOp.variable("live", 1, max(chunk, 1))
    # (a read window's tokens: up to the context's)
    (self.tok, (self.tok_np,)), (self.given, (self.given_np,)) = (host_tensor(self.devs[0], max(n, model.max_context), dtypes.int32)
                                                                  for _ in range(2))
    # ([done | tokens | guesses | their scores (a scored draft's, f32)])
    self.res, (self.res_np,) = host_tensor(self.devs[0], 1 + n * (1 + 2 * max(self.width, 1)), dtypes.int32)
    self.temperature, (self.temperature_np,) = host_tensor(self.devs[0], 1, dtypes.float32)
    self.key, (self.key_np,) = host_tensor(self.devs[0], 2, dtypes.uint32)
    self.rng = np.random.default_rng()
    self.jits = {T: TinyJit(functools.partial(self.run, T), capture_first=True) for T in sorted({1, tokens, *widths})}
    self.keep_jits = (TinyJit(self.keep, capture_first=True), TinyJit(self.unkeep, capture_first=True))
    self.widths = sorted({tokens, *widths}) if draft is not None and draft.scored else [tokens]
    # (measured: a step's seconds, the chain's that guessed for it too)
    self.step_s:dict[int, collections.deque] = {T: collections.deque(maxlen=64) for T in self.jits}
    self.chain_s = 0.0
    # a draft's chain of guesses after its first (a verify step's other tokens): its steps on its own hidden states (dx: a step's), from a row
    # ([done | a chain step's guesses | their probabilities (f32)])
    self.links = 1 if draft is not None and draft.blk else max(tokens - 2, 0)
    self.cres, (self.cres_np,) = host_tensor(self.devs[0], 1 + 2 * self.links, dtypes.int32)
    self.dx, self.v_row = Tensor.empty(n, model.config.dim, device=self.devs).realize(), UOp.variable("row", 0, tokens - 1)
    self.chain_jit = TinyJit(self.run_chain, capture_first=True) if draft is not None and draft.width < tokens - 1 else None
    self.times, self.verified = np.zeros(2), np.zeros(3)  # (launching the graphs, after the last layer), (verify steps, guesses, right ones)
    self.scored, self.copied = np.zeros((2, 10)), np.zeros(2)  # (a scored draft's verified guesses by the score's tenth, the copy's: seen, right)
    self.window_s:collections.deque = collections.deque(maxlen=64)  # (measured: a window's tokens, seconds)
    # each gpu's pool in the experts' words (a bitcast of it is a copy), of what's free after the blocks' caches and states and the window's
    # graphs (captured on a pool of a slot), then the slots a window's state is lent (its own while it runs): slots of decode's otherwise
    words = {t: dtypes.uint16 if t in HALFWORD_QUANTS else dtypes.uint32 for t in itertools.chain(*(t for ts in self.copies for t in ts if t))}
    self.word = min(words.values(), key=lambda dt: dt.itemsize)
    def pool(n:int) -> Tensor: return Tensor.zeros(G * n * self.part // self.word.itemsize, dtype=self.word, device=self.devs, axis=0).realize()
    self.W = 0
    if window: self.window_init(window)
    # (the caches lent from the pool's first slots (a gpu its shard of each), their rows of positions in stripes: the rows a context reached
    # kept from the experts (grow), the rest the experts' (never read))
    shapes = {str(i): (getattr(o, k).uop.shard_shape, getattr(o, k).dtype) for i, (o, k) in enumerate(self.caches)}
    axes = [getattr(o, k).uop.axis for o, k in self.caches]
    # a kept session (a long one a short request diverges from early): copies of the caches' rows that request may write, its recurrent states
    self.B, self.session = min(SESSION, model.max_context // 2), cast(tuple[list[int], int, list[tuple[list[int], int]]]|None, None)
    self.v_at = UOp.variable("at", 0, model.max_context - self.B)
    def rows0(t:Tensor) -> tuple[int, ...]: return tuple(self.B if j == next(j for j, d in enumerate(t.shape) if d >= model.max_context) else d
                                                         for j, d in enumerate(t.shape))
    self.stash = [Tensor.empty(*rows0(t:=getattr(o, k)), dtype=t.dtype, device=self.devs, axis=t.uop.axis) for o, k in self.caches]
    self.stash_states = [s.empty_like() for s in self.session_states()]
    Tensor.realize(*self.stash, *self.stash_states)
    for o, k in self.caches: setattr(o, k, None)
    free = (lambda: min(free_bytes(d) for d in self.devs)) if not slots else (lambda: 0)  # (what's free on the gpus: sizing the pool by it)
    self.stage(f"states ({free() / 1e9:.2f} GB free before the pool)")
    # (lent: the window's compute state and a piece's, the slots of more pieces lent at a prefill)
    self.win, lent, self.layer = {}, self.slots_of(self.window_bytes) + self.slots_of(self.piece_bytes) if window else 0, 0
    self.slots = slots = ((slots or (free() - SPARE) // self.part - lent) + lent) // (w:=math.lcm(*self.spans)) * w
    self.tail = self.lent_from = slots - lent
    self.pool = pool(slots)
    pools = {dt: self.pool if dt == self.word else Tensor(self.lend({"pool": ((slots * self.part // dt.itemsize,), dt)}, 0)[0]["pool"].uop.unshard(0))
             for dt in set(words.values())}
    self.pools = {t: pools[dt] for t, dt in words.items()}
    caches, self.cache_bytes = self.lend(shapes, 0)
    # (a cache's: its positions' axis, its first byte, a position's bytes, its blocks of every position's rows and their bytes)
    self.stripes = [(a, caches[i].uop.base.buffer.bufs[0].offset, row, prod(shape[:a]), shape[a] * row) for i, (shape, dt) in shapes.items()
                    for a in [next(j for j, d in enumerate(shape) if d >= model.max_context)] for row in [prod(shape[a + 1:]) * dt.itemsize]]
    for i, (o, k) in enumerate(self.caches): setattr(o, k, caches[str(i)] if axes[i] is None else Tensor(caches[str(i)].uop.unshard(axes[i])))
    # (the caches grow by a step of rows, zeroed in a graph: a step's rows at a position)
    self.grown, self.grow_step = 0, min(chunk or 8192, model.max_context)
    self.v_grown, self.zero_jit = UOp.variable("grown", 0, model.max_context - self.grow_step), TinyJit(self.zero, capture_first=True)
    # the host's state, kept by step_io: each code's (layer * E + expert) slot (-1: not cached) and each slot's code (-1: free), the decayed
    # use counts, the guesses' reads in flight, the guesses their layer hasn't routed, pinned slots, the victims of the next reads
    codes = len(self.ms) * L * E
    self.state = {"slot_of": np.full(codes, -1, np.int32), "expert_in": np.full(slots, -1, np.int64), "freq": np.zeros(codes, np.float32),
                  "flying": np.zeros(codes, np.int32), "guess": np.zeros(codes, np.uint8), "mask": np.zeros(slots, np.uint8),
                  "victims": np.zeros(len(self.spans) * VICTIMS, np.int64), "nvict": np.zeros(len(self.spans), np.int64),
                  "span": np.array(self.spans, np.int64), "span_group": self.span_groups,
                  "bias": np.array([b.ctypes.data for b in self.bias], np.uint64),
                  "bufs": np.array([[[v.ctypes.data for v in views] for _, views in b] for b in bufs], np.uint64)}
    self.io = IoT(layers=L, E=E, ngpu=G, k=K, kmin=K // 4, k1=self.ks[0], rows=2, slots=slots, tau=TAU, tauc=TAUC, beta=BETA, pf=PREFETCH,
                  fill=FILL, decay=0.5 ** (1 / 8), copies=len(self.ms), big=BIG, slot_groups=len(self.spans),
                  pf_tail=2, tail=1)
    for name, a in self.state.items(): setattr(self.io, name, a.ctypes.data)
    if window:
      self.stage(f"pool: {slots} slots ({slots * self.part / 1e9:.2f} GB, of it lent: the caches' {self.cache_bytes / 1e9:.2f} GB as they grow, "
                 f"the window's {(slots - self.tail) * self.part / 1e9:.2f} GB)")
      self.window_lend()
      self.stage(f"window graphs (their scratch {(self.arena[1] - self.arena[0]) * self.part / 1e9:.2f} GB, the pool's)")
    else: self.restore()

  def drives(self, *disks:list[list[str]]):
    """step_io's reads from the store's drives (each copy's: each gpu's disk) into the pools"""
    def sizes(c:dict, lengths:list[int]) -> list[tuple[int, int]|None]:
      return [(c["layers"].index(b), n) if n else None for b, n in zip(self.m["layers"], lengths)]
    self.tables, scalars = reads([(c, d, sizes(c, lengths)) for c, d, lengths in zip(self.ms, disks, self.expert_sizes)],
                                self.pool.uop.buffer.bufs, self.part)
    for name, v in {**scalars, **{k: a.ctypes.data for k, a in self.tables.items()}}.items(): setattr(self.io, name, v)

  # *** the step's graph

  @property
  def taps(self) -> list[tuple[int, bool]]: return self.draft.taps if self.draft is not None else self.dump_taps
  @property
  def width(self) -> int: return self.draft.width if self.draft is not None else 0  # (a row's guesses)

  def mixer(self, i:int) -> AttentionResidual:  # (the attention residual that next reads the streams after block i)
    return self.model.blk[i + 1].attn_res_score if i + 1 < len(self.model.blk) else self.model.output_res_score

  def forward(self, tokens:Tensor, pos:UOp) -> tuple[Tensor, Tensor|None]:
    """the logits (T, vocab) after each of the tokens (1, T) at positions pos.., the draft's taps (1, T, taps, dim): after a block its last
    stream (the block's sum so far) or (mixed) the streams as the next block's attention residual mixes them"""
    m, hidden, taps, streams = self.model, {}, self.draft.taps if self.draft is not None else [], self.model.config.attn_res_block_size
    x = m.token_embd(tokens.to(m.token_embd.weight.device)).float()
    if streams: x = x.unsqueeze(2)  # (the attention residuals' streams: (1, T, 1, dim))
    mask = None
    for i, b in enumerate(m.blk):
      x, mask = b(x, (pos,), mask)
      for t in taps:
        if t[0] == i: hidden[t] = (self.mixer(i).mix(x) if t[1] else x[:, :, -1]) if streams else x
    x = m.output_res_score.mix(x) if streams else x
    return m.output(m.output_norm(x))[0].float(), Tensor.stack(*(hidden[t] for t in taps), dim=2) if taps else None

  def run(self, T:int, pos:UOp) -> tuple[Tensor, Tensor]:
    """the token after each of the step's tokens (sampled at the temperature) and (a draft) its guesses of the tokens after that (after the
    given one where there's one: a prompt's next) into host memory, then its done word (the host reads them without a sync), their logits
    (kept, else done)"""
    done, logits, *dx = self.outputs(T, pos)
    Tensor.realize(done, logits, *dx)
    return done, logits

  def outputs(self, T:int, pos:UOp) -> list[Tensor]:
    """run's tensors: the done word, the logits (kept, else the done word) and the draft's hidden states"""
    logits, hidden = self.forward(self.tok[:T].reshape(1, T), pos)
    u = Tensor._rand(self.key, self.key.zeros_like(), logits.shape, dtypes.float32).maximum(1e-12)
    out, dx = (toks:=(logits + (self.temperature * -(-u.log()).log()).shard_like(logits)).argmax(-1)), []
    if self.draft is not None:
      # the token after each row (the given one, else the target's) and the rows kept (the step's tokens it matches from the second on): the
      # draft's guesses after the last kept row's
      truth = ((g:=self.given[:T].to(toks.device)) >= 0).where(g, toks)
      kept = ((self.tok[1:T].to(toks.device) == truth[:T - 1]).cast(dtypes.int32).cumsum(0) == Tensor.arange(1, T).to(toks.device)).sum() \
        if T > 1 else 0
      guesses, scores, x = self.draft.guess(truth.reshape(1, T), hidden, pos, kept)
      out = toks.cat(guesses, *([] if scores is None else [scores.float().bitcast(dtypes.int32)]))
      dx = [Tensor(self.dx.uop)[:T].assign(x[0].float())] if x is not None else []
    outs = Tensor(self.res.uop)[1:1 + out.shape[0]].assign(out.to(self.devs[0]).cast(dtypes.int32))
    done = Tensor(self.res.uop.after(outs.uop))[:1].assign(Tensor.ones(1, dtype=dtypes.int32, device=self.devs[0]))
    return [done, logits if self.keep_logits else done, *dx]  # (realized, the logits are a buffer of their own: only for an eval)

  def run_chain(self, pos:UOp, row:UOp) -> Tensor:
    """the draft's next guesses after the given one (a token at pos+1), a step each on its own hidden state of the step before (from its
    row of the verify step's) into host memory and their probabilities, then the done word"""
    tok, hid, cache, out, probs = self.given[:1].to(self.devs), Tensor(self.dx.uop)[row:row + 1], None, [], []
    for k in range(self.links):
      logits, hid, cache = self.draft(tok.reshape(1, 1), None, pos + k, hid=hid.reshape(1, 1, -1), cache=cache)
      out.append(tok:=logits[0].argmax(-1))
      probs.append(logits[0].softmax(-1).max(-1).float().bitcast(dtypes.int32))
    outs = [Tensor(self.cres.uop)[1:].assign(Tensor.cat(*[t.to(self.devs[0]).cast(dtypes.int32) for t in out + probs])).uop]
    if self.draft.blk: outs.append(Tensor(self.dx.uop)[row:row + 1].assign(hid.reshape(1, -1).float()).uop)
    return Tensor(self.cres.uop.after(*outs))[:1].assign(Tensor.ones(1, dtype=dtypes.int32, device=self.devs[0])).realize()

  def feed_forward(self, li:int, b:FFNBlock, x:Tensor) -> Tensor:
    """moe block b's (layer li's) feed forward on the tokens x (1, T, dim): the gpus' scores to the host, then the experts' inputs and the shared
    experts on the gpus while it routes"""
    T, s = x.shape[1], w(x) if isinstance(w:=self.scorers[li], Linear) else x[0] @ w.T
    scores = Tensor(self.scores[li])[:T].assign(s.float().reshape(-1, len(self.devs), 2, self.m["n_experts"] // len(self.devs))
                                                .permute(0, 2, 1, 3).reshape(-1, 2, self.m["n_experts"]))
    x = Tensor(x.contiguous().uop.after((tag:=self.routing_tag(li, scores)).uop))
    h, shared = b._expert_input(x), b._shared(x)
    return b._expert_output(self.routed(li, h, self.host(li, 0, scores, tag, h, *([] if shared is None else [shared]))), shared)

  def routed(self, li:int, h:Tensor, first:Tensor) -> Tensor:
    """layer li's routed experts on the expert inputs h (1, T, lat): each gpu's units of the experts it gives in two parts (first: part 1's
    table), summed over the gpus"""
    T = h.shape[1]
    y1, w1 = self.experts(li, h[0], first, self.ks[0] * T)
    y1 = y1.contiguous()  # (a buffer: part 2 waited for after it)
    y2, w2 = self.experts(li, h[0], self.host(li, 1, y1), self.ks[1] * T)
    if self.m["layers"][li] < len(self.model.blk) and ordered_expert_sum_supported(y1.device):
      y = ordered_expert_sum(y1, y2, w1, w2, T)
    else:
      y = self.heaviest_first([y1, y2], [w1, w2], T)
    return (partial_sum(y) * self.model.config.routed_scaling_factor).unsqueeze(0)

  def routing_tag(self, li:int, scores:Tensor) -> Tensor:
    s = scores[0, 0].reshape(len(self.devs), -1)[:, :1]
    return Tensor(self.tags[li]).assign((s == s).where(1, 2).cast(dtypes.uint32))

  def host(self, li:int, i:int, *deps:Tensor) -> Tensor:
    """part i of layer li's slots and weights once the host releases it, waited for after deps (in their order: part 1's the scores, the
    tag, then the experts' inputs and the shared experts the gpus run while the host routes)"""
    return Tensor((t:=self.parts[i][li].uop).after(wait(t, *(d.uop for d in deps))))

  def experts(self, li:int, h:Tensor, table:Tensor, n:int) -> tuple[Tensor, Tensor]:
    """each gpu's units of layer li's n experts of a table (a copy's n slots, padded with slot 0, their weights a token's: j*T + t, then the next
    copy's) on h (T, lat): (gpus, n, lat) and the weights (n,). entry j*T+t is token t's j-th expert, from the copy where its weight isn't 0,
    the act in the q8 of the down gemv's input"""
    u, sets = self.units, []
    for c, types in enumerate(self.copies):
      if types[li] is None: continue
      (gu_type, down_type), stride, at = types[li], self.layer_spans[c][li] * self.part, 1 + 2 * c * n
      off = self.nbytes(gu_type, 2 * u * self.lat)
      sets.append((gu_type, down_type, stride, off, table[at:at + n].bitcast(dtypes.int32), table[at + n:at + 2 * n].bitcast(dtypes.float32)))
    ws = functools.reduce(Tensor.add, [w for *_, w in sets])
    if amd_custom_kernels_supported(h.device):
      gu = copies_linear([(self.pools[g], slots, g, stride, 0, w) for g, _, stride, _, slots, w in sets], h, 2 * u, self.lat, split=True)
      return copies_linear([(self.pools[d], slots, d, stride, off, w) for _, d, stride, off, slots, w in sets], gu, self.lat, u, act=self.act), ws
    ys = []
    for gu_type, down_type, stride, off, slots, w in sets:
      gu = self.gemv(stride, slots, h, 2 * u, self.lat, gu_type, 0)
      ys.append((w.reshape(1, -1, 1) != 0).where(self.gemv(stride, slots, self.act(*gu.split(u, -1)), self.lat, u, down_type, off), 0))
    return functools.reduce(Tensor.add, ys), ws

  @staticmethod
  def heaviest_first(ys:list[Tensor], ws:list[Tensor], T:int) -> Tensor:
    """(gpus, T, lat): each token's expert outputs of the parts ys (gpus, n, lat), entry j*T+t token t's, weighted by ws (n,) and summed
    heaviest first (a tie: the earlier entry first), the same sum whichever part an expert came in"""
    rows = [(y[:, j * T:(j + 1) * T], w[j * T:(j + 1) * T].reshape(1, T, 1)) for y, w in zip(ys, ws) for j in range(w.shape[0] // T)]
    rank = Tensor.stack(*[sum(((v >= w) if i < j else (v > w)).cast(dtypes.int32) for i, (_, v) in enumerate(rows) if i != j)
                          for j, (_, w) in enumerate(rows)]).contiguous()
    return functools.reduce(Tensor.add, [functools.reduce(Tensor.add, [(rank[j] == r).where(y * w, 0) for j, (y, w) in enumerate(rows)])
                                         for r in range(len(rows))])

  def gemv(self, stride:int, slots:Tensor, x:Tensor, out_f:int, in_f:int, qtype:int, off:int) -> Tensor:
    """(gpus, n, out_f): W @ x[k % rows] (or a gpu's x[k]) of the W of quant type qtype at off bytes in copy c's slots[k] of each gpu's pool,
    decoded (no kernel)"""
    n, size, isz = slots.shape[0], self.nbytes(qtype, out_f * in_f), self.word.itemsize
    w = self.pool.reshape(len(self.devs), -1, stride // isz)[:, slots][..., off // isz:(off + size) // isz]
    w = w.bitcast(dtypes.uint8)
    w = ggml_data_to_tensor(w.flatten(), prod(w.shape) // size * out_f * in_f, qtype).reshape(-1, n, out_f, in_f)
    return (w * (x if x.ndim == 3 else x[Tensor.arange(n) % x.shape[0]]).unsqueeze(-2)).sum(-1)

  # *** a prompt's window: the blocks a layer at a time over up to W tokens (layer major: a layer's experts read once a window). a layer's
  # attention, the mixes and the shared experts in steps of a chunk, its tokens' routed experts once over the window. the residual streams of
  # the window's tokens in a bank, a layer adds to its last one. the graphs are a kind's (blocks of the same buffers): a block's buffers its
  # args, the first of the kind's block captures it

  def window_init(self, W:int, arenas:dict|None=None):
    """the window's variables and graphs (captured once its state is lent from the pool: window_capture), its state's bytes"""
    m = self.model
    self.chunk_vars = {SHORT_CHUNK: UOp.variable("short_live", 1, SHORT_CHUNK)} if self.chunk > SHORT_CHUNK else {}
    self.chunk_vars[self.chunk] = self.v_live
    self.W, S = W, 1 + sum(map(self.new_stream, m.blk))
    self.window_bytes, self.piece_bytes = (sum(round_up(prod(sh) * dt.itemsize, 256) for sh, dt in f().values())
                                           for f in (self.window_shapes, self.piece_shapes))
    self.v_p0 = UOp.variable("p0", 0, self.max_context - 1)  # (a piece's first token in the read window)
    self.v_c0, self.v_wlive = UOp.variable("c0", 0, W - self.chunk), UOp.variable("wlive", 1, W)
    # (an attends graph's chains: each its chunk's position, the full ones' live rows a variable as the last one's)
    self.v_poss = [self.v_pos, *(UOp.variable(f"start_pos{k}", 0, self.max_context - 1) for k in range(1, -(-W // self.chunk)))]
    self.v_full = UOp.variable("full", 1, self.chunk)
    self.v_n, self.v_s = UOp.variable("streams", 1, S), UOp.variable("stream", 0, S - 1)
    self.v_tap = UOp.variable("tap", 0, self.piece_shapes()["taps"][0][3] - 1)
    self.arenas:dict = {} if arenas is None else arenas  # (the window's graphs never run at the same time)
    self.wjit = functools.partial(TinyJit, arenas=self.arenas, capture_first=True)
    self.tap_jit, self.mix_jit = self.wjit(self.tap), self.wjit(self.tap_mix)
    self.fill_jit = self.wjit(self.fill) if self.draft else None
    # (a window's routed experts in tiles of an expert's rows sized by its rows an expert: few tokens, 16 rows, else 64. a latent's experts graph
    # of each count of a piece's chunks a tile size takes, the most first: each lane of their scratch allocated once)
    def counts(t:int) -> range:
      most = W if t == TILES[-1] else min(W, t * self.m["n_experts"] // m.config.num_experts_per_tok)
      return range(-(-most // self.chunk), 0, -1) if m.config.expert_latent_dim else range(1)
    self.routes = {t: (self.wjit(functools.partial(self.window_route, t)),
                       {(ty, n): self.wjit(functools.partial(self.window_experts, t, ty, n)) for ty in dict.fromkeys(self.layer_copies)
                        for n in counts(t)}) for t in TILES}
    self.forecast_jit = self.wjit(self.forecast_route) if self.forecast_counts is not None else None
    self.start_jit = self.wjit(self.start)

  def window_lend(self):
    """the window's state lent from the pool's last slots, its graphs captured on it, their scratch (one arena) lent from the pool's first slots
    past the caches' (what of them it takes the experts' between prefills)"""
    self.tail = self.lent_from = self.slots - self.slots_of(self.window_bytes) - self.slots_of(self.piece_bytes)
    self.win = {**self.window_state(self.tail), **self.piece(self.slots - self.slots_of(self.piece_bytes))}
    first, key = self.slots_of(self.cache_bytes), (self.devs, 0)
    if first >= self.tail:
      raise MemoryError(f"the pool's {self.slots * self.part / 1e9:.2f} GB under the caches' {self.cache_bytes / 1e9:.2f} GB and the window's "
                        f"{(self.slots - self.tail) * self.part / 1e9:.2f} GB: no room for the window's graphs' scratch (a smaller CTX or WINDOW)")
    room = self.lend({"arena": (((self.tail - first) * self.part,), dtypes.int8)}, first)[0]["arena"].uop.base
    self.arenas.clear()
    self.arenas[key] = room
    self.restore()
    host_c().flags(ctypes.byref(self.io), 1)
    try: self.window_capture()
    except MemoryError as e: raise MemoryError(f"the window's graphs' scratch past the pool's {room.max_numel() / 1e9:.2f} GB for it") from e
    if (need:=self.arenas[key].max_numel()) > room.max_numel():
      raise MemoryError(f"the window's graphs' scratch {need / 1e6:.1f} MB past the pool's {room.max_numel() / 1e6:.1f} MB for it")
    jits = [j for _, j, _ in self.kinds.values()] + [j for r, e in self.routes.values() for j in (r, *e.values())] + \
           [self.start_jit, self.tap_jit, self.mix_jit, self.fill_jit, self.forecast_jit]
    used = max((sum(u.marg[0]) for j in jits if j is not None and j.captured is not None for u in j.captured._linear.toposort()
                if u.op is Ops.SHRINK and u.src[0] is room), default=0)  # (a planned buffer: the arena's (start, bytes))
    self.arena = (first, first + self.slots_of(used))

  def capture_short_window(self):
    """Capture an alternate window over the same pool and caches before serving the first prompt."""
    for d in self.devs: Device[d].synchronize()
    view = copy.copy(self)
    view.short_window, view.short_engine = 0, None
    view.chunk = min(SHORT_CHUNK, self.short_window // 2)
    view.v_live = UOp.variable("short_window_live", 1, view.chunk)
    # The persisted graph cache belongs to the full window. The alternate view captures its own graphs.
    view.saved, view.names, view.source = {}, {}, ""
    view.__dict__.pop("graph_key", None)
    # Reject any growth of the pool's compute arena during graph lowering, before executing the dry graph.
    start, stop = self.arena
    room = (view.lend({"arena": (((stop - start) * view.part,), dtypes.int8)}, start)[0]["arena"].uop.base if stop > start else
            UOp.new_buffer(view.devs, 0, dtypes.int8))
    view.window_init(self.short_window, _WindowArenas((view.devs, 0), room))
    view.tail = view.lent_from = view.slots - view.slots_of(view.window_bytes) - view.slots_of(view.piece_bytes)
    assert view.tail >= self.tail
    view.win = {**view.window_state(view.tail), **view.piece(view.slots - view.slots_of(view.piece_bytes))}
    # Both windows grow and pin the same model cache through the main engine.
    view.grow = self.grow  # type: ignore[method-assign]
    tables = [(host, host.copy()) for b in self.bufs for i in (0, 2, 3) for host in b[i][1]]
    # Dry execution writes at position zero. Preserve the warm engine's recurrent state as well as the host tables.
    states = [(t, t.empty_like().assign(t)) for t in self.model._states()]
    if states: Tensor.realize(*(saved for _, saved in states))
    try:
      for host, _ in tables: host.fill(0)
      host_c().flags(ctypes.byref(self.io), 1)
      view.window_capture()
    finally:
      if states: Tensor.realize(*(t.assign(saved) for t, saved in states))
      for d in self.devs: Device[d].synchronize()
      for host, previous in tables: host[:] = previous
    self.short_engine = view

  def window_capture(self):
    """the window's graphs captured on its lent state: every kind's on its first block (its buffers the args) on dry chunks at position 0 (the
    host's flags preset: slot 0), the routes and experts, the rest. their scratch: one arena (SPARE's)"""
    m, self.kinds = self.model, {}
    moe = [b for b in m.blk if hasattr(b, "ffn_gate_inp")]
    stages = [("attend", self.attends)]
    if self.forecast_jit is not None: stages.append(("forecast", functools.partial(self.attends, forecast=True)))
    for b, (name, fxn), n, cap in itertools.product(m.blk, stages, range(-(-self.W // self.chunk), 0, -1), self.chunk_vars):
      if name == "forecast" and (b not in moe or cap != min(self.chunk_vars) or n > 1): continue
      jit, args, first, live = self.piece_graph(f"{name}{self.index_roles.get(b, '')}", b, fxn, (n - 1) * self.chunk + cap)
      if first is b: self.graph(jit, *self.chained_vars(0, n), live, self.v_n.bind(1), self.v_s.bind(0), *args)
    for b, n, cap in itertools.product(moe, range(-(-self.W // self.chunk), 0, -1), self.chunk_vars):
      jit, args, first, live = self.piece_graph("post", b, self.posts, (n - 1) * self.chunk + cap)
      if first is b: self.graph(jit, live, self.v_s.bind(0), *args)
    self.graph(self.start_jit, self.v_p0.bind(0))
    self.graph(self.tap_jit, self.v_s.bind(0), self.v_tap.bind(0))
    if self.fill_jit: self.graph(self.fill_jit, self.v_pos.bind(0), self.v_p0.bind(0), self.v_c0.bind(0))
    for b in moe: self.dry(b)  # (and once on every moe layer's host buffers: their first use maps them on each gpu, a first window would wait)

  def dry(self, b:FFNBlock):  # (moe block b's route and experts in each tile size on a dry chunk: each experts graph once, a layer's one)
    li = self.m["layers"].index(self.model.blk.index(b))
    if self.forecast_jit is not None and li + 1 < len(self.types) and (target:=self.m["layers"][li + 1]) < len(self.model.blk):
      self.graph(self.forecast_jit, self.v_wlive.bind(min(self.chunk, SHORT_CHUNK)), b.exp_probs_b["bias"],
                 self.model.blk[target].exp_probs_b["bias"], Tensor(self.tags[li]), self.parts[1][li])
    for route, experts in self.routes.values():
      self.graph(route, self.v_wlive.bind(self.chunk), b.exp_probs_b["bias"], Tensor(self.tags[li]), self.parts[1][li])
      for i, jit in enumerate(j for (tys, _), j in experts.items() if tys == self.layer_copies[li]):
        if not i or jit.captured is None: self.graph(jit, self.pool, self.parts[0][li], Tensor(self.tags[li]))

  def window_shapes(self) -> dict[str, tuple[tuple[int, ...], DType]]:
    """the window's compute state (a piece of W tokens'): the moe's inputs (the routed experts' inputs, each gpu's partials of the shared
    experts' outputs in a piece's order, the router's logits), its routes, a gpu's rows of the experts' activations and outputs, a latent's
    gpu's rows of the shared experts' outputs (summed by the experts)"""
    c, W, E, K = self.model.config, self.W, self.m["n_experts"], self.model.config.num_experts_per_tok
    R, f32, i32 = max(map(self.rows, TILES)), dtypes.float32, dtypes.int32
    return {"h": ((W, self.lat), self.tiles_dtype), "shared": ((1, W, c.dim), self.bank), "logits": ((W, E), f32), "w": ((W, K), f32),
            "pos": ((W * K,), i32), "entries": ((R,), i32), "tiles": ((R // TILES[0],), i32),
            "act": ((R, self.units), dtypes.half), "down": ((R, self.lat), dtypes.half),
            **({"guess_logits": ((min(self.chunk, SHORT_CHUNK), E), f32)} if self.forecast_counts is not None else {}),
            **({"shared_rows": ((W // len(self.devs), c.dim), self.bank)} if c.expert_latent_dim else {})}

  def window_state(self, at:int) -> dict[str, Tensor]:  # (lent from the pool's slots from at on)
    return {k: Tensor(t.uop.unshard(0)) if k in ("act", "down", "shared", "shared_rows") else t
            for k, t in self.lend(self.window_shapes(), at)[0].items()}

  def piece_shapes(self) -> dict[str, tuple[tuple[int, ...], DType]]:
    """a piece's: the bank of its W tokens' residual streams, the draft's inputs. a gpu's rows: its part of every chunk (1, 1, W/gpus, ...)"""
    c, W = self.model.config, self.W // len(self.devs)
    S = self.bank_streams or 1 + sum(map(self.new_stream, self.model.blk))
    shapes:dict[str, tuple[tuple[int, ...], DType]] = {"bank": ((1, 1, W, S, c.dim), self.bank),
                                                       "taps": ((1, 1, W, max(len(self.taps), 1), c.dim), dtypes.float)}
    if self.index_roles: shapes["indices"] = ((self.W, unwrap(c.indexer).top_k), dtypes.int32)
    return shapes

  def piece(self, at:int) -> dict[str, Tensor]:  # (lent from the pool's slots from at on: (1, gpus, W/gpus, ...) split on the gpus)
    return {k: t if k == "indices" else Tensor(t.uop.unshard(1)) for k, t in self.lend(self.piece_shapes(), at)[0].items()}

  def own(self, t:Tensor, c0:UOp|int) -> Tensor:  # (the rows of the chunk at c0 of a piece's (1, gpus, W/gpus, ...): each gpu's own)
    n = self.chunk // len(self.devs)
    return t[:, :, c0 // len(self.devs):c0 // len(self.devs) + n]

  def whole(self, t:Tensor, dep:list[Tensor]|None=None) -> Tensor:
    """t (1, R, ...) split on its rows, on every gpu: each gpu's rows copied to the others once, a buffer (written after dep)"""
    n = t.shape[1] // len(self.devs)
    rows = Tensor.cat(*(t[:, i * n:(i + 1) * n] for i in range(len(self.devs))), dim=1)
    return rows.contiguous() if dep is None else _stored(rows, dep)

  def in_order(self, t:Tensor, r0:int=0, n:int|None=None) -> Tensor:
    """a piece's (1, gpus, W/gpus, ...) rows r0..r0+n on every gpu in its tokens' order (1, n, ...): their chunks' gathered"""
    G, C, rest, n = len(self.devs), self.chunk, t.shape[3:], self.W - r0 if n is None else n
    k0, k1 = r0 // C, -(-(r0 + n) // C)
    w = self.whole(t[:, :, k0 * C // G:k1 * C // G].flatten(1, 2)).reshape(1, G, k1 - k0, C // G, *rest)
    return w.permute(0, 2, 1, 3, *range(4, 4 + len(rest))).reshape(1, (k1 - k0) * C, *rest)[:, r0 - k0 * C:r0 - k0 * C + n]

  def lend(self, shapes:dict[str, tuple[tuple[int, ...], DType]], at:int) -> tuple[dict[str, Tensor], int]:
    """buffers of shapes on every gpu lent from the pool's slots from at on, and their bytes"""
    state, off = {}, at * self.part
    for k, (shape, dt) in shapes.items():
      n = prod(shape) * dt.itemsize
      (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [Buffer(d, n, base=p, offset=off).allocate()
                                                     for d, p in zip(self.devs, self.pool.uop.buffer.bufs)]
      state[k], off = Tensor(UOp.from_buffer(mb, dt, self.devs)).reshape(shape), off + round_up(n, 256)
    return state, off - at * self.part

  def scorer(self, b:FFNBlock) -> Tensor|Linear|None:  # (a moe block's router, the engine's)
    return self.scorers[self.m["layers"].index(i)] if (i:=self.model.blk.index(b)) in self.m["layers"] else None

  kinds:dict[tuple, tuple[FFNBlock, Callable, list[Tensor]]] = {}  # (a kind's, and a stage of a block's: its args)

  def kind(self, name:str, b:FFNBlock, fxn:Callable) -> tuple[Callable, list[Tensor], FFNBlock]:
    """b's graph of a stage (its kind's), b's buffers and its router's (its args) and the kind's first block (the one it's captured on)"""
    if (name, b) not in self.kinds:
      seen:dict[UOp, None] = {}
      for t in nn.state.get_state_dict([b, self.scorer(b)]).values():
        for u in t.uop.toposort():
          if u.op is Ops.BUFFER: seen.setdefault(u, None)
      key = (name, type(b), tuple((u.shape, u.dtype, u.nbytes()) for u in seen))
      if key not in self.kinds:
        self.kinds[key] = (b, self.saved_graph(f"kind{sum(len(k) == 3 for k in self.kinds)}", self.wjit(functools.partial(fxn, b))), [])
      self.kinds[(name, b)] = (*self.kinds[key][:2], [Tensor(u) for u in seen])
    first, jit, args = self.kinds[(name, b)]
    return jit, args, first

  def piece_graph(self, name:str, b:FFNBlock, fxn:Callable, T:int) -> tuple[Callable, list[Tensor], FFNBlock, UOp]:
    """b's graph of a stage over a piece of T rows (fxn(n, ..): its n chunks'): a kind of each count of chunks and its last chunk's cap, the
    last chunk's live rows"""
    n = -(-T // self.chunk)
    cap = next(c for c in self.chunk_vars if T - (n - 1) * self.chunk <= c)
    return (*self.kind(f"{name}{n}x{cap}", b, functools.partial(fxn, n)), self.chunk_vars[cap].bind(T - (n - 1) * self.chunk))

  def chained_vars(self, pos:int, n:int) -> list[UOp]:  # (an attends graph's chunks' positions from pos, the full chunks' live rows)
    return [*(v.bind(pos + k * self.chunk) for k, v in enumerate(self.v_poss[:n])), self.v_full.bind(self.chunk)]

  def window_mask(self, b:FFNBlock, h:Tensor, pos:UOp, c0:int, q_a:Tensor|None=None) -> Tensor|None:
    def select() -> Tensor|None:
      if isinstance(b, MLATransformerBlock): return b._indexer_mask(h, (pos,), None, q_a)
      return b._indexer_mask(h, (pos,), None)
    role = self.index_roles.get(b)
    if role is None: return select()
    # The selection is replicated on the peers, with positions absolute in the KV cache and rows local to this piece.
    rows = Tensor(self.win["indices"].uop)[c0:c0+self.chunk]
    if role == "write":
      mask = unwrap(select())
      assert mask.dtype == dtypes.int32
      rows = rows.assign(mask[0].pad_to(rows.shape, value=-1))
    return rows[:h.shape[1]].unsqueeze(0)

  def attends(self, n:int, b:FFNBlock, *args, forecast:bool=False) -> Tensor:
    """a piece's n chunks' attends in one graph (args: their positions, the full chunks' live rows, the last one's, the streams, stream s),
    each a chain of attend's stages, in turn: a round the next chain's first stage, then each running chain's next, the newest first. a
    stage's first kernels after the stage before's leaves: a chain's copies, issued at the end of a stage, wait under the others' stages"""
    poss, (full, live, streams, s) = args[:n], args[n:n + 4]
    held = [(o, k, getattr(o, k)) for o, k in self.held(b)]
    chains = [self.attend(b, poss[k], live if k == n - 1 else full, k * self.chunk, streams, s, forecast) for k in range(n)]
    running:list[Generator[list[Tensor], list[Tensor], list[Tensor]]] = []
    dep:list[Tensor] = []
    outs:list[Tensor] = []
    try:
      while chains or running:
        if chains:
          running.insert(0, chains.pop(0))
          next(running[0])
        for chain in list(running):
          try: dep = chain.send(dep)
          except StopIteration as done:
            running.remove(chain)
            outs += done.value
    finally:
      for o, k, t in held: setattr(o, k, t)
    return Tensor.realize(*outs)[0]

  @staticmethod
  def held(b:FFNBlock) -> list[tuple[object, str]]:  # (block b's state a chunk's attention leaves the next one's: recurrent states, caches)
    owners = ((b, "conv_states"), (b, "conv_state"), (b, "recurrent_state"), (b, "cache_k"), (getattr(b, "indexer", None), "cache_kg"))
    return [(o, k) for o, k in owners if hasattr(o, k)]

  def after(self, k:str, dep:list[Tensor]) -> Tensor:  # (the window's buffer k after dep: its next state, its reads and writes after it)
    return self.win[k].replace(_after(self.win[k], dep))

  @staticmethod
  def chained(t:Tensor, a:Tensor) -> Tensor:  # (a block's state t as a's graph leaves it, read after a: its buffer's last state there, else t)
    states = [u for u in a.uop.toposort() if u.op is Ops.AFTER and u.storage_base is t.uop.storage_base]
    return Tensor((states[-1] if states else t.uop).after(a.uop))

  def attend(self, b:FFNBlock, pos:UOp, live:UOp, c0:int, n:UOp, s:UOp, forecast:bool) -> Generator[list[Tensor], list[Tensor], list[Tensor]]:
    """a chunk's live rows' attention at c0 of the window (positions pos..) on its n streams, added to stream s, then its feed forward's mix: a
    moe's parts before the routed experts into the window's buffers, a dense block's feed forward added to stream s. the rows symbolic: a
    kernel's work the live rows'. in stages, each ending where its copies are issued, its first kernels after dep (the stage before's leaves:
    another chain's), yielding its leaves (a buffer its kernels end in) or dep. its outputs returned"""
    rows = (c0 // len(self.devs), self.chunk // len(self.devs))
    dep = yield []
    m = self.mix(getattr(b, "attn_res_score", None), self.win["bank"].flatten(1, 2), b.attn_norm, n, rows).cast(self.bank)
    dep = yield dep
    h = self.whole(m, dep)[:, :live]
    if isinstance(b, MLATransformerBlock):
      latents = b._latents(h)
      mask = self.window_mask(b, h, pos, c0, latents[0])
      a = self.partials(lambda: b._attention(h, (pos,), mask, live, latents))
    else:
      mask = self.window_mask(b, h, pos, c0)
      a = self.partials(lambda: b._attention(h, (pos,), mask, live))
    steps = self.scattered(a)
    for o, k in self.held(b):
      v = getattr(o, k)
      setattr(o, k, [self.chained(t, steps[0]) for t in v] if isinstance(v, list) else self.chained(v, steps[0]))
    dep = yield steps[:1]
    summed = self.summed(steps, dep)
    self.add(s, c0, summed.cast(a.dtype).reshape(1, len(self.devs), -1, summed.shape[-1]))
    xm = self.mix(getattr(b, "ffn_res_score", None), self.win["bank"].flatten(1, 2), b.ffn_norm, s + 1, rows).cast(self.bank)
    dep = yield [summed]
    x = self.whole(xm, dep)[:, :live]
    if not hasattr(b, "ffn_gate_inp"):
      y = self.partials(lambda: b._feed_forward(x))
      steps = self.scattered(y)
      dep = yield steps[:1]
      summed = self.summed(steps, dep)
      return [self.add(s, c0, summed.cast(y.dtype).reshape(1, len(self.devs), -1, summed.shape[-1]))]
    scores = (w(x[0]) if isinstance(w:=self.scorer(b), Linear) else x[0] @ w.T).float().reshape(live, len(self.devs), 2, -1)
    logits = scores[:, :, 0].reshape(live, -1).contiguous()
    predicted = scores[:, :, 1].reshape(live, -1).contiguous().half() if forecast else None
    routed = b._expert_input(x)[0]
    shared = self.own(self.shared(), c0)
    shexp = self.as_partials(self.partials(lambda: b._shared(x)))[:, 0]
    shared = shared.assign(shexp.pad_to((len(self.devs), self.chunk, shexp.shape[-1])).reshape(shared.shape).cast(shared.dtype))
    dep = yield dep
    # (a gpu's experts' logits, everyone's on every gpu)
    outs = [(self.after("h", dep), routed), (self.after("logits", dep), allgather_last(logits))]
    result = [t[c0:c0 + self.chunk].assign(v.pad_to((self.chunk, v.shape[1])).cast(t.dtype)) for t, v in outs]
    if predicted is not None:
      guess = self.after("guess_logits", dep)
      result.append(guess.assign(allgather_last(predicted).float().pad_to(guess.shape)))
    return [*result, shared]

  @staticmethod
  def new_stream(b:FFNBlock) -> bool: return hasattr(b, "attn_res_score") and b.attn_res_score.new_block

  def mix(self, res:AttentionResidual|None, xs:Tensor, norm:Callable[[Tensor], Tensor], n:UOp, rows:tuple[UOp|int, int]) -> Tensor:
    """the bank's streams xs (1, W, S, dim) of each gpu's rows at rows mixed by an attention residual (none: the one stream) and normed"""
    if res is not None: return res.mix(xs, norm, n, rows)
    return norm(xs.reshape(1, len(self.devs), -1, *xs.shape[2:])[:, :, rows[0]:rows[0] + rows[1]].flatten(1, 2)[:, :, 0].float())

  def partials(self, f:Callable[[], Tensor]) -> Tensor:  # (f's output, its row parallel linears' partials (gpus, ...) as they are, if any)
    PARTIALS[0] = True
    try: return f()
    finally: PARTIALS[0] = False

  def add(self, s:UOp, c0:UOp|int, *ys:Tensor) -> Tensor:
    """ys (each gpu its rows of the chunk) added in turn to stream s of the bank's chunk at c0"""
    # (a slice of a symbolic offset, a symbolic length and a symbolic index isn't written: the chunk's whole rows. each gpu its own)
    bank = self.own(Tensor(self.win["bank"].uop), c0)[:, :, :, s]
    return bank.assign(functools.reduce(Tensor.add, ys, bank).cast(bank.dtype))

  def shared(self) -> Tensor:  # (each gpu's partials of the window's shared experts' outputs in a piece's order: (gpus, gpus, W/gpus, dim))
    return Tensor(self.win["shared"].uop).reshape(len(self.devs), len(self.devs), -1, self.model.config.dim)

  def split(self, y:Tensor) -> Tensor:
    """y (1, live, dim) a chunk's, or its partials (gpus, 1, live, dim) summed (reduce scattered): (1, gpus, chunk/gpus, dim), each gpu its rows
    (the rows past live 0)"""
    summed = self.summed(self.scattered(y))
    return summed.cast(y.dtype).reshape(1, len(self.devs), -1, summed.shape[-1])

  def scattered(self, y:Tensor) -> list[Tensor]:
    """y (1, live, dim) a chunk's (its rows kept on their gpus, a buffer), or its partials (gpus, 1, live, dim) scattered to their rows' gpus:
    scatter's steps (the first each gpu's own, a buffer)"""
    G, C = len(self.devs), self.chunk
    if y.ndim == 4: return scatter(y[:, 0].pad_to((G, C, y.shape[-1])))
    return [_stored(self.kept(y.pad_to((1, C, y.shape[2])).reshape(1, G, C // G, -1)), [])]

  def summed(self, steps:list[Tensor], dep:list[Tensor]|None=None) -> Tensor:  # (scattered's steps summed in the ring's order: a buffer, after dep)
    if len(steps) == 1: return steps[0]
    return ring_sum(steps).contiguous() if dep is None else _stored(ring_sum(steps), dep)

  def kept(self, t:Tensor) -> Tensor:  # (t (1, gpus, ...) on every gpu split on axis 1: each gpu keeps its own, nothing copied)
    return Tensor(t.uop._shard(1, UOp.range(len(self.devs), 0, AxisType.DEVICE)).unshard(1))

  def as_partials(self, y:Tensor) -> Tensor:  # (y's partials (gpus, 1, live, dim): a whole y (1, live, dim) the first gpu's, the others' zeros)
    if y.ndim == 4: return y
    first = Tensor.arange(len(self.devs)).to(y.device).reshape(1, -1, 1, 1, 1) == 0
    return self.kept(first.where(y.reshape(1, 1, *y.shape), 0))[0]

  def rows(self, tile:int) -> int:  # (a window's rows in tiles of tile rows: its selections, each expert's padded)
    return round_up(self.W * self.model.config.num_experts_per_tok + (tile - 1) * self.m["n_experts"], tile)
  def state_rows(self, tile:int) -> tuple[int, int]:  # (the state's rows, its tiles of tile rows: the largest tile size's)
    return (R:=max(map(self.rows, TILES))), R // tile

  def forecast_route(self, live:UOp, bias:Tensor, next_bias:Tensor, tag:Tensor, part2:Tensor, *_) -> Tensor:
    return self.window_route(TILES[0], live, bias, tag, part2, next_bias=next_bias)

  def window_route(self, tile:int, live:UOp, bias:Tensor, tag:Tensor, part2:Tensor, *_, next_bias:Tensor|None=None) -> Tensor:
    """the gpus route the window's live tokens exactly (its tiles of tile rows of an expert), the expert counts to the host (part 2, then the
    tag)"""
    K, E = self.model.config.num_experts_per_tok, self.m["n_experts"]
    w, pos, entries, tiles, counts = moe_route(Tensor(self.win["logits"].uop), bias, K, tile, live=live)
    R, NT = self.state_rows(tile)  # (the routes padded to the state's rows: the experts graph's tiles past them skipped)
    vals = {"w": w, "pos": pos, "entries": entries.pad((0, R - entries.shape[0]), value=-1), "tiles": tiles.pad((0, NT - tiles.shape[0]), value=-1)}
    kept = Tensor.realize(*(Tensor(self.win[k].uop)[:v.shape[0]].assign(v) for k, v in vals.items()))
    counted = [Tensor(part2.uop)[1:1 + E].assign(counts.bitcast(dtypes.uint32))]
    if next_bias is not None:
      assert self.forecast_counts is not None
      prediction = moe_counts(Tensor(self.win["guess_logits"].uop), next_bias, K, live)
      counted.append(Tensor(self.forecast_counts[0].uop).assign(prediction.bitcast(dtypes.uint32)))
    most = Tensor.cat(*(Tensor(c.uop.unshard(0)).reshape(len(self.devs), -1) for c in counted), dim=1).max(1, keepdim=True)
    return Tensor(tag.uop.after(kept.uop)).assign((most <= live).where(1, 2).cast(dtypes.uint32)).realize()

  def window_experts(self, tile:int, types:tuple[tuple[int, int]|None, ...], n:int, pool:Tensor, part1:Tensor, tag:Tensor, *_) -> Tensor:
    """the window's routed experts once the host gave their slots (part 1: an expert's slot * copies + its copy): each gpu its units of the
    experts in tiles of an expert's rows, a copy's tiles at a time (of its gate/up and down quant types, none: a layer it doesn't hold) (their
    rows of down), a latent's shared experts' outputs of a piece's n chunks summed into each gpu's rows (their copies under the experts), then
    the tag 3 (the host pumps the reads till it: a graph's launch waits without)"""
    K, E, C, u = self.model.config.num_experts_per_tok, self.m["n_experts"], len(self.ms), self.units
    entries, h1, h2 = (Tensor(self.win[k].uop) for k in ("entries", "act", "down"))
    tiles = Tensor(self.win["tiles"].uop)[:self.state_rows(tile)[1]]
    if self.waits: part1 = Tensor(part1.uop.after(wait(part1.uop)))
    slots = Tensor(part1.uop)[1:1 + E].bitcast(dtypes.int32)
    at, xrow = (tiles >= 0).where(slots[tiles.maximum(0)], -1), (entries >= 0).where(entries // K, -1)
    h = Tensor(self.win["h"].uop)
    for c, ty in enumerate(types):
      if ty is None: continue
      tslot = ((at >= 0) & (at % C == c)).where(at // C, -1)
      h1 = self.tiles(tile, pool, tslot, h, 2 * u, self.lat, ty[0], 0, h1, xrow=xrow, act=self.act, live=xrow)
      h2 = self.tiles(tile, pool, tslot, h1, self.lat, u, ty[1], self.nbytes(ty[0], 2 * u * self.lat), h2, live=xrow)
    if n:
      G, rows = len(self.devs), n * self.chunk // len(self.devs)
      shared = Tensor(self.win["shared_rows"].uop.after(h2.uop)).reshape(1, G, -1, self.model.config.dim)[:, :, :rows]
      h2 = shared.assign(ring_sum(scatter(Tensor(self.win["shared"].uop), rows)).reshape(shared.shape))
    return Tensor(tag.uop.after(h2.uop)).assign(tag.const_like(3)).realize()

  def tiles(self, tile:int, pool:Tensor, tslot:Tensor, x:Tensor, out_f:int, in_f:int, qtype:int, off:int, out:Tensor, xrow:Tensor|None=None,
            act:Callable|None=None, live:Tensor|None=None) -> Tensor:
    """(gpus, tile * tiles, out_f) into out (the rows of tiles -1 kept): each gpu's units of the experts (of quant type qtype) at off bytes in
    the pool's slots of tiles of tile rows, on their rows of x (xrow: its rows, -1 its first), act(gate, up) of the outputs' halves: the AMD
    tensor cores' expert tiles for TILE_QUANTS (live: the rows' padding -1, left out), else decoded"""
    out = out.reshape(-1, out.shape[-1])
    if amd_custom_kernels_supported(x.device) and qtype in TILE_QUANTS:
      return expert_tiles(self.pools[qtype], tslot, x, out_f, in_f, qtype, self.part, off, xrow=xrow, act=act, out=out, tile=tile, live=live,
                          dtype=x.dtype)
    NT, size, isz = tslot.shape[0], self.nbytes(qtype, out_f * in_f), self.word.itemsize
    w = pool.reshape(len(self.devs), -1, self.part // isz)[:, tslot.maximum(0)][..., off // isz:(off + size) // isz]
    w = w.bitcast(dtypes.uint8)
    w = ggml_data_to_tensor(w.flatten(), prod(w.shape) // size * out_f * in_f, qtype).reshape(-1, NT, out_f, in_f)
    y = (x if xrow is None else x[xrow.maximum(0)]).reshape(-1, NT, tile, in_f) @ w.transpose(-1, -2)
    y = (y if act is None else act(*y.split(out_f // 2, -1))).cast(out.dtype)
    y = (tslot >= 0).reshape(1, NT, 1, 1).where(y, Tensor(out.uop).reshape(y.shape))
    return Tensor(out.uop).assign(y.reshape(out.shape)).reshape(len(self.devs), NT * tile, -1)

  def posts(self, n:int, b:FFNBlock, live:UOp, s:UOp, *_) -> Tensor:
    """a piece's n chunks' posts in one graph, each its own chain (the last chunk's live rows, the others' whole): a latent's chunks' routed
    experts' outputs gathered one after another, each scattered to their rows' gpus (a chunk's copies under the next one's gather)"""
    chunks = [(k * self.chunk, live if k == n - 1 else self.chunk) for k in range(n)]
    if not self.model.config.expert_latent_dim: return Tensor.realize(*(self.post(c0, live, s) for c0, live in chunks))[0]
    steps:list[list[Tensor]] = []
    for c0, live in chunks:
      down = Tensor(self.win["down"].uop.after(*(st[0].uop for st in steps[-1:])))
      steps.append(scatter(self.routed_out(down, c0, live).pad_to((len(self.devs), self.chunk, self.lat))))
    return Tensor.realize(*(self.latent_post(b, c0, s, st) for (c0, _), st in zip(chunks, steps)))[0]

  def routed_out(self, down:Tensor, c0:int, live:UOp|int) -> Tensor:
    """a chunk's live rows' routed experts' outputs: their rows of down weighted, each gpu's"""
    G, K = len(self.devs), self.model.config.num_experts_per_tok
    pos, w = Tensor(self.win["pos"].uop).reshape(-1, K)[c0:c0 + live], Tensor(self.win["w"].uop)[c0:c0 + live]
    out = (down.reshape(G, -1, self.lat)[:, pos.maximum(0)].float() * w.unsqueeze(-1)).sum(2)
    return out * self.model.config.routed_scaling_factor

  def post(self, c0:int, live:UOp|int, s:UOp) -> Tensor:
    """a chunk's live rows' routed experts' outputs and shared experts' outputs summed over the gpus in one reduce scatter, added to stream s"""
    shared = self.own(self.shared(), c0).reshape(len(self.devs), 1, self.chunk, -1)[:, :, :live].float()
    return self.add(s, c0, self.split(self.routed_out(Tensor(self.win["down"].uop), c0, live).unsqueeze(1) + shared))

  def latent_post(self, b:FFNBlock, c0:int, s:UOp, steps:list[Tensor]) -> Tensor:
    """a chunk's routed experts' outputs summed on their rows' gpus (their scatter's steps), normed there and gathered for the up projection's
    columns, sent back to their rows' gpus, added to stream s with the shared experts' outputs (the experts graph's sum)"""
    G = len(self.devs)
    rows = ring_sum(steps).contiguous().float()
    up = b.ffn_routed_up(self.whole(b.ffn_routed_norm(rows).half().reshape(1, self.chunk, self.lat)))[0]
    shared = self.own(Tensor(self.win["shared_rows"].uop).reshape(1, G, -1, self.model.config.dim), c0).float()
    return self.add(s, c0, shared, all_to_all(up.half()).reshape(1, G, self.chunk // G, -1))

  def tap(self, s:UOp, j:UOp, *_) -> Tensor:  # (the draft's input: stream s after a block its tap j)
    return Tensor(self.win["taps"].uop)[:, :, :, j].assign(Tensor(self.win["bank"].uop)[:, :, :, s].float()).realize()

  def tap_mix(self, s:UOp, j:UOp, w:Tensor, *_) -> Tensor:  # (tap j: the streams up to s mixed by an attention residual's weight w)
    bank = Tensor(self.win["bank"].uop).float()
    scores = (bank.rmsnorm(eps=self.model.config.norm_eps) * w).sum(-1, keepdim=True)
    scores = (Tensor.arange(bank.shape[3]).to(bank.device).reshape(-1, 1) <= Tensor(s)).where(scores, -math.inf)
    return Tensor(self.win["taps"].uop)[:, :, :, j].assign((scores.softmax(3) * bank).sum(3)).realize()

  def fill(self, pos:UOp, p0:UOp, c0:UOp, *_) -> Tensor:  # (the draft's cache of a chunk of the piece at p0 of the read window)
    C = self.chunk
    given = self.given[p0 + c0:p0 + c0 + C].reshape(1, C).to(self.devs)
    return self.draft.fill(given, self.whole(self.own(Tensor(self.win["taps"].uop), c0).flatten(1, 2).cast(self.bank)).float(), pos).realize()

  def start(self, p0:UOp, *_) -> Tensor:  # (a piece's bank of streams: its tokens' embeddings, the rest zero. each gpu its rows of the chunks)
    G, C, bank = len(self.devs), self.chunk, Tensor(self.win["bank"].uop)
    emb = self.model.token_embd(self.tok[p0:p0 + self.W].reshape(1, self.W).to(self.devs))
    emb = emb.reshape(1, self.W // C, G, C // G, -1).permute(0, 2, 1, 3, 4).reshape(1, G, self.W // G, 1, -1)
    return bank.assign(self.kept(emb).pad(((0, 0), (0, 0), (0, 0), (0, bank.shape[3] - 1), (0, 0))).cast(bank.dtype)).realize()

  def stage(self, name:str):  # (a start's stage: its seconds since the stage before)
    print(f"start: {name} {time.perf_counter() - self.t_stage:.1f} s", flush=True)
    self.t_stage = time.perf_counter()

  def slots_of(self, n:int) -> int: return -(-n // self.part)  # (the pool's slots n bytes take)

  def drop(self, units:slice|np.ndarray, wait:bool=False):  # (the experts in the pool's slots units dropped: all of their slots)
    st, n = self.state, len(self.types) * self.m["n_experts"]
    codes = np.unique(st["expert_in"][units])
    codes = codes[codes >= 0]
    if wait:
      flying = np.ascontiguousarray(codes[st["flying"][codes] > 0], np.int64)
      if len(flying) and (err:=host_c().read_wait(ctypes.byref(self.io), len(flying), flying.ctypes.data)): raise RuntimeError(f"reads failed {err}")
    for c in codes:
      w = self.layer_spans[c // n][c % n // self.m["n_experts"]]
      st["expert_in"][st["slot_of"][c]:st["slot_of"][c] + w], st["slot_of"][c] = -1, -1
  @staticmethod
  def nbytes(qtype:int, n:int) -> int: return n // _GGML_QUANT[qtype][0] * _GGML_QUANT[qtype][1]

  def grow(self, n:int):
    """the caches' rows of the first n positions (in steps of a chunk) kept from the experts: their slots' experts dropped (once read,
    if in flight), the slots pinned (step_io's mask, read's keep), the rows zeroed (they were experts': a cache's keys past a context masked)"""
    if n <= self.grown: return
    new, st = min(round_up(n, self.grow_step), self.max_context), self.state
    for (a, first, row, blocks, size), (o, k) in zip(self.stripes, self.caches):
      starts = first + np.arange(blocks) * size
      sl = np.unique(np.concatenate([np.arange((s + self.grown * row) // self.part, (s + new * row - 1) // self.part + 1) for s in starts]))
      self.drop(sl, wait=True)
      st["mask"][sl] = 1
    for g in range(self.grown, new, self.grow_step): self.zero_jit(self.v_grown.bind(min(g, self.max_context - self.grow_step)))
    self.grown = new

  def zero(self, g:UOp):  # (every cache's step of rows at g zeroed)
    rows = [getattr(o, k).shrink(tuple((g, g + self.grow_step) if j == a else None for j in range(getattr(o, k).ndim)))
            for (a, *_), (o, k) in zip(self.stripes, self.caches)]
    Tensor.realize(*[r.assign(r.zeros_like()) for r in rows])

  def pieces(self) -> int:
    """a read window's pieces: lent below the window's state, past the caches' stripes and the scratch (their rows pinned as they grow, the
    rest the experts'), the pool past them and two layers' experts (the one routed, the one read ahead)"""
    free = min(self.tail - self.arena[1], self.free_slots() - 2 * self.m["n_experts"] * self.max_primary_span)
    return 1 + max(0, free // self.slots_of(self.piece_bytes))

  def free_slots(self) -> int:
    (a0, a1), mask = self.arena, self.state["mask"]
    return self.tail - (a1 - a0) - int(mask[:a0].sum() + mask[a1:].sum())

  def window(self, toks:list[int], pos:int, given:list[int]):
    """the tokens at positions pos.. (a read window) through every block, a layer at a time over its pieces of up to W tokens (each its bank):
    a layer's experts read once a read window. the caches and states as after steps of them, the draft's cache. on an error the gpus' waits
    for the host released (the graphs queued run out)"""
    T, C, W, t0 = len(toks), self.chunk, self.W, time.perf_counter()
    arena = slice(*self.arena)
    self.drop(arena, wait=True)
    self.state["mask"][arena] = 1
    self.grow(pos + round_up(T, C))
    assert T <= W * self.pieces() and pos + round_up(T, C) <= self.max_context
    self.tok_np[:T], self.given_np[:T] = toks, given
    ps, st = self.slots_of(self.piece_bytes), self.state
    # (the pieces' states: the first the always lent one, the rest lent below the compute state's slots, their experts dropped)
    banks = [{k: self.win[k] for k in self.piece_shapes()}] + [self.piece(self.tail - i * ps) for i in range(1, -(-T // W))]
    self.lent_from = self.tail - (len(banks) - 1) * ps
    assert self.lent_from >= self.arena[1] and not st["mask"][self.lent_from:self.tail].any(), "a piece over the caches or the scratch"
    self.drop(slice(self.lent_from, None), wait=True)
    pieces = [(p0, min(W, T - p0), bank) for p0, bank in zip(range(0, T, W), banks)]
    try:
      for p0, _, bank in pieces:
        self.win.update(bank)
        self.graph(self.start_jit, self.v_p0.bind(p0))
      host_c().flags(ctypes.byref(self.io), 0)  # (the tags cleared)
      self.layers(pieces, pos)
      for p0, n, bank in pieces if self.fill_jit else []:
        self.win.update(bank)
        for c0 in range(0, n, C): self.graph(self.fill_jit, self.v_pos.bind(pos + p0 + c0), self.v_p0.bind(p0), self.v_c0.bind(c0))
      for d in self.devs: Device[d].synchronize()
    except BaseException:
      host_c().flags(ctypes.byref(self.io), 1)
      raise
    finally:
      self.win.update(banks[0])
      self.lent_from, self.state["mask"][arena] = self.tail, 0
    self.window_s.append((T, time.perf_counter() - t0))

  def steps_faster(self, n:int) -> bool:
    """n tokens in steps (of the most tokens there's a graph of) faster than a window of them: a step's measured seconds, a window's a fixed
    part (the store's experts not cached read) and its tokens' (a token's the measured windows' median past the fixed part: a cold one doesn't
    move it). before any are measured, tinys3's"""
    T = max(self.jits)
    step = float(np.median(self.step_s[T])) if self.step_s[T] else STEP_S
    # The minimum stored expert size estimates bytes read independently of the pool's allocation quantum.
    fixed = (len(self.bias) * self.m["n_experts"] * self.store_part - self.slots * self.part) * len(self.devs) / READ_BW
    token = float(np.median([max(t - fixed, 0) / m for m, t in self.window_s])) if self.window_s else WINDOW_TOKEN_S
    return -(-n // T) * step < fixed + n * token

  def layers(self, pieces:list[tuple[int, int, dict[str, Tensor]]], pos:int):
    """a read window's blocks, a layer at a time over its pieces. a moe layer's experts a piece's tokens route to read in (the host waits for
    the gpus' counts: the layers before are done with their slots), their slots given to the gpus, kept till the layer's done. when a
    piece's tokens route to half of a layer's experts or more, every next layer's are read ahead (queued behind the layer's own, a layer
    read ahead whole the next before its counts) while the gpus work"""
    n, E, K = 1, self.m["n_experts"], self.model.config.num_experts_per_tok
    ahead:dict[int, np.ndarray] = {}
    moe = sum(hasattr(b, "ffn_gate_inp") for b in self.model.blk)
    for i, b in enumerate(self.model.blk):
      s = n - 1 + int(self.new_stream(b))
      for p0, T, bank in pieces:
        self.win.update(bank)
        forecast = (self.forecast_jit if len(pieces) == 1 and T <= min(self.chunk, SHORT_CHUNK) and
                    hasattr(b, "ffn_gate_inp") and i < self.m["layers"][moe - 1] else None)
        name, fxn = ("forecast", functools.partial(self.attends, forecast=True)) if forecast is not None else ("attend", self.attends)
        jit, args, _, live = self.piece_graph(f"{name}{self.index_roles.get(b, '')}", b, fxn, T)
        self.graph(jit, *self.chained_vars(pos + p0, -(-T // self.chunk)), live, self.v_n.bind(n), self.v_s.bind(s), *args)
        self.pump()
        if hasattr(b, "ffn_gate_inp"):
          self.layer = li = self.m["layers"].index(i)
          route, experts = self.routes[next(t for t in TILES if T * K <= t * E or t == TILES[-1])]
          for t, p in zip(self.bufs[li][0][1], self.bufs[li][2][1]): t[0] = p[0] = 0  # (the tag: this piece's counts, part 1 not given)
          args = (Tensor(self.tags[li]), self.parts[1][li])
          if forecast is None: self.graph(route, self.v_wlive.bind(T), b.exp_probs_b["bias"], *args)
          else: self.graph(forecast, self.v_wlive.bind(T), b.exp_probs_b["bias"], self.model.blk[self.m["layers"][li + 1]].exp_probs_b["bias"], *args)
          early = self.waits and not self.state["flying"][self.window_codes(li, np.arange(E))].any()
          if early: self.experts_post(experts, b, li, T, s)
          if li in ahead and len(ahead[li]) == E and li + 1 < moe and li + 1 not in ahead:
            if self.read(self.window_codes(li + 1, ahead[li]), self.window_codes(li, ahead[li]), wait=False): ahead[li + 1] = ahead[li]
          counts = self.counted(li, T)
          used = np.flatnonzero(counts)
          codes = self.window_codes(li, np.union1d(used, ahead[li]) if li in ahead else used)
          if li + 1 < moe and li + 1 not in ahead and (li in ahead or len(used) >= E // 2):
            guesses = np.arange(E)
            if T * K < E * TILES[0]:
              freq = self.state["freq"][(li + 1) * E:(li + 2) * E]
              if forecast is not None:
                assert self.forecast_counts is not None
                prediction = self.forecast_counts[1][0]
                if prediction.sum() == T * K and prediction.max() <= T:
                  guesses = np.lexsort((-freq, -prediction.astype(np.int32)))[:len(used)]
                else: guesses = np.argsort(-freq, kind="stable")[:len(used)]
              else: guesses = np.argpartition(freq, E - len(used))[E - len(used):] if len(used) else guesses[:0]
              self.read(codes, codes, wait=False)
            if self.read(self.window_codes(li + 1, guesses), codes, wait=False): ahead[li + 1] = guesses
          t0 = time.perf_counter()
          self.read(codes, codes, wait=True)
          self.io.t_reads += time.perf_counter() - t0
          table = np.maximum(self.state["slot_of"][self.window_codes(li, np.arange(E))], 0) * len(self.ms)
          for p in self.bufs[li][2][1]:
            p[1:1 + E] = table.astype(np.int32).view(np.uint32)
            p[0] = 1
          if not early: self.experts_post(experts, b, li, T, s)
          self.ran(li)
        for j, (blk, mixed) in enumerate(self.taps):
          if blk != i: continue
          if mixed: self.graph(self.mix_jit, self.v_s.bind(s), self.v_tap.bind(j), self.mixer(i).weight)
          else: self.graph(self.tap_jit, self.v_s.bind(s), self.v_tap.bind(j))
      n = s + 1

  def experts_post(self, experts:dict, b:FFNBlock, li:int, T:int, s:int):  # (moe block b's (layer li's) experts and posts graphs, a piece of T rows)
    self.graph(experts[(self.layer_copies[li], -(-T // self.chunk) if self.model.config.expert_latent_dim else 0)], self.pool, self.parts[0][li],
               Tensor(self.tags[li]))
    jit, args, _, live = self.piece_graph("post", b, self.posts, T)
    self.graph(jit, live, self.v_s.bind(s), *args)

  def window_codes(self, li:int, experts:np.ndarray) -> np.ndarray:
    """the codes of layer li's experts a window reads: the first copy's"""
    return li * self.m["n_experts"] + experts

  def graph(self, jit:Callable, *args) -> Tensor:  # (a window graph on the window's state)
    return jit(*args, *self.win.values())

  def pump(self):  # (the reads ahead queued as the drives have room)
    if host_c().pump(ctypes.byref(self.io)): raise RuntimeError("window reads failed")

  def ran(self, li:int):  # (layer li's experts done on the gpus (their tag 3), the reads pumped meanwhile)
    t0 = time.perf_counter()
    while not all(t[0] == 3 for t in self.bufs[li][0][1]):
      if time.perf_counter() - t0 > 60: raise RuntimeError(f"window layer {li}: the gpus' experts didn't finish")
      self.pump()

  def counted(self, li:int, live:int) -> np.ndarray:
    """layer li's expert counts of a window's tokens once the gpus give them, added to its decayed use counts"""
    E, io, t0 = self.m["n_experts"], self.io, time.perf_counter()
    tags = self.bufs[li][0][1]
    while not all(t[0] == 1 for t in tags):
      if any(t[0] == 2 for t in tags) or time.perf_counter() - t0 > 60: raise RuntimeError(f"window layer {li}: no counts from the gpus")
      self.pump()
    replicas = [p[1:1 + E] for p in self.bufs[li][3][1]]
    counts = replicas[0].copy()
    expected = live * self.model.config.num_experts_per_tok
    if counts.sum() != expected or counts.max() > live:
      raise RuntimeError(f"window layer {li}: invalid expert counts for {live} tokens (sum {counts.sum()}, expected {expected}, max {counts.max()})")
    if any(not np.array_equal(counts, r) for r in replicas[1:]): raise RuntimeError(f"window layer {li}: expert counts differ across GPUs")
    f = self.state["freq"][li * E:(li + 1) * E]
    f[:] = f * io.decay + counts
    io.t_gpu += time.perf_counter() - t0
    return counts

  def read(self, codes:np.ndarray, busy:np.ndarray, wait:bool) -> bool:
    """the codes not cached read into the least used slots but those of busy's codes (in use or read): waited for (and those read ahead), or
    not (read ahead)"""
    st, io = self.state, self.io
    reads = [np.ascontiguousarray(a, np.int64) for a in (codes[st["flying"][codes] > 0], codes[st["slot_of"][codes] < 0])]
    if wait and len(reads[0]) and (err:=host_c().read_wait(ctypes.byref(io), len(reads[0]), reads[0].ctypes.data)):
      raise RuntimeError(f"window reads failed {err}")
    if not len(miss:=reads[1]): return True
    old, E, LE = st["expert_in"], self.m["n_experts"], len(self.types) * self.m["n_experts"]
    keep = (np.arange(self.slots) >= self.lent_from) | (st["mask"] > 0) | np.isin(old, busy) | ((old >= 0) & (st["flying"][np.maximum(old, 0)] > 0))
    rank = np.where(keep, 2, (old >= 0) & (old % LE >= self.layer * E) & (old % LE < (self.layer + 2) * E))
    plan = miss if wait else np.concatenate([miss, np.setdiff1d(busy[st["slot_of"][busy] < 0], miss)])
    victims, groups, worst = np.zeros(len(plan), np.int64), self.span_groups.reshape(-1)[plan // E], 0
    for c in np.unique(groups):
      at, w = np.nonzero(groups == c)[0], self.spans[c]
      units = rank[:self.slots // w * w].reshape(-1, w).max(1)
      key = np.where(old >= 0, st["freq"][np.maximum(old, 0)], -1.0)[:self.slots // w * w].reshape(-1, w).sum(1)
      victims[at] = np.lexsort((key, units))[:len(at)] * w
      worst = max(worst, int(units[victims[at] // w].max()))
      rank[(victims[at][:, None] + np.arange(w)).flatten()] = 2
    if not wait and worst: return False
    if worst == 2: raise RuntimeError(f"window layer {self.layer}: {len(miss)} reads, the pool's free slots fewer")
    victims, groups = np.ascontiguousarray(victims[:len(miss)]), groups[:len(miss)]
    for c in np.unique(groups):
      at, w = np.nonzero(groups == c)[0], self.spans[c]
      self.drop(units:=(victims[at][:, None] + np.arange(w)).flatten())
      st["slot_of"][miss[at]], st["expert_in"][units] = victims[at], np.repeat(miss[at], w)
    io.misses += len(miss)
    if not wait: host_c().read_ahead(ctypes.byref(io), len(miss), miss.ctypes.data)
    elif (err:=host_c().read_codes(ctypes.byref(io), len(miss), miss.ctypes.data, victims.ctypes.data)):
      raise RuntimeError(f"window reads failed {err}")
    return True

  # *** the host's side

  def warm(self):
    """every step's graph (and the rollback's, the saves') captured on dry steps (or the start's file's, checked by them): the host's flags
    preset (its tables still zero: no experts). the graphs to the start's file, unless they're its. the start's objects then kept out of the
    collector's passes (a full one would walk every graph)"""
    host_c().flags(ctypes.byref(self.io), 1)
    self.t_stage = time.perf_counter()
    self.dry_steps()
    if self.saved and self.saved["fps"] != self.fingerprints():
      print("start: the graphs file's graphs don't run as they were saved: dropped, capturing", flush=True)
      self.recapture()
    if not self.saved and self.source:
      persist.save(self, self.names, self.graphs(), self.fingerprints())
      self.stage("graphs kept for the next start")
    if self.short_window and self.short_engine is None:
      self.capture_short_window()
      self.stage("short continuation window")
    gc.freeze()

  def dry_steps(self):
    for T, jit in self.jits.items():
      jit(self.v_pos.bind(0))[0].tolist()
      self.stage(f"step graph of {T}")
    for n in range(1, self.tokens): self.model.rollback(n)  # (their jits captured too, after a verify step)
    for i, save in itertools.product(range(len(self.model._sets)), (True, False)): self.model._copy(i, save)  # (and save's)
    if self.chain_jit: self.chain_jit(self.v_pos.bind(0), self.v_row.bind(0)).tolist()
    for jit in self.keep_jits: jit(self.v_at.bind(0))  # (a kept session's copies)
    self.zero_jit(self.v_grown.bind(0))  # (and the caches' growth: its rows zeroed, there's no context yet)
    self.stage("rollbacks, saves, chain, session copies")

  # *** a start's graphs kept (persist.py): every graph named

  def graphs(self) -> dict[str, TinyJit]:
    m = self.model
    return {**{f"step{T}": j for T, j in self.jits.items()}, "keep": self.keep_jits[0], "unkeep": self.keep_jits[1], "zero": self.zero_jit,
            **({"chain": self.chain_jit} if self.chain_jit else {}), **{f"rollback{n}": j for n, j in m._rollbacks.items()},
            **{f"copy{i}{save}": j for (i, save), j in m._copies.items()},
            **({"start": self.start_jit, "tap": self.tap_jit, "mix": self.mix_jit, **({"fill": self.fill_jit} if self.fill_jit else {}),
                **({"forecast": self.forecast_jit} if self.forecast_jit else {}),
                **{f"route{t}": r for t, (r, _) in self.routes.items()},
                **{f"experts{t}x{n}_" + "_".join(f"{ty[0]}_{ty[1]}" if ty else "x" for ty in tys): j for t, (_, e) in self.routes.items()
                   for (tys, n), j in e.items()},
                **{f"kind{i}": j for i, (_, j, _) in enumerate(v for k, v in self.kinds.items() if len(k) == 3)}} if self.W else {})}

  def saved_graph(self, name:str, jit:TinyJit) -> TinyJit:  # (jit, the start's file's graph of the name if there's one)
    if (captured:=self.saved.get("graphs", {}).get(name)) is not None: jit.captured, jit.cnt = captured, 2
    return jit

  @functools.cached_property
  def graph_key(self) -> str: return persist.key(self)

  def restore(self):
    """the start's file's graphs into theirs (the model's rollbacks' and saves' made), else they're captured"""
    self.names = persist.buffers(self)
    self.saved = persist.load(self, self.names)
    for name, jit in self.graphs().items(): self.saved_graph(name, jit)
    for name, captured in self.saved.get("graphs", {}).items():
      if name.startswith("rollback"): self.model._rollbacks[int(name[8:])] = _TinyJit(None, captured=captured)
      if name.startswith("copy"): self.model._copies[(int(name[4]), name[5:] == "True")] = _TinyJit(None, captured=captured)

  def fingerprints(self) -> dict[str, str]:
    """the graphs' outputs on defined inputs, run after every graph's first (the gpus' first uses of the host's buffers done): a step of each
    count of tokens of token 0 at position 0 on zeroed recurrent states (and the draft's chain), the window's route of zero logits (its
    selections' rows sorted: within an expert their order is the atomics'). only what's exact: tokens, guesses, the routes' weights"""
    if (states:=self.model._states()): Tensor.realize(*[s.assign(s.zeros_like()) for s in states])
    self.tok_np[:], self.given_np[:] = 0, -1
    fps = {}
    for T, jit in self.jits.items():
      jit(self.v_pos.bind(0))[0].tolist()
      fps[f"step{T}"] = persist.fingerprint(self.res_np[:1 + T + self.width])  # (the tokens and guesses, not the scores' floats)
    if self.chain_jit:
      self.chain_jit(self.v_pos.bind(0), self.v_row.bind(0)).tolist()
      fps["chain"] = persist.fingerprint(self.cres_np[:1 + self.links])
    if self.W:
      Tensor(self.win["logits"].uop).assign(Tensor(self.win["logits"].uop).zeros_like()).realize()
      b = next(b for b in self.model.blk if hasattr(b, "ffn_gate_inp"))
      route, li = self.routes[TILES[0]][0], self.m["layers"].index(self.model.blk.index(b))
      self.graph(route, self.v_wlive.bind(self.chunk), b.exp_probs_b["bias"], Tensor(self.tags[li]), self.parts[1][li])
      fps["route"] = persist.fingerprint(*(self.win[k].numpy() for k in ("w", "tiles")), *(np.sort(self.win[k].numpy()) for k in ("pos", "entries")))
    return fps

  def recapture(self):
    """the start's file dropped, every graph captured again (the window's on a state lent again)"""
    persist.discard(self)
    self.saved = {}
    for jit in self.graphs().values():
      if jit.fxn is not None: jit.reset()
    self.model._rollbacks.clear()
    self.model._copies.clear()
    self.short_engine = None
    if self.W: self.window_lend()
    self.dry_steps()

  def step(self, toks:list[int], pos:int, given:list[int]|None=None) -> tuple[list[int], list[int], np.ndarray|None]:
    """a step of the tokens at positions pos..: the gpus run its graph, the host every layer's routing and reads. the greedy token after each,
    the draft's guesses after the one after the kept tokens and their scores (a scored draft's: each right if the ones before are) (their logits
    kept)"""
    assert DEBUG < 2, "DEBUG>=2 syncs in the step's launch, before the host's part runs: a hang"
    io, T, t0 = self.io, len(toks), time.perf_counter()
    assert T in self.jits and pos + T <= self.max_context
    self.grow(max(pos + T, getattr(self.draft, "window", 0)) + self.tokens)
    io.tokens, io.victims_n, self.res_np[0] = T, 32 * T, 0
    self.tok_np[:T], self.given_np[:T], self.key_np[:] = toks, ((given or []) + [-1] * T)[:T], self.rng.integers(1 << 32, size=2)
    host_c().flags(ctypes.byref(io), 0)
    _, self.logits = self.jits[T](self.v_pos.bind(pos))
    t1 = time.perf_counter()
    err = host_c().step_io(ctypes.byref(io))
    if err: host_c().flags(ctypes.byref(io), 1)  # (the gpus run the graph out: one killed waiting on a layer wedges)
    t2 = time.perf_counter()
    while self.res_np[0] != 1:
      if time.perf_counter() - t2 > 60: raise RuntimeError("the step's graph didn't finish")
    if err: raise RuntimeError({-1: "a read failed", -2: "timeout", -3: "no victims left", -4: "NaN scores"}[err])
    ret, G = self.res_np[1:1 + T + 2 * self.width], self.width
    self.times += (t1 - t0, time.perf_counter() - t2)
    self.step_s[T].append(time.perf_counter() - t0 + self.chain_s)
    self.chain_s = 0.0
    scores = ret[T + G:].view(np.float32).copy() if self.draft is not None and self.draft.scored else None
    return ret[:T].tolist(), ret[T:T + G].tolist(), scores

  def proposal(self, g:list[int], score:np.ndarray|None, pos:int, row:int, copy:list[int], pace:float) -> tuple[list[int], np.ndarray|None]:
    """a verify step's guesses after the token at pos: the row's of the step before (guessed at its row), the draft's chain after them. scored:
    as many as the step of the widths that expects most tokens a second, or once there's a pace (the request's tokens a second so far) most
    tokens over what its seconds take at the pace (by the scores as the steps bore them out, a guess the chain hasn't made yet scored as the
    last, the steps' seconds a line in the width through every width's measured ones: one not measured first), the chain run a guess further
    while a wider step's expected. a scored draft's block of the whole step and the context's copy: the draft's guesses up to the first one
    the copy differs at, the copy's from there, scored by the copy's verified guesses' rate (one both guess the higher). the guesses and the
    draft's scores of its own"""
    chain, seen = None, [(T, t) for T in self.widths for t in self.step_s[T]]
    line = np.polyfit(*zip(*seen), 1) if len(self.widths) > 1 and all(self.step_s[T] for T in self.widths) else None
    own, copied = len(g), score is not None and len(copy) == len(g)
    if copied:
      own, copy_s = next((j for j, (a, b) in enumerate(zip(g, copy)) if a != b), own), (self.copied[1] + 1) / (self.copied[0] + 2)
      g = [*g[:own], *copy[own:]]
    while True:
      n = self.tokens - 1
      if score is not None and len(self.widths) > 1:
        s = self.borne_out(np.concatenate([score, np.full(n, score[-1])]))
        if copied: s[:own], s[own:len(g)] = np.maximum(s[:own], copy_s), copy_s
        def rate(T:int) -> float:
          if not (t:=self.step_s[T]): return math.inf
          tokens, seconds = 1 + np.cumprod(s[:T - 1]).sum(), np.mean(t) if line is None else np.polyval(line, T)
          return tokens - pace * seconds if pace else tokens / seconds
        n = max(self.widths, key=rate) - 1
      if len(g) >= n: return g[:n], None if score is None else score[:min(n, own)]
      guess, prob = next(chain:=chain or self.chain(g[-1], pos, row))
      g, score, own = [*g, guess], None if score is None else np.append(score, prob), own + 1

  def borne_out(self, s:np.ndarray) -> np.ndarray:
    """scores as the steps bore them out: their tenth's verified guesses right, a score itself as 16 of them"""
    seen, right = self.scored[:, np.minimum((s * 10).astype(int), 9)]
    return (right + 16 * s) / (seen + 16)

  def tally(self, score:np.ndarray, n:int, row:int):
    """a verify step's verified guesses of its n (the first row right, the next wrong): the draft's into the table of their scores' tenth, the
    copy's after them into its count"""
    for j in range(min(row + 1, n)):
      if j < len(score): self.scored[:, min(int(score[j] * 10), 9)] += (1, j < row)
      else: self.copied += (1, j < row)

  def chain(self, g:int, pos:int, row:int) -> Iterator[tuple[int, float]]:
    """the draft's guesses after g (at pos+1, guessed at the step's row) for a verify step and their probabilities, a chain step's at a time"""
    while self.chain_jit is not None:
      t0 = time.perf_counter()
      self.given_np[0], self.cres_np[0], self.io.tokens, self.io.victims_n = g, 0, 1, 32
      host_c().flags(ctypes.byref(self.io), 0)
      self.chain_jit(self.v_pos.bind(pos), self.v_row.bind(row))
      if (err:=host_c().layer(ctypes.byref(self.io), len(self.bias) - 1) if self.draft.blk else 0): host_c().flags(ctypes.byref(self.io), 1)
      st = time.perf_counter()
      while self.cres_np[0] != 1:
        if time.perf_counter() - st > 60: raise RuntimeError("the draft's chain didn't finish")
      if err: raise RuntimeError(f"the draft's layer: {err}")
      self.chain_s += time.perf_counter() - t0
      guesses, pos = self.cres_np[1:1 + self.links].tolist(), pos + self.links
      yield from zip(guesses, self.cres_np[1 + self.links:].view(np.float32).tolist())
      g = guesses[-1]

  # *** master's model's generate (LLMServer's): the caches only go forward, a prompt that extends the tokens fed (or the last prompt) feeds
  # the rest

  @property
  def max_context(self) -> int: return self.model.max_context
  def get_start_pos(self, tokens:list[int]) -> int:
    self.resume(tokens)
    return self.model.get_start_pos(tokens)

  def session_rows(self, i:int, at:UOp) -> Tensor:  # (cache i's rows of a kept session's at..)
    o, k = self.caches[i]
    return getattr(o, k).shrink(tuple((at, at + self.B) if j == self.stripes[i][0] else None for j in range(getattr(o, k).ndim)))

  def session_states(self) -> list[Tensor]: return [*self.model._states(), *itertools.chain(*self.model._sets)]
  def keep(self, at:UOp):  # (a session's rows at.. and recurrent states into the copies)
    states = zip(self.stash_states, self.session_states())
    Tensor.realize(*[c.assign(self.session_rows(i, at)) for i, c in enumerate(self.stash)], *[c.assign(s) for c, s in states])

  def unkeep(self, at:UOp):  # (and back)
    states = zip(self.stash_states, self.session_states())
    Tensor.realize(*[self.session_rows(i, at).assign(c) for i, c in enumerate(self.stash)], *[s.assign(c) for c, s in states])

  def resume(self, tokens:list[int]):
    """the kept session back when tokens share more with it than its copied rows' start and the tokens fed are (if the requests since wrote
    only its copied rows: gone for tokens that share less), else the tokens fed kept when tokens diverge from them early and are fewer (a
    short request: a title, a summary), unless they're a short request's too and one longer is kept"""
    m = self.model
    def shared(a:list[int]) -> int: return sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens, a)))
    if self.session is not None and (shared(self.session[0]) < self.session[1] or m._cached_tokens != self.session[0] and
       round_up(len(m._cached_tokens) + self.tokens, max(self.chunk, 1)) > self.session[1] + self.B): self.session = None
    if self.session is not None and shared(kept:=self.session[0]) > max(self.session[1], len(m._cached_tokens)):
      self.keep_jits[1](self.v_at.bind(self.session[1]))
      m._cached_tokens, m._saved, self.session = kept[:], self.session[2], None
    elif len(tokens) < len(m._cached_tokens) and (at:=shared(m._cached_tokens)) + self.B <= self.max_context and len(tokens) - at < self.B // 2 \
         and (self.session is None or len(fed:=m._cached_tokens) > len(self.session[0]) or len(fed) - at >= self.B and fed != self.session[0]):
      self.keep_jits[0](self.v_at.bind(at))
      self.session = (m._cached_tokens[:], at, m._saved[:])

  def prefill(self, tokens:list[int], pos:int) -> int:
    """tokens from pos on but the last fed in windows of up to W of them (an even split, a layer at a time over each), the last one's chunk
    partial (its rows padded), unless steps of them are faster (steps_faster), the position after them"""
    if not self.W: return pos
    end, E, K, w = len(tokens) - 1, self.m["n_experts"], self.model.config.num_experts_per_tok, self.max_primary_span
    self.grow(end + self.chunk)
    free = self.free_slots() - w * int((self.state["flying"] > 0).sum())
    if not (P:=self.W * self.pieces() if free >= E * w else min(free // (K * w), self.W)): return pos  # (a read window's most tokens)
    size = -(-(end - pos) // -(-(end - pos) // P)) if end > pos else P
    while pos < end:
      n = min(size, end - pos)
      view = self.short_engine if self.short_engine is not None and pos > 0 and n <= self.short_window else self
      if pos + round_up(n, view.chunk) > self.max_context or self.steps_faster(n): break
      view.window(tokens[pos:pos + n], pos, tokens[pos + 1:pos + 1 + n])
      pos += n
    return pos

  def cap_prefill_prior(self):
    """Keep each layer's expert/copy proportions while limiting the weight of its prefill history."""
    if self.io.decay >= 1: return
    freq = self.state["freq"].reshape(self.io.copies, self.io.layers, -1)
    total = freq.sum(axis=(0, 2))
    factor = np.minimum(1, (20 / (1 - self.io.decay)) / np.maximum(total, 1))
    freq *= factor[None, :, None]

  def steps(self, tokens:list[int], pos:int, end:int, *, cap_prior:bool=False) -> tuple[list[int], list[int], np.ndarray|None]:
    """the tokens from pos to end in steps of the most of them there's a graph of (each given the token after it), the last step's outputs"""
    out:tuple[list[int], list[int], np.ndarray|None] = ([], [], None)
    tail, self.io.tail = self.io.tail, 0
    while pos < end:
      T = max(t for t in self.jits if t <= end - pos)
      if cap_prior and pos + T == end: self.cap_prefill_prior()
      out = self.step(tokens[pos:pos + T], pos, tokens[pos + 1:pos + 1 + T])
      pos += T
    self.io.tail = tail
    return out

  def generate(self, tokens:list[int], temperature:float=0.0):
    """the tokens after tokens sampled at temperature (greedy at 0), appended as they come: fed from get_start_pos in chunks (prefill) and the
    rest in steps (steps), then a token a step or (a draft) a verify step of [the next, the draft's chain of guesses after it]: the guesses kept
    up to the first wrong one, the target's token after them next, the recurrent states rolled back to after the kept ones"""
    self.temperature_np[:] = temperature
    start = self.get_start_pos(tokens)
    # (where tokens diverge from the tokens fed before, a window's any length: its states saved)
    fork_prefilled = start
    fork, self.model._cached_tokens = self.model.fork(tokens, start, 1 if self.W else self.chunk or 1), []
    if fork > start:
      fork_prefilled = self.prefill(tokens[:fork + 1], start)
      self.steps(tokens, fork_prefilled, fork)
      self.model.save(tokens[:fork])
    prefilled = self.prefill(tokens, fork)
    cap_prior = fork_prefilled > start or prefilled > fork
    (*_, t), g, score = self.steps(tokens, prefilled, len(tokens), cap_prior=cap_prior)
    self.model.save(tokens)
    row, c0, v0, widths, copies, done = 0, self.counters(), self.verified.copy(), collections.Counter(), Copies(), np.zeros(2)
    try:
      while len(tokens) + self.tokens <= self.max_context:  # (a verify step's last token in the context)
        tokens.append(t)
        self.model._cached_tokens = tokens[:-1]
        yield t
        if self.draft is None:
          (t,), _, _ = self.step([t], len(tokens) - 1)
          widths[1] += 1
          continue
        gs, scores = self.proposal(g, score, len(tokens) - 1, row, copies(tokens, self.tokens - 1), done[0] / done[1] if done[1] > 2 else 0.0)
        toks, g, score = self.step([t, *gs], len(tokens) - 1)
        widths[len(toks)] += 1
        row = next((j for j, g in enumerate(gs) if toks[j] != g), len(gs))
        self.verified += (1, len(gs), row)
        done += (row + 1, self.step_s[len(toks)][-1])
        if scores is not None: self.tally(scores, len(gs), row)
        if row < len(gs): self.model.rollback(row + 1)  # (before the kept ones are yielded: a request may stop at any)
        for kept in gs[:row]:
          tokens.append(kept)
          self.model._cached_tokens = tokens[:]
          yield kept
        t = toks[row]
    except Exception:
      self.model._cached_tokens = []
      raise
    finally:
      if (n:=sum(widths.values())):
        dt, _, refs, hits, *_, gpu, _, reads, _, _, after = (self.counters() - c0) / n
        print(f"decode: {n} steps, {1 + (self.verified[2] - v0[2]) / n:.2f} tokens a step (widths {dict(sorted(widths.items()))}), "
              f"{dt * 1e3:.1f} ms a step: {gpu * 1e3:.1f} waiting on the gpus, {reads * 1e3:.1f} on reads, {after * 1e3:.1f} after the last layer; "
              f"{hits / max(refs, 1):.1%} of experts cached", flush=True)

  def counters(self) -> np.ndarray:
    io = self.io
    return np.array([time.perf_counter(), io.bytes, io.refs, io.hits, io.late, io.misses, io.guesses, io.used, io.fills,
                     io.t_gpu, io.t_route, io.t_reads, io.t_after, *self.times])

  def report(self, c0:np.ndarray, n:int):
    """a step's time, reads and their split over the n steps since the counters c0"""
    dt, reads, refs, hits, late, misses, guesses, used, fills, *ms = (self.counters() - c0) / n
    print(f"{dt * 1e3:.1f} ms/token = {1 / dt:.2f} tok/s, {reads / 1e9:.2f} GB/token read: {misses:.0f} misses, "
          f"{guesses:.0f} guesses ({used:.0f} used), {fills:.0f} fills; of {refs:.0f} experts {hits / refs:.1%} cached, {late / refs:.1%} in flight")
    print("  ms/token: " + ", ".join(f"{t * 1e3:.1f} {k}" for k, t in zip(("waiting on the gpus", "routing", "waiting on reads",
                                                                           "after the reads", "launching", "after the last layer"), ms)), flush=True)

if __name__ == "__main__":
  from tinygrad.llm.cli import SimpleTokenizer, chat_template
  from tinygrad.llm.serve import LLMServer
  from extra.k3.store import open_store
  # DEADLINE=s: the run stops between steps once it's s seconds in (killed mid-step, the gpus are left waiting on the host). SIGTERM as SIGINT:
  # a clean exit (the devices finalized)
  signal.signal(signal.SIGTERM, signal.default_int_handler)
  st = time.perf_counter()
  def late() -> bool: return time.perf_counter() - st > getenv("DEADLINE", 540)
  model, kv = Transformer.from_gguf(sys.argv[1], max_context=getenv("CTX", 4096), shard=getenv("SHARD", 4))
  tok = SimpleTokenizer.from_gguf_kv(kv)
  print(f"model in {time.perf_counter() - st:.0f} s", flush=True)
  # STORE=at,..: the store's copies of the experts (their manifests' offsets on disk 0)
  stores = [open_store(list(model.token_embd.weight.device), int(at)) for at in getenv("STORE", str(512 << 30)).split(",")]  # (k3's iq3_xxs store)
  # DRAFT=dir: the draft (eagle's head, dspark's blocks; mtp: the model's own next token prediction layer), speculative decoding with verify
  # steps of VERIFY tokens (memory: fewer SLOTS)
  # VERIFY=n: a verify step's tokens, the next and the draft's n-1 guesses (eagle's: a chain) (an EVAL: its steps of n tokens)
  # (VERIFY=n,n..: a scored draft's verify steps of any of them, each step's by the guesses' scores)
  widths = [int(x) for x in str(getenv("VERIFY", "2")).split(",")]
  draft = (MTP(model) if d == "mtp" else load(d, lambda t: model.token_embd(t).float(), model.output, model.token_embd.weight.device,
                                               model.max_context, max(widths) - 1)) if (d:=getenv("DRAFT", "")) else None
  print(f"start: store, draft {time.perf_counter() - st:.0f} s in", flush=True)
  # CHUNK=n: a prompt fed in steps of n tokens (0: a token a step), WINDOW=n: in windows of up to n tokens (a layer at a time, chunks of CHUNK)
  # TAPS=b,b..: without a draft, the streams a window keeps after blocks b (bm: as the next block's attention residual mixes them), for a DUMP
  taps = [(int(t.rstrip("m")), t.endswith("m")) for t in getenv("TAPS", "").split(",") if t]
  eng = Engine(model, stores[0][1], tokens=max(widths) if draft else 1, guess=getenv("GUESS", ""), draft=draft, logits=bool(getenv("EVAL", "")),
               chunk=getenv("CHUNK", 256), window=getenv("WINDOW", 0), taps=taps, widths=tuple(widths), source=sys.argv[1],
               copies=tuple(m for _, m in stores[1:]), short_window=getenv("SHORT_WINDOW", 0))
  eng.drives(*(d for d, _ in stores))
  eng.warm()
  print(f"warm in {time.perf_counter() - st:.0f} s, {eng.slots} slots a gpu", flush=True)
  if (spec:=getenv("EVAL", "") or getenv("WEVAL", "") or getenv("BENCH", "")):  # EVAL=/WEVAL=/BENCH=corpus.jsonl:nseq:ntok:first (eval.py)
    from extra.k3.eval import evaluate, window_evaluate, window_dump, bench
    if (dump:=getenv("DUMP", "")): window_dump(eng, spec, late, dump)  # DUMP=dir with WEVAL=: the taps and the target's tokens (eval.py)
    elif getenv("BENCH", ""): bench(eng, spec, late, getenv("DEPTH", ""))  # (DEPTH=corpus.jsonl:n: n tokens of text before each prompt)
    else: (evaluate if getenv("EVAL", "") else window_evaluate)(eng, spec, late)
    sys.exit(0)
  # SERVE=port: master's openai server (and the model's chat template), else N tokens after the prompt
  if (port:=getenv("SERVE", 0)): LLMServer(('', port), eng, kv.get('general.name', 'k3'), tok, chat_template(tok)).serve_forever()
  ids = tok.encode(sys.argv[2] if len(sys.argv) > 2 else "The capital of France is")
  if tok.bos_id is not None and ids[:1] != [tok.bos_id]: ids = [tok.bos_id] + ids
  t0, out = time.perf_counter(), []
  for t in eng.generate(list(ids)):
    if not out:
      print(f"prompt of {len(ids)} tokens: {len(ids) / (time.perf_counter() - t0):.2f} tok/s", flush=True)
      c0 = eng.counters()
    out.append(t)
    if len(out) == getenv("N", 32) or tok.is_end(t) or late(): break
  print(repr(tok.decode(out)), out, flush=True)
  steps, guesses, right = eng.verified
  if draft: print(f"{right / guesses:.1%} of {guesses:.0f} guesses right, {(steps + right) / steps:.2f} tokens a step")
  eng.report(c0, len(out) - 1)
