#!/usr/bin/env python3
# trace simulation of extra/k3/engine.py's cache policy on the teacher traces' test tokens (decode order): global decayed LFU slots, tau
# keep-resident routing, the next layer's guesses (GUESS's fitted maps on the layer's input, those its router would keep, in its order:
# sigmoid + bias) holding their slots till their layer. no timing: a guess is ready at its layer. reports read GB/token, the share of
# experts cached at the router, the misses and the guesses used.
# python3 sim_engine.py teacher_dir guess_dir [ntest [slots [prefetch [half_life]]]]
import sys
from pathlib import Path
import numpy as np

D, G = Path(sys.argv[1]), Path(sys.argv[2])
NTEST, SLOTS, PREFETCH, HALF_LIFE = (int(sys.argv[3]) if len(sys.argv) > 3 else 3000, int(sys.argv[4]) if len(sys.argv) > 4 else 6900,
                                     int(sys.argv[5]) if len(sys.argv) > 5 else 5, float(sys.argv[6]) if len(sys.argv) > 6 else 8.0)
E, K, KMIN, NL, PART, TAU = 896, 16, 4, 92, 12641952, 0.03
split, bias = np.load(D / "split.npy"), np.load(D / "router_bias.npy")
te = np.nonzero(split == 2)[0]
te = te[np.lexsort((np.load(D / "position.npy")[te], np.load(D / "request.npy")[te]))][:NTEST]

def top(logits:np.ndarray, T:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """the K experts of the largest sigmoid + bias, in that order, their normalized weights and keys"""
  s = 1 / (1 + np.exp(-logits))
  sel = np.argsort(-(s + bias[T - 1]), axis=1)[:, :K]
  w = np.take_along_axis(s, sel, 1)
  return sel, w / w.sum(1, keepdims=True), np.take_along_axis(s + bias[T - 1], sel, 1)

true_sel, true_w, guess = np.zeros((NL, len(te), K), np.int16), np.zeros((NL, len(te), K), np.float32), {}
for T in range(1, NL + 1):
  true_sel[T - 1], true_w[T - 1], _ = top(np.load(D / f"logits-{T}.npy", mmap_mode="r")[te].astype(np.float32), T)
  if T > 1:  # the guesses of layer T on layer T-1's input: those its router would keep (top KMIN, weight >= tau), -1 the rest
    sel, w, key = top(np.load(D / f"router_input-{T - 1}.npy", mmap_mode="r")[te].astype(np.float32) @
                      np.load(G / f"guess-{T}.npy").astype(np.float32).T, T)
    guess[T - 1] = np.where((np.arange(K) < KMIN) | (w >= TAU), sel, -1)
  print(f"layer {T} prepared", file=sys.stderr, flush=True)

slot_of, expert_in = np.full(NL * E, -1, np.int64), np.full(SLOTS, -1, np.int64)
freq = np.zeros(NL * E + 1, np.float32)
freq[-1] = -1  # a free slot goes first
guessed:set[int] = set()
reads = refs = hits = used = guesses = 0
def fetch(codes:np.ndarray, pinned:np.ndarray):
  """slots for codes: the least used but the pinned and the guessed"""
  global reads
  if not len(codes): return
  score = freq[expert_in]
  held = np.fromiter(guessed, np.int64, len(guessed)) if guessed else np.zeros(0, np.int64)
  score[np.concatenate([pinned, slot_of[held]]).astype(np.int64)] = np.inf
  i = np.argpartition(score, len(codes) - 1)[:len(codes)]
  old = expert_in[i]
  slot_of[old[old >= 0]] = -1
  expert_in[i], slot_of[codes] = codes, i
  reads += len(codes)
for t in range(len(te)):
  for li in range(NL):
    sel, w = true_sel[li, t].astype(np.int64), true_w[li, t]
    codes = li * E + sel
    cached = slot_of[codes] >= 0
    keep = (np.arange(K) < KMIN) | (w >= TAU) | cached
    codes, cached = codes[keep], cached[keep]
    refs, hits = refs + len(codes), hits + int(cached.sum())
    mine = {c for c in guessed if c // E == li}
    used += len(mine.intersection(codes.tolist()))
    guessed -= mine
    fetch(codes[~cached], slot_of[codes[cached]])
    if li + 1 < NL and PREFETCH:
      gs = guess[li + 1][t].astype(np.int64)
      gc = (li + 1) * E + gs[gs >= 0]
      gc = gc[slot_of[gc] < 0][:PREFETCH]
      fetch(gc, slot_of[codes])
      guessed.update(gc.tolist())
      guesses += len(gc)
    f = freq[li * E:(li + 1) * E]
    f *= 0.5 ** (1 / HALF_LIFE)
    f[codes - li * E] += 1
n = len(te)
print(f"{n} test tokens, {SLOTS} slots, prefetch {PREFETCH}, half life {HALF_LIFE}: {reads * PART / n / 1e9:.2f} GB/token, "
      f"{hits / refs:.1%} cached at the router, misses {(refs - hits) / n:.0f}/token, guesses {guesses / n:.0f}/token, {used / n:.0f} used")
