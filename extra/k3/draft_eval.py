#!/usr/bin/env python3
# a draft's guesses offline on a dump of the target's taps (eval.py's DUMP), on one gpu: the anchors are the positions whose next token is a
# completion's; a block of K guesses after each, each given the ones before it right (the text's tokens: as verify steps on the fixed text).
# per kind of text (thinking, answer, edit): the first guess right against the text and against the target's greedy token, by the target's top-1
# probability (the ceiling: the target's greedy token the text's), the guesses right by position, a verify step of T tokens' expected tokens and
# its modeled tok/s (a step T2 seconds at T=2, DT more a token). results-<name>.npz: each anchor's guesses and the draft's confidence
# DEV=AMD PYTHONPATH=. python3 extra/k3/draft_eval.py dump_dir trunk.gguf dspark|eagle path [name]
import json, sys, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit, nn
from tinygrad.helpers import getenv
from tinygrad.llm.gguf import gguf_parse, ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear
from extra.k3.draft import DSpark, Eagle3, weight

CLOSE, THINK = 163588, 39964  # (K3.1's "<|close|>think": the end of the thinking)
K, STRIDE, BATCH, T2, DT = getenv("K", 7), getenv("STRIDE", 1), getenv("BATCH", 64), getenv("T2", 0.050), getenv("DT", 0.015)
DEADLINE = getenv("DEADLINE", 50.0)  # (seconds: the anchors guessed by then reported)

def shared(path:str, dev:str) -> tuple:
  """the target's embedding and head (in q8_0: the kernel's fast), decoded on dev from the trunk gguf"""
  _, entries = gguf_parse(path)
  def dec(name:str) -> Tensor:
    data, shape, typ = entries[name]
    return ggml_data_to_tensor(data.to(dev), int(np.prod(shape)), typ).reshape(shape).half().realize()
  emb, out = nn.Embedding(1, 1), weight(dec("output.weight"), (dev,))
  emb.weight = dec("token_embd.weight")
  def head(x:Tensor) -> Tensor: return out(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:-1], -1) if isinstance(out, Linear) else x.half() @ out.T
  return emb, head

def kinds(meta:dict, toks:np.ndarray) -> np.ndarray:
  """each position's kind of text: 0 prompt, 1 thinking, 2 answer, 3 an edit's completion"""
  kind = np.zeros(len(toks), np.int8)
  for s in meta["seqs"]:
    o = s["start"] + s["prompt"] - 1  # (the positions whose next token is the completion's)
    if s["task_id"].startswith("edit"): kind[o:s["start"] + s["len"]] = 3
    else:
      out = toks[s["start"] + s["prompt"]:s["start"] + s["len"]]
      end = next((i for i in range(len(out) - 1) if out[i] == CLOSE and out[i + 1] == THINK), len(out))
      kind[o:o + end + 1], kind[o + end + 1:s["start"] + s["len"]] = 1, 2
  return kind

def run(draft_kind:str, path:str, dump:str, gguf:str) -> dict:
  dev, meta = Device.DEFAULT, json.load(open(f"{dump}/meta.json"))
  toks = np.load(f"{dump}/tokens.npy")
  tap = {n: np.load(f"{dump}/tap{n}.npy", mmap_mode="r") for n in meta["taps"]}
  embed, head = shared(gguf, dev)
  L = max(s["len"] for s in meta["seqs"])
  t0 = time.perf_counter()
  if draft_kind == "dspark":
    d = DSpark(path, lambda t: embed(t).float(), head, (dev,), L + K, q8=bool(getenv("Q8", 1)))
    names = [f"{b}{'m' if getenv('MIXED', 0) else ''}" for b, _ in d.taps]
  else:
    d = Eagle3(path, lambda t: embed(t).float(), (dev,), L)
    names = [str(b) for b, _ in d.taps]
  print(f"{draft_kind} loaded in {time.perf_counter() - t0:.1f} s, taps {names}", flush=True)
  guesses, conf, anchors = [], [], []
  @TinyJit
  def fill(hid:Tensor) -> Tensor: return d.fill(None, hid, 0).realize()
  @TinyJit
  def block(at:Tensor, ap:Tensor, prev:Tensor) -> tuple[Tensor, Tensor]:  # (BATCH blocks, the tokens before their guesses given)
    g, c, _ = d(at, ap, K, (0, L + K), prev)
    Tensor.realize(g, c)
    return g, c
  for s in meta["seqs"]:
    if time.perf_counter() - t0 > DEADLINE: break
    a0, n = s["start"], s["len"]
    hid = np.stack([tap[nm][a0:a0 + n] for nm in names], axis=1)[None]  # (1, n, taps, dim)
    an = np.arange(max(s["prompt"] - 1, 1), n - 1)[::STRIDE]  # (anchors: a position, its block guesses the tokens after it)
    if draft_kind == "dspark":
      fill(Tensor(np.pad(hid, ((0, 0), (0, L - n), (0, 0), (0, 0)))).to(dev).realize())  # (every sequence L long: one graph)
      st = np.pad(toks[a0:a0 + n], (0, K))
      for b in range(0, len(an), BATCH):
        if time.perf_counter() - t0 > DEADLINE:
          an = an[:b]
          break
        a = np.pad(an[b:b + BATCH], (0, BATCH - len(an[b:b + BATCH])), mode="edge")
        prev = np.stack([st[x:x + K] for x in a])  # (the text's tokens before each guess)
        g, c = block(*(Tensor(x.astype(np.int32)).to(dev).realize() for x in (st[a], a, prev)))
        m = min(BATCH, len(an) - b)
        guesses.append(g.numpy()[:m]), conf.append(c.numpy()[:m])
    else:  # (eagle: row j takes t[j+1] on the target's taps after t[j], guesses t[j+2]: the first guess of the block after j+1)
      lg, _, _ = d(Tensor(toks[a0 + 1:a0 + n].astype(np.int32)[None]).to(dev), Tensor(hid[:, :n - 1]).to(dev), 0)
      first = lg[0].argmax(-1).numpy()
      guesses.append(np.pad(first[an - 1][:, None], ((0, 0), (0, K - 1)), constant_values=-1)), conf.append(np.zeros((len(an), K), np.float32))
    anchors.append(a0 + an)
  print(f"guessed in {time.perf_counter() - t0:.1f} s", flush=True)
  return {"guesses": np.concatenate(guesses), "conf": np.concatenate(conf), "anchors": np.concatenate(anchors)}

def report(r:dict, dump:str, k_max:int):
  meta, toks, nxt, p1 = json.load(open(f"{dump}/meta.json")), np.load(f"{dump}/tokens.npy"), np.load(f"{dump}/next.npy"), np.load(f"{dump}/p1.npy")
  a, g, kind = r["anchors"], r["guesses"][:, :k_max], kinds(meta, toks)[r["anchors"]]
  end = np.concatenate([np.full(s["len"], s["start"] + s["len"]) for s in meta["seqs"]])[a]
  text = np.stack([np.where(a + 1 + j < end, np.pad(toks, (0, k_max + 1))[a + 1 + j], -2) for j in range(k_max)], 1)  # (the text's next k)
  right = np.cumprod(g == text, 1)  # (the guess and all before it right)
  for name, sel in [("all", kind > 0), ("thinking", kind == 1), ("answer", kind == 2), ("edit", kind == 3)]:
    if not sel.any(): continue
    s = sel
    print(f"{name} ({s.sum()} anchors): first guess {(g[s, 0] == text[s, 0]).mean():.1%} of the text, {(g[s, 0] == nxt[a[s]]).mean():.1%} of "
          f"the target's greedy token (the target's greedy the text's: {(nxt[a[s]] == text[s, 0]).mean():.1%})")
    for lo, hi in ((0.9, 1.01), (0.5, 0.9), (0, 0.5)):
      b = s & (p1[a] >= lo) & (p1[a] < hi)
      if b.any(): print(f"  top-1 p in [{lo}, {min(hi, 1)}): {b.sum() / s.sum():.0%} of anchors, first guess {(g[b, 0] == text[b, 0]).mean():.1%} "
                        f"of the text, {(g[b, 0] == nxt[a[b]]).mean():.1%} of greedy, ceiling {(nxt[a[b]] == text[b, 0]).mean():.1%}")
    rates = right[s].mean(0)
    print("  right by position (all before right): " + " ".join(f"{x:.3f}" for x in rates))
    print("  verify T: " + ", ".join(f"T={T} {1 + rates[:T - 1].sum():.2f} tokens {(1 + rates[:T - 1].sum()) / (T2 + DT * (T - 2)):.1f} tok/s"
                                   for T in range(2, k_max + 2) if r["guesses"][0, T - 2] >= 0))

if __name__ == "__main__":
  dump, gguf, kind, path = sys.argv[1:5]
  name = sys.argv[5] if len(sys.argv) > 5 else kind
  r = run(kind, path, dump, gguf)
  np.savez(f"{dump}/results-{name}.npz", **r)
  report(r, dump, K)
