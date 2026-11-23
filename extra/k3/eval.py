# EVAL=corpus.jsonl:nseq:ntok:first (engine.py): production completions teacher forced through an engine, each from position 0. per output
# token the top-1 agreement with production and KL(production's top 5 and the rest || ours), per sequence and so far
import itertools, json, os, sys, time
from typing import Callable
import numpy as np
from tinygrad import Tensor

def score(lg:np.ndarray, top:list) -> tuple[float, float]:
  """logits lg against production's top 5 (logprob, token): top-1 agreement, KL(production's top 5 and the rest || ours)"""
  lp = lg - lg.max() - np.log(np.exp(lg - lg.max()).sum())
  p, q = np.exp([x[0] for x in top]), lp[[x[1] for x in top]]
  rest_p, rest_q = max(1 - p.sum(), 1e-12), max(1 - np.exp(q).sum(), 1e-12)
  return float(lp.argmax() == top[0][1]), float((p * (np.log(p) - q)).sum() + rest_p * (np.log(rest_p) - np.log(rest_q)))

def evaluate(eng, spec:str, late:Callable[[], bool]):
  path, *nums = spec.split(":")
  nseq, ntok, first = [int(x) for x in nums] + [16, 256, 0][len(nums):]
  stats:list[tuple[float, float]] = []
  for r in [json.loads(line) for line in open(path)][first:first + nseq]:
    ids, outs, n0, c0 = r["prompt_ids"], r["output_ids"][:ntok], len(stats), eng.counters()
    seq, T, at = ids + outs[:-1], eng.tokens, eng.prefill(ids, 0)  # (the prompt in chunks, then steps of T tokens: a verify step's numerics)
    while at < len(seq):
      if late(): sys.exit(f"deadline: {len(stats)} tokens, top-1 agree and KL {np.mean(stats, 0) if stats else '-'}")
      n = T if len(seq) - at >= T else 1
      eng.step(seq[at:at + n], at)
      for pos, lg in enumerate(eng.logits.numpy().astype(np.float64), at):
        if pos >= len(ids) - 1: stats.append(score(lg, r["output_top"][pos - len(ids) + 1]))
      at += n
    (a1, k1), (a, k) = np.mean(stats[n0:], 0), np.mean(stats, 0)
    print(f"{r['task_id']}: top-1 agree {a1:.1%}, KL {k1:.4f}; all {len(stats)} tokens: top-1 agree {a:.1%}, KL {k:.4f}")
    eng.report(c0, len(ids) + len(outs) - 2)

def bank_logits(m, bank:Tensor) -> Tensor:
  return m.output(m.output_norm(m.output_res_score.mix(bank) if m.config.attn_res_block_size else bank[:, :, 0].float()))[0].float()

def window_evaluate(eng, spec:str, late:Callable[[], bool], rows:int=128, size:int=0) -> list[tuple[float, float]]:
  """WEVAL=corpus.jsonl:nseq:ntok:first: evaluate's completions through windows (of up to size tokens, else the engine's), every position's logits
  from the window's last streams (a pass of the sequence, no steps), rows at a time"""
  path, *nums = spec.split(":")
  nseq, ntok, first = [int(x) for x in nums] + [16, 256, 0][len(nums):]
  m, stats = eng.model, []
  for r in [json.loads(line) for line in open(path)][first:first + nseq]:
    if late(): sys.exit(f"deadline: {len(stats)} tokens, top-1 agree and KL {np.mean(stats, 0) if stats else '-'}")
    ids, outs, n0 = r["prompt_ids"], r["output_ids"][:ntok], len(stats)
    seq = ids + outs[:-1]
    for at in range(0, len(seq), size or eng.W):
      toks = seq[at:at + (size or eng.W)]
      eng.window(toks, at, (seq[at + 1:at + 1 + len(toks)] + [-1])[:len(toks)])
      for r0 in range(0, len(toks), rows):
        lgs = bank_logits(m, eng.in_order(Tensor(eng.win["bank"].uop), r0, min(rows, eng.W - r0))).numpy()[:len(toks) - r0]
        for pos, lg in enumerate(lgs.astype(np.float64), at + r0):
          if pos >= len(ids) - 1: stats.append(score(lg, r["output_top"][pos - len(ids) + 1]))
    (a1, k1), (a, k) = np.mean(stats[n0:], 0), np.mean(stats, 0)
    print(f"{r['task_id']}: top-1 agree {a1:.1%}, KL {k1:.4f}; all {len(stats)} tokens: top-1 agree {a:.1%}, KL {k:.4f}", flush=True)
  return stats

def window_dump(eng, spec:str, late:Callable[[], bool], out:str, rows:int=128):
  """DUMP=dir with WEVAL's spec: every position of the completions (the prompts' too) through windows, for a draft's offline evaluation: the
  engine's taps (a file each, (positions, dim) f32), the token, the target's greedy next token and its probability. meta.json: the sequences"""
  path, *nums = spec.split(":")
  nseq, ntok, first = [int(x) for x in nums] + [16, 256, 0][len(nums):]
  m, seqs = eng.model, [(r["task_id"], r["prompt_ids"] + r["output_ids"][:ntok], len(r["prompt_ids"])) for r in
                        [json.loads(line) for line in open(path)][first:first + nseq]]
  N, dim, names = sum(len(s) for _, s, _ in seqs), m.config.dim, [f"{b}{'m' if mixed else ''}" for b, mixed in eng.taps]
  os.makedirs(out, exist_ok=True)
  taps = [np.lib.format.open_memmap(f"{out}/tap{n}.npy", "w+", np.float32, (N, dim)) for n in names]
  toks, nxt, p1 = (np.lib.format.open_memmap(f"{out}/{k}.npy", "w+", dt, (N,)) for k, dt in (("tokens", np.int32), ("next", np.int32),
                                                                                                ("p1", np.float32)))
  meta:dict = {"taps": names, "seqs": []}
  at0 = 0
  for task, seq, n_prompt in seqs:
    if late(): sys.exit(f"deadline: {at0} positions")
    for at in range(0, len(seq), eng.W):
      toks_w = seq[at:at + eng.W]
      eng.window(toks_w, at, (seq[at + 1:at + 1 + len(toks_w)] + [-1])[:len(toks_w)])
      for j, t in enumerate(taps): t[at0 + at:at0 + at + len(toks_w)] = eng.in_order(Tensor(eng.win["taps"].uop), 0, len(toks_w))[0, :, j].numpy()
      for r0 in range(0, len(toks_w), rows):
        bank = eng.in_order(Tensor(eng.win["bank"].uop), r0, min(rows, eng.W - r0))
        lg = bank_logits(m, bank)[:len(toks_w) - r0]
        o = at0 + at + r0
        nxt[o:o + lg.shape[0]], p1[o:o + lg.shape[0]] = lg.argmax(-1).numpy(), lg.softmax(-1).max(-1).numpy()
    toks[at0:at0 + len(seq)] = seq
    meta["seqs"].append({"task_id": task, "start": at0, "len": len(seq), "prompt": n_prompt})
    at0 += len(seq)
    print(f"{task}: {len(seq)} positions dumped", flush=True)
  for a in (*taps, toks, nxt, p1): a.flush()
  json.dump(meta, open(f"{out}/meta.json", "w"))

def bench(eng, specs:str, late:Callable[[], bool], depth:str=""):
  """BENCH=corpus.jsonl:nseq:ntok:first[+corpus..]: verify steps of [the next, the draft's guesses] over production completions teacher forced
  (the text fixed: the steps' time and the guesses right don't move with what the engine would have written), each corpus's so far.
  depth (DEPTH=corpus.jsonl:n[,n..]): n tokens of that corpus's texts (its last rows first) before each prompt, every corpus at each n"""
  dpath, dns = depth.split(":") if depth else ("", "0")
  text = [t for line in open(dpath).readlines()[::-1] for r in [json.loads(line)] for t in r["prompt_ids"] + r["output_ids"]] if depth else []
  for prefix, spec in itertools.product([text[:int(dn)] for dn in dns.split(",")], specs.split("+")):
    path, *nums = spec.split(":")
    nseq, ntok, first = [int(x) for x in nums] + [16, 256, 0][len(nums):]
    steps, right, firsts, tokens, t_steps, T = 0, 0, 0, 0, 0.0, eng.tokens
    for r in [json.loads(line) for line in open(path)][first:first + nseq]:
      toks, n = prefix + r["prompt_ids"] + r["output_ids"][:ntok], len(prefix) + len(r["prompt_ids"])
      t0 = time.perf_counter()
      for pos in range(eng.prefill(toks[:n], 0), n): _, g, score = eng.step(toks[pos:pos + 1], pos, toks[pos + 1:pos + 2])
      if prefix: print(f"{r['task_id']}: prompt of {n} tokens in {time.perf_counter() - t0:.0f} s", flush=True)
      pos, row = n, 0
      while pos + T < len(toks) and not late():
        st = time.perf_counter()
        gs, _ = eng.proposal(g, score, pos, row)
        _, g, score = eng.step([toks[pos], *gs], pos, toks[pos + 1:pos + 1 + len(gs) + 1])
        t_steps, steps = t_steps + time.perf_counter() - st, steps + 1
        row = next((j for j, x in enumerate(gs) if x != toks[pos + 1 + j]), len(gs))
        right, firsts, tokens = right + row, firsts + (row > 0), tokens + len(gs) + 1
        if row < len(gs): eng.model.rollback(row + 1)
        pos += 1 + row
      print(f"{r['task_id']}: {steps} verify steps of {tokens / steps:.2f} tokens, {t_steps / steps * 1e3:.1f} ms a step, "
            f"{firsts / steps:.1%} of the first guesses right, {1 + right / steps:.2f} tokens kept a step, {(steps + right) / t_steps:.2f} tok/s",
            flush=True)
    print("  ms a step of tokens: " + ", ".join(f"{T} {np.mean(v) * 1e3:.1f} ({len(v)})" for T, v in eng.step_s.items() if v), flush=True)
