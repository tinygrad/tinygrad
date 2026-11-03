from __future__ import annotations
import time
from dataclasses import dataclass, field
from tinygrad.helpers import getenv, round_up

# ******** sequences sharing the model's caches (Transformer.seq_*): the host's side ********
# every resident sequence (a conversation) has a slot of the recurrent states and a region of rows of the attention caches. a prompt continues
# the idle sequence it extends (prefilled from its end, or from its checkpoint: the end of its last prompt), else a new one, the least recently
# used idle ones evicted for its slot and region. steps decode every active sequence at once

@dataclass(eq=False)
class Seq:
  slot:int
  base:int
  cap:int
  tokens:list[int] = field(default_factory=list)  # its tokens: the ones in the caches, then the last (sampled, not in them yet)
  n:int = 0  # the tokens in the caches
  hrow:int = 0  # the slot's row of the hidden state of the last one in the caches
  ckpt:int = 0  # the position of its checkpoint (the end of its last prompt, the states there kept), 0: none
  used:float = 0.0
  active:bool = False
  # decoding: the position the MTP layer catches up from, the drafts accepted, the parity, the last step's sampled tokens, the temperature
  mtp:int = 0
  j:int = 0
  par:int = 1
  y:list[int] = field(default_factory=list)
  temp:tuple[float, float] = (0.0, 1.0)

def common_prefix(a:list[int], b:list[int]) -> int:
  n = min(len(a), len(b))
  if a[:n] == b[:n]: return n
  lo, hi = 0, n  # (a[:lo] == b[:lo], a[:hi] != b[:hi])
  while hi - lo > 1:
    mid = (lo + hi) // 2
    if a[:mid] == b[:mid]: lo = mid
    else: hi = mid
  return lo

class SeqEngine:
  # (the room for a reply a region has past its prompt when it's made, more is made as it's needed)
  ROOM = getenv("SEQ_ROOM", 2048)
  def __init__(self, model):
    from tinygrad.llm.model import REGION_ALIGN
    self.model, self.k, self.align = model, model.config.spec_drafts, REGION_ALIGN
    self.rows, self.nslots = model.seq_rows(), model.seq_slots()
    self.seqs:list[Seq] = []
    self.max_active, self.evictions = max(model.SEQ_SIZES), 0

  # ** regions and slots
  def _gaps(self, skip:Seq|None=None) -> list[tuple[int, int]]:
    """the free (start, size) ranges of rows"""
    taken, gaps, at = sorted((s.base, s.cap) for s in self.seqs if s is not skip), [], 0
    for b, c in taken:
      if b > at: gaps.append((at, b - at))
      at = max(at, b + c)
    if at < self.rows: gaps.append((at, self.rows - at))
    return gaps
  def _evict(self, keep:Seq|None=None) -> bool:
    """the least recently used idle sequence out (its slot and region free): False if none"""
    idle = [s for s in self.seqs if not s.active and s is not keep]
    if not idle: return False
    self.seqs.remove(out:=min(idle, key=lambda s: s.used))
    self.evictions += 1
    if getenv("SEQ_LOG", 0): print(f"seq slot {out.slot} evicted ({out.n} tokens)", flush=True)
    return True
  def _place(self, s:Seq, need:int) -> bool:
    """a free region of need rows for s (not a resident one yet): the regions compacted, the idle sequences evicted, for it"""
    while need <= self.rows:
      if (b:=next((b for b, c in self._gaps() if c >= need), None)) is not None:
        s.base, s.cap = b, need
        return True
      if self.rows - sum(t.cap for t in self.seqs) >= need: self._compact()
      elif not self._evict(): return False
    return False
  def _slot(self) -> int|None:
    while len(self.seqs) >= self.nslots:
      if not self._evict(): return None
    return min(set(range(self.nslots)) - {s.slot for s in self.seqs})

  def _need(self, length:int) -> int:
    # (a step writes 2*(k+1) positions past the last token at most, a pool past that)
    return round_up(length + 2*(self.k + 1) + 8, self.align)
  def _move(self, s:Seq, base:int):
    if base != s.base: self.model.seq_move(s.base, base, s.n)
    s.base = base
  def _compact(self) -> list[Seq]:
    """the regions one after another from row 0, in their order (a region moved down never overwrites the ones after it): the free rows at the
    end. returns the sequences in order"""
    at, order = 0, sorted(self.seqs, key=lambda t: t.base)
    for t in order:
      self._move(t, at)
      at += t.cap
    return order
  def room(self, s:Seq, length:int) -> bool:
    """the region of s has rows for a sequence of `length` tokens (grown in place, moved, or the regions compacted, if not): False if no region
    does (the idle sequences evicted for it)"""
    need = self._need(length)
    if need <= s.cap: return True
    if need > self.rows: return False
    end = next((b for b, _ in sorted((t.base, t.cap) for t in self.seqs if t is not s) if b > s.base), self.rows)
    if s.base + need <= end:
      s.cap = need
      return True
    if (base:=next((b for b, c in self._gaps(s) if c >= need and (b >= s.base + s.cap or b + need <= s.base)), None)) is not None:
      self._move(s, base)
      s.cap = need
      return True
    # (all the free rows: the regions compacted, the ones after s moved up by its growth, the last first)
    while self.rows - sum(t.cap for t in self.seqs) < need - s.cap:
      if not self._evict(s): return False
    order = self._compact()
    delta = need - s.cap
    for t in reversed(order[order.index(s)+1:]): self._move(t, t.base + delta)
    s.cap = need
    return True

  # ** a prompt's sequence
  def take(self, ids:list[int], room:int=0) -> tuple[Seq, int, bool]|None:
    """
    the sequence of a prompt and the position its prefill starts at, and whether that's its checkpoint (restored): an idle one it continues, else
    a new one (from 0). None: no slot or region for it (every slot active)
    """
    best:tuple[Seq, int, bool]|None = None
    for s in self.seqs:
      if s.active: continue
      if s.n and s.n < len(ids) and ids[:s.n] == s.tokens[:s.n]: cand = (s, s.n, False)
      elif s.ckpt and s.ckpt < len(ids) and ids[:s.ckpt] == s.tokens[:s.ckpt]: cand = (s, s.ckpt, True)
      else: continue
      if best is None or cand[1] > best[1]: best = cand
    if best is not None:
      s, start, restore = best
      s.active = True  # (not evicted for its own room)
      if self.room(s, len(ids) + room): return best
      s.active = False
      self.seqs.remove(s)
    if (slot:=self._slot()) is None: return None
    s = Seq(slot, 0, 0, active=True)
    if not self._place(s, self._need(len(ids) + min(room, self.ROOM))) and not self._place(s, self._need(len(ids))): return None
    self.seqs.append(s)
    return s, 0, False

  def prefill(self, s:Seq, ids:list[int], start:int, restore:bool, temp:tuple[float, float]) -> int:
    """the prompt ids prefilled in s from start (its checkpoint restored first): the token sampled after it"""
    if restore: self.model.seq_ckpt(False, s.slot, s.ckpt)
    first = self.model.seq_prefill(s.slot, s.base, ids, start, 0 if restore else s.hrow, temp)
    # (the end of the prompt: the checkpoint of the states there)
    self.model.seq_ckpt(True, s.slot, len(ids))
    s.tokens, s.n, s.hrow, s.ckpt, s.used = ids + [first], len(ids), 0, len(ids), time.perf_counter()
    s.mtp, s.j, s.par, s.y, s.temp = len(ids), 0, 1, [first] + [0]*self.k, temp
    return first

  def step(self, seqs:list[Seq]) -> list[list[int]]:
    """a step of decoding of the sequences: the tokens every one accepted (its caches have the ones before them)"""
    k, out = self.k, self.model.seq_step([(s.slot, s.base, s.mtp, s.j, s.par, s.y, s.temp) for s in seqs])
    ret = []
    for s, (sampled, drafts) in zip(seqs, out):
      j = next((i for i in range(k) if drafts[i] != sampled[i]), k)
      start = s.mtp + s.j
      # (the token buffer: the accepted ones, the n-gram rows of the next step read them)
      self.model._token_host[s.base + start + 1:s.base + start + 2 + j] = __import__("array").array('i', sampled[:j+1])
      s.mtp, s.j, s.par, s.y = start + 1, j, 1 - s.par, sampled
      s.tokens += sampled[:j+1]
      s.n, s.hrow = start + j + 1, (1 - s.par)*(k+1) + j
      ret.append(sampled[:j+1])
    return ret

  def stop(self, s:Seq, kept:int):
    """the end of the decoding of s: its tokens past `kept` (of its reply, after its end) aren't its"""
    if kept < len(s.tokens):
      # (the caches have the tokens before the last one kept: the hidden state of the one before it is the step's row of it)
      drop = len(s.tokens) - kept
      s.tokens, s.n, s.hrow = s.tokens[:kept], s.n - drop, s.hrow - drop
    s.active, s.used = False, time.perf_counter()
    # (idle: its region no longer than its tokens need, the rest free)
    s.cap = min(s.cap, self._need(len(s.tokens)))
