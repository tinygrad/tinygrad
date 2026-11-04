from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import cast
from tinygrad.helpers import getenv, round_up

# ******** sequences sharing the model's caches (Transformer.seq_*): the host's side ********
# every resident sequence (a conversation) has a slot of the recurrent states and a region of rows of the attention caches. a prompt continues
# the idle sequence it extends (prefilled from its end, or from its checkpoint: the end of its last prompt), else a new one, the least recently
# used idle ones evicted for its slot and region. steps decode every active sequence at once

@dataclass(eq=False)
class Stashed:
  """
  a sequence's copy in the stash (host memory, see Transformer.seq_stash): its tokens, their chunks (in order) and the entry of its states, its
  row. an evicted one's, or a resident one's (its prefix: the chunks it hasn't written since are copied when it's evicted again)
  """
  tokens:list[int]
  chunks:list[int]
  entry:int
  used:float
  hrow:int
  ckpt:int

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
  host:Stashed|None = None  # (its copy in the stash: of the prefix of its tokens it shares)
  # decoding: the position the MTP layer catches up from, the drafts accepted, the parity, the last step's sampled tokens, the temperature
  mtp:int = 0
  j:int = 0
  par:int = 1
  y:list[int] = field(default_factory=list)
  temp:tuple[float, float] = (0.0, 1.0)

def starts(ids:list[int], toks:list[int], n:int) -> bool:
  """ids[:n] == toks[:n] (a few tokens compared first: another conversation's differ there, not copied)"""
  if n <= 0 or len(ids) < n or len(toks) < n or ids[n-1] != toks[n-1] or ids[n//2] != toks[n//2]: return n <= 0
  return ids[:n] == toks[:n]

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
  EVICT_FIRST = 3  # (the idle ones evicted for a region before the regions are compacted)
  def __init__(self, model):
    from tinygrad.llm.model import REGION_ALIGN
    self.model, self.k, self.align = model, model.config.spec_drafts, REGION_ALIGN
    self.rows, self.nslots = model.seq_rows(), model.seq_slots()
    self.seqs:list[Seq] = []
    self.max_active, self.evictions = max(model.SEQ_SIZES), 0
    # (the stash: the free chunks (in order) and entries, the evicted sequences in it)
    self.free_chunks, self.free_entries = list(range(getattr(model, "stash_chunks", 0))), list(range(getattr(model, "stash_entries", 0)))
    self.stashed:list[Stashed] = []
    self.times:dict[str, float|int] = {}  # (the time of the moves, stashes and restores: see the server's status line)

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
    stashed = self._stash(out)
    if getenv("SEQ_LOG", 0): print(f"seq slot {out.slot} evicted ({out.n} tokens{', stashed' if stashed else ''})", flush=True)
    return True
  def _stash(self, s:Seq) -> bool:
    """
    s (evicted) into the stash: its chunks its copy there doesn't have (a resident's copy, of the prefix it shares) and its states, the least
    recently used copies out for it (the residents' first: they're in the caches). False if it doesn't fit
    """
    C, n, h = self.model.SEQ_MOVE, -(-s.n // self.model.SEQ_MOVE), s.host
    if s.n <= 0 or not (self.free_entries or self.stashed or h): return False
    s.host = None
    chunks, entry, keep = (h.chunks, h.entry, min(common_prefix(h.tokens, s.tokens[:s.n]) // C, len(h.chunks))) if h else ([], -1, 0)
    if n < len(chunks): self._free(chunks[n:], None)
    chunks = chunks[:n]
    while len(self.free_chunks) < n - len(chunks) or (entry < 0 and not self.free_entries):
      if self._drop_one() is None:
        if h: self._free(chunks, entry)
        return False
    chunks, entry = chunks + self._take(n - len(chunks)), entry if entry >= 0 else self.free_entries.pop()
    st = time.perf_counter()
    self.model.seq_stash(True, s.slot, s.base, self._segs(chunks, keep, n), entry, s.n, s.hrow)
    self._time("stash", st)
    self.times["stashed"] = self.times.get("stashed", 0) + n - keep
    self.stashed.append(Stashed(s.tokens[:s.n], chunks, entry, s.used, s.hrow, s.ckpt))
    return True
  def _drop_one(self) -> Stashed|None:
    """the least recently used copy in the stash out (a resident's first), None if none"""
    residents = [t for t in self.seqs if t.host is not None]
    if residents:
      t = min(residents, key=lambda t: t.used)
      x, t.host = cast(Stashed, t.host), None
    elif self.stashed: self.stashed.remove(x:=min(self.stashed, key=lambda x: x.used))
    else: return None
    self._free(x.chunks, x.entry)
    return x
  def _take(self, n:int) -> list[int]:
    """n free chunks: consecutive ones if there are (the copies' calls are of runs)"""
    f = self.free_chunks
    at = next((i for i in range(len(f) - n + 1) if f[i + n - 1] - f[i] == n - 1), 0) if n else 0
    taken, self.free_chunks = f[at:at+n], f[:at] + f[at+n:]
    return taken
  def _free(self, chunks:list[int], entry:int|None):
    self.free_chunks = sorted(self.free_chunks + chunks)
    if entry is not None and entry >= 0: self.free_entries.append(entry)
  @staticmethod
  def _segs(chunks:list[int], lo:int, hi:int) -> list[tuple[int, int, int]]:
    """the chunks lo..hi of a sequence as segments (its chunk, the stash's, count) of consecutive ones"""
    segs:list[tuple[int, int, int]] = []
    for i in range(lo, hi):
      if segs and segs[-1][0] + segs[-1][2] == i and segs[-1][1] + segs[-1][2] == chunks[i]: segs[-1] = (segs[-1][0], segs[-1][1], segs[-1][2] + 1)
      else: segs.append((i, chunks[i], 1))
    return segs
  def _place(self, s:Seq, need:int) -> bool:
    """a free region of need rows for s (not a resident one yet): the regions compacted, the idle sequences evicted, for it"""
    evicted = 0
    while need <= self.rows:
      if (b:=next((b for b, c in self._gaps() if c >= need), None)) is not None:
        s.base, s.cap = b, need
        return True
      # (with the stash: a few idle ones out before the regions are compacted (it moves every one), their copies there cheap)
      if (evicted < self.EVICT_FIRST and self.free_entries + self.stashed and self._evict()) or \
         (self.rows - sum(t.cap for t in self.seqs) < need and self._evict()): evicted += 1
      elif self.rows - sum(t.cap for t in self.seqs) >= need: self._compact()
      else: return False
    return False
  def _slot(self) -> int|None:
    while len(self.seqs) >= self.nslots:
      if not self._evict(): return None
    return min(set(range(self.nslots)) - {s.slot for s in self.seqs})

  def _need(self, length:int) -> int:
    # (a step writes 2*(k+1) positions past the last token at most, a pool past that)
    return round_up(length + 2*(self.k + 1) + 8, self.align)
  def _time(self, name:str, st:float): self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - st
  def _move(self, s:Seq, base:int):
    st = time.perf_counter()
    if base != s.base:
      self.model.seq_move(s.base, base, s.n)
      self.times["moved"] = self.times.get("moved", 0) + -(-s.n // self.align)  # (the chunks: see the status line)
    s.base = base
    self._time("move", st)
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
    evicted = 0
    while True:
      end = next((b for b, _ in sorted((t.base, t.cap) for t in self.seqs if t is not s) if b > s.base), self.rows)
      if s.base + need <= end:
        s.cap = need
        return True
      if (base:=next((b for b, c in self._gaps(s) if c >= need and (b >= s.base + s.cap or b + need <= s.base)), None)) is not None:
        self._move(s, base)
        s.cap = need
        return True
      # (with the stash: a few idle ones out before the regions are compacted, see _place)
      if not (evicted < self.EVICT_FIRST and self.free_entries + self.stashed and self._evict(s)): break
      evicted += 1
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
      if s.n and s.n < len(ids) and starts(ids, s.tokens, s.n): cand = (s, s.n, False)
      elif s.ckpt and s.ckpt < len(ids) and starts(ids, s.tokens, s.ckpt): cand = (s, s.ckpt, True)
      else: continue
      if best is None or cand[1] > best[1]: best = cand
    # (a stashed one it continues further: back from host memory, prefilled from its end, or its checkpoint)
    def scont(x:Stashed) -> int:
      if len(x.tokens) < len(ids) and starts(ids, x.tokens, len(x.tokens)): return len(x.tokens)
      return x.ckpt if 0 < x.ckpt < len(ids) and starts(ids, x.tokens, x.ckpt) else 0
    stashed = max(((n, i, x) for i, x in enumerate(self.stashed) if (n:=scont(x))), default=(0, 0, None))[2]
    if best is not None and (stashed is None or best[1] >= scont(stashed)):
      s, start, restore = best
      s.active = True  # (not evicted for its own room)
      if self.room(s, len(ids) + room): return best
      s.active = False
      self.seqs.remove(s)
    # (the stashed one out of the stash while its slot and region are made: the evictions don't stash over it)
    start = scont(stashed) if stashed is not None else 0
    if stashed is not None: self.stashed.remove(stashed)
    restored = False
    try:
      if (slot:=self._slot()) is None: return None
      s = Seq(slot, 0, 0, active=True)
      if not self._place(s, self._need(len(ids) + min(room, self.ROOM))) and not self._place(s, self._need(len(ids))): return None
      self.seqs.append(s)
      if stashed is None: return s, 0, False
      n, restored, st = len(stashed.tokens), True, time.perf_counter()
      self.model.seq_stash(False, s.slot, s.base, self._segs(stashed.chunks, 0, len(stashed.chunks)), stashed.entry, n, stashed.hrow)
      self._time("restore", st)
      self.times["restored"] = self.times.get("restored", 0) + len(stashed.chunks)
      s.host = stashed  # (its copy stays: evicted again, only what it writes since is copied)
      self.model._token_host[s.base:s.base+n] = __import__("array").array('i', stashed.tokens)
      # (as it was: prefilled from its last token's states, or its checkpoint restored)
      s.tokens, s.n, s.hrow, s.ckpt = list(stashed.tokens), n, stashed.hrow, stashed.ckpt
      if getenv("SEQ_LOG", 0): print(f"seq slot {s.slot} unstashed ({n} tokens, from {start})", flush=True)
      return s, start, start < n
    finally:
      # (back in the stash if it's not in the caches now)
      if stashed is not None and not restored: self.stashed.append(stashed)

  def prefill(self, s:Seq, ids:list[int], start:int, restore:bool, temp:tuple[float, float]) -> int:
    """the prompt ids prefilled in s from start (its checkpoint restored first): the token sampled after it"""
    if restore: self.model.seq_ckpt(False, s.slot, s.ckpt)
    first = self.model.seq_prefill(s.slot, s.base, ids, start, 0 if restore else s.hrow, temp)
    # (the end of the prompt: the checkpoint of the states there)
    self.model.seq_ckpt(True, s.slot, len(ids))
    s.tokens, s.n, s.hrow, s.ckpt, s.used = ids + [first], len(ids), 0, len(ids), time.perf_counter()
    s.mtp, s.j, s.par, s.y, s.temp = len(ids), 0, 1, [first] + [0]*self.k, temp
    return first

  # ** a mixed step: a prompt prefilled with a step of the others (see Transformer.seq_step)
  def mixable(self, n:int, ids:list[int], start:int) -> bool:
    return callable(getattr(self.model, "mix_size", None)) and self.model.mix_size(n, len(ids) - start, start) is not None
  def mix_prompt(self, s:Seq, ids:list[int], start:int, restore:bool, temp:tuple[float, float]) -> tuple:
    """the prompt of a mixed step: its tokens into the token buffer, its slot's states the model's (its checkpoint's if restore)"""
    if restore: self.model.seq_ckpt(False, s.slot, s.ckpt)
    self.model._token_host[s.base+start:s.base+len(ids)+1] = __import__("array").array('i', ids[start:] + [-1])
    self.model._seq_copy_call(False, s.slot, start, 0 if restore else s.hrow, keep=start > 0)
    return (s.slot, s.base, start, len(ids) - start, temp)
  def mix_done(self, s:Seq, ids:list[int], first:int, temp:tuple[float, float]):
    """after a mixed step: the model's states into the prompt's slot, its checkpoint, s decodes from the token after it (as prefill)"""
    self.model._seq_copy_call(True, s.slot, len(ids), 0)
    self.model.seq_ckpt(True, s.slot, len(ids))
    s.tokens, s.n, s.hrow, s.ckpt, s.used = ids + [first], len(ids), 0, len(ids), time.perf_counter()
    s.mtp, s.j, s.par, s.y, s.temp = len(ids), 0, 1, [first] + [0]*self.k, temp

  def step(self, seqs:list[Seq], prompt:tuple|None=None) -> list[list[int]]:
    """a step of decoding of the sequences: the tokens every one accepted (its caches have the ones before them)"""
    k, out = self.k, self.model.seq_step([(s.slot, s.base, s.mtp, s.j, s.par, s.y, s.temp) for s in seqs], prompt)
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

  def fits(self, length:int) -> bool: return self._need(length) <= self.rows
  def pause(self, s:Seq):
    """s (decoding) out of the caches, into the stash if it fits: its slot and region free, a prompt of its tokens goes on from it"""
    s.active, s.used = False, time.perf_counter()
    self.seqs.remove(s)
    self.evictions += 1
    stashed = self._stash(s)
    if getenv("SEQ_LOG", 0): print(f"seq slot {s.slot} paused ({s.n} tokens{', stashed' if stashed else ''})", flush=True)

  def stop(self, s:Seq, kept:int):
    """the end of the decoding of s: its tokens past `kept` (of its reply, after its end) aren't its"""
    if kept < len(s.tokens):
      # (the caches have the tokens before the last one kept: the hidden state of the one before it is the step's row of it)
      drop = len(s.tokens) - kept
      s.tokens, s.n, s.hrow = s.tokens[:kept], s.n - drop, s.hrow - drop
    s.active, s.used = False, time.perf_counter()
    # (idle: its region no longer than its tokens and the room of a turn need, the rest free)
    s.cap = min(s.cap, self._need(len(s.tokens) + self.ROOM))
