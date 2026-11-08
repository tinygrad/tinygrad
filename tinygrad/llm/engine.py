from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import cast
from tinygrad.helpers import round_up, getenv

def _trace(model):
  """(SEQ_TRACE) every seq_ call of the model printed when it starts and ends (the last one started of a hang is its)"""
  import functools, sys
  def short(a):
    if isinstance(a, list) and a and isinstance(a[0], tuple) and len(a[0]) >= 4:
      return f"[{len(a)} seqs: " + " ".join(f"{t[0]}@{t[2]}+{t[3]}" for t in a) + "]"
    if isinstance(a, list): return f"<{len(a)} ints>"
    return repr(a)
  for name in [n for n in dir(model) if n.startswith("seq_") and callable(getattr(model, n))]:
    fn = getattr(model, name)
    @functools.wraps(fn)
    def w(*a, _fn=fn, _name=name, **kw):
      st = time.perf_counter()
      print(f"TRACE {st:.3f} {_name}({', '.join(short(x) for x in a)}{', ' if kw else ''}{kw or ''})", file=sys.stderr, flush=True)
      r = _fn(*a, **kw)
      print(f"TRACE {time.perf_counter():.3f} {_name} done {1000*(time.perf_counter()-st):.1f} ms", file=sys.stderr, flush=True)
      return r
    setattr(model, name, w)

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

def starts(ids:list[int], toks:list[int], n:int, exact:bool=True) -> bool:
  """
  ids[:n] == toks[:n] (a few tokens compared first: another conversation's differ there, not copied). not exact: 16 tokens of the n compared
  (an estimate, for an order: a long prompt's copies are ~0.5 ms)
  """
  if n <= 0 or len(ids) < n or len(toks) < n or ids[n-1] != toks[n-1] or ids[n//2] != toks[n//2]: return n <= 0
  if not exact: return all(ids[p] == toks[p] for p in range(0, n, max(1, n // 16)))
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
  # (the room for a reply a region has past its prompt when it's made (a request's max_tokens, RESERVE at most: more is made as it's needed),
  # and an idle one's for its next turn)
  ROOM, RESERVE = 2048, getenv("SEQ_RESERVE", 4096)
  # (an idle conversation used in the last STICKY s is held: its agent runs a tool, its next turn comes soon. a prompt of another conversation
  # doesn't make it lost (out of the caches and the stash), unless it's forced (it waited too long, see the server's STARVE))
  STICKY = getenv("SEQ_STICKY", 180.0)
  def __init__(self, model):
    from tinygrad.llm.qwen4exp import REGION_ALIGN
    self.model, self.k, self.align = model, model.config.spec_drafts, REGION_ALIGN
    if getenv("SEQ_TRACE"): _trace(model)
    self.rows, self.nslots = model.seq_rows(), model.seq_slots()
    self.seqs:list[Seq] = []
    # (the sequences decoding and prefilled at once: SEQ_MAX_ACTIVE, the largest step's by default)
    self.max_active, self.evictions, self.lost = min(max(model.SEQ_SIZES), getenv("SEQ_MAX_ACTIVE", max(model.SEQ_SIZES))), 0, 0
    # (changed when what a prompt could take does: a sequence taken, stopped or evicted (a prompt that didn't fit is tried again after that)
    self.version = 0
    # (the stash: the free chunks (in order) and entries, the evicted sequences in it)
    self.free_chunks, self.free_entries = list(range(getattr(model, "stash_chunks", 0))), list(range(getattr(model, "stash_entries", 0)))
    self.stashed:list[Stashed] = []
    self.times:dict[str, float|int] = {}  # (the time of the moves, stashes and restores: see the server's status line)

  def held(self, used:float) -> bool: return time.perf_counter() - used < self.STICKY

  # ** regions and slots
  def _gaps(self, skip:Seq|None=None) -> list[tuple[int, int]]:
    """the free (start, size) ranges of rows"""
    taken, gaps, at = sorted((s.base, s.cap) for s in self.seqs if s is not skip), [], 0
    for b, c in taken:
      if b > at: gaps.append((at, b - at))
      at = max(at, b + c)
    if at < self.rows: gaps.append((at, self.rows - at))
    return gaps
  def _chunks(self, n:int) -> int: return -(-n // self.model.SEQ_MOVE)
  def _victims(self, rows:int, slot:bool, keep:Seq|None=None) -> list[Seq]|None:
    """
    the idle sequences evicted for `rows` free rows (in all: the regions compacted) and (slot) a free slot: the ones not held first, the least
    recently used first. None: the idle ones aren't enough
    """
    idle = sorted((s for s in self.seqs if not s.active and s is not keep), key=lambda s: (self.held(s.used), s.used))
    free, slots, out = self.rows - sum(s.cap for s in self.seqs), self.nslots - len(self.seqs), []
    while free < rows or (slot and slots <= 0):
      if not idle: return None
      out.append(v:=idle.pop(0))
      free, slots = free + v.cap, slots + 1
    return out
  def _stash_fits(self, victims:list[Seq]) -> bool:
    """the held ones of victims fit in the stash, no held one there dropped for them (the ones not held, the residents' copies are)"""
    if not (held:=[v for v in victims if self.held(v.used)]): return True
    if not self.free_entries and not self.stashed and not any(v.host for v in held): return False
    others = [t.host for t in self.seqs if t.host is not None and t not in victims] + [x for x in self.stashed if not self.held(x.used)]
    chunks, entries = len(self.free_chunks) + sum(len(x.chunks) for x in others), len(self.free_entries) + len(others)
    for v in held:
      have = len(v.host.chunks) if v.host is not None else 0
      chunks -= max(0, self._chunks(v.n) - have)
      entries -= v.host is None
    return chunks >= 0 and entries >= 0
  def _make_room(self, rows:int, slot:bool, keep:Seq|None=None, force:bool=False) -> bool:
    """
    `rows` free rows (in all) and (slot) a free slot: the idle sequences evicted for them (into the stash, the held ones first). checked before
    anything is evicted: False (nothing evicted) if they can't be, or a held one would be lost (not forced)
    """
    if (victims:=self._victims(rows, slot, keep)) is None or (not force and not self._stash_fits(victims)): return False
    self._evict_all(victims, force)
    return True
  def _evict_all(self, victims:list[Seq], force:bool):
    for v in sorted(victims, key=lambda v: not self.held(v.used)):
      self.seqs.remove(v)
      self.evictions += 1
      if not self._stash(v, force): self.lost += 1
    self.version += 1
  def _window(self, need:int, force:bool, after:Seq|None=None) -> tuple[int, list[Seq]]|None:
    """
    a run of free rows and idle regions of need rows (after: the one right after its region), nothing moved: its first row and the idle ones
    evicted for it (the fewest held ones, then the fewest rows of them). None: none (a held one would be lost, not forced)
    """
    regs = sorted(self.seqs, key=lambda t: t.base)
    ends = [0] + [t.base + t.cap for t in regs]
    gaps = [t.base - ends[i] for i, t in enumerate(regs)] + [self.rows - ends[-1]]
    best:tuple[tuple[bool, int], int, list[Seq]]|None = None
    for a in ([regs.index(after) + 1] if after is not None else range(len(regs) + 1)):
      total, victims = 0, cast(list[Seq], [])
      for b in range(a, len(regs) + 1):
        if (total:=total + gaps[b]) >= need:
          key = (any(self.held(v.used) for v in victims), sum(v.n for v in victims))
          if (best is None or key < best[0]) and (force or self._stash_fits(victims)): best = (key, ends[a], victims)
          break
        if b == len(regs) or regs[b].active: break
        victims.append(regs[b])
        total += regs[b].cap
    return best[1:] if best is not None else None
  def _stash(self, s:Seq, force:bool=False) -> bool:
    """
    s (evicted) into the stash: its chunks its copy there doesn't have (a resident's copy, of the prefix it shares) and its states, the least
    recently used copies out for it (the residents' first: they're in the caches, then the ones not held, the held ones if forced). False if it
    doesn't fit
    """
    C, n, h = self.model.SEQ_MOVE, self._chunks(s.n), s.host
    if s.n <= 0 or not (self.free_entries or self.stashed or h): return False
    s.host = None
    chunks, entry, keep = (h.chunks, h.entry, min(common_prefix(h.tokens, s.tokens[:s.n]) // C, len(h.chunks))) if h else ([], -1, 0)
    if n < len(chunks): self._free(chunks[n:], None)
    chunks = chunks[:n]
    while len(self.free_chunks) < n - len(chunks) or (entry < 0 and not self.free_entries):
      if self._drop_one(force) is None:
        if h: self._free(chunks, entry)
        return False
    chunks, entry = chunks + self._take(n - len(chunks)), entry if entry >= 0 else self.free_entries.pop()
    st = time.perf_counter()
    self.model.seq_stash(True, s.slot, s.base, self._segs(chunks, keep, n), entry, s.n, s.hrow)
    self._time("stash", st)
    self.times["stashed"] = self.times.get("stashed", 0) + n - keep
    self.stashed.append(Stashed(s.tokens[:s.n], chunks, entry, s.used, s.hrow, s.ckpt))
    return True
  def _drop_one(self, held:bool=True) -> Stashed|None:
    """the least recently used copy in the stash out (a resident's first, then the ones not held, then (held) the held ones), None if none"""
    residents = [t for t in self.seqs if t.host is not None]
    if residents:
      t = min(residents, key=lambda t: t.used)
      x, t.host = cast(Stashed, t.host), None
    elif (cands:=[x for x in self.stashed if held or not self.held(x.used)]):
      self.stashed.remove(x:=min(cands, key=lambda x: (self.held(x.used), x.used)))
      self.lost += 1
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
  def _pack(self, need:int, fixed:Seq|None=None) -> tuple[int, int, int]|None:
    """
    the cheapest gap of need rows (the free rows in all are enough): a run of regions packed down onto the one before them merges the gaps
    between them into one after them. (the rows moved, the run's first and end index of the regions by base), None: none (fixed doesn't move)
    """
    regs = sorted(self.seqs, key=lambda t: t.base)
    ends = [0] + [t.base + t.cap for t in regs]
    gaps = [t.base - ends[i] for i, t in enumerate(regs)] + [self.rows - ends[-1]]
    best:tuple[int, int, int]|None = None
    for a in range(len(regs) + 1):
      total, cost = 0, 0
      for b in range(a, len(regs) + 1):
        if b > a:
          if regs[b-1] is fixed: break
          cost += regs[b-1].n
        if best is not None and cost >= best[0]: break
        if (total:=total + gaps[b]) >= need:
          best = (cost, a, b)
          break
    return best
  def _packed(self, a:int, b:int) -> int:
    """the regions a..b (by base) packed down onto the one before them: returns the first row of the gap after them"""
    regs = sorted(self.seqs, key=lambda t: t.base)
    at = regs[a-1].base + regs[a-1].cap if a else 0
    for t in regs[a:b]:
      self._move(t, at)
      at += t.cap
    return at
  def _place(self, s:Seq, need:int):
    """a region of need rows for s (not a resident one yet, the free rows in all are enough): a gap, else the fewest rows moved for one"""
    _, a, b = cast(tuple[int, int, int], self._pack(need))
    s.base, s.cap = self._packed(a, b), need

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
  def room(self, s:Seq, length:int, force:bool=True) -> bool:
    """the region of s has rows for a sequence of `length` tokens (grown in place, moved, or the regions compacted, if not): False if no region
    does (the idle sequences evicted for it, held ones too if forced: a reply decoding goes before them)"""
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
    # (idle ones right after it evicted, nothing moved)
    if (win:=self._window(need - s.cap, force, after=s)) is not None:
      self._evict_all(win[1], force)
      s.cap = need
      return True
    if not self._make_room(need - s.cap, False, s, force): return False
    # (the fewest rows moved: the regions after s packed up for its growth (the last first), or s moved into a gap made for it)
    delta, after = need - s.cap, sorted((t for t in self.seqs if t.base > s.base), key=lambda t: t.base)
    at, total, up = s.base + s.cap, 0, None
    for i in range(len(after) + 1):
      if (total:=total + (after[i].base if i < len(after) else self.rows) - at) >= delta:
        up = (sum(t.n for t in after[:i]), i)
        break
      if i < len(after): at = after[i].base + after[i].cap
    other = self._pack(need, fixed=s)
    if up is not None and (other is None or up[0] <= other[0] + s.n):
      end = after[up[1]].base if up[1] < len(after) else self.rows
      for t in reversed(after[:up[1]]):
        end -= t.cap
        self._move(t, end)
    elif other is not None: self._move(s, self._packed(*other[1:]))
    else:
      # (the free rows only with its own: the regions compacted, the ones after s moved up by its growth, the last first)
      order = self._compact()
      for t in reversed(order[order.index(s)+1:]): self._move(t, t.base + delta)
    s.cap = need
    return True

  # ** a prompt's sequence
  def take(self, ids:list[int], room:int=0, force:bool=False) -> tuple[Seq, int, bool]|None:
    """
    the sequence of a prompt and the position its prefill starts at, and whether that's its checkpoint (restored): an idle one it continues, else
    a new one (from 0), with rows for a reply of `room` tokens (RESERVE at most, 0: RESERVE). None: no slot or region for it without losing a held
    conversation (not forced), or every slot active: nothing evicted, it waits
    """
    if (ret:=self._take_seq(ids, room, force)) is not None:
      self.version += 1
      # (the continuations from a conversation's end and from its checkpoint (the reply not sent back as it was): see the status line)
      if ret[1]: self.times["from_ckpt" if ret[2] else "from_end"] = self.times.get("from_ckpt" if ret[2] else "from_end", 0) + 1
    return ret
  def _take_seq(self, ids:list[int], room:int, force:bool) -> tuple[Seq, int, bool]|None:
    # (no slot or rows for it even with every idle one out: none of its prefix compared)
    if sum(t.active for t in self.seqs) >= self.nslots or self._need(len(ids)) > self.rows - sum(t.cap for t in self.seqs if t.active): return None
    best:tuple[Seq, int, bool]|None = None
    for s in self.seqs:
      if s.active: continue
      if s.n and s.n < len(ids) and starts(ids, s.tokens, s.n): cand = (s, s.n, False)
      # (its checkpoint: a prompt that continues it, or is it (the same prompt again: the token after it sampled from it, nothing prefilled))
      elif s.ckpt and s.ckpt <= len(ids) and starts(ids, s.tokens, s.ckpt): cand = (s, s.ckpt, True)
      else: continue
      if best is None or cand[1] > best[1]: best = cand
    # (a stashed one it continues further: back from host memory, prefilled from its end, or its checkpoint)
    def scont(x:Stashed) -> int:
      if len(x.tokens) < len(ids) and starts(ids, x.tokens, len(x.tokens)): return len(x.tokens)
      return x.ckpt if 0 < x.ckpt <= len(ids) and starts(ids, x.tokens, x.ckpt) else 0
    stashed = max(((n, i, x) for i, x in enumerate(self.stashed) if (n:=scont(x))), default=(0, 0, None))[2]
    # (the rows of its prompt and reply. more than the active ones leave even with every idle one out: its prompt's, the reply's grown later)
    length = len(ids) + min(room or self.RESERVE, self.RESERVE)
    if self._need(length) > self.rows - sum(t.cap for t in self.seqs if t.active): length = len(ids)
    if best is not None and (stashed is None or best[1] >= scont(stashed)):
      s, start, restore = best
      s.active = True  # (not evicted for its own room)
      if self.room(s, length, force): return best
      s.active = False
      return None
    # (the stashed one out of the stash while its slot and region are made: the evictions don't stash over it)
    start = scont(stashed) if stashed is not None else 0
    if stashed is not None: self.stashed.remove(stashed)
    # (a run of free rows and idle ones evicted, nothing moved: else the least recently used ones evicted, the fewest rows moved for a region)
    need, base = self._need(length), None
    if need <= self.rows and (win:=self._window(need, force)) is not None and (win[1] or len(self.seqs) < self.nslots):
      self._evict_all(win[1], force)
      base = win[0]
    elif need > self.rows or not self._make_room(need, True, None, force):
      if stashed is not None: self.stashed.append(stashed)
      return None
    s = Seq(min(set(range(self.nslots)) - {t.slot for t in self.seqs}), 0, 0, active=True)
    if base is None: self._place(s, need)
    else: s.base, s.cap = base, need
    self.seqs.append(s)
    if stashed is None: return s, 0, False
    n, st = len(stashed.tokens), time.perf_counter()
    self.model.seq_stash(False, s.slot, s.base, self._segs(stashed.chunks, 0, len(stashed.chunks)), stashed.entry, n, stashed.hrow)
    self._time("restore", st)
    self.times["restored"] = self.times.get("restored", 0) + len(stashed.chunks)
    s.host = stashed  # (its copy stays: evicted again, only what it writes since is copied)
    self.model._token_host[s.base:s.base+n] = __import__("array").array('i', stashed.tokens)
    # (as it was: prefilled from its last token's states, or its checkpoint restored)
    s.tokens, s.n, s.hrow, s.ckpt = list(stashed.tokens), n, stashed.hrow, stashed.ckpt
    return s, start, start < n or start == len(ids)

  def cached(self, ids:list[int]) -> int:
    """the tokens of a prompt in the caches or the stash (about: the position take would prefill it from, no sequence taken)"""
    def cont(tokens:list[int], n:int, ckpt:int) -> int:
      if n and n < len(ids) and starts(ids, tokens, n, exact=False): return n
      return ckpt if ckpt and ckpt <= len(ids) and starts(ids, tokens, ckpt, exact=False) else 0
    return max([cont(s.tokens, s.n, s.ckpt) for s in self.seqs if not s.active] +
               [cont(x.tokens, len(x.tokens), x.ckpt) for x in self.stashed], default=0)

  def prefill_slice(self, s:Seq, ids:list[int], start:int, restore:bool, end:int):
    """the tokens of the prompt ids from start to end prefilled in s (its checkpoint restored first): the rest is prefilled from end"""
    if restore: self.model.seq_ckpt(False, s.slot, s.ckpt)
    self.model.seq_prefill(s.slot, s.base, ids, start, 0 if restore else s.hrow, upto=end)
    s.tokens, s.n, s.hrow, s.used = ids[:end], end, 0, time.perf_counter()

  def settle(self):
    """the last slice of a prefill (prefill_slice: not waited for) done"""
    if (wait:=getattr(self.model, "_slice_wait", None)) is not None: wait()

  def prefill(self, s:Seq, ids:list[int], start:int, restore:bool, temp:tuple[float, float]) -> int:
    """the prompt ids prefilled in s from start (its checkpoint restored first): the token sampled after it"""
    if restore: self.model.seq_ckpt(False, s.slot, s.ckpt)
    # (a prompt that is its checkpoint: the token after it from the hidden state of its last token, restored)
    if restore and start == len(ids): first = self.model.seq_first(s.slot, s.base, start, temp)
    else:
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

  def forget(self):
    """the idle sequences and the stash dropped (their states may be wrong): every prompt is prefilled from the start"""
    self.version += 1
    for s in [s for s in self.seqs if not s.active]: self.seqs.remove(s)
    while self._drop_one() is not None: pass

  def fits(self, length:int) -> bool: return self._need(length) <= self.rows
  def pause(self, s:Seq):
    """s (decoding) out of the caches, into the stash if it fits: its slot and region free, a prompt of its tokens goes on from it"""
    s.active, s.used, self.version = False, time.perf_counter(), self.version + 1
    self.seqs.remove(s)
    self.evictions += 1
    if not self._stash(s, True): self.lost += 1

  def stop(self, s:Seq, kept:int):
    """the end of the decoding of s: its tokens past `kept` (of its reply, after its end) aren't its"""
    self.version += 1
    if kept < len(s.tokens):
      # (the caches have the tokens before the last one kept: the hidden state of the one before it is the step's row of it)
      drop = len(s.tokens) - kept
      s.tokens, s.n, s.hrow = s.tokens[:kept], s.n - drop, s.hrow - drop
    s.active, s.used = False, time.perf_counter()
    # (idle: its region no longer than its tokens and the room of a turn need, the rest free)
    s.cap = min(s.cap, self._need(len(s.tokens) + self.ROOM))
