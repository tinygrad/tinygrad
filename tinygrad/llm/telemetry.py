"""Bounded serving telemetry: no token text, prompts, or model-state access.

Time values are seconds unless a field name says otherwise. Histogram percentiles
are upper bounds of their duration bins. The implementation retains at most the
current 600 one-second buckets; cardinality is bounded by the small key space.
"""
from __future__ import annotations
import bisect, collections, datetime, json, os, threading, time, uuid
from pathlib import Path
from typing import Any

BINS = (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05,
        0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0)


def utc_ms(epoch: float | None = None) -> str:
  return datetime.datetime.fromtimestamp(time.time() if epoch is None else epoch, datetime.timezone.utc) \
    .isoformat(timespec="milliseconds").replace("+00:00", "Z")


def context_bucket(tokens: int) -> str:
  for n, name in ((4096, "04k"), (16384, "16k"), (65536, "64k"), (131072, "128k"), (262144, "262k")):
    if tokens <= n: return name
  return "262k+"


def prefill_bucket(tokens: int) -> str:
  if tokens < 1024: return "lt1k"
  if tokens < 4096: return "1k_4k"
  if tokens < 16384: return "4k_16k"
  return "16k_plus"


class Rolling:
  """Histograms of (value, weight), keyed and aggregated independently."""
  def __init__(self, retention: int = 600):
    self.retention, self._lock = retention, threading.Lock()
    self.buckets: dict[int, dict[str, list[Any]]] = {}

  def add(self, key: str, value: float, weight: int = 0, at: float | None = None) -> None:
    second = int(time.time() if at is None else at)
    bi = bisect.bisect_left(BINS, value)
    with self._lock:
      if len(self.buckets) > self.retention + 2:
        for old in [s for s in self.buckets if second - s > self.retention]: del self.buckets[old]
      bucket = self.buckets.setdefault(second, {})
      hist = bucket.get(key)
      if hist is None: hist = bucket[key] = [0, 0.0, 0, [0] * (len(BINS) + 1)]
      hist[0] += 1
      hist[1] += value
      hist[2] += weight
      hist[3][bi] += 1

  @staticmethod
  def _pct(counts: list[int], total: int, p: float) -> float:
    want = max(1, int(total * p + 0.999999999))  # nearest rank, ceil(N*p)
    at = 0
    for i, count in enumerate(counts):
      at += count
      if at >= want: return BINS[i] if i < len(BINS) else -1.0  # -1: above the last finite bin
    return 0.0

  def snapshot(self, window: int) -> dict[str, dict[str, Any]]:
    cutoff = time.time() - window
    grouped: dict[str, list] = {}
    with self._lock:
      buckets = [bucket for second, bucket in self.buckets.items() if second >= cutoff]
      for bucket in buckets:
        for key, hist in bucket.items():
          out = grouped.get(key)
          if out is None: out = grouped[key] = [0, 0.0, 0, [0] * (len(BINS) + 1)]
          out[0] += hist[0]
          out[1] += hist[1]
          out[2] += hist[2]
          for i, count in enumerate(hist[3]): out[3][i] += count
    ret = {}
    for key, (count, total, weight, bins) in grouped.items():
      ret[key] = {"events": count, "time_s": round(total, 6), "mean_s": round(total / count, 6) if count else 0,
                  "p50_s": self._pct(bins, count, 0.50), "p90_s": self._pct(bins, count, 0.90),
                  "p99_s": self._pct(bins, count, 0.99), "tokens": weight,
                  "tokens_per_event": round(weight / count, 4) if count else 0,
                  "hist_s": {**{f"le_{BINS[i]:g}": n for i, n in enumerate(bins[:-1])}, f"gt_{BINS[-1]:g}": bins[-1]}}
    return ret


class Telemetry:
  CACHE_CAUSES = ("evicted", "rotated_to_stash_restored", "restart", "never_seen", "length", "other")

  def __init__(self, log_path: str = ""):
    self.boot_id, self.boot_time, self.log_path = uuid.uuid4().hex[:12], time.time(), log_path
    self.steps, self.prefills, self.requests = Rolling(), Rolling(), Rolling()
    self.counters: dict[str, int] = collections.Counter()
    self.cache_miss: dict[str, dict[str, int]] = {cause: {"requests": 0, "tokens": 0} for cause in self.CACHE_CAUSES}
    self.sessions: dict[str, dict[str, Any]] = {}
    self._io_lock = threading.Lock()
    self._load_session_history()

  def _load_session_history(self) -> None:
    """Recover only metadata needed for restart attribution, never message/token content."""
    if not self.log_path or not (path := Path(self.log_path)).exists(): return
    try:
      with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 4 * 1024 * 1024))
        lines = f.read().decode(errors="replace").splitlines()[-10000:]
      for line in lines:
        try: event = json.loads(line)
        except json.JSONDecodeError: continue
        if event.get("event") == "request_end" and (sid := event.get("session_id")) and event.get("boot_id") != self.boot_id:
          self.sessions[sid] = {"boot_id": event["boot_id"], "prompt_tokens": event.get("prompt_tokens", 0)}
    except OSError:
      pass

  def event(self, kind: str, **fields: Any) -> None:
    """Optionally append one schema-versioned event; I/O is never on the model hot path unless explicitly enabled."""
    self.counters[f"event_{kind}"] += 1
    if not self.log_path: return
    row = {"schema": 1, "event": kind, "timestamp": utc_ms(), "boot_id": self.boot_id, **fields}
    with self._io_lock:
      try:
        with open(self.log_path, "a", encoding="utf-8") as f:
          f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
      except OSError as err:
        self.counters["log_errors"] += 1
        if self.counters["log_errors"] == 1: print(f"telemetry log error: {err}", file=__import__("sys").stderr)

  def step(self, batch: int, context_tokens: int, seconds: float, tokens: int) -> None:
    bucket = context_bucket(context_tokens)
    self.steps.add(f"batch={batch}|context={bucket}", seconds, tokens)
    self.steps.add("decode", seconds, tokens)

  def prefill(self, new_tokens: int, seconds: float, warm: bool, batch: int = 1) -> None:
    # A batch's seconds are one observation (not replicated once per member).
    key = f"{'warm' if warm else 'cold'}|{prefill_bucket(new_tokens)}|batch={batch}"
    self.prefills.add(key, seconds, new_tokens)
    self.prefills.add("prefill", seconds, new_tokens)

  def _remember_session(self, sid: str, prompt_tokens: int) -> None:
    # Bounded LRU metadata; drop/reinsert so a hit is recent as well as newly seen.
    self.sessions.pop(sid, None)
    self.sessions[sid] = {"boot_id": self.boot_id, "prompt_tokens": prompt_tokens}
    while len(self.sessions) > 10000: self.sessions.pop(next(iter(self.sessions)))

  def cache_observe(self, sid: str, prompt_tokens: int, max_context: int, cached: int, restored: bool) -> None:
    prior = self.sessions.get(sid) if sid else None
    new = max(0, prompt_tokens - cached)
    if new == 0:
      cause = None
    elif restored:
      cause = "rotated_to_stash_restored"
    elif prompt_tokens >= max_context - 2048:
      cause = "length"
    elif prior is None:
      cause = "never_seen"
    elif prior.get("boot_id") != self.boot_id:
      cause = "restart"
    elif cached < min(prior.get("prompt_tokens", 0), prompt_tokens):
      cause = "evicted"
    else:
      cause = None
    if cause is not None:
      self.cache_miss[cause]["requests"] += 1
      # Never-seen is a cold-prefill count; the other counts estimate old work not reused.
      prior_tokens = prior.get("prompt_tokens", 0) if prior else 0
      missing = new if cause in ("never_seen", "rotated_to_stash_restored") else max(0, min(prompt_tokens, prior_tokens) - cached)
      self.cache_miss[cause]["tokens"] += missing
      self.event("cache_observe", session_id=sid or None, cause=cause, prompt_tokens=prompt_tokens,
                 cached_tokens=cached, uncached_tokens=new, previous_boot_id=prior.get("boot_id") if prior else None)
    if sid: self._remember_session(sid, prompt_tokens)

  def finish(self, req: Any, seconds: float, error: Exception | None) -> None:
    ttft = (req.pt or seconds + req.st) - req.st
    self.requests.add("token_gap", req.max_gap)
    self.requests.add("queue", req.queued)
    self.requests.add("ttft", ttft)
    self.requests.add("total", seconds, len(req.generated))
    bench = getattr(req, "bench_tag", "")
    if sid := getattr(req, "session_id", ""): self._remember_session(sid, len(req.ids))
    self.event("request_end", request_id=req.rid, session_id=getattr(req, "session_id", "") or None,
               bench_tag=bench or None, prompt_tokens=len(req.ids), cached_tokens=req.cache_start,
               uncached_prompt_tokens=len(req.ids) - req.cache_start, completion_tokens=len(req.generated),
               steps=req.steps, queue_s=req.queued, prefill_s=req.prefill_time, ttft_s=ttft,
               max_token_gap_s=req.max_gap, total_s=seconds, finish_reason=req.finish_reason,
               error=f"{type(error).__name__}: {error}" if error else None, cancelled=req.cancelled)

  def storage_event(self, kind: str, tokens: int, chunks: int, entry: bool, model: Any, **fields: Any) -> None:
    """Record logical storage movement and compute bytes from model-provided sizes when available."""
    chunk_bytes, entry_bytes = getattr(model, "stash_chunk_nbytes", 0), getattr(model, "stash_entry_nbytes", 0)
    nbytes = chunks * chunk_bytes + int(entry) * entry_bytes
    self.counters[f"{kind}_events"] += 1
    self.counters[f"{kind}_tokens"] += tokens
    self.counters[f"{kind}_chunks"] += chunks
    self.counters[f"{kind}_bytes"] += nbytes
    self.event(kind, tokens=tokens, chunks=chunks, entry=entry, bytes=nbytes,
               bytes_exact=bool((not chunks or chunk_bytes) and (not entry or entry_bytes)), **fields)

  def snapshot(self) -> dict[str, Any]:
    step1, step10 = self.steps.snapshot(60), self.steps.snapshot(600)
    pre1, pre10 = self.prefills.snapshot(60), self.prefills.snapshot(600)
    req1, req10 = self.requests.snapshot(60), self.requests.snapshot(600)
    def shares(step: dict, prefill: dict) -> dict[str, Any]:
      decode, pre = step.get("decode", {}).get("time_s", 0), prefill.get("prefill", {}).get("time_s", 0)
      total = decode + pre
      return {"decode_s": decode, "prefill_s": pre,
              "decode_pct": round(100 * decode / total, 3) if total else None,
              "prefill_pct": round(100 * pre / total, 3) if total else None,
              "note": "Measured active seconds, not exclusive whole-GPU utilization; generic batch prefills may mix work."}
    return {"schema": 1, "boot_id": self.boot_id, "started_at": utc_ms(self.boot_time),
            "histogram_percentiles": "upper bound of duration bin; -1 means above 1800s",
            "step_time": {"1m": step1, "10m": step10}, "active_share": {"1m": shares(step1, pre1), "10m": shares(step10, pre10)},
            "prefill": {"1m": pre1, "10m": pre10}, "requests": {"1m": req1, "10m": req10},
            "cache_miss": self.cache_miss,
            "cache_attribution": {"restart_history": bool(self.log_path),
                                  "attribution_limited": not bool(self.log_path),
                                  "note": "TELEMETRY_LOG persists bounded prior-boot session metadata. "
                                          "Without it restart and never-seen cannot be distinguished."},
            "storage": dict(self.counters)}
