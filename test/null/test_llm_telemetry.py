import json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tinygrad.llm.telemetry import Rolling, Telemetry, context_bucket, prefill_bucket
from tinygrad.llm.engine import Seq, SeqEngine


class TestRollingTelemetry(unittest.TestCase):
  def test_windows_histogram_weights_and_expiry(self):
    rolling = Rolling()
    rolling.add("x", 0.005, 3, at=1000.0)
    rolling.add("x", 0.020, 4, at=1001.0)
    with patch("time.time", return_value=1001.1):
      row = rolling.snapshot(10)["x"]
      self.assertEqual(row["events"], 2)
      self.assertAlmostEqual(row["time_s"], 0.025)
      self.assertEqual(row["tokens"], 7)
      self.assertEqual(row["p50_s"], 0.005)
      self.assertEqual(row["p90_s"], 0.020)
      self.assertEqual(row["hist_s"]["le_0.005"], 1)
      self.assertEqual(row["hist_s"]["le_0.02"], 1)
    rolling.add("x", 0.1, at=1701.0)
    with patch("time.time", return_value=1701.1):
      self.assertEqual(rolling.snapshot(10)["x"]["events"], 1)
      self.assertNotIn("x", rolling.snapshot(0.05))

  def test_buckets(self):
    self.assertEqual([context_bucket(x) for x in (4096, 4097, 65536, 65537, 262144, 262145)],
                     ["04k", "16k", "64k", "128k", "262k", "262k+"])
    self.assertEqual([prefill_bucket(x) for x in (1023, 1024, 4096, 16384)],
                     ["lt1k", "1k_4k", "4k_16k", "16k_plus"])

  def test_step_prefill_and_shares(self):
    telemetry = Telemetry()
    telemetry.step(16, 70000, 0.030, 40)
    telemetry.prefill(2000, 0.010, True)
    status = telemetry.snapshot()
    step = status["step_time"]["10m"]["batch=16|context=128k"]
    self.assertEqual(step["events"], 1)
    self.assertEqual(step["tokens_per_event"], 40)
    self.assertEqual(status["active_share"]["10m"]["decode_pct"], 75.0)
    self.assertEqual(status["active_share"]["10m"]["prefill_pct"], 25.0)
    self.assertEqual(status["prefill"]["10m"]["warm|1k_4k|batch=1"]["tokens"], 2000)
    json.dumps(status)  # every status value is JSON serializable

  def test_cache_cause_tracking(self):
    telemetry = Telemetry()
    telemetry.cache_observe("s", 1000, 262144, 0, False)
    self.assertEqual(telemetry.cache_miss["never_seen"]["tokens"], 1000)
    telemetry.cache_observe("s", 2000, 262144, 0, False)
    self.assertEqual(telemetry.cache_miss["evicted"]["tokens"], 1000)
    telemetry.cache_observe("s", 3000, 262144, 100, True)
    self.assertEqual(telemetry.cache_miss["rotated_to_stash_restored"]["tokens"], 2900)
    telemetry.cache_observe("s", 262000, 262144, 250000, False)
    self.assertEqual(telemetry.cache_miss["length"]["requests"], 1)
    telemetry.sessions["old"] = {"boot_id": "previous", "prompt_tokens": 5000}
    telemetry.cache_observe("old", 6000, 262144, 0, False)
    self.assertEqual(telemetry.cache_miss["restart"]["tokens"], 5000)
    self.assertEqual(set(telemetry.cache_miss), set(Telemetry.CACHE_CAUSES))

  def test_storage_units(self):
    telemetry, model = Telemetry(), Mock(stash_chunk_nbytes=100, stash_entry_nbytes=20)
    telemetry.storage_event("stash_in", 300, 3, True, model, slot=2)
    self.assertEqual(telemetry.counters["stash_in_bytes"], 320)
    self.assertEqual(telemetry.counters["stash_in_tokens"], 300)


class TestTelemetryPersistence(unittest.TestCase):
  def test_request_jsonl_and_restart_recovery(self):
    with tempfile.TemporaryDirectory() as directory:
      path = str(Path(directory) / "server-events.jsonl")
      first = Telemetry(path)
      req = SimpleNamespace(rid=9, session_id="agent-1", bench_tag="swe-pro/iq4", ids=list(range(100)), cache_start=90,
                            generated=[1, 2, 3], steps=2, queued=0.25, prefill_time=0.5, max_gap=0.1,
                            pt=102.0, st=100.0, finish_reason="tool_calls", cancelled=False)
      first.finish(req, 4.0, None)
      rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
      self.assertEqual(rows[-1]["event"], "request_end")
      self.assertEqual(rows[-1]["request_id"], 9)
      self.assertEqual(rows[-1]["session_id"], "agent-1")
      self.assertEqual(rows[-1]["ttft_s"], 2.0)
      self.assertNotIn("ids", rows[-1])
      self.assertNotIn("messages", rows[-1])
      second = Telemetry(path)
      self.assertIn("agent-1", second.sessions)
      second.cache_observe("agent-1", 120, 262144, 0, False)
      self.assertEqual(second.cache_miss["restart"]["tokens"], 100)


class TestEngineTelemetry(unittest.TestCase):
  def test_capacity_and_pause_events_preserve_counts(self):
    victim = Seq(slot=2, base=0, cap=128, tokens=[1, 2, 3], n=3)
    engine = SeqEngine.__new__(SeqEngine)
    engine.seqs, engine.evictions, engine.lost, engine.version = [victim], 0, 0, 0
    engine.telemetry, engine._stash = Mock(), Mock(return_value=True)
    engine._evict_all([victim], False)
    self.assertEqual(engine.evictions, 1)
    self.assertEqual(engine.lost, 0)
    engine.telemetry.event.assert_called_once_with("eviction", cause="capacity", slot=2, tokens=3, stashed=True)

    paused = Seq(slot=3, base=128, cap=128, tokens=[4, 5], n=2)
    engine.seqs = [paused]
    engine.pause(paused)
    self.assertEqual(engine.evictions, 2)
    engine.telemetry.event.assert_called_with("eviction", cause="paused_length", slot=3, tokens=2, stashed=True)


if __name__ == "__main__":
  unittest.main()
