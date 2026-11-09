# LLM serving telemetry

`tinygrad.llm` telemetry is **default on**, bounded, and has no model/numerics effect. It records counts, IDs, timestamps and timings—not prompts, generated token IDs, message bodies or model state. Optional JSONL is enabled with:

```bash
TELEMETRY_LOG=/raid/logs/qwen-events.jsonl python3 -m tinygrad.llm ...
```

Use a fast persistent filesystem. The log is append-only and opened per event, so do not point it at a slow root/USB drive for a high-rate benchmark.

## Logs

- `batch --` lines begin with a UTC ISO-8601 timestamp with milliseconds.
- `/v1/chat/completions` and `/v1/messages` model-log lines begin with a timestamp and include internal `rid`, sanitized `sid`, and `bench` fields.
- `X-Session-ID` is carried into these fields and `/status` active rows. The optional `X-Bench-Tag` HTTP header is echoed as `bench`; use values such as `swe-pro/iq4xs/attempt-3`. Whitespace is replaced with `_`, and each tag is limited to 96 characters. These IDs do not enter the model prompt.

Example shape:

```text
2026-11-09T13:42:01.123Z /v1/chat/completions -- rid:42 sid:abc123 bench:matrix/C3 -- reqs:4 ...
2026-11-09T13:42:06.123Z batch -- 16 requests decoding, 2 waiting ...
```

## `/status.telemetry`

All duration values and histogram bins are **seconds (`_s`)**, not milliseconds. Percentiles are upper bounds of the duration bins (`hist_s`), with nearest-rank selection. `-1` means above the final finite 1800-second bin. `tokens_per_event` for step rows is decoded tokens per batch step (including speculative acceptance separately from the step duration).

```json
{
  "schema": 1,
  "boot_id": "process-random-id",
  "started_at": "...Z",
  "step_time": {
    "1m": {
      "batch=16|context=64k": {
        "events": 123,
        "time_s": 3.69,
        "mean_s": 0.03,
        "p50_s": 0.03,
        "p90_s": 0.04,
        "p99_s": 0.05,
        "tokens": 2952,
        "tokens_per_event": 24,
        "hist_s": {"le_0.03": 100, "le_0.04": 20, "...": 3}
      }
    },
    "10m": {}
  },
  "active_share": {
    "1m": {"decode_s": 40, "prefill_s": 10, "decode_pct": 80, "prefill_pct": 20},
    "10m": {}
  },
  "prefill": {
    "1m": {"warm|1k_4k|batch=1": {"events": 5, "time_s": 1.2, "tokens": 10000}},
    "10m": {}
  },
  "requests": {
    "1m": {"token_gap": {"p50_s": 0.1, "p90_s": 0.5}, "queue": {}, "ttft": {}, "total": {}},
    "10m": {}
  },
  "cache_miss": {
    "evicted": {"requests": 0, "tokens": 0},
    "rotated_to_stash_restored": {"requests": 0, "tokens": 0},
    "restart": {"requests": 0, "tokens": 0},
    "never_seen": {"requests": 0, "tokens": 0},
    "length": {"requests": 0, "tokens": 0},
    "other": {"requests": 0, "tokens": 0}
  },
  "storage": {"stash_in_bytes": 123, "stash_out_bytes": 100, "...": 0}
}
```

- **Context buckets:** `04k`, `16k`, `64k`, `128k`, `262k`, `262k+`.
- **Prefill chunk classes:** `lt1k`, `1k_4k`, `4k_16k`, `16k_plus`, each warm/cold and batch cardinality. Qwen sequence slices have exact per-slice time. A GLM batched prefill is one time observation with `batch=N`, not N replicated durations.
- **active_share:** decode/prefill active seconds for each rolling window. It is not exclusive whole-GPU utilization; mixed generic batch work can contain both.
- **requests:** maximum token gap per completed request, queue, TTFT and total time distributions over 1/10 minutes. Existing `resident`, `stash`, and `capacity` status fields supply tokens, rows and sequences live at query time.
- **storage:** monotonic per-boot counters: `{stash_in,stash_out}_{events,tokens,chunks,bytes}`, plus event/log counters. `stash_in` is GPU-to-host, `stash_out` host-to-GPU. Stash chunks/state entries and their exact model byte sizes are exposed by Qwen and GLM sequence models.

## Cache-cause attribution

A cache observation happens after sequence admission, when cached and uncached prefix lengths are known:

| Cause | Meaning |
|---|---|
| `never_seen` | Cold prefill without prior session metadata in this process or recovered JSONL history; token count is all newly prefilled input |
| `evicted` | Same-boot prior session whose observed old-prefix reuse is less than expected |
| `rotated_to_stash_restored` | Actual `SeqEngine` stash restore followed by more prefill; includes new-turn prefill as well as old-prefix work |
| `restart` | Returning session metadata recovered from the previous `boot_id` in JSONL, but its in-memory prefix is no longer resident |
| `length` | Cache observation at/within 2,048 tokens of `max_context`, where length/truncation is the limiting cause class |
| `other` | Reserved initialized class for future observed causes; not fabricated from a guess |

Without `X-Session-ID`, a cold first request is `never_seen`; more detailed same-client resume attribution is intentionally not guessed. Without persistent `TELEMETRY_LOG`, restart and never-seen cannot be distinguished: `cache_attribution.attribution_limited=true`. At startup the server reads at most the last 4 MiB/10,000 lines of the event log to recover only session ID, prior boot ID and prompt length. At most 10,000 recent sessions are retained live.

## JSONL events

Every row includes `schema:1`, `event`, UTC `timestamp`, and `boot_id`.

- `request_end`: `request_id`, session/bench IDs, prompt/cached/uncached/output token counts, steps, queue/prefill/TTFT/max-gap/total seconds, finish reason, cancellation and error. Includes successful and failed/cancelled completions.
- `cache_observe`: actual admitted cache prefix, cause and previous boot ID where known.
- `eviction`: `capacity` or `paused_length`, sequence slot/tokens and whether a stash copy exists.
- `stash_in` / `stash_out`: token/chunk/entry counts, exact host-copy bytes where the model exposes units, and a `bytes_exact` flag.
- `stash_drop`: host stash copy discarded (resident copy or entire conversation).

The JSONL file itself has no automatic rotation: the operator should rotate/size it as part of the run manifest. Restart recovery tolerates a partial trailing line from a crash.

## Resource bounds and overhead

- Decode steps record one fixed-bin observation plus one decode-total observation using fixed cardinality `(batch, context)`.
- Prefills and requests use small fixed key sets and the same per-second histogram implementation.
- Only the current 600 one-second buckets are retained per series (one-minute views are a subset).
- `/status` aggregates those bounded histograms on demand; recording does not sort or write files.
- Persistent I/O happens only when `TELEMETRY_LOG` is explicitly set.

## Validation

```bash
python -m unittest -v test.null.test_llm_server test.null.test_llm_telemetry
ruff check tinygrad/llm/telemetry.py tinygrad/llm/serve.py tinygrad/llm/engine.py \
  test/null/test_llm_telemetry.py test/null/test_llm_server.py
python -m py_compile tinygrad/llm/serve.py tinygrad/llm/engine.py tinygrad/llm/telemetry.py
```
