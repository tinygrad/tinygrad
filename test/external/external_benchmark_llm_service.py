#!/usr/bin/env python3
# the latencies of an OpenAI compatible server (tinygrad.llm --serve) under the loads of coding agents:
#   matrix: cold long prompts at once (unique numbers: no shared prefix), each sent again right after (all of it cached), their TTFT and totals
#   agent:  agents' conversations: a task with a file, then turns that add the reply and a tool's output (a file) to the history. the cached
#           tokens of a turn are the ones of the history before it (its prompt continues the last one)
#   probe:  a short streaming request while long prompts are prefilled: its TTFT (a request isn't behind long prompts)
# python3 test/external/external_benchmark_llm_service.py [matrix|agent|probe] [--url http://localhost:8000/v1] [--sizes 16000,64000,128000]
import argparse, concurrent.futures, glob, json, hashlib, random, statistics, threading, time, urllib.request

def stream(url:str, body:dict) -> dict:
  """a streaming chat completion: its TTFT, total, usage, and the reply (content, reasoning, tool calls)"""
  body = {**body, "stream": True, "stream_options": {"include_usage": True}}
  req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
  st, first, usage, content, reasoning, calls = time.perf_counter(), None, {}, [], [], []
  with urllib.request.urlopen(req, timeout=1800) as r:
    for raw in r:
      if not raw.startswith(b"data: ") or raw[6:].strip() == b"[DONE]": continue
      e = json.loads(raw[6:])
      for c in e.get("choices", []):
        d = c.get("delta", {})
        if (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")) and first is None: first = time.perf_counter()
        content.append(d.get("content") or ""); reasoning.append(d.get("reasoning_content") or "")  # noqa: E702
        calls += d.get("tool_calls") or []
      if e.get("usage"): usage = e["usage"]
  et = time.perf_counter()
  return {"ttft": (first or et) - st, "total": et - st, "prompt": usage.get("prompt_tokens", 0),
          "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0), "completion": usage.get("completion_tokens", 0),
          "content": "".join(content), "reasoning": "".join(reasoning), "tool_calls": calls}

def pct(xs:list[float], q:float) -> float: return sorted(xs)[min(len(xs)-1, int(q * len(xs)))]
def numbers(seed:int, tokens:int) -> str:
  rng = random.Random(seed)
  return f"Unique request {hashlib.sha256(str(seed).encode()).hexdigest()[:16]}. Ignore the numbers and reply OK.\n" + \
         " ".join(str(rng.randrange(10000, 100000)) for _ in range(tokens // 6))

def matrix(args):
  for size in [int(x) for x in args.sizes.split(",")]:
    for conc in [int(x) for x in args.concurrency.split(",")]:
      seed = int(time.time() * 1000)
      def pair(i:int) -> tuple[dict, dict]:
        body = {"model": "x", "max_tokens": 64, "temperature": 1.0, "messages": [{"role": "user", "content": numbers(seed + i, size)}]}
        return stream(args.url, body), stream(args.url, body)
      st = time.perf_counter()
      with concurrent.futures.ThreadPoolExecutor(conc) as ex: res = list(ex.map(pair, range(conc)))
      wall = time.perf_counter() - st
      cold, warm = [c for c, _ in res], [w for _, w in res]
      print(f"{size:7d} tokens x{conc}: cold ttft p50 {statistics.median(r['ttft'] for r in cold):6.2f}s "
            f"p90 {pct([r['ttft'] for r in cold], .9):6.2f}s "
            f"total p90 {pct([r['total'] for r in cold], .9):6.2f}s | repeat ttft p50 {statistics.median(r['ttft'] for r in warm):5.2f}s, "
            f"{sum(r['cached'] for r in warm)/max(1, sum(r['prompt'] for r in warm)):4.0%} cached | "
            f"{sum(r['prompt'] - r['cached'] for r in cold + warm)/wall:6.0f} tok/s prefilled", flush=True)

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file of the repository",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
def agent(args):
  files = sorted(glob.glob(args.files))
  texts = [open(f, errors="replace").read() for f in files]
  def run(a:int) -> list[dict]:
    rng = random.Random(a)
    pick = lambda n: "".join(rng.choice(texts) for _ in range(n))[:args.file_chars]  # noqa: E731
    msgs = [{"role": "system", "content": f"You are coding agent {a}. Use read_file to look at the repository, then answer briefly."},
            {"role": "user", "content": f"Here is the main module:\n```python\n{pick(8)}\n```\nFind the function that parses its arguments."}]
    out = []
    for t in range(args.turns):
      r = stream(args.url, {"model": "x", "max_tokens": args.reply_tokens, "temperature": 0.7, "messages": msgs, "tools": TOOLS})
      out.append(r)
      # the reply into the history as an agent sends it back (its reasoning kept or not), then the tool's output
      reply:dict = {"role": "assistant", "content": r["content"] or ""}
      if args.keep_reasoning and r["reasoning"]: reply["reasoning_content"] = r["reasoning"]
      call_id = f"call_{a}_{t}"
      fn = {"name": "read_file", "arguments": json.dumps({"path": f"f{t}.py"})}
      reply["tool_calls"] = [{"id": call_id, "type": "function", "function": fn}]
      msgs += [reply, {"role": "tool", "tool_call_id": call_id, "content": pick(2)[:args.tool_chars]}]
    return out
  with concurrent.futures.ThreadPoolExecutor(args.agents) as ex: res = list(ex.map(run, range(args.agents)))
  for t in range(args.turns):
    rs = [r[t] for r in res]
    print(f"turn {t}: prompt {statistics.mean(r['prompt'] for r in rs):8.0f} tokens, cached {statistics.mean(r['cached'] for r in rs):8.0f} "
          f"({sum(r['cached'] for r in rs)/max(1, sum(r['prompt'] for r in rs)):4.0%}), ttft p50 {statistics.median(r['ttft'] for r in rs):6.2f}s "
          f"max {max(r['ttft'] for r in rs):6.2f}s, total p50 {statistics.median(r['total'] for r in rs):6.2f}s", flush=True)

def probe(args):
  size, conc = int(args.sizes.split(",")[-1]), int(args.concurrency.split(",")[-1])
  seed, done = int(time.time() * 1000), threading.Event()
  def long(i:int):
    stream(args.url, {"model": "x", "max_tokens": 16, "messages": [{"role": "user", "content": numbers(seed + i, size)}]})
  threads = [threading.Thread(target=long, args=(i,)) for i in range(conc)]
  for th in threads: th.start()
  time.sleep(2)
  r = stream(args.url, {"model": "x", "max_tokens": 64, "messages": [{"role": "user", "content": "Count from 1 to 20."}]})
  print(f"probe while {conc} prompts of {size} tokens are prefilled: ttft {r['ttft']:.2f}s, total {r['total']:.2f}s", flush=True)
  for th in threads: th.join()
  done.set()

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("mode", choices=["matrix", "agent", "probe"])
  p.add_argument("--url", default="http://localhost:8000/v1")
  p.add_argument("--sizes", default="16000,64000,128000")
  p.add_argument("--concurrency", default="1,4,8")
  p.add_argument("--agents", type=int, default=4)
  p.add_argument("--turns", type=int, default=6)
  p.add_argument("--reply_tokens", type=int, default=300)
  p.add_argument("--files", default="/usr/lib/python3*/*.py")
  p.add_argument("--file_chars", type=int, default=120000)
  p.add_argument("--tool_chars", type=int, default=20000)
  p.add_argument("--keep_reasoning", type=int, default=1)
  args = p.parse_args()
  {"matrix": matrix, "agent": agent, "probe": probe}[args.mode](args)
