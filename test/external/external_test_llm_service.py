#!/usr/bin/env python3
# service tests against the server: python3 svc_test.py [host:port] [only-test-names...]
import http.client, json, sys, time, threading, re, glob, random
HOST = sys.argv[1] if len(sys.argv) > 1 else "localhost:8000"
ONLY = set(sys.argv[2:])
results = []

def post(path, body, timeout=120, stream=False):
  h, p = HOST.split(":")
  c = http.client.HTTPConnection(h, int(p), timeout=timeout)
  c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
  r = c.getresponse()
  if stream: return c, r
  data = r.read()
  c.close()
  return r.status, data

def chat(msgs, max_tokens=300, temperature=0, **kw):
  st = time.perf_counter()
  status, data = post("/v1/chat/completions", {"model": "q", "messages": msgs, "max_tokens": max_tokens, "temperature": temperature, **kw})
  assert status == 200, f"status {status}: {data[:300]!r}"
  j = json.loads(data)
  return j, time.perf_counter() - st

def content(j): return j["choices"][0]["message"].get("content") or ""

def test(fn):
  if ONLY and fn.__name__ not in ONLY: return fn
  st = time.perf_counter()
  try:
    info = fn()
    results.append((fn.__name__, True, info))
    print(f"PASS {fn.__name__} ({time.perf_counter()-st:.1f}s) {info or ''}", flush=True)
  except Exception as e:
    results.append((fn.__name__, False, str(e)))
    print(f"FAIL {fn.__name__} ({time.perf_counter()-st:.1f}s) {type(e).__name__}: {str(e)[:400]}", flush=True)
  return fn

def code_text(n_chars):
  t = "".join(open(f).read() for f in sorted(glob.glob("/usr/lib/python3.12/[a-c]*.py")))
  return t[:n_chars]

@test
def basic():
  j, t = chat([{"role": "user", "content": "What is 17*23? Answer with just the number."}])
  assert content(j).strip().endswith("391"), repr(content(j)[-100:])
  u = j["usage"]
  assert u["prompt_tokens"] > 0 and u["completion_tokens"] > 0 and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"], u
  assert j["choices"][0]["finish_reason"] == "stop", j["choices"][0]["finish_reason"]
  return f"{t:.2f}s"

@test
def streaming():
  c, r = post("/v1/chat/completions", {"model": "q", "messages": [{"role": "user", "content": "What is 12*12? Answer with just the number."}],
                                       "max_tokens": 300, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}, stream=True)
  assert r.status == 200
  text, finish, usage, n = "", None, None, 0
  for line in r:
    line = line.decode().strip()
    if not line.startswith("data:"): continue
    d = line[5:].strip()
    if d == "[DONE]": break
    j = json.loads(d)
    n += 1
    if j.get("usage"): usage = j["usage"]
    for ch in j.get("choices", []):
      text += ch.get("delta", {}).get("content") or ""
      finish = ch.get("finish_reason") or finish
  c.close()
  assert text.strip().endswith("144"), repr(text[-100:])
  assert finish == "stop" and usage and usage["completion_tokens"] > 0, (finish, usage)
  return f"{n} chunks"

@test
def max_tokens_length():
  j, _ = chat([{"role": "user", "content": "Write a long story about a dragon."}], max_tokens=20)
  assert j["choices"][0]["finish_reason"] == "length", j["choices"][0]["finish_reason"]
  assert j["usage"]["completion_tokens"] == 20, j["usage"]

@test
def long_decode():
  # 1000 tokens of counting: the numbers in order, as many tokens as asked
  j, t = chat([{"role": "user", "content": "Count from 1 to 4000, one number per line. Do not stop early, do not explain."}],
              max_tokens=1000, chat_template_kwargs={"enable_thinking": False})
  nums = [int(x) for x in re.findall(r"^\s*(\d+)\s*$", content(j), re.M)]
  assert j["usage"]["completion_tokens"] == 1000 or j["choices"][0]["finish_reason"] == "stop", j["usage"]
  if nums:
    bad = [i for i in range(1, len(nums)) if nums[i] != nums[i-1] + 1]
    assert not bad, f"not consecutive at {bad[:5]}: {nums[max(0,bad[0]-3):bad[0]+3]}"
  n = j['usage']['completion_tokens']
  return f"{n} tokens in {t:.1f}s ({n/t:.0f} tok/s), counted to {nums[-1] if nums else None}"

def prefill_case(chars):
  text = code_text(chars)
  msgs = [{"role": "user", "content": "Here is some code:\n" + text + "\n\nReply with just the word OK."}]
  times = []
  for rep in range(2):
    # (a different first line every rep: not the cached conversation)
    m = [{"role": "user", "content": f"[{random.random()}]\n" + msgs[0]["content"]}]
    j, t = chat(m, max_tokens=1)
    times.append((t, j["usage"]["prompt_tokens"]))
  return ", ".join(f"{p} tok {t:.2f}s ({p/t:.0f} tok/s)" for t, p in times)

@test
def prefill_100(): return prefill_case(300)
@test
def prefill_1k(): return prefill_case(3500)
@test
def prefill_8k(): return prefill_case(28000)
@test
def prefill_16k(): return prefill_case(56000)

@test
def cached_repeat():
  # the same prompt twice at temperature 0: the same reply (the second from the cached conversation)
  q = "What is the name of the first function defined? Answer briefly."
  msgs = [{"role": "user", "content": "Here is some code:\n" + code_text(20000) + "\n\n" + q}]
  a, ta = chat(msgs, max_tokens=60)
  b, tb = chat(msgs, max_tokens=60)
  assert content(a) == content(b), (content(a)[:200], content(b)[:200])
  return f"{ta:.2f}s then {tb:.2f}s"

@test
def multi_turn():
  msgs = [{"role": "user", "content": "Remember the number 4817. Reply OK."}]
  j, _ = chat(msgs, max_tokens=200)
  msgs += [{"role": "assistant", "content": content(j)}, {"role": "user", "content": "What number did I ask you to remember? Just the number."}]
  j, _ = chat(msgs, max_tokens=200)
  assert "4817" in content(j), repr(content(j))

@test
def concurrent():
  # 16 at once (a batch), every answer right
  pairs = [(random.randint(10, 99), random.randint(10, 99)) for _ in range(16)]
  out, errs = {}, []
  def one(i, a, b):
    try:
      j, t = chat([{"role": "user", "content": f"What is {a}*{b}? Answer with just the number."}], max_tokens=400)
      out[i] = content(j)
    except Exception as e: errs.append(repr(e))
  ts = [threading.Thread(target=one, args=(i, a, b)) for i, (a, b) in enumerate(pairs)]
  st = time.perf_counter()
  for t in ts: t.start()
  for t in ts: t.join()
  assert not errs, errs[:3]
  wrong = [(a, b, out.get(i, "")[-30:]) for i, (a, b) in enumerate(pairs) if not out.get(i, "").strip().endswith(str(a*b))]
  assert len(wrong) <= 1, f"{len(wrong)} wrong: {wrong[:4]}"
  return f"{time.perf_counter()-st:.1f}s, {len(wrong)} wrong"

def long_body(N_LONG):
  # long generations at once with different prompts: no corruption (every one counts in order)
  errs, outs = [], {}
  def one(i):
    try:
      j, _ = chat([{"role": "user", "content": f"Count from {i*1000} to {i*1000+3000}, one number per line. No explanation."}], max_tokens=600,
                  chat_template_kwargs={"enable_thinking": False})
      # (cut at max_tokens: its last line may be a number's start)
      outs[i] = content(j).rsplit("\n", 1)[0] if j["choices"][0]["finish_reason"] == "length" else content(j)
    except Exception as e: errs.append(repr(e))
  ts = [threading.Thread(target=one, args=(i,)) for i in range(N_LONG)]
  for t in ts: t.start()
  for t in ts: t.join()
  assert not errs, errs[:3]
  for i, text in outs.items():
    nums = [int(x) for x in re.findall(r"^\s*(\d+)\s*$", text, re.M)]
    assert len(nums) > 50, (i, text[:200])
    bad = [k for k in range(1, len(nums)) if nums[k] != nums[k-1] + 1]
    assert not bad, f"seq {i} not consecutive at {nums[max(0,bad[0]-3):bad[0]+3]}"
  return f"{len(outs)} ok"

@test
def concurrent_long(): return long_body(8)
@test
def concurrent_long24(): return long_body(24)

@test
def mixed_load():
  # long decodes while prompts of a few thousand tokens come: both right
  errs, res = [], {}
  def dec(i):
    try:
      j, _ = chat([{"role": "user", "content": f"Count from {i*100} to {i*100+2000}, one number per line. No explanation."}], max_tokens=500,
                  chat_template_kwargs={"enable_thinking": False})
      t = content(j).rsplit("\n", 1)[0]
      nums = [int(x) for x in re.findall(r"^\s*(\d+)\s*$", t, re.M)]
      res[("d", i)] = len(nums) > 50 and all(nums[k] == nums[k-1] + 1 for k in range(1, len(nums)))
    except Exception as e: errs.append(repr(e))
  def pre(i):
    try:
      a, b = random.randint(100, 999), random.randint(100, 999)
      j, _ = chat([{"role": "user", "content": code_text(10000 + 1000*i)[1000*i:] + f"\n\nIgnore the code. What is {a}+{b}? Just the number."}],
                  max_tokens=400)
      res[("p", i)] = content(j).strip().endswith(str(a+b))
    except Exception as e: errs.append(repr(e))
  ts = [threading.Thread(target=dec, args=(i,)) for i in range(12)] + [threading.Thread(target=pre, args=(i,)) for i in range(8)]
  for t in ts:
    t.start()
    time.sleep(0.05)
  for t in ts: t.join()
  assert not errs, errs[:3]
  bad = [k for k, v in res.items() if not v]
  assert len(bad) <= 1, bad
  return f"{len(res)} done, bad {bad}"

@test
def disconnect_then_ok():
  # a client gone mid stream: the next request is fine
  c, r = post("/v1/chat/completions", {"model": "q", "messages": [{"role": "user", "content": "Write a very long essay about rivers."}],
                                       "max_tokens": 2000, "temperature": 0, "stream": True}, stream=True)
  for i, line in enumerate(r):
    if i > 20: break
  c.close()
  time.sleep(1)
  j, _ = chat([{"role": "user", "content": "What is 9*9? Answer with just the number."}])
  assert content(j).strip().endswith("81"), repr(content(j)[-50:])

@test
def too_long():
  status, data = post("/v1/chat/completions", {"model": "q", "messages": [{"role": "user", "content": "x " * 300000}], "max_tokens": 5})
  assert status == 400, (status, data[:200])

@test
def tool_call():
  tools = [{"type": "function", "function": {"name": "get_weather", "description": "the weather of a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
  j, _ = chat([{"role": "user", "content": "What's the weather in Paris? Use the tool."}], max_tokens=500, tools=tools)
  ch = j["choices"][0]
  tc = ch["message"].get("tool_calls")
  assert tc and tc[0]["function"]["name"] == "get_weather" and "Paris" in json.loads(tc[0]["function"]["arguments"])["city"], ch
  assert ch["finish_reason"] == "tool_calls", ch["finish_reason"]

@test
def anthropic_messages():
  status, data = post("/v1/messages", {"model": "q", "max_tokens": 300, "messages": [{"role": "user", "content": "What is 6*7? Just the number."}]})
  assert status == 200, (status, data[:200])
  j = json.loads(data)
  text = "".join(b.get("text", "") for b in j["content"] if b["type"] == "text")
  assert text.strip().endswith("42"), j

@test
def still_ok():
  j, _ = chat([{"role": "user", "content": "What is 17*23? Answer with just the number."}])
  assert content(j).strip().endswith("391"), repr(content(j)[-100:])


@test
def long_context_200k():
  # a prompt of ~200k tokens: prefilled, the answer from its start (the needle), then a short request is fine
  hay = code_text(10**7)
  text = hay[:700000]
  st = time.perf_counter()
  status, data = post("/v1/chat/completions", {"model": "q", "max_tokens": 300, "temperature": 0, "messages": [{"role": "user", "content":
                      "The secret word is PELICAN.\n" + text + "\n\nWhat is the secret word given at the very start? Just the word."}]}, timeout=300)
  t = time.perf_counter() - st
  assert status == 200, (status, data[:300])
  j = json.loads(data)
  assert "PELICAN" in content(j).upper(), repr(content(j)[-200:])
  return f"{j['usage']['prompt_tokens']} tokens in {t:.1f}s"

print(f"\n{sum(ok for _, ok, _ in results)} of {len(results)} passed", flush=True)
