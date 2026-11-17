#!/usr/bin/env python3
# The customer smoke of a started server, through its public port (rule 1 of every start). Exit code != 0 on any failure.
# usage: python3 extra/llm_service/smoke.py <url> [--slots 64] [--rows-4k N] [--no-pictures] [--no-reasoning] [--only abcd]
#  a  20 chats with NO sampling parameters: status 200, stray-script characters in content = 0, no <think> in content, reasoning in
#     reasoning_content on every reply (--no-reasoning for a model that does not think), five printed for the eye
#  b  sampling honoured: one-token draws at T 1 / top_p 1 and T 1 / top_p 0.95: the top_p 0.95 arm draws (almost) nothing from the
#     top_p 1 arm's last 5% tail (fails if that tail is empty); a request with top_k and an unknown field is answered (200), and one
#     with presence / frequency / repetition penalties (200; this server ignores them: printed, RUNNING.md)
#  c  returning turns under pressure: N = max(80, 1.25 x slots) conversations, a secret each, then every one returns and must answer its
#     OWN secret (100%); text, then one picture each
#  d  10 pictures in one request accepted; a 3840x2160 picture's prompt tokens (checked against --rows-4k if given)
# It cannot check (e), that the start was made by the launcher exactly as RUNNING.md prints it: say that in the 'serves' line yourself.
import sys, json, re, time, random, base64, io, argparse, collections, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--slots", type=int, default=64)
ap.add_argument("--rows-4k", type=int, default=0)
ap.add_argument("--no-pictures", action="store_true")
ap.add_argument("--draws", type=int, default=300)
ap.add_argument("--max-tokens", type=int, default=8000, help="max_tokens of the part a chats (a thinking model's reply must fit)")
ap.add_argument("--ret-max-tokens", type=int, default=48, help="max_tokens of a part c reply (thinking off)")
ap.add_argument("--draw-prompt", default="Name one fruit. Reply with the fruit's name only, no other words.",
                help="part b's prompt: its first generated token must have a spread (a tail) at temperature 1")
ap.add_argument("--no-reasoning", action="store_true")
ap.add_argument("--only", default="abcd")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
URL = args.url.rstrip("/") + "/v1/chat/completions"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
fails: list[str] = []

def post(body:dict, timeout:float=900) -> tuple[int, dict]:
  req = urllib.request.Request(URL, json.dumps({"model": "x", **body}).encode(), {"Content-Type": "application/json"})
  try:
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, json.loads(r.read())
  except urllib.error.HTTPError as e:
    try: return e.code, json.loads(e.read() or b"{}")
    except ValueError: return e.code, {}
def content(r:dict) -> str: return ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
def say(*a): print(*a, flush=True)
def data_url(im) -> str:
  b = io.BytesIO()
  im.save(b, "PNG")
  return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
STRAY = re.compile(r"[\u0400-\u04FF\u0600-\u06FF\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF]")

if "a" in args.only:
  topics = ["how a bicycle gear works", "why the sky is blue", "how to make bread", "what a hash table is", "the water cycle", "how vaccines work",
            "what inflation is", "how a compiler works", "the rules of chess", "how bees make honey", "what a black hole is", "how to tie a bowline",
            "the history of the printing press", "how solar panels work", "what recursion is", "how tides work", "how a refrigerator works",
            "what DNS is", "how to brew coffee", "why leaves change colour"]
  def chat(t, mt=args.max_tokens):
    return post({"messages": [{"role": "user", "content": f"Explain {t} in about 200 words, in English."}], "max_tokens": mt})
  with ThreadPoolExecutor(8) as ex: res = list(ex.map(chat, topics))
  # (an empty reply that ran out of tokens (finish_reason length: the thinking took them all) is asked again once at twice the limit, and said)
  cut = [i for i, (c, r) in enumerate(res)
         if c == 200 and r.get("choices") and not content(r).strip() and r["choices"][0].get("finish_reason") == "length"]
  if cut:
    with ThreadPoolExecutor(8) as ex: again = list(ex.map(lambda i: chat(topics[i], 2 * args.max_tokens), cut))
    for i, cr in zip(cut, again): res[i] = cr
    say(f"A: {len(cut)} empty replies cut by max_tokens {args.max_tokens} (thinking took them all) asked again at {2 * args.max_tokens}")
  ok = [r for c, r in res if c == 200 and r.get("choices")]
  stray = sum(len(STRAY.findall(content(r))) for r in ok)
  think = sum("<think>" in content(r) or "</think>" in content(r) for r in ok)
  empty = sum(not content(r).strip() for r in ok)
  reasoning = sum(bool(r["choices"][0]["message"].get("reasoning_content")) for r in ok)
  say(f"A default sampling: {len(topics)} chats, status != 200: {len(topics) - len(ok)}, stray-script chars in content: {stray}, "
      f"<think> in content: {think}, empty content: {empty}, with reasoning_content: {reasoning}")
  for t, r in list(zip(topics, [r for _, r in res]))[:5]: say(f"  EYE [{t}]: {content(r)[:300]!r}")
  if len(ok) != len(topics) or stray or think or empty or (not args.no_reasoning and reasoning != len(ok)): fails.append("a")

if "b" in args.only:
  msgs = [{"role": "user", "content": args.draw_prompt}]
  def draw(kw): return post({"messages": msgs, "max_tokens": 1, **NOTHINK, **kw})
  # the first generated token: in content, or in reasoning_content for a model that always thinks (GLM 5.3 Flash ignores enable_thinking)
  def first(r:dict) -> str: return content(r) or ((r.get("choices") or [{}])[0].get("message") or {}).get("reasoning_content") or ""
  arms = {}
  for name, kw in (("P1", {"temperature": 1.0, "top_p": 1.0}), ("P95", {"temperature": 1.0, "top_p": 0.95})):
    with ThreadPoolExecutor(8) as ex: rs = list(ex.map(draw, [kw] * args.draws))
    arms[name] = collections.Counter(first(r) for c, r in rs if c == 200 and r.get("choices") and first(r))
  p1, n1 = arms["P1"], sum(arms["P1"].values())
  # the P1 arm's tail: its rarest tokens, holding its last 5% of draws (an estimate: a few draws of a borderline token are allowed)
  tail, acc = set(), 0
  for tok, k in sorted(p1.items(), key=lambda x: x[1]):
    if acc + k > 0.05 * n1: break
    acc += k
    tail.add(tok)
  out, allow = sum(k for tok, k in arms["P95"].items() if tok in tail), max(2, int(0.01 * args.draws))
  code, _r = post({"messages": msgs, "max_tokens": 4, "top_k": 20, "temperature": 0.7, "frobnicate": True})
  pcode, _r = post({"messages": msgs, "max_tokens": 4, "presence_penalty": 1.5, "frequency_penalty": 0.5, "repetition_penalty": 1.1})
  say(f"B sampler: {args.draws} draws an arm; P1 distinct {len(p1)}, its 5% tail {len(tail)} tokens ({acc} draws); P95 draws in that tail: {out} "
      f"(allow {allow}); top P1 {p1.most_common(4)} P95 {arms['P95'].most_common(4)}; top_k + unknown field: HTTP {code}")
  say(f"B penalties: presence_penalty / frequency_penalty / repetition_penalty: HTTP {pcode}, accepted and IGNORED by this server "
      "(the reply is sampled with temperature / top_p / top_k only: RUNNING.md)")
  # (no tail = no power: the arm cannot show a top_p that is ignored; pick a --draw-prompt with a spread instead of passing)
  if not tail: say("B has NO POWER: the P1 arm has no 5% tail (its first token is too certain): use a --draw-prompt with a spread")
  if out > allow or code != 200 or pcode != 200 or not tail or min(sum(a.values()) for a in arms.values()) < args.draws * 0.9: fails.append("b")

if "b" in args.only:
  # DEFAULT SAMPLING keeps its spread: the same short request 40 times with temperature 1 alone (what pi and most clients send: the server adds
  # the model's defaults, Qwen's GGUF top_p 0.95 + top_k 20) against temperature 1, top_p 1, top_k 0. A nucleus that keeps the top token alone
  # gives the same words 40 times (Qwen Nov 16: 1 distinct of 40 against 40 of 40); a working one about as many as the plain arm
  wmsgs = [{"role": "user", "content": "Write five random English words, separated by spaces."}]
  def words(kw:dict) -> str:
    c, r = post({"messages": wmsgs, "max_tokens": 16, **NOTHINK, "temperature": 1.0, **kw})
    m = ((r.get("choices") or [{}])[0].get("message") or {}) if c == 200 else {}
    return (m.get("content") or "") + "|" + (m.get("reasoning_content") or "")
  dist = {}
  for name, kw in (("plain", {"top_p": 1.0, "top_k": 0}), ("default", {})):
    with ThreadPoolExecutor(8) as ex: dist[name] = len(set(ex.map(words, [kw] * 40)))
  say(f"B default sampling: distinct replies of 40, temperature 1 alone {dist['default']}, with top_p 1 + top_k 0 {dist['plain']} (fail under half)")
  if dist["default"] < 0.5 * dist["plain"] or dist["plain"] < 10: fails.append("b-default")

if "c" in args.only:
  n = max(80, int(1.25 * args.slots + 0.999))
  rng = random.Random(args.seed or int(time.time()))
  T = " Answer with the answer only."
  def ask(messages) -> str:
    code, r = post({"messages": messages, "temperature": 0, "max_tokens": args.ret_max_tokens, **NOTHINK}, timeout=600)
    return content(r).strip() if code == 200 else f"<HTTP {code}>"
  def returning(name:str, convs:list, wants:list, q2:dict, wants2:list) -> int:
    # every conversation's first turn, then every one returns (N > slots: the early ones are evicted or stashed by then)
    ans = [ask(c) for c in convs]
    bad1 = sum(str(w) not in a for w, a in zip(wants, ans))
    bad = sum(str(w) not in ask(c + [{"role": "assistant", "content": a}, q2]) for c, a, w in zip(convs, ans, wants2))
    say(f"C returning turns under pressure ({name}, N={n} > {args.slots} slots, greedy, thinking off): first turns wrong {bad1}/{n}, "
        f"returning turns wrong {bad}/{n}")
    return bad
  filler = " ".join(f"Line {i}: the weather report says nothing important today." for i in range(25))
  xs = [rng.randrange(1000, 9999) for _ in range(n)]
  convs = [[{"role": "user", "content": f"{filler}\nRemember this: the secret number is {x}. Reply with OK only."}] for x in xs]
  if returning("text", convs, ["OK"] * n, {"role": "user", "content": "What is the secret number?" + T}, xs): fails.append("c-text")
  if not args.no_pictures:
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=28)
    def bars(vals):
      im = Image.new("RGB", (640, 480), (255, 255, 255))
      d = ImageDraw.Draw(im)
      d.line((60, 420, 600, 420), fill=(0, 0, 0), width=3)
      for i, (lab, v) in enumerate(zip("ABC", vals)):
        x = 110 + i * 170
        d.rectangle((x, 420 - v * 3, x + 100, 420), fill=(40, 90, 200))
        d.text((x + 30, 430), lab, fill=(0, 0, 0), font=font)
        d.text((x + 25, 420 - v * 3 - 36), str(v), fill=(0, 0, 0), font=font)
      return im
    vs = [[rng.randrange(10, 99) for _ in range(3)] for _ in range(n)]
    pconvs: list = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(bars(v))}},
                                           {"type": "text", "text": "What is the value written on bar B?" + T}]}] for v in vs]
    if returning("picture", pconvs, [v[1] for v in vs], {"role": "user", "content": "What is the value written on bar C?" + T}, [v[2] for v in vs]):
      fails.append("c-picture")

if "d" in args.only and not args.no_pictures:
  from PIL import Image, ImageDraw
  pics = []
  for i in range(10):
    im = Image.new("RGB", (320, 288), (255, 255, 255))
    ImageDraw.Draw(im).text((20, 120), f"picture {i}", fill=(0, 0, 0))
    pics.append(data_url(im))
  q10 = [{"type": "image_url", "image_url": {"url": u}} for u in pics] + [{"type": "text", "text": "How many pictures did I send? Number only."}]
  code, r = post({"messages": [{"role": "user", "content": q10}], "max_tokens": 8, **NOTHINK})
  say(f"D 10 pictures in one request: HTTP {code} answer {content(r)!r}")
  if code != 200: fails.append("d-10")
  im = Image.new("RGB", (3840, 2160), (40, 90, 160))
  ImageDraw.Draw(im).rectangle((1000, 600, 2800, 1500), fill=(240, 220, 20))
  text = {"type": "text", "text": "What colour is the rectangle? One word."}
  q4k = [{"type": "image_url", "image_url": {"url": data_url(im)}}, text]
  code, r = post({"messages": [{"role": "user", "content": q4k}], "max_tokens": 4, **NOTHINK})
  pt = r.get("usage", {}).get("prompt_tokens")
  _code, r0 = post({"messages": [{"role": "user", "content": [text]}], "max_tokens": 1, **NOTHINK})
  rows = pt - r0.get("usage", {}).get("prompt_tokens", 0) if pt else None
  say(f"D 3840x2160 picture: HTTP {code} prompt_tokens {pt} (~{rows} picture rows + pads; expected rows {args.rows_4k or 'not given'}) "
      f"answer {content(r)!r}")
  if code != 200 or (args.rows_4k and (rows is None or abs(rows - args.rows_4k) > 4)): fails.append("d-4k")

say(f"SMOKE {'PASSED' if not fails else 'FAILED: ' + ' '.join(fails)}")
sys.exit(1 if fails else 0)
