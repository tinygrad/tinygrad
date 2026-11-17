#!/usr/bin/env python3
# The customer smoke of a started server, through its public port (rule 1 of every start). Exit code != 0 on any failure.
# usage: python3 extra/llm_service/smoke.py <url> [--slots 64] [--rows-4k N] [--no-pictures] [--no-reasoning] [--only abcd]
#  a  20 chats with NO sampling parameters: status 200, stray-script characters in content = 0, no <think> in content, reasoning in
#     reasoning_content on every reply (--no-reasoning for a model that does not think), five printed for the eye
#  b  sampling against the server's own distribution: one-token draws at T 1 / top_p 1 / top_k 0 (plain), T 1 / top_p 0.95 and T 1 alone
#     (the model's defaults): on plain's head (its top 92%) the other two arms' relative frequencies equal plain's (chi-square, fails at
#     p < 0.001 or without power, unless the top token holds 92%+); top_p 0.95 CUTS: P95's mass outside the 95% head of one half of plain
#     is smaller than the other half's (one-sided, fails unless p < 0.001: an ignored top_p fails); a request with top_k and an unknown
#     field is answered (200), and one with presence / frequency / repetition penalties (200; this server ignores them: printed, RUNNING.md)
#  c  returning turns under pressure: N = max(80, 1.25 x slots) conversations, a secret each, then every one returns and must answer its
#     OWN secret (100%); text, then one picture each
#  d  10 pictures in one request accepted; a 3840x2160 picture's prompt tokens (checked against --rows-4k if given)
# It cannot check (e), that the start was made by the launcher exactly as RUNNING.md prints it: say that in the 'serves' line yourself.
import sys, json, re, time, math, random, base64, io, argparse, collections, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--slots", type=int, default=64)
ap.add_argument("--rows-4k", type=int, default=0)
ap.add_argument("--no-pictures", action="store_true")
ap.add_argument("--draws", type=int, default=600)
ap.add_argument("--head", type=float, default=0.92, help="part b: plain's head = its top tokens while the mass before them is under this")
ap.add_argument("--max-tokens", type=int, default=8000, help="max_tokens of the part a chats (a thinking model's reply must fit)")
ap.add_argument("--ret-max-tokens", type=int, default=48, help="max_tokens of a part c reply (thinking off)")
ap.add_argument("--draw-prompt", default="Do not think about this at all; your thinking must start with one random fruit name. Which fruit?",
                help="part b's prompt: its first generated token must have a spread at temperature 1 (this one has on Qwen and GLM Flash)")
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

def chi2_sf(x:float, k:int) -> float:
  # P(chi-square with k degrees > x): regularized upper incomplete gamma, series form
  if x <= 0: return 1.0
  a, tot, term, n = k / 2, 0.0, 2.0 / k, 0
  while term > 1e-15 * tot and n < 100000:
    tot += term
    n += 1
    term *= (x / 2) / (a + n)
  return max(0.0, 1 - tot * math.exp(-x / 2 + a * math.log(x / 2) - math.lgamma(a)))

if "b" in args.only:
  # SAMPLING against the server's own distribution: the first generated token of --draw-prompt, drawn --draws times in three arms:
  # plain (temperature 1, top_p 1, top_k 0) = the model's distribution, P95 (temperature 1, top_p 0.95) and default (temperature 1 alone:
  # the server adds the model's defaults, both GGUFs top_p 0.95, Qwen also top_k 20). A correct nucleus only CUTS the tail and renormalises:
  # on the plain arm's head H (its tokens in rank order while the mass before them is under --head 0.92, each with 10+ plain draws: inside a
  # nucleus of 0.95 by 2-3 standard errors at 600 draws, and inside a top_k of 20: a token of 1.7% or more is in any top 20 that matters)
  # the relative frequencies of P95 and default equal plain's: a two-sample chi-square on H's bins, fail at p < 0.001. A nucleus that keeps
  # the top token alone (Qwen, Nov 16) or that reshapes the head fails; a peaked model does not (the old 'distinct of 40' rule failed GLM
  # Flash and GLM 5.3 for that). No power (|H| < 2 or under 20 plain draws below the top token in H) fails too (pick a --draw-prompt with a
  # spread), except PEAKED: plain's top token alone holds 92%+ (GLM 5.3's thinking starts 'The' ~98% whatever the prompt): a nucleus of 0.95
  # is then the top token or little more, nothing to compare on the head, and the cut test below decides alone. An IGNORED top_p passes
  # the chi-square (nothing reshaped): so the plain arm draws twice as many, its halves A and B (by request order),
  # T = A's tokens in rank order while A's mass before them is under 0.95, and P95's draws outside T must be FEWER than B's (one-sided
  # two-proportion test, fails unless p < 0.001). T is chosen on A alone: with top_p ignored B and P95 are draws of one distribution
  # independent of T, so the test is exact whatever T is; with a nucleus, P95 has none of plain's mass past 0.95 (power: the prompt's
  # tail: a peaked model needs more --draws, ~1200 for GLM 5.3's 2% tail).
  msgs = [{"role": "user", "content": args.draw_prompt}]
  def draw(kw): return post({"messages": msgs, "max_tokens": 1, **NOTHINK, **kw})
  # the first generated token: in content, or in reasoning_content for a model that always thinks (GLM 5.3 Flash ignores enable_thinking)
  def first(r:dict) -> str: return content(r) or ((r.get("choices") or [{}])[0].get("message") or {}).get("reasoning_content") or ""
  arms = {}
  for name, kw in (("plain", {"temperature": 1.0, "top_p": 1.0, "top_k": 0}), ("P95", {"temperature": 1.0, "top_p": 0.95}),
                   ("default", {"temperature": 1.0})):
    with ThreadPoolExecutor(8) as ex: rs = list(ex.map(draw, [kw] * (args.draws * (2 if name == "plain" else 1))))
    toks = [first(r) if c == 200 and r.get("choices") else "" for c, r in rs]
    if name == "plain": halves = (toks[0::2], toks[1::2])
    arms[name] = collections.Counter(t for t in toks if t)
  # (the cut: T from half A, P95's mass outside T against half B's)
  ta = collections.Counter(t for t in halves[0] if t)
  tail, acc_a = set(), 0
  for tok, k in ta.most_common():
    if acc_a >= 0.95 * sum(ta.values()): break
    tail.add(tok)
    acc_a += k
  nb, xb = sum(1 for t in halves[1] if t), sum(1 for t in halves[1] if t and t not in tail)
  np95, xp = sum(arms["P95"].values()), sum(k for t, k in arms["P95"].items() if t not in tail)
  pool = (xb + xp) / max(1, nb + np95)
  z = (xb / max(1, nb) - xp / max(1, np95)) / math.sqrt(pool * (1 - pool) * (1 / max(1, nb) + 1 / max(1, np95))) if 0 < pool < 1 else 0.0
  pcut = 0.5 * math.erfc(z / math.sqrt(2))
  plain, n0 = arms["plain"], sum(arms["plain"].values())
  head, acc = [], 0
  for tok, k in plain.most_common():
    if acc >= args.head * n0 or k < 10: break
    head.append(tok)
    acc += k
  power = len(head) >= 2 and acc - (plain[head[0]] if head else 0) >= 20
  peaked = not power and n0 > 0 and plain.most_common(1)[0][1] >= 0.92 * n0
  verdict = []
  for name in ("P95", "default"):
    arm = arms[name]
    bins = [(plain[t], arm[t]) for t in head]
    n1, n2 = sum(b[0] for b in bins), sum(b[1] for b in bins)
    # (two-sample chi-square: expected count of a bin in a sample = the bin's pooled count x that sample's share of the pooled total)
    def cell(o:int, q:int, n:int, x:int) -> float: return (x - (o + q) * n / (n1 + n2)) ** 2 / ((o + q) * n / (n1 + n2))
    chi = sum(cell(o, q, n1, o) + cell(o, q, n2, q) for o, q in bins) if n1 and n2 else 0.0
    p = chi2_sf(chi, len(bins) - 1) if len(bins) > 1 and n1 and n2 else 0.0
    verdict.append((name, chi, len(bins) - 1, p, 1 - n2 / max(1, sum(arm.values()))))
  code, _r = post({"messages": msgs, "max_tokens": 4, "top_k": 20, "temperature": 0.7, "frobnicate": True})
  pcode, _r = post({"messages": msgs, "max_tokens": 4, "presence_penalty": 1.5, "frequency_penalty": 0.5, "repetition_penalty": 1.1})
  say(f"B sampler: {args.draws} draws an arm (plain {2 * args.draws}); plain distinct {len(plain)},"
      f" head H {len(head)} tokens = {acc / max(1, n0):.3f} of plain; "
      f"top plain {plain.most_common(4)} P95 {arms['P95'].most_common(4)} default {arms['default'].most_common(4)}")
  say(f"B top_p 0.95 cuts: outside A's 95% head ({len(tail)} tokens): plain half B {xb}/{nb} = {xb / max(1, nb):.3f}, P95 {xp}/{np95} = "
      f"{xp / max(1, np95):.3f}; one-sided p = {pcut:.5f} (fail unless under 0.001: top_p ignored or no tail to cut)")
  for name, chi, df, p, out in verdict:
    say(f"B {name} vs plain on H: chi-square {chi:.2f} on {df} df, p = {p:.4f} (fail under 0.001); "
        f"mass outside H {out:.3f} (plain {1 - acc / max(1, n0):.3f})")
  say(f"B top_k + unknown field: HTTP {code}; penalties (presence / frequency / repetition): HTTP {pcode}, accepted and IGNORED by this server "
      "(the reply is sampled with temperature / top_p / top_k only: RUNNING.md)")
  if peaked: say("B head PEAKED (top token 92%+ of plain): no chi-square on the head, the cut test decides")
  elif not power: say("B has NO POWER: the plain arm's head is one token or too few draws: use a --draw-prompt with a spread")
  if ((not power and not peaked) or (power and any(p < 0.001 for _, _, _, p, _ in verdict)) or pcut >= 0.001 or code != 200 or pcode != 200
      or min(sum(a.values()) for a in arms.values()) < args.draws * 0.9): fails.append("b")

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
