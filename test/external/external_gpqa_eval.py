# GPQA Diamond eval for an OpenAI API server: the prompt, the answer shuffling (seeded) and the answer extraction of openai/simple-evals
# results are appended to --out (a jsonl, one line a question and repeat), a run with the same --out skips the ones it has
import argparse, csv, json, os, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tinygrad.helpers import fetch, colored

TEMPLATE = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' " \
           "(without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
ANSWER = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--port", "-p", type=int, default=8000)
  parser.add_argument("--limit", "-L", type=int, default=None)
  parser.add_argument("--repeats", "-n", type=int, default=1, help="every question this many times (sampled answers: the mean)")
  parser.add_argument("--max_tokens", "-T", type=int, default=None)
  parser.add_argument("--temperature", "-t", type=float, default=1.0)
  parser.add_argument("--concurrency", "-c", type=int, default=8)
  parser.add_argument("--out", "-o", type=str, default="gpqa_results.jsonl")
  args = parser.parse_args()

  client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="tinygrad", timeout=24*3600, max_retries=0)
  rows = list(csv.DictReader(open(fetch("https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"), encoding="utf-8")))
  rng = random.Random(0)
  rows = [row | {"permutation": rng.sample(range(4), 4)} for row in rows][:args.limit]
  done = {(r["question"], r["repeat"]) for r in map(json.loads, open(args.out))} if os.path.exists(args.out) else set()
  todo = [(i, rep) for rep in range(args.repeats) for i in range(len(rows)) if (i, rep) not in done]
  lock, st = threading.Lock(), time.perf_counter()

  def ask(i:int, rep:int) -> dict:
    row = rows[i]
    choices = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    choices = [choices[p] for p in row["permutation"]]
    correct = "ABCD"[choices.index(row["Correct Answer"])]
    prompt = TEMPLATE.format(Question=row["Question"], A=choices[0], B=choices[1], C=choices[2], D=choices[3])
    t0 = time.perf_counter()
    resp = client.chat.completions.create(model="test", messages=[{"role": "user", "content": prompt}], max_tokens=args.max_tokens,
                                          temperature=args.temperature)
    text = resp.choices[0].message.content or ""
    m = re.search(ANSWER, text)
    return {"question": i, "repeat": rep, "correct": correct, "given": m.group(1).upper() if m else None, "finish": resp.choices[0].finish_reason,
            "tokens": resp.usage.completion_tokens if resp.usage else None, "seconds": time.perf_counter() - t0, "domain": row["High-level domain"]}

  results = [json.loads(l) for l in open(args.out)] if os.path.exists(args.out) else []
  with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
    for fut in as_completed([pool.submit(ask, i, rep) for i, rep in todo]):
      r = fut.result()
      with lock:
        results.append(r)
        with open(args.out, "a") as f: f.write(json.dumps(r) + "\n")
        acc = sum(x["given"] == x["correct"] for x in results) / len(results)
        toks = sum(x["tokens"] or 0 for x in results)
        given = colored(str(r['given']), 'green' if r['given'] == r['correct'] else 'red')
        print(f"{len(results):4d}/{len(rows)*args.repeats}  q {r['question']:3d}  {r['correct']} {given}"
              f"  {r['finish']:6s} {r['tokens'] or 0:6d} tok  acc {acc*100:5.1f}%  mean {toks/len(results):6.0f} tok  "
              f"{(time.perf_counter()-st)/60:6.1f} min", flush=True)
