# vision eval for the OpenAI API server: RealWorldQA (765 multiple-choice questions on real-world images)
# https://huggingface.co/datasets/lmms-lab-encoder/RealWorldQA
# usage: python3 test/external/external_vlm_eval.py --port 8000
import argparse, base64, re, pyarrow.parquet as pq
from openai import OpenAI
from tinygrad.helpers import fetch, colored

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--port", "-p", type=int, default=8000)
  parser.add_argument("--limit", "-L", type=int, default=None)
  parser.add_argument("--max_tokens", "-T", type=int, default=4096)
  parser.add_argument("--offset", "-O", type=int, default=0)
  parser.add_argument("--temperature", "-t", type=float, default=0.0)
  parser.add_argument("--no_think", action="store_true", help="disable thinking (prefills empty think block via assistant message)")
  parser.add_argument("--debug", action="store_true")
  args = parser.parse_args()

  client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="tinygrad")
  parts = [fetch(f"https://huggingface.co/datasets/lmms-lab-encoder/RealWorldQA/resolve/main/data/test-0000{i}-of-00002.parquet") for i in range(2)]
  rows = [row for part in parts for row in pq.read_table(part).to_pylist()]

  num_correct, num_answered = 0, 0
  total_questions = min(len(rows), args.offset + args.limit) if args.limit else len(rows)
  for row in rows[args.offset:total_questions]:
    data_uri = "data:image/jpeg;base64," + base64.b64encode(row["image"]["bytes"]).decode()
    messages = [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": data_uri}},
      {"type": "text", "text": row["question"]}]}]
    if args.no_think: messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
    resp = client.chat.completions.create(model="test", messages=messages,
                                          max_tokens=args.max_tokens, temperature=args.temperature)
    correct = row["answer"].strip()
    text = (resp.choices[0].message.content or "").strip()
    if args.debug: print(f"\n--- PROMPT ---\n{row['question']}\n--- RESPONSE ---\n{text}\n---")
    # answers are either an option letter (A-D) or a free-form word/number; accept an exact (case-insensitive)
    # match of the response's first word against the answer or its first letter
    m = re.findall(r'\b([A-D])\b', text)
    given = m[0] if m else text.split()[0] if text else ""
    c, g = correct.lower(), given.lower().strip(".,")
    good = c == g or (len(g) == 1 and c.startswith(g)) or (len(c) == 1 and g.startswith(c))
    num_correct += good
    num_answered += 1
    print(f"{num_answered:4d}/{total_questions:4d}  "+\
          f"Correct Answer: {correct}  "+\
          f"Given Answer: {colored(given, 'green' if good else 'red')}  "+\
          f"Percent: {num_correct*100.0/num_answered:.2f}%", flush=True)
