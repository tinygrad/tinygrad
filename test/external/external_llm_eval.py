# eval for tinygrad.apps.llm -- hits the server via OpenAI API
# uses Meta's exact ARC-Challenge prompt template from lm-evaluation-harness llama3 tasks
import argparse, re, pyarrow.parquet as pq
from openai import OpenAI
from tinygrad.helpers import fetch, colored

LABEL = ["A", "B", "C", "D"]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--port", "-p", type=int, default=11434)
  parser.add_argument("--limit", "-L", type=int, default=None)
  args = parser.parse_args()

  client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="tinygrad")
  dat = fetch("https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet")
  table = pq.read_table(dat)

  num_correct, num_answered = 0, 0
  # filter to 4-choice questions and normalize labels to A/B/C/D (matches Meta's eval)
  rows = [(q, c, a) for q, c, a in zip(table["question"], table["choices"], table["answerKey"]) if len(c["label"]) == 4]
  total_questions = min(len(rows), args.limit) if args.limit else len(rows)
  for question, choices, answer in rows[:total_questions]:
    phrasing = f"Given the following question and four candidate answers (A, B, C and D), choose the best answer.\n" +\
               f"Question: {question}\n" + '\n'.join([f"{l}. {t}" for l, t in zip(LABEL, choices['text'])]) +\
               f'\nYour response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A, B, C or D.'
    resp = client.chat.completions.create(model="test", messages=[{"role": "user", "content": phrasing}], max_tokens=100)
    # normalize answer key (some use 1/2/3/4 instead of A/B/C/D)
    correct = answer.as_py().strip()
    if correct not in LABEL: correct = LABEL[int(correct) - 1]
    # extract answer: take last single capital letter A-D from response (prompt asks model to end with the answer)
    text = resp.choices[0].message.content.strip()
    m = re.findall(r'\b([A-D])\b', text)
    given = m[-1] if m else text[:1]
    num_correct += correct == given
    num_answered += 1
    print(f"{num_answered:4d}/{total_questions:4d}  "+\
          f"Correct Answer: {correct}  "+\
          f"Given Answer: {colored(given, 'green' if correct==given else 'red')}  "+\
          f"Percent: {num_correct*100.0/num_answered:.2f}%")
