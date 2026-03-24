# eval for tinygrad.apps.llm -- hits the server via OpenAI API
import argparse, pyarrow.parquet as pq
from openai import OpenAI
from tinygrad.helpers import fetch, colored

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--port", "-p", type=int, default=11434)
  parser.add_argument("--limit", "-L", type=int, default=None)
  args = parser.parse_args()

  client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="tinygrad")
  dat = fetch("https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet")
  table = pq.read_table(dat)

  num_correct, num_answered = 0, 0
  total_questions = min(len(table["question"]), args.limit) if args.limit else len(table["question"])
  for question, choices, answer in list(zip(table["question"], table["choices"], table["answerKey"]))[:total_questions]:
    phrasing = f"Question: {question}\n\n" + \
               '\n'.join([f"{k}) {v}" for k,v in zip(choices['label'], choices['text'])]) +\
               "\n\nReply with the letter of the correct answer only."
    resp = client.chat.completions.create(model="test", messages=[
      {"role": "system", "content": "You answer multiple choice questions with a single letter."},
      {"role": "user", "content": phrasing}], max_tokens=1)
    correct, given = answer.as_py().strip(), resp.choices[0].message.content.strip()
    num_correct += correct == given
    num_answered += 1
    print(f"{num_answered:4d}/{total_questions:4d}  "+\
          f"Correct Answer: {correct}  "+\
          f"Given Answer: {colored(given, 'green' if correct==given else 'red')}  "+\
          f"Percent: {num_correct*100.0/num_answered:.2f}%")
