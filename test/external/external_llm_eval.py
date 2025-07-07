import pyarrow.parquet as pq
from tinygrad.helpers import fetch, colored
from tinygrad.apps.llm import Transformer, SimpleLlamaTokenizer, models
from tinygrad import Tensor

if __name__ == "__main__":
  #dat = fetch("https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Easy/train-00000-of-00001.parquet")
  #dat = fetch("https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Easy/test-00000-of-00001.parquet")
  dat = fetch("https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet")
  table = pq.read_table(dat)

  model, kv = Transformer.from_gguf(Tensor.from_url(models["1B"]), max_context=4096)

  tok = SimpleLlamaTokenizer(kv["tokenizer.ggml.tokens"])
  eos_id: int = tok.token_to_id["<|end_of_text|>"]
  eot_id: int = tok.token_to_id["<|eot_id|>"]

  num_correct, num_answered = 0, 0
  total_questions = len(table["question"])
  for question, choices, answer in zip(table["question"], table["choices"], table["answerKey"]):
    phrasing = f"Question: {question}\n\n" + \
               '\n'.join([f"{k}) {v}" for k,v in zip(choices['label'], choices['text'])]) +\
               "\n\nReply with the letter of the correct answer only."
    try:
      ids = [t for x in ["<|begin_of_text|>",
        "<|start_header_id|>", "user", "<|end_header_id|>\n\n", phrasing, "<|eot_id|>",
        "<|start_header_id|>", "assistant", "<|end_header_id|>\n\n", "Answer: "] for t in tok.encode(x)]
    except RuntimeError:
      # TODO: fix the tokenizer
      pass
    next_id = next(model.generate(ids))
    correct, given = answer.as_py().strip(), tok.decode([next_id]).strip()
    num_correct += correct == given
    num_answered += 1
    print(f"{num_answered:4d}/{total_questions:4d}  "+\
          f"Correct Answer: {correct}  "+\
          f"Given Answer: {colored(given, 'green' if correct==given else 'red')}  "+\
          f"Percent: {num_correct*100.0/num_answered:.2f}%")
