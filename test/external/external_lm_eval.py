import sys
import pyarrow.parquet as pq
from tinygrad.helpers import fetch
from tinygrad.apps.llm import Transformer, SimpleLlamaTokenizer, models
from tinygrad import Tensor

if __name__ == "__main__":
  #dat = fetch("https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet")
  dat = fetch("https://huggingface.co/datasets/cais/mmlu/resolve/main/elementary_mathematics/test-00000-of-00001.parquet")
  table = pq.read_table(dat)

  model, kv = Transformer.from_gguf(Tensor.from_url(models["3B"]), max_context=4096)

  tok = SimpleLlamaTokenizer(kv["tokenizer.ggml.tokens"])
  eos_id: int = tok.token_to_id["<|end_of_text|>"]
  eot_id: int = tok.token_to_id["<|eot_id|>"]

  for question, choices, answer in zip(table["question"], table["choices"], table["answer"]):
    phrasing = f"Question: {question}\n\n" + \
               f"A. {choices[0]}\n" + \
               f"B. {choices[1]}\n" + \
               f"C. {choices[2]}\n" + \
               f"D. {choices[3]}\n\n" + \
               "Think about it, but end your response with only the letter of the correct answer."
    try:
      ids = [t for x in ["<|begin_of_text|>",
        #"<|start_header_id|>", "system", "<|end_header_id|>", "You are an expert exam taker.", "<|eot_id|>",
        "<|start_header_id|>", "user", "<|end_header_id|>\n\n", phrasing, "<|eot_id|>",
        "<|start_header_id|>", "assistant", "<|end_header_id|>\n\n"] for t in tok.encode(x)]
    except RuntimeError:
      # TODO: fix the tokenizer
      pass
    print("Answer:", "ABCD"[answer.as_py()], tok.decode(ids))
    for next_id in model.generate(ids):
      if next_id in (eos_id, eot_id): break
      sys.stdout.write(tok.decode([next_id]) if next_id != eot_id else "\n\n")
      sys.stdout.flush()
    print("\n------------------------------------\n")
