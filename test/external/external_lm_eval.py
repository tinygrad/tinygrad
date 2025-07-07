import pyarrow.parquet as pq
from tinygrad.helpers import fetch
from tinygrad.apps.llm import Transformer, SimpleLlamaTokenizer
from tinygrad import Tensor

if __name__ == "__main__":
  dat = fetch("https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet")
  table = pq.read_table(dat)

  model_url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf"
  model, kv = Transformer.from_gguf(Tensor.from_url(model_url), max_context=4096)

  tok = SimpleLlamaTokenizer(kv["tokenizer.ggml.tokens"])
  eos_id: int = tok.token_to_id["<|end_of_text|>"]
  eot_id: int = tok.token_to_id["<|eot_id|>"]

  i = 0
  for question, choices, answer in zip(table["question"], table["choices"], table["answer"]):
    phrasing = f"Question: {question}\n\n" + \
               f"A. {choices[0]}\n" + \
               f"B. {choices[1]}\n" + \
               f"C. {choices[2]}\n" + \
               f"D. {choices[3]}\n\n" + \
               "End your response with the letter of the correct answer."
    #print("\n\n\n"+phrasing)
    ids = [t for x in ["<|begin_of_text|>",
      #"<|start_header_id|>", "system", "<|end_header_id|>", "You are an expert exam taker.", "<|eot_id|>",
      "<|start_header_id|>", "user", "<|end_header_id|>\n\n", phrasing, "<|eot_id|>",
      "<|start_header_id|>", "assistant", "<|end_header_id|>\n\n"] for t in tok.encode(x)]
    for next_id in model.generate(ids):
      if next_id == eot_id: break
    print(tok.decode(ids), "Answer:", "ABCD"[answer.as_py()])
