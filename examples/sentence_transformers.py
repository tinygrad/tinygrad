import time
import argparse

import torch
from transformers import BertTokenizer, BertModel
import numpy as np

from tinygrad import Tensor
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.helpers import fetch
from extra.models.bert import Bert

# For now only bert models are supported
MODELS = {
  "all-MiniLM-L6-v2": {
    "download_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors",
    "bert_config": {
      "hidden_size": 384,
      "intermediate_size": 1536,
      "num_attention_heads": 12,
      "num_hidden_layers": 6,
      "max_position_embeddings": 512,
      "type_vocab_size": 2,
      "vocab_size": 30522,
      "attention_probs_dropout_prob": 0.1,
      "hidden_dropout_prob": 0.1,
    },
  },
  "all-MiniLM-L12-v2": {
    "download_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/model.safetensors",
    "bert_config": {
      "hidden_size": 384,
      "intermediate_size": 1536,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "max_position_embeddings": 512,
      "type_vocab_size": 2,
      "vocab_size": 30522,
      "attention_probs_dropout_prob": 0.1,
      "hidden_dropout_prob": 0.1,
    },
  },
  "multi-qa-MiniLM-L6-cos-v1": {
    "download_url": "https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1/resolve/main/model.safetensors",
    "bert_config": {
      "hidden_size": 384,
      "intermediate_size": 1536,
      "num_attention_heads": 12,
      "num_hidden_layers": 6,
      "max_position_embeddings": 512,
      "type_vocab_size": 2,
      "vocab_size": 30522,
      "attention_probs_dropout_prob": 0.1,
      "hidden_dropout_prob": 0.1,
    },
  },
}

sentences = [
  "The quick brown fox jumps over the lazy dog.",
  "Artificial intelligence is transforming the world.",
  "She sells seashells by the seashore.",
  "Climate change is a pressing global issue.",
  "The capital of France is Paris.",
  "He read the book in one sitting.",
  "The stock market closed higher today.",
  "A picture is worth a thousand words.",
  "The concert was canceled due to rain.",
  "They are planning a trip to Japan next year.",
  "Technology advances at a rapid pace.",
  "I enjoy hiking in the mountains during summer.",
  "The new café down the street serves excellent coffee.",
  "Learning a new language can be challenging but rewarding.",
  "The movie received rave reviews from critics.",
  "Exercise is essential for maintaining good health.",
  "She won the award for her outstanding performance.",
  "Time flies when you're having fun.",
  "They adopted a puppy from the animal shelter.",
  "The scientist presented her findings at the conference.",
  "He plays the guitar in a local band.",
  "Online education has become more prevalent recently.",
  "The bakery is known for its delicious pastries.",
  "He solved the complex equation effortlessly.",
  "Gardening is a relaxing hobby for many people.",
  "They watched the sunrise from the beach.",
  "The museum exhibit features ancient artifacts.",
  "She is writing a novel set in medieval times.",
  "Teamwork is crucial for the success of the project.",
  "He enjoys painting landscapes in his free time.",
  "The city lights illuminated the night sky.",
  "Innovation drives progress in all industries.",
]


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run sentence-transformer models in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", choices=MODELS.keys(), help="Model to use")
  parser.add_argument("--validate", action="store_true", help="validate the tinygrad output against the transformers output")
  parser.add_argument("--torch-device", type=str, default="cpu", help="Device to use for PyTorch")
  args = parser.parse_args()

  tokenizer = BertTokenizer.from_pretrained(f"sentence-transformers/{args.model}")

  bert_config = MODELS[args.model]["bert_config"]
  download_url = MODELS[args.model]["download_url"]
  model_bert = Bert(**bert_config)
  model_file = fetch(download_url)
  load_state_dict(model_bert, safe_load(str(model_file)), strict=False)

  t0 = time.perf_counter()
  encoded_input = tokenizer(sentences, return_tensors="np", padding=True)
  input_ids = Tensor(encoded_input["input_ids"])
  attention_mask = Tensor(encoded_input["attention_mask"])
  model_output = model_bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=Tensor.zeros_like(input_ids))
  input_mask = attention_mask.unsqueeze(-1)
  embeddings = ((model_output * input_mask).sum(1) / input_mask.sum(1).clamp(1e-9)).numpy()
  print(f"tinygrad time: {time.perf_counter() - t0:.2f}s")

  if args.validate:
    model_pt = BertModel.from_pretrained(f"sentence-transformers/{args.model}").to(args.torch_device)
    model_pt.eval()
    t0 = time.perf_counter()
    encoded_input = tokenizer(sentences, return_tensors="pt", padding=True)
    input_ids = encoded_input["input_ids"].to(args.torch_device)
    attention_mask = encoded_input["attention_mask"].to(args.torch_device)
    with torch.no_grad():
      model_output = model_pt(input_ids, attention_mask)
    input_mask = attention_mask.unsqueeze(-1).float()
    embeddings_pt = (torch.sum(model_output[0] * input_mask, dim=1) / torch.clamp(input_mask.sum(dim=1), min=1e-9)).cpu().numpy()
    print(f"transformers time: {time.perf_counter() - t0:.2f}s")
    np.testing.assert_allclose(embeddings, embeddings_pt, rtol=1e-4, atol=1e-4)
    print("Validation successful!")
