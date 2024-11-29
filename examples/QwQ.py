import tiktoken
from examples.llama import concat_weights, load
from tiktoken.load import load_tiktoken_bpe

class Tokenizer:
  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
  def __init__(self, model_path: str):
    mergeable_ranks = load_tiktoken_bpe(model_path)
    self.num_base_tokens = len(mergeable_ranks)
    special_tokens = [
      "<|endoftext|>",
      "<|im_start|>",
      "<|im_end|>",
      "<|object_ref_start|>",
      "<|object_ref_end|>",
      "<|box_start|>",
      "<|box_end|>",
      "<|quad_start|>",
      "<|quad_end|>",
      "<|vision_start|>",
      "<|vision_end|>",
      "<|vision_pad|>",
      "<|image_pad|>",
      "<|video_pad|>",
    ]

    self.special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}

    self.model = tiktoken.Encoding(name=model_path, pat_str=self.pat_str, mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens)

  def decode(self, toks): return self.model.decode([t for t in toks if t < self.num_base_tokens])
  def encode(self, s): return self.model.encode(s)

  def eos_id(self): return self.special_tokens["<|endoftext|>"]
  def vocab_size(self): return self.model.n_vocab