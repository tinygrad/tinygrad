import base64
import regex
from typing import AbstractSet, Collection, Literal, Sequence
from tinygrad.helpers import fetch

class Encoding:
  def __init__(self, name: str, *, pat_str: str, mergeable_ranks: dict[bytes, int], special_tokens: dict[str, int],
               explicit_n_vocab: int | None = None,) -> None:

    self.max_token_value = max(max(mergeable_ranks.values()), max(special_tokens.values(), default=0))

    if explicit_n_vocab:
      assert len(mergeable_ranks) + len(special_tokens) == explicit_n_vocab
      assert self.max_token_value == explicit_n_vocab - 1

    self._special_tokens = special_tokens
    self._special_regex = regex.compile("|".join(regex.escape(t) for t in special_tokens.keys()))

    self._mergeable_ranks = mergeable_ranks
    self._inv_vocab = {v:k for k, v in self._mergeable_ranks.items() }
    self._inv_vocab.update({v: k.encode("utf-8") for k, v in self._special_tokens.items()})
    self._pat = regex.compile(pat_str)

  @property
  def n_vocab(self): return self.max_token_value + 1

  def encode(self, text:str, *, allowed_special: Literal["all"] | AbstractSet[str] = set(),
             disallowed_special: Literal["all"] | Collection[str] = "all"):
    allowed_special = set(self._special_tokens.keys()) if allowed_special == "all" else allowed_special

    if disallowed_special == "all":
      disallowed_special = set(self._special_tokens.keys()) - allowed_special

    special_matches = list(self._special_regex.finditer(text))
    special_strs = [m.group(0) for m in special_matches]
    if (disallowed_token := next((s for s in special_strs if s in disallowed_special), None)) is not None:
      raise ValueError(f"Encountered text corresponding to disallowed special token '{disallowed_token}'.")
    special_matches = [ m for m in special_matches if m.group(0) in allowed_special ]
    special_tokens = [self._special_tokens[s] for s in special_strs]

    patch_ends = [m.start() for m in special_matches] + [len(text)]
    patch_starts = [0] + [m.end() for m in special_matches]

    tokens: list[int] = []

    for s, e in zip(patch_starts, patch_ends):
      patch = text[s:e]
      tokens.extend([tok for word in self._pat.findall(patch) for tok in self._encode_word(word.encode("utf-8"))])
      if len(special_tokens) > 0: tokens.append(special_tokens.pop(0))

    return tokens

  def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
    return b"".join(self._inv_vocab[token] for token in tokens).decode("utf-8", errors=errors)

  # source: https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py
  def _encode_word(self, word: bytes):
    parts = [bytes([b]) for b in word]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = self._mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # If there were no pairs we could merge, we're done!
        if min_rank is None:
            break

        # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx+1]] + parts[min_idx+2:]

    tokens = [self._mergeable_ranks[part] for part in parts]
    return tokens

def load_tiktoken_bpe(url: str):
  with fetch(url).open("rb") as f:
    return {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in f if line)}

_encoding_loaders = {
  "gpt2": lambda: {
    "pat_str": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""",
    "mergeable_ranks": load_tiktoken_bpe("https://raw.githubusercontent.com/openai/whisper/refs/heads/main/whisper/assets/gpt2.tiktoken"),
    "special_tokens": {"<|endoftext|>": 50256},
    "explicit_n_vocab": 50257
  }
}

def get_encoding(model: str): return Encoding(model, **_encoding_loaders[model]())
