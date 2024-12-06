import os, pathlib
from examples.llama3 import Tokenizer
from tinygrad import fetch

def stringify(base_path):
  ret = []
  for path, _, files in os.walk(os.path.join(base_path, "tinygrad")):
    for name in files:
      if not name.endswith(".py"): continue
      if 'tinygrad/runtime/autogen' in path.replace('\\', '/'): continue
      code = pathlib.Path(os.path.join(path, name)).read_text()
      ret += [name, code]
  return '\x00'.join(ret)

if __name__ == "__main__":
  code_str = stringify(".")
  print(f"code has {len(code_str)} chars")
  print(f"code has {code_str.count("\n")} newlines")

  # llama 3 tokenizer
  tokenizer = Tokenizer(fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model").as_posix())

  encoded = tokenizer.encode(code_str)
  print(f"code has {len(encoded)} tokens")
