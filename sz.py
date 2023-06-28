#!/usr/bin/env python3
import os
import token
import tokenize
from itertools import groupby
from tabulate import tabulate

TOKEN_WHITELIST = [token.OP, token.NAME, token.NUMBER, token.STRING]


class FileStats:
  def __init__(self, name: str, lc: int, tokens: int):
    self.name = name
    self.dir = self.name.rsplit("/", 1)[0]
    self.lines_count = lc
    self.tokens = tokens

  def format(self):
    return {"name": self.name, "lines": self.lines_count, "tokens/line": self.tokens / self.lines_count}


def print_stats(data):
  files_dict = [f.format() for f in data]
  print(tabulate(files_dict, headers="keys", floatfmt=".1f") + "\n")

  for dir_name, dir_group in groupby(sorted([(f.dir, f.lines_count) for f in data]), key=lambda x: x[0]):
    print(f"{dir_name:30s} : {sum([f[1] for f in dir_group]):6d}")

  print(f"\ntotal line count: {sum([x.lines_count for x in data])}")


def gen_stats(base_path="."):
  table = []
  for path, subdirs, files in os.walk(os.path.join(base_path, "tinygrad")):
    for name in files:
      if not name.endswith(".py"):
        continue
      filepath = os.path.join(path, name)
      relfilepath = os.path.relpath(filepath, base_path)
      with tokenize.open(filepath) as _file:
        tokens = [t for t in tokenize.generate_tokens(_file.readline) if t.type in TOKEN_WHITELIST]
        token_count, line_count = len(tokens), len(set([t.start[0] for t in tokens]))
        table.append(FileStats(relfilepath, line_count, token_count))

  return sorted(table, key=lambda x: -x.lines_count)


if __name__ == "__main__":
  print_stats(gen_stats())
