#!/usr/bin/env python3
import os
from pathlib import Path
import token
import tokenize
import itertools
from tabulate import tabulate

TOKEN_WHITELIST = [token.OP, token.NAME, token.NUMBER, token.STRING]

def file_sz():
  table = {}
  for path, subdirs, files in os.walk("tinygrad"):
    for name in files:
      if not name.endswith(".py"): continue
      filepath = Path(path) / name
      with tokenize.open(filepath) as file_:
        tokens = [t for t in tokenize.generate_tokens(file_.readline) if t.type in TOKEN_WHITELIST]
        token_count, line_count = len(tokens), len(set([t.start[0] for t in tokens]))
        table[filepath.as_posix()] = [line_count, token_count/line_count]

  return table

if __name__ == "__main__":
  headers = ["Name", "Lines", "Tokens/Line"]
  if os.getenv("CI") == 1: 
    print(file_sz())
  else:
    table = [[key] + value for key, value in file_sz().items()]

    print(tabulate([headers] + sorted(table, key=lambda x: -x[1]), headers="firstrow", floatfmt=".1f")+"\n")

    for dir_name, group in itertools.groupby(sorted([(x[0].rsplit("/", 1)[0], x[1]) for x in table]), key=lambda x:x[0]):
      print(f"{dir_name:30s} : {sum([x[1] for x in group]):6d}")

    print(f"\ntotal line count: {sum([x[1] for x in table])}")