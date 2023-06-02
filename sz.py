#!/usr/bin/env python3
import os
import token
import tokenize
import itertools
from tabulate import tabulate

TOKEN_WHITELIST = [token.OP, token.NAME, token.NUMBER, token.STRING]

def print_stats(data):
    print(tabulate(data, headers=["Name", "Lines", "Tokens/Line"], floatfmt=".1f")+"\n")

    for dir_name, group in itertools.groupby(sorted([(x[0].rsplit("/", 1)[0], x[1]) for x in data]), key=lambda x: x[0]):
        print(f"{dir_name:30s} : {sum([x[1] for x in group]):6d}")

    print(f"\ntotal line count: {sum([x[1] for x in data])}")

def gen_stats(base_path="."):
    table = [] # [filename, line count, tokens/line] should be mantained, used in .github/workflows/sz-diff.py
    for path, subdirs, files in os.walk(os.path.join(base_path, "tinygrad")):
        for name in files:
            if not name.endswith(".py"): continue
            filepath = os.path.join(path, name)
            relfilepath = os.path.relpath(filepath, base_path)
            with tokenize.open(filepath) as file_:
                tokens = [t for t in tokenize.generate_tokens(file_.readline) if t.type in TOKEN_WHITELIST]
                token_count, line_count = len(tokens), len(set([t.start[0] for t in tokens]))
                table.append([relfilepath, line_count, token_count/line_count])
    return sorted(table, key=lambda x: -x[1])

if __name__ == "__main__":
    print_stats(gen_stats())
