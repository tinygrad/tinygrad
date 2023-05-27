#!/usr/bin/env python3

import os
import token
import tokenize


VALID_TOKENS = [
    token.OP,
    token.NAME,
    token.NUMBER,
    token.STRING
]


if __name__ == "__main__":
    count_by_file = {}

    for path, subdirs, files in os.walk("tinygrad"):
        for name in files:
            if name.endswith(".py"):
                token_count = 0
                filepath = os.path.join(path, name)
                with tokenize.open(filepath) as file_:
                    tokens = tokenize.generate_tokens(file_.readline)
                    for token in tokens:
                        if token.type in VALID_TOKENS:
                            token_count += 1
                count_by_file[filepath] = token_count

    max_length = max(len(k) for k in count_by_file.keys()) + 10
    print(f"{'File':<{max_length}}  {'Token count'}")
    print('-' * (max_length + 14))
    for key, value in count_by_file.items():
        print(f"{key:<{max_length}}  {value}")
    print('-' * (max_length + 14))
    print(f"{'Total':<{max_length}} {sum(count_by_file.values())}")
