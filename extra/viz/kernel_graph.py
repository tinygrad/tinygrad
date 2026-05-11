#!/usr/bin/env python3
import argparse, json, sys
from tinygrad.helpers import ansistrip

def main() -> int:
  parser = argparse.ArgumentParser(description="Read tinygrad.viz.cli --json output from stdin")
  parser.add_argument("kernel", help="Kernel name substring to match")
  args = parser.parse_args()

  found = False
  for line in sys.stdin:
    if not line.strip(): continue
    row = json.loads(line)
    text = ansistrip(json.dumps(row, sort_keys=True))
    if args.kernel in text:
      found = True
      print(json.dumps(row))
  return 0 if found else 1

if __name__ == "__main__": sys.exit(main())
