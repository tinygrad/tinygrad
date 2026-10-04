"""Offline packer for an explicitly validated physical-call schedule.

python -m extra.gptoss_kernels.hcq_schedule.build --graph graph.json \
  --programs programs.json --order order.json --name descriptive_name

Prints a self-contained recipe to stdout. It does not translate binary hashes,
infer timing equivalence, claim a GPU speedup, or install an untested order.
"""
import argparse, json, pathlib
from extra.gptoss_kernels.hcq_schedule import make_plan

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for field in ("graph", "programs", "order"): parser.add_argument(f"--{field}", required=True, type=pathlib.Path)
  parser.add_argument("--name", required=True)
  args = parser.parse_args()
  print(json.dumps(make_plan(*(json.loads(getattr(args, field).read_text()) for field in ("graph", "programs", "order")), args.name), indent=2))

if __name__ == "__main__": main()
