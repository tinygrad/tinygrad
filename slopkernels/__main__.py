"""CLI: compile all slopkernels offline to UIR text (`python3 -m slopkernels [out_dir] [--target ARCH]`)."""
import argparse

from slopkernels import compile_all

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("out_dir", nargs="?")
  ap.add_argument("--target", default=None, help="AMD target arch (e.g. gfx1100, gfx1201)")
  args = ap.parse_args()
  uirs = compile_all(args.out_dir, target=args.target)
  total_lines = sum(u.count("\n") + 1 for u in uirs.values())
  dest = f" -> {args.out_dir}" if args.out_dir else ""
  tgt = f" [{args.target}]" if args.target else ""
  print(f"compiled {len(uirs)} slopkernels to UIR{tgt} ({total_lines} lines){dest}")
