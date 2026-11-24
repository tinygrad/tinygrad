"""CLI: generate all LLM `.uir` kernels (`python3 -m slopkernels`) or compile all slopkernels (`python3 -m slopkernels OUT_DIR [--target ARCH]`)."""
import argparse

from slopkernels import compile_all, generate_all

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("out_dir", nargs="?")
  ap.add_argument("--target", default=None, help="AMD target arch (e.g. gfx1100, gfx1201)")
  args = ap.parse_args()
  if args.out_dir is None and args.target is None:
    uirs = generate_all()
    print(f"generated {len(uirs)} LLM kernels -> slopkernels/uir/ ({sum(len(u) for u in uirs.values()):,} bytes)")
  else:
    uirs = compile_all(args.out_dir, target=args.target)
    total_lines = sum(u.count("\n") + 1 for u in uirs.values())
    dest = f" -> {args.out_dir}" if args.out_dir else ""
    tgt = f" [{args.target}]" if args.target else ""
    print(f"compiled {len(uirs)} slopkernels to UIR{tgt} ({total_lines} lines){dest}")
