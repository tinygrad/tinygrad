"""CLI: compile all slopkernels offline to UIR text (`python3 -m slopkernels [out_dir]`)."""
import sys

from slopkernels import compile_all

if __name__ == "__main__":
  out_dir = sys.argv[1] if len(sys.argv) > 1 else None
  uirs = compile_all(out_dir)
  total_lines = sum(u.count("\n") + 1 for u in uirs.values())
  dest = f" -> {out_dir}" if out_dir else ""
  print(f"compiled {len(uirs)} slopkernels to UIR ({total_lines} lines){dest}")
