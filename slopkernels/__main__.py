"""Generate all static templated UIR kernels into `slopkernels/uir/`: `DEV=NULL python3 -m slopkernels`."""
from slopkernels import generate_all

if __name__ == "__main__":
  uirs = generate_all()
  print(f"generated {len(uirs)} kernels -> slopkernels/uir/ ({sum(len(u) for u in uirs.values()):,} bytes)")
