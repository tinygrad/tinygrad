#!/usr/bin/env python3
"""Compare Clang's codegen for tinygrad's two renderings of a conditional BACKEDGE."""
import argparse, ctypes, random, shutil, subprocess, sys, tempfile, time
from pathlib import Path

C_SOURCE = r"""
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
#define NOINLINE __attribute__((noinline, aligned(64)))

#define START uint32_t a = src[x], b = a ^ 0x9e3779b9u, c = a + 0x85ebca6bu, d = a ^ 0xc2b2ae35u, i = 0
#define LIGHT a = (a ^ i) * 1664525u + 1013904223u
#define HEAVY do { \
  a = (a ^ (b >> 7)) * 1664525u + i; \
  b = (b + (c << 3)) * 22695477u + 1u; \
  c = (c ^ (d >> 11)) * 1103515245u + 12345u; \
  d = (d + (a << 5)) ^ (i * 2654435761u); \
} while (0)
#define MEMORY a = (a ^ src[(x + i) & mask]) * 1664525u + 1013904223u

#define DEFINE_PAIR(NAME, BODY) \
EXPORT NOINLINE uint64_t NAME##_do(const uint32_t *src, size_t n, uint32_t iters) { \
  uint64_t out = 0; const size_t mask = n - 1; \
  for (size_t x = 0; x < n; x++) { START; do { BODY; ++i; } while (i < iters); out += a + b + c + d; } \
  return out; \
} \
EXPORT NOINLINE uint64_t NAME##_break(const uint32_t *src, size_t n, uint32_t iters) { \
  uint64_t out = 0; const size_t mask = n - 1; \
  for (size_t x = 0; x < n; x++) { START; for (;;) { BODY; ++i; if (!(i < iters)) break; } out += a + b + c + d; } \
  return out; \
}

DEFINE_PAIR(light, LIGHT)
DEFINE_PAIR(heavy, HEAVY)
DEFINE_PAIR(memory, MEMORY)
"""

CASES = (("light", 1 << 15, 512), ("heavy", 1 << 14, 256), ("memory", 1 << 16, 128))

def median(xs:list[int]) -> int:
  vals = sorted(xs)
  return vals[len(vals)//2]

def compile_library(clang:str, workdir:Path, opt:str) -> Path:
  src = workdir / "loop_forms.c"
  lib = workdir / ("loop_forms.dylib" if sys.platform == "darwin" else "loop_forms.so")
  src.write_text(C_SOURCE)
  shared_args = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
  subprocess.run([clang, opt, "-march=native", "-ffreestanding", "-fno-math-errno", "-fno-omit-frame-pointer",
                  *shared_args, str(src), "-o", str(lib)], check=True)
  return lib

def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--clang", default=shutil.which("clang"), help="Clang executable")
  parser.add_argument("--opt", default="-O2", help="optimization flag (default: -O2, matching ClangCompiler)")
  parser.add_argument("--samples", type=int, default=15, help="timed samples per loop form")
  args = parser.parse_args()
  if args.clang is None: parser.error("clang was not found")
  if args.samples < 3: parser.error("--samples must be at least 3")

  with tempfile.TemporaryDirectory(prefix="tinygrad-loop-forms-") as td:
    lib = ctypes.CDLL(str(compile_library(args.clang, Path(td), args.opt)))
    nmax = max(n for _,n,_ in CASES)
    state, values = 1, []
    for _ in range(nmax):
      state = (state * 1664525 + 1013904223) & 0xffffffff
      values.append(state)
    data = (ctypes.c_uint32 * nmax)(*values)

    print(subprocess.check_output([args.clang, "--version"], text=True).splitlines()[0])
    print(f"{args.opt} -march=native, median of {args.samples} samples (lower is better)")
    print(f"{'kernel':<10} {'do-while':>12} {'for+break':>12} {'break/do':>10}")
    rng = random.Random(0)
    for name,n,iters in CASES:
      funcs = {form:getattr(lib, f"{name}_{form}") for form in ("do", "break")}
      for fn in funcs.values():
        fn.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.c_uint32)
        fn.restype = ctypes.c_uint64
      expected = funcs["do"](data, n, iters)
      assert funcs["break"](data, n, iters) == expected, f"{name}: loop forms produced different results"
      times = {form:[] for form in funcs}
      for _ in range(2):
        for fn in funcs.values(): fn(data, n, iters)
      for _ in range(args.samples):
        order = list(funcs)
        rng.shuffle(order)
        for form in order:
          start = time.perf_counter_ns()
          funcs[form](data, n, iters)
          times[form].append(time.perf_counter_ns() - start)
      do_ns, break_ns = median(times["do"]), median(times["break"])
      scale = n * iters
      print(f"{name:<10} {do_ns/scale:>9.3f} ns {break_ns/scale:>9.3f} ns {break_ns/do_ns:>9.3f}x")

if __name__ == "__main__": main()
