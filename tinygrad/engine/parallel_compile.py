"""Parallel kernel compilation for tinygrad.

This module provides utilities for compiling multiple kernels in parallel
using multiprocessing, which can significantly speed up first-run compile times.
"""
from __future__ import annotations
import multiprocessing as mp
from dataclasses import dataclass
from tinygrad.helpers import getenv, DEBUG

# Environment variable to control parallel compilation
PARALLEL_COMPILE = getenv("PARALLEL_COMPILE", 1)
PARALLEL_COMPILE_WORKERS = getenv("PARALLEL_COMPILE_WORKERS", 0)  # 0 = auto

@dataclass
class CompileTask:
  """A kernel compilation task."""
  key: bytes  # Cache key (ast.key)
  source: str  # Rendered source code
  compiler_name: str  # Compiler class name for recreation

@dataclass
class CompileResult:
  """Result of a compilation."""
  key: bytes
  binary: bytes | None
  error: str | None = None

def _compile_one(task: tuple[bytes, str, str, str | None]) -> tuple[bytes, bytes | None, str | None]:
  """Compile a single kernel in worker process.

  Args:
    task: (key, source, compiler_module, compiler_cachekey)

  Returns:
    (key, binary, error)
  """
  key, source, compiler_info, cachekey = task
  try:
    # Dynamically import and instantiate the compiler
    # compiler_info is "module:class" format
    module_name, class_name = compiler_info.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_name)
    compiler_cls = getattr(module, class_name)
    compiler = compiler_cls(cachekey) if cachekey else compiler_cls()
    binary = compiler.compile(source)
    return (key, binary, None)
  except Exception as e:
    return (key, None, str(e))

def _worker_init():
  """Initialize worker process."""
  # Disable any GPU context in workers - we only do CPU compilation
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = ""

def parallel_compile(tasks: list[tuple[bytes, str, type, str | None]],
                     num_workers: int | None = None) -> dict[bytes, bytes]:
  """Compile multiple kernels in parallel.

  Args:
    tasks: List of (key, source, compiler_class, cachekey)
    num_workers: Number of workers (default: auto based on CPU count)

  Returns:
    Dictionary mapping key to compiled binary
  """
  if not tasks:
    return {}

  # Determine number of workers
  if num_workers is None:
    if PARALLEL_COMPILE_WORKERS > 0:
      num_workers = PARALLEL_COMPILE_WORKERS
    else:
      num_workers = min(mp.cpu_count(), len(tasks), 8)

  # For small batches or single worker, compile sequentially
  if num_workers <= 1 or len(tasks) <= 2:
    results = {}
    for key, source, compiler_cls, cachekey in tasks:
      compiler = compiler_cls(cachekey) if cachekey else compiler_cls()
      results[key] = compiler.compile(source)
    return results

  # Serialize tasks for multiprocessing
  # Format: (key, source, "module:class", cachekey)
  serialized_tasks = []
  for key, source, compiler_cls, cachekey in tasks:
    compiler_info = f"{compiler_cls.__module__}:{compiler_cls.__name__}"
    serialized_tasks.append((key, source, compiler_info, cachekey))

  if DEBUG >= 2:
    print(f"parallel_compile: {len(tasks)} tasks with {num_workers} workers")

  # Compile in parallel using process pool
  results = {}
  try:
    # Use spawn to avoid issues with forking
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers, initializer=_worker_init) as pool:
      for key, binary, error in pool.imap_unordered(_compile_one, serialized_tasks):
        if error:
          raise RuntimeError(f"Compilation failed for key {key.hex()[:8]}: {error}")
        if binary is not None:
          results[key] = binary
  except Exception as e:
    if DEBUG >= 1:
      print(f"parallel_compile failed: {e}, falling back to sequential")
    # Fallback to sequential compilation
    results = {}
    for key, source, compiler_cls, cachekey in tasks:
      compiler = compiler_cls(cachekey) if cachekey else compiler_cls()
      results[key] = compiler.compile(source)

  return results
