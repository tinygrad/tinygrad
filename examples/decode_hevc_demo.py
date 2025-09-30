#!/usr/bin/env python3
"""Minimal HEVC Annex B decoding demo for NVDEC CUVID integration.

This example can run against real NVIDIA hardware or fall back to the in-tree
stub CUVID implementation used by the unit tests. The stub mode is handy for CI
and development on machines without NVDEC: it exercises the Python control flow
and prints metadata for the synthetic frames.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

# Ensure repository root is on the import path when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from tinygrad.dtype import dtypes
from tinygrad.runtime.cuvid import (
  CuvidLibrary,
  CuvidUnavailable,
  NVVideoDecoder,
  is_available as cuvid_is_available,
  nv12_frames_to_tensors,
)

try:  # Optional dependency used for the demo fallback
  from tinygrad.runtime.testing.cuvid_stub import (
    StubCuvidLib,
    build_demo_annexb_stream,
    prepare_demo_callbacks,
  )
except Exception:  # pragma: no cover - python path edge cases
  StubCuvidLib = None  # type: ignore[misc]
  build_demo_annexb_stream = None  # type: ignore[misc]
  prepare_demo_callbacks = None  # type: ignore[misc]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, help="Path to an Annex B HEVC bitstream (.h265).")
  parser.add_argument("--use-stub", action="store_true", help="Force the demo to run with the CUVID stub.")
  parser.add_argument("--convert", action="store_true", help="Convert decoded NV12 frames to RGB tensors (requires CUDA).")
  parser.add_argument("--color-space", default="bt709", help="Color space for RGB conversion (default: bt709).")
  parser.add_argument("--device", default="CUDA", help="Device name used when converting to tensors (default: CUDA).")
  parser.add_argument("--normalize", action="store_true", help="Apply 0-1 normalization when converting to tensors.")
  parser.add_argument("--max-frames", type=int, default=2, help="Number of synthetic frames in stub mode (default: 2).")
  parser.add_argument("--timestamps", action="store_true", help="Print timestamps alongside frame metadata.")
  return parser.parse_args()


def _load_annexb_data(args: argparse.Namespace) -> tuple[bytes, Optional[Sequence[int]]]:
  if args.input:
    data = args.input.read_bytes()
    return data, None
  if build_demo_annexb_stream is None:
    raise RuntimeError("Stub helpers unavailable and no input provided.")
  return build_demo_annexb_stream(frames=max(args.max_frames, 1))


def _print_summary(header: str, lines: Iterable[str]):
  print(header)
  for line in lines:
    print(f"  {line}")


def main() -> int:
  args = _parse_args()
  use_stub = args.use_stub or not cuvid_is_available()

  try:
    data, timestamps = _load_annexb_data(args)
  except FileNotFoundError:
    print(f"Input file not found: {args.input}", file=sys.stderr)
    return 2
  except RuntimeError as exc:
    print(exc, file=sys.stderr)
    print("Provide --input with a real HEVC bitstream or re-run with --use-stub.", file=sys.stderr)
    return 2

  decoder: NVVideoDecoder
  stub: Optional[StubCuvidLib] = None
  if use_stub:
    if StubCuvidLib is None or prepare_demo_callbacks is None:
      print("CUVID stub helpers are unavailable; cannot run demo.", file=sys.stderr)
      return 2
    stub = StubCuvidLib()
    stub.attach()
    prepare_demo_callbacks(stub, frames=max(args.max_frames, 1), timestamps=timestamps)
    decoder = NVVideoDecoder(cuvid=CuvidLibrary(stub))
    print("Running HEVC demo in stub mode (no NVDEC hardware detected).")
  else:
    try:
      decoder = NVVideoDecoder()
    except CuvidUnavailable as exc:
      print(f"NVDEC CUVID runtime unavailable: {exc}", file=sys.stderr)
      print("Re-run with --use-stub to execute the synthetic demo.")
      return 1

  try:
    if not use_stub and args.convert:
      outputs = decoder.decode_annexb_to_tensors(
        data,
        device=args.device,
        dtype=dtypes.float32,
        normalize=args.normalize,
        include_timestamps=True,
        color_space=args.color_space,
      )
      lines = []
      for idx, (timestamp, tensor) in enumerate(outputs):
        lines.append(
          f"Frame {idx}: ts={timestamp}, shape={tuple(tensor.shape)}, mean={tensor.mean().item():.5f}"
        )
      _print_summary("Decoded tensors", lines)
      print(f"Total frames: {len(outputs)}")
      return 0

    # Metadata-only path (stub mode or conversion disabled)
    feed_count = decoder.feed_annexb_stream(data, timestamps=timestamps)
    if stub is not None:
      stub.run_callbacks()
    frames = list(decoder.drain_frames())
    frame_meta = [(frame.timestamp, frame.width, frame.height) for frame in frames]
    print(f"Packets fed: {feed_count}")
    _print_summary(
      "Frames decoded",
      [
        (
          f"ts={ts}, size={w}x{h}" if args.timestamps else f"size={w}x{h}"
        )
        for ts, w, h in frame_meta
      ],
    )

    if frames and not use_stub and args.convert:
      tensor_stream = nv12_frames_to_tensors(
        frames,
        device=args.device,
        dtype=dtypes.float32,
        normalize=args.normalize,
        include_timestamps=True,
        color_space=args.color_space,
      )
      lines = []
      for (ts_meta, w, h), (ts_tensor, tensor) in zip(frame_meta, tensor_stream):
        assert ts_meta == ts_tensor
        lines.append(
          f"ts={ts_tensor}, tensor_shape={tuple(tensor.shape)}, mean={tensor.mean().item():.5f}"
        )
      _print_summary("Converted tensors", lines)
    else:
      for frame in frames:
        frame.release()

    print(f"Total frames: {len(frame_meta)}")
    return 0
  finally:
    decoder.close()


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
