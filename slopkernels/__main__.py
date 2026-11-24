"""Offline CLI for slopkernels UIR compilation and manifest generation/verification.

Usage:
  python3 -m slopkernels [out_dir] [--target ARCH]
  python3 -m slopkernels --write-manifest MANIFEST [out_dir] [--target ARCH]
  python3 -m slopkernels --manifest MANIFEST [out_dir]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

from slopkernels import _REGISTRY, _canonical_spec, compile_all, list_kernels
from slopkernels.precompiled import normalize_target, to_tuples


def _to_json(v:Any) -> Any:
  if isinstance(v, tuple): return [_to_json(x) for x in v]
  if isinstance(v, dict): return {k: _to_json(val) for k, val in v.items()}
  return v


def write_manifest(manifest_path:pathlib.Path, out_dir:pathlib.Path|None=None, target:str="gfx1201") -> dict[str, Any]:
  """Generate all canonical UIR artifacts for `target` and write `manifest.json` + `.uir` files."""
  manifest_path = manifest_path.resolve()
  uir_dir = (out_dir.resolve() if out_dir is not None else manifest_path.parent / "uir")
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  uir_dir.mkdir(parents=True, exist_ok=True)
  norm_target = normalize_target(target)
  artifacts: list[dict[str, Any]] = []
  for name in list_kernels():
    spec = _canonical_spec(name, target=norm_target)
    uir = _REGISTRY[name][0](**spec)
    assert uir is not None, f"canonical sample for {name!r} returned None"
    raw = (uir + "\n").encode("utf-8")
    dst = uir_dir / f"{name}.uir"
    dst.write_bytes(raw)
    try: rel_path = dst.relative_to(manifest_path.parent).as_posix()
    except ValueError: rel_path = dst.as_posix()
    artifacts.append({
      "name": name,
      "target": norm_target,
      "spec": _to_json(spec),
      "path": rel_path,
      "size": len(raw),
      "sha256": hashlib.sha256(raw).hexdigest(),
    })
  manifest = {"version": 1, "target": norm_target, "artifacts": artifacts}
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  return manifest


def verify_manifest(manifest_path:pathlib.Path, out_dir:pathlib.Path|None=None) -> tuple[int, int]:
  """Regenerate every artifact in `manifest_path` and verify its declared byte size and SHA-256 hash."""
  manifest_path = manifest_path.resolve()
  data = json.loads(manifest_path.read_text(encoding="utf-8"))
  if data.get("version") != 1 or not isinstance(data.get("artifacts"), list):
    raise ValueError(f"invalid manifest format: {manifest_path}")
  if data.get("kind") == "llm_kernels":
    from slopkernels.export_llm import verify_llm_manifest
    return verify_llm_manifest(manifest_path)
  total_bytes = 0
  for entry in data["artifacts"]:
    name = entry["name"]
    spec = to_tuples(entry["spec"])
    tmpls = entry.get("templates")
    if tmpls:
      from slopkernels.trace import trace_kernel
      uir = trace_kernel(name, spec, tmpls)
    else:
      if name not in _REGISTRY: raise KeyError(f"unknown slopkernel {name!r} in {manifest_path}")
      built = _REGISTRY[name][0](**spec)
      if built is None: raise ValueError(f"builder for {name!r} returned None while regenerating manifest entry")
      uir = built
    raw = (uir + "\n").encode("utf-8")
    size, digest = len(raw), hashlib.sha256(raw).hexdigest()
    if size != entry["size"]:
      raise ValueError(f"{name}: regenerated size {size} != manifest size {entry['size']}")
    if digest != entry["sha256"]:
      raise ValueError(f"{name}: regenerated sha256 {digest} != manifest sha256 {entry['sha256']}")
    if out_dir is not None:
      dst = out_dir.resolve() / entry["path"]
      dst.parent.mkdir(parents=True, exist_ok=True)
      dst.write_bytes(raw)
    else:
      on_disk = manifest_path.parent / entry["path"]
      if on_disk.is_file():
        disk_raw = on_disk.read_bytes()
        if len(disk_raw) != entry["size"] or hashlib.sha256(disk_raw).hexdigest() != entry["sha256"]:
          raise ValueError(f"{name}: on-disk artifact {on_disk} does not match manifest size/sha256")
    total_bytes += size
  return len(data["artifacts"]), total_bytes


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("out_dir", nargs="?")
  ap.add_argument("--target", default=None, help="AMD target arch (e.g. gfx1100, gfx1201)")
  ap.add_argument("--manifest", default=None, help="Regenerate every artifact in MANIFEST and verify size + SHA-256")
  ap.add_argument("--write-manifest", default=None, help="Generate canonical UIR artifacts and write MANIFEST")
  ap.add_argument("--export-llm", default=None, help="Export templated LLM UIR kernels + manifest.json to directory")
  args = ap.parse_args()
  out_path = pathlib.Path(args.out_dir) if args.out_dir else None
  if args.export_llm:
    from slopkernels.export_llm import export_llm_kernels
    m = export_llm_kernels(pathlib.Path(args.export_llm))
    total_bytes = sum(a["size"] for a in m["artifacts"])
    print(f"exported {len(m['artifacts'])} templated LLM kernels ({total_bytes:,} bytes) -> {args.export_llm}")
  elif args.write_manifest:
    m = write_manifest(pathlib.Path(args.write_manifest), out_dir=out_path, target=args.target or "gfx1201")
    total_bytes = sum(a["size"] for a in m["artifacts"])
    print(f"wrote {len(m['artifacts'])} artifacts ({total_bytes:,} bytes) -> {args.write_manifest}")
  elif args.manifest:
    try:
      count, total_bytes = verify_manifest(pathlib.Path(args.manifest), out_dir=out_path)
    except Exception as e:
      print(f"manifest verification failed: {e}", file=sys.stderr)
      sys.exit(1)
    print(f"verified {count} artifacts ({total_bytes:,} bytes) from {args.manifest}")
  else:
    uirs = compile_all(args.out_dir, target=args.target)
    total_lines = sum(u.count("\n") + 1 for u in uirs.values())
    dest = f" -> {args.out_dir}" if args.out_dir else ""
    tgt = f" [{args.target}]" if args.target else ""
    print(f"compiled {len(uirs)} slopkernels to UIR{tgt} ({total_lines} lines){dest}")
