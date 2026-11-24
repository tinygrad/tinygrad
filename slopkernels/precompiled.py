"""Lazy precompiled UIR manifest loading and lookup for slopkernels.

Runtime-safe and 100% pure:
  - Performs zero disk I/O on module import; manifest JSON and `.uir` files are loaded lazily on first lookup.
  - Never invokes kernel builders (`slopkernels.src`) or offline template tracing utilities (`slopkernels.trace`).
  - Supports both concrete UIR artifacts (returned directly) and templated UIR artifacts (specialized via `UOp.specialize`).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from tinygrad import UOp
from tinygrad.uop.uir import parse_uir, render_uir

DEFAULT_MANIFEST = pathlib.Path(__file__).resolve().parent / "manifest.json"


def normalize_target(target:str|tuple[int, ...]|None) -> str|None:
  """Canonicalize an AMD target tuple/string (e.g. `(12, 0, 1)` -> `'gfx1201'`)."""
  if target is None: return None
  if isinstance(target, tuple) and len(target) == 3 and all(isinstance(x, int) and not isinstance(x, bool) for x in target):
    return f"gfx{target[0]}{target[1]}{target[2]:x}"
  return str(target)


def to_tuples(v:Any) -> Any:
  """Recursively convert JSON lists into tuples while preserving pure primitive types."""
  if isinstance(v, list): return tuple(to_tuples(x) for x in v)
  if isinstance(v, tuple): return tuple(to_tuples(x) for x in v)
  if isinstance(v, dict): return {k: to_tuples(val) for k, val in v.items()}
  return v


def _freeze(v:Any) -> Any:
  """Hashable canonical key distinguishing bool from int and converting lists to tuples."""
  if v is None: return ("N", None)
  if isinstance(v, bool): return ("B", v)
  if isinstance(v, int): return ("I", v)
  if isinstance(v, float): return ("F", float(v))
  if isinstance(v, str): return ("S", v)
  if isinstance(v, (list, tuple)): return ("T", tuple(_freeze(x) for x in v))
  if isinstance(v, dict): return ("D", tuple((k, _freeze(val)) for k, val in sorted(v.items())))
  raise TypeError(f"unsupported spec value type: {type(v).__name__}")


def _canonical_spec(spec:dict[str, Any]) -> dict[str, Any]:
  out = {k: to_tuples(v) for k, v in spec.items()}
  if "target" in out: out["target"] = normalize_target(out["target"])
  return out


def _freeze_spec(spec:dict[str, Any]) -> tuple[tuple[str, Any], ...]:
  return tuple((k, _freeze(v)) for k, v in sorted(spec.items()))


@dataclass(frozen=True)
class ArtifactEntry:
  name: str
  target: str | None
  spec: dict[str, Any]
  path: str
  size: int
  sha256: str
  templates: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManifestIndex:
  manifest_dir: pathlib.Path
  target: str | None
  artifacts: tuple[ArtifactEntry, ...]
  concrete: dict[tuple[str, tuple[tuple[str, Any], ...]], ArtifactEntry]
  templated: dict[str, tuple[ArtifactEntry, ...]]


_LOADED_MANIFESTS: dict[pathlib.Path, ManifestIndex] = {}


def is_loaded(manifest_path:str|pathlib.Path|None=None) -> bool:
  """Return True if the manifest at `manifest_path` (default `slopkernels/manifest.json`) has already been loaded."""
  return pathlib.Path(manifest_path or DEFAULT_MANIFEST).resolve() in _LOADED_MANIFESTS


def clear_cache() -> None:
  """Clear cached manifest indices and cached UIR artifacts."""
  _LOADED_MANIFESTS.clear()
  _read_artifact_text.cache_clear()
  _read_template_uop.cache_clear()


def load_manifest(manifest_path:str|pathlib.Path|None=None) -> ManifestIndex|None:
  """Lazily load and index `manifest.json`. Returns None if the manifest file does not exist."""
  p = pathlib.Path(manifest_path or DEFAULT_MANIFEST).resolve()
  if (cached := _LOADED_MANIFESTS.get(p)) is not None: return cached
  if not p.is_file(): return None
  data = json.loads(p.read_text(encoding="utf-8"))
  if data.get("version") != 1 or not isinstance(data.get("artifacts"), list):
    raise ValueError(f"invalid slopkernels manifest format in {p}")
  entries: list[ArtifactEntry] = []
  concrete: dict[tuple[str, tuple[tuple[str, Any], ...]], ArtifactEntry] = {}
  templated_map: dict[str, list[ArtifactEntry]] = {}
  for raw in data["artifacts"]:
    spec = _canonical_spec(raw["spec"])
    tmpls_raw = raw.get("templates")
    tmpls = tuple(sorted(tmpls_raw.keys() if isinstance(tmpls_raw, dict) else tmpls_raw)) if tmpls_raw else ()
    entry = ArtifactEntry(
      name=raw["name"], target=normalize_target(raw.get("target")), spec=spec,
      path=raw["path"], size=int(raw["size"]), sha256=str(raw["sha256"]), templates=tmpls,
    )
    entries.append(entry)
    if tmpls: templated_map.setdefault(entry.name, []).append(entry)
    else: concrete[(entry.name, _freeze_spec(spec))] = entry
  idx = ManifestIndex(
    manifest_dir=p.parent, target=normalize_target(data.get("target")), artifacts=tuple(entries),
    concrete=concrete, templated={k: tuple(v) for k, v in templated_map.items()},
  )
  _LOADED_MANIFESTS[p] = idx
  return idx


def _materialize_manifest(manifest_path:pathlib.Path, artifact_path:pathlib.Path) -> None:
  """Materialize all `.uir` files declared in `manifest_path` onto disk (race-safe across processes)."""
  repo_root = pathlib.Path(__file__).resolve().parents[1]
  env = {**os.environ, "DEV": "NULL", "PYTHONPATH": str(repo_root) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
  lock_path = artifact_path.parent / ".lock"
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  def _run() -> None:
    proc = subprocess.run([sys.executable, "-m", "slopkernels", "--manifest", str(manifest_path), "--materialize"],
                          env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0: raise RuntimeError(f"failed to materialize {manifest_path}: {proc.stderr or proc.stdout}")
  try:
    import fcntl
    with open(lock_path, "w") as lf:
      fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
      try:
        if not artifact_path.is_file(): _run()
      finally:
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
  except ImportError:
    if not artifact_path.is_file(): _run()


@functools.cache
def _read_artifact_text(manifest_path:pathlib.Path, artifact_path:pathlib.Path, expected_size:int, expected_sha256:str) -> str:
  if not artifact_path.is_file(): _materialize_manifest(manifest_path, artifact_path)
  raw = artifact_path.read_bytes()
  if len(raw) != expected_size:
    raise ValueError(f"precompiled UIR size mismatch for {artifact_path}: got {len(raw)}, expected {expected_size}")
  digest = hashlib.sha256(raw).hexdigest()
  if digest != expected_sha256:
    raise ValueError(f"precompiled UIR sha256 mismatch for {artifact_path}: got {digest}, expected {expected_sha256}")
  return raw.decode("utf-8").rstrip("\n")


@functools.cache
def _read_template_uop(manifest_path:pathlib.Path, artifact_path:pathlib.Path, expected_size:int, expected_sha256:str) -> UOp:
  return parse_uir(_read_artifact_text(manifest_path, artifact_path, expected_size, expected_sha256))


def lookup(name:str, spec:dict[str, Any], *, manifest_path:str|pathlib.Path|None=None) -> str|None:
  """Look up a precompiled UIR artifact for `(name, spec)` in the manifest. Returns None on a miss."""
  mpath = pathlib.Path(manifest_path or DEFAULT_MANIFEST).resolve()
  idx = load_manifest(mpath)
  if idx is None: return None
  norm = _canonical_spec(spec)
  # 1. Exact concrete hit
  if (entry := idx.concrete.get((name, _freeze_spec(norm)))) is not None:
    return _read_artifact_text(mpath, idx.manifest_dir / entry.path, entry.size, entry.sha256)
  # 2. Templated artifact hit: match all structural keys, specialize template scalar keys
  for entry in idx.templated.get(name, ()):
    expected_keys = set(entry.spec) | set(entry.templates)
    if set(norm) != expected_keys: continue
    if any(_freeze(norm[k]) != _freeze(entry.spec[k]) for k in entry.spec if k not in entry.templates): continue
    tvals = {k: norm[k] for k in entry.templates}
    tmpl_uop = _read_template_uop(mpath, idx.manifest_dir / entry.path, entry.size, entry.sha256)
    try: specialized = tmpl_uop.specialize(**tvals)
    except (ValueError, TypeError): continue
    return render_uir(specialized, color=False)
  return None
