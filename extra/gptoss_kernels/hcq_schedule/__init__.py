"""Replay a measured GPT-OSS HCQ2 order only for its exact post-allocation graph.

No scheduling decisions use generated kernel names or estimates. The recipe is
bound to source, binary, geometry, argument ABI, physical aliases and every
RAW/WAR/WAW dependency. Changed graphs use the original order. HCQ2 must rebuild
BatchCtx after this function; it still generates all signals and waits itself.
"""
from __future__ import annotations
import base64, functools, hashlib, json, pathlib, struct, zlib
from typing import Any

from tinygrad.helpers import DEBUG

def _json(value) -> bytes: return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()

def fingerprint(rows:list[dict], programs:dict) -> str:
  # Program keys and generated names are deliberately excluded: their identity
  # is represented by the complete source, loaded binary and launch geometry.
  signatures = {key:(hashlib.sha256(p["source"].encode()).hexdigest(), p["binary_sha256"], p["geometry"])
                for key, p in programs.items()}
  digest = hashlib.sha256(b"gptoss-hcq-physical-v1\n")
  for r in rows:
    buffers = [(b["base"], b["lane"], b["offset"], b["bytes"], b["shape"], b["dtype"], b["device"]) for b in r["buffers"]]
    signature = signatures[r["key"]] if r["key"] is not None else None
    digest.update(_json((r["device"], r["queue"], signature, r["writes"], r["reads"], buffers, r["deps"])))
    digest.update(b"\n")
  return digest.hexdigest()

def validate_order(rows:list[dict], order:list[int]) -> bool:
  if len(order) != len(rows) or sorted(order) != list(range(len(rows))): return False
  positions = [0] * len(order)
  for pos, old in enumerate(order): positions[old] = pos
  return all(0 <= d < i and positions[d] < positions[i] for i, r in enumerate(rows) for d in r["deps"])

def make_plan(rows:list[dict], programs:dict, order:list[int], name:str) -> dict:
  if not validate_order(rows, order): raise ValueError("order is not an exact hazard-preserving permutation")
  encoded = struct.pack(f"<{len(order)}i", *(i - prev for i, prev in zip(order, [0, *order[:-1]])))
  compressed = base64.b85encode(zlib.compress(encoded, 9)).decode()
  return dict(version=1, name=name, qualification="cpu_candidate", nodes=len(rows), hazards=sum(len(r["deps"]) for r in rows),
              graph_sha256=fingerprint(rows, programs),
              order_sha256=hashlib.sha256(encoded).hexdigest(), order_b85=[compressed[i:i+100] for i in range(0, len(compressed), 100)])

def decode_order(plan:dict) -> list[int]:
  if plan["version"] != 1: raise ValueError("unsupported HCQ schedule recipe")
  decoded = zlib.decompress(base64.b85decode("".join(plan["order_b85"])))
  if len(decoded) != plan["nodes"] * 4 or hashlib.sha256(decoded).hexdigest() != plan["order_sha256"]:
    raise ValueError("corrupt HCQ schedule recipe")
  import itertools
  return list(itertools.accumulate(struct.unpack(f"<{plan['nodes']}i", decoded)))

def capture(batch:list) -> tuple[list[dict], dict]:
  """Canonical physical graph. This inspects UOps and never opens a device."""
  from tinygrad.engine.realize import get_call_arg_uops, get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import HCQDepsTracker, unwrap_lane
  from tinygrad.uop.ops import Ops
  tracker, bases, rows, programs = HCQDepsTracker(), {}, [], {}
  for i, (call, devices, queue) in enumerate(batch):
    if call.op is not Ops.CALL or call.body.op not in (Ops.STORE, Ops.PROGRAM): raise ValueError("unsupported HCQ call")
    args = list(get_call_arg_uops(call))
    writes, reads = get_call_outs_ins(call)
    deps = sorted(set(tracker.access_resources(args, list(range(len(args))) if writes is None else writes, i)))
    buffers = []
    for b in args:
      base, lane, offset = unwrap_lane(b)
      buffers.append(dict(base=bases.setdefault(base, len(bases)), lane=lane, offset=int(offset), bytes=b.nbytes(),
                          shape=b.shape, dtype=str(b.dtype), device=b.device))
    program = call.body if call.body.op is Ops.PROGRAM else None
    key = program.key.hex() if program is not None else None
    if program is not None and key not in programs:
      programs[key] = dict(source=program.src[2].arg, binary_sha256=hashlib.sha256(program.src[3].arg).hexdigest(),
                           geometry=[list(program.arg.global_size), list(program.arg.local_size)])
    rows.append(dict(device=devices[0], queue=queue, key=key, buffers=buffers, writes=writes, reads=reads, deps=deps))
  return rows, programs

@functools.cache
def _plans() -> tuple[tuple[dict, list[int]], ...]:
  return tuple((p, decode_order(p)) for path in sorted(pathlib.Path(__file__).parent.glob("*.plan.json")) if (p:=json.loads(path.read_text())))

def matching_order(rows:list[dict], programs:dict, plans:tuple[tuple[dict, list[int]], ...]) -> tuple[list[int]|None, str]:
  graph = fingerprint(rows, programs)
  for plan, order in plans:
    if plan["nodes"] != len(rows) or plan["graph_sha256"] != graph: continue
    # Independently check every actual hazard even though the graph digest matches.
    if not validate_order(rows, order): return None, "recipe violates physical dependencies"
    return order, plan["name"]
  return None, f"unknown physical graph {graph[:16]}"

def reorder_batch(batch:list[Any]) -> list[Any]:
  eligible = False
  try:
    plans = tuple((p, o) for p, o in _plans() if p["nodes"] == len(batch))
    if not plans: return batch
    eligible = True
    rows, programs = capture(batch)
    order, reason = matching_order(rows, programs, plans)
  except (KeyError, ValueError, TypeError, OverflowError, OSError, struct.error, zlib.error) as e:
    if eligible or DEBUG >= 1: print(f"GPTOSS HCQ schedule: stock order ({e})")
    return batch
  if order is None:
    print(f"GPTOSS HCQ schedule: stock order ({reason})")
    return batch
  print(f"GPTOSS HCQ schedule: {reason} ({len(batch)} calls, all physical hazards preserved)")
  return [batch[i] for i in order]
