"""Versioned, non-pickle SQTT timing evidence. Packet timestamps are observations, not completion times."""
import hashlib, json, re, struct
from pathlib import Path
from tinygrad.renderer.amd.sqtt import decode, map_insts, LAYOUT_HEADER, WAVESTART, WAVEEND
from tinygrad.viz.serve import amd_decode

class CaptureError(ValueError): pass

def digest(data:bytes) -> str: return hashlib.sha256(data).hexdigest()

def validate_program(lib:bytes) -> None:
  # elf_loader allocates from section addresses. Check this harness's relocation-free ELF before calling it.
  limit = 64 << 20
  if not 64 <= len(lib) <= limit or lib[:5] != b"\x7fELF\x02": raise CaptureError("invalid ELF64 program")
  offset, count, strings = struct.unpack_from("<Q", lib, 40)[0], *struct.unpack_from("<HH", lib, 60)
  if not count or strings >= count or offset + count * 64 > len(lib): raise CaptureError("invalid ELF section table")
  sections = [struct.unpack_from("<IIQQQQIIQQ", lib, offset + i * 64) for i in range(count)]
  if sections[strings][1] != 3: raise CaptureError("invalid ELF section names")
  fixed, appended = 0, 0
  for _, kind, _, address, start, size, _, _, alignment, _ in sections:
    if kind not in (1, 3): raise CaptureError("only relocation-free corpus ELF sections are supported")
    if start + size > len(lib) or address + size > limit or alignment > limit: raise CaptureError("ELF section exceeds evidence bounds")
    if kind == 1:
      if address: fixed = max(fixed, address + size)
      else: appended += size + max(alignment, 1)
  if fixed + appended > limit: raise CaptureError("ELF image exceeds evidence bounds")

def observations(blob:bytes, lib:bytes, target:str, capacity:int) -> dict:
  if not target.startswith("gfx11"): raise CaptureError("only RDNA3 observations are supported")
  if not blob or len(blob) % 32: raise CaptureError("missing or truncated SQTT buffer")
  if capacity <= 32 or len(blob) >= capacity - 32: raise CaptureError("SQTT buffer may be saturated")
  try:
    validate_program(lib)
    packets = list(decode(blob))
    if not packets or not isinstance(packets[0], LAYOUT_HEADER) or packets[0].layout != 3:
      raise CaptureError("missing RDNA3 layout header")
    if sum(isinstance(p, LAYOUT_HEADER) for p in packets) != 1: raise CaptureError("multiple layout headers")
    simd, active, wave_count, lifetimes, origin = packets[0].simd, {}, 0, {}, None
    for p in packets:
      if not isinstance(p, (WAVESTART, WAVEEND)): continue
      key = (p.cu, p.simd, p.wave)
      if isinstance(p, WAVESTART):
        if key in active: raise CaptureError("overlapping wave lifetimes")
        active[key] = p._time
        if p.cu == 0 and p.simd == simd:
          lifetimes.setdefault(p.wave, []).append([wave_count, p._time, None])
          wave_count += 1
          if origin is None: origin = p._time
      else:
        if key not in active: raise CaptureError("wave ended without starting")
        del active[key]
        if p.cu == 0 and p.simd == simd: lifetimes[p.wave][-1][2] = p._time
    if active: raise CaptureError("incomplete wave lifetimes")
    if not wave_count: raise CaptureError("no waves in the observed CU/SIMD")
    pc_map, events = amd_decode(lib, target), []
    if not pc_map: raise CaptureError("empty program")
    first_pc = min(pc_map)
    for packet, info in map_insts(blob, lib, target):
      if isinstance(packet, (WAVESTART, WAVEEND)) and (packet.cu != 0 or packet.simd != simd): continue
      if info is None: continue
      if not lifetimes.get(info.wave): raise CaptureError("instruction outside an observed wave")
      ordinal, start, end = lifetimes[info.wave][0]
      if not start <= packet._time <= end: raise CaptureError("instruction outside an observed wave lifetime")
      if isinstance(packet, WAVEEND) and (info.pc not in pc_map or pc_map[info.pc].op_name != "S_ENDPGM"):
        raise CaptureError("trace ended before the program's end instruction")
      events.append({"wave": ordinal, "pc": info.pc - first_pc, "instruction": info.inst.to_bytes().hex(),
                     "packet": type(packet).__name__, "tick": packet._time - origin})
      if isinstance(packet, WAVEEND): lifetimes[info.wave].pop(0)
    if any(lifetimes.values()) or len(events) <= wave_count: raise CaptureError("missing instruction observations")
    return {"cu": 0, "simd": simd, "waves": wave_count, "events": events}
  except (AssertionError, KeyError, IndexError, StopIteration, RuntimeError) as exc:
    raise CaptureError(f"invalid instruction trace: {exc}") from exc

def compare(reference:dict, actual:dict) -> dict:
  """One global wave-start origin only. Never shift individual waves or rescale the time axis."""
  if any(reference[k] != actual[k] for k in ("cu", "simd", "waves")): raise CaptureError("observation domains differ")
  lhs, rhs = reference["events"], actual["events"]
  def identity(e): return e["wave"], e["pc"], e["instruction"]
  if len(lhs) != len(rhs) or any(identity(a) != identity(b) for a, b in zip(lhs, rhs)):
    return {"status": "instruction-mismatch", "reference_events": len(lhs), "actual_events": len(rhs)}
  differences = [b["tick"] - a["tick"] for a, b in zip(lhs, rhs)]
  return {"status": "match" if not any(differences) else "timing-mismatch", "tick_differences": differences,
          "max_absolute_tick_difference": max(map(abs, differences), default=0)}

def measured_region(observed:dict, start:int, end:int) -> dict:
  events = [e for e in observed["events"] if start <= e["pc"] < end]
  if len({e["wave"] for e in events}) != observed["waves"]: raise CaptureError("missing measured region in an observed wave")
  origin = min(e["tick"] for e in events)
  return {**observed, "events": [{**e, "tick": e["tick"] - origin} for e in events]}

def validate_metadata(metadata:dict, lib:bytes, code:bytes) -> None:
  from tinygrad.viz.serve import get_elf_section
  if not isinstance(metadata, dict): raise CaptureError("metadata must be an object")
  if metadata["source"] != "hardware": raise CaptureError("reference is not a hardware capture")
  for key in ("repository_sha", "tracked_diff_sha256", "harness_sha256"):
    if not isinstance(metadata[key], str) or not re.fullmatch(r"[0-9a-f]{40}" if key == "repository_sha" else r"[0-9a-f]{64}", metadata[key]):
      raise CaptureError(f"invalid {key}")
  if not isinstance(metadata["architecture"], str) or not metadata["architecture"].startswith("gfx11"):
    raise CaptureError("RDNA3 architecture required")
  if metadata["split"] not in ("calibration", "held-out"): raise CaptureError("invalid corpus split")
  if not isinstance(metadata["case"], str) or not re.fullmatch(r"[a-z0-9_]+", metadata["case"]): raise CaptureError("invalid case name")
  for key in ("repetition", "buffer_capacity", "wave_size", "submitted_waves", "se", "rsrc1", "rsrc2", "rsrc3", "lds_bytes",
              "measured_code_start", "measured_code_end"):
    if type(metadata[key]) is not int or metadata[key] < 0: raise CaptureError(f"invalid {key}")
  if metadata["repetition"] >= 20 or metadata["wave_size"] != 32: raise CaptureError("unsupported repetition or wave size")
  groups = metadata["workgroups"]
  if not isinstance(groups, list) or any(type(x) is not int for x in groups) or groups != [1, 1, 1]:
    raise CaptureError("only one workgroup is supported")
  size = metadata["local_size"]
  if not isinstance(size, list) or len(size) != 3 or any(type(x) is not int or x <= 0 for x in size):
    raise CaptureError("invalid local size")
  threads = size[0] * size[1] * size[2]
  if threads > 1024 or threads % 32 or metadata["submitted_waves"] != threads // 32: raise CaptureError("invalid wave count")
  values = metadata["outputs"]
  if not isinstance(values, list) or len(values) != threads or any(type(x) is not int or not 0 <= x < 1 << 32 for x in values):
    raise CaptureError("invalid output data")
  start, end = metadata["measured_code_start"], metadata["measured_code_end"]
  if start % 4 or end % 4 or not 0 <= start < end <= len(code): raise CaptureError("invalid measured code range")
  validate_program(lib)
  if not get_elf_section(lib, ".text").content.startswith(code): raise CaptureError("kernel bytes differ from executed program")

def write_capture(path:Path, metadata:dict, lib:bytes, code:bytes, blob:bytes) -> None:
  validate_metadata(metadata, lib, code)
  record = {"schema": 1, "metadata": metadata, "program": lib.hex(), "code": code.hex(), "trace": blob.hex(),
            "sha256": {"program": digest(lib), "code": digest(code), "trace": digest(blob)}}
  record["observations"] = observations(blob, lib, metadata["architecture"], metadata["buffer_capacity"])
  # Exclusive creation avoids silently overwriting measurements from a previous run.
  with path.open("x", encoding="utf-8") as output: json.dump(record, output, sort_keys=True, allow_nan=False)

def read_capture(path:Path) -> dict:
  def unique(pairs):
    result = {}
    for key, value in pairs:
      if key in result: raise CaptureError(f"duplicate JSON key: {key}")
      result[key] = value
    return result
  try:
    if path.stat().st_size > 64 << 20: raise CaptureError("capture exceeds the harness size limit")
    record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(record, dict) or type(record["schema"]) is not int or record["schema"] != 1:
      raise CaptureError("unsupported capture schema")
    for name in ("program", "code", "trace"):
      if digest(bytes.fromhex(record[name])) != record["sha256"][name]: raise CaptureError(f"{name} digest mismatch")
    metadata = record["metadata"]
    validate_metadata(metadata, bytes.fromhex(record["program"]), bytes.fromhex(record["code"]))
    actual = observations(bytes.fromhex(record["trace"]), bytes.fromhex(record["program"]),
                          metadata["architecture"], metadata["buffer_capacity"])
    if actual != record["observations"]: raise CaptureError("stored observations do not match the raw trace")
    return record
  except CaptureError: raise
  except (OSError, KeyError, TypeError, ValueError, AttributeError, OverflowError, AssertionError, IndexError, StopIteration, RuntimeError) as exc:
    raise CaptureError(f"invalid capture: {exc}") from exc
