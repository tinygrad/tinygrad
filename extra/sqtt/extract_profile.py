#!/usr/bin/env python3
"""Extract a small, deterministic SQTT oracle from legacy profile pickles without unpickling them."""
from __future__ import annotations

import argparse, base64, hashlib, json, pickletools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ExtractError(ValueError): pass


@dataclass(frozen=True)
class Limits:
  max_input_bytes: int = 64 * 1024 * 1024
  max_opcodes: int = 1_000_000
  max_depth: int = 128
  max_memo_entries: int = 100_000
  max_container_elements: int = 1_000_000
  max_string_bytes: int = 1024 * 1024
  max_blob_bytes: int = 32 * 1024 * 1024
  max_integer_bits: int = 256
  max_events: int = 100_000
  max_aggregate_blob_bytes: int = 32 * 1024 * 1024
  max_output_bytes: int = 64 * 1024 * 1024


@dataclass(frozen=True)
class _Global:
  module: str
  name: str


@dataclass
class _Record:
  kind: _Global
  args: tuple[Any, ...] = ()
  state: dict[str, Any] = field(default_factory=dict)


_MARK = object()
_GLOBALS = {
  ("decimal", "Decimal"),
  ("tinygrad.dtype", "AddrSpace"),
  ("tinygrad.dtype", "DType"),
  ("tinygrad.dtype", "PtrDType"),
  ("tinygrad.device", "ProfileDeviceEvent"),
  ("tinygrad.device", "ProfileProgramEvent"),
  ("tinygrad.helpers", "ProfilePointEvent"),
  ("tinygrad.helpers", "ProfileRangeEvent"),
  ("tinygrad.helpers", "TracingKey"),
  ("tinygrad.runtime.ops_amd", "PMCSample"),
  ("tinygrad.runtime.ops_amd", "ProfilePMCEvent"),
  ("tinygrad.runtime.ops_amd", "ProfileSQTTEvent"),
  ("tinygrad.uop", "Ops"),
  ("tinygrad.uop.ops", "AxisType"),
  ("tinygrad.uop.ops", "KernelInfo"),
  ("tinygrad.uop.ops", "UOp"),
  *((f"tinygrad.runtime.autogen.amd.{arch}.ins", name) for arch, names in {
    "rdna3": ("DS", "GLOBAL", "SMEM", "SOPP", "VOP1", "VOP2", "VOPC_LIT"),
    "rdna4": ("DS", "SMEM", "SOP1", "SOP2", "SOPC", "SOPP", "VGLOBAL", "VOP1", "VOP2", "VOP3P", "VOP3SD", "VOP3_SDST", "VOPC_LIT"),
  }.items() for name in names),
}
_PASSIVE = {"PROTO", "FRAME"}
_ALLOWED = _PASSIVE | {
  "STOP", "MARK", "NONE", "NEWTRUE", "NEWFALSE", "BININT", "BININT1", "BININT2", "LONG1", "LONG4",
  "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8",
  "EMPTY_LIST", "EMPTY_TUPLE", "EMPTY_DICT", "TUPLE", "TUPLE1", "TUPLE2", "TUPLE3", "APPEND", "APPENDS",
  "SETITEM", "SETITEMS", "MEMOIZE", "BINPUT", "LONG_BINPUT", "BINGET", "LONG_BINGET", "GLOBAL", "STACK_GLOBAL",
  "NEWOBJ", "REDUCE", "BUILD",
}


def _fail(message: str, pos: int | None = None) -> ExtractError:
  return ExtractError(message if pos is None else f"{message} at byte {pos}")


def _parse(data: bytes, limits: Limits) -> Any:
  if len(data) > limits.max_input_bytes: raise _fail("input bytes limit exceeded")
  stack: list[Any] = []
  memo: dict[int, Any] = {}
  elements = marks = count = 0
  frame_end: int | None = None
  stopped = False

  def push(value: Any) -> None: stack.append(value)
  def pop(pos: int) -> Any:
    if not stack: raise _fail("stack underflow", pos)
    return stack.pop()
  def marked(pos: int) -> list[Any]:
    nonlocal marks
    try: idx = len(stack) - 1 - stack[::-1].index(_MARK)
    except ValueError: raise _fail("missing MARK", pos) from None
    vals = stack[idx + 1:]
    del stack[idx:]
    marks -= 1
    return vals
  def add_elements(number: int, pos: int) -> None:
    nonlocal elements
    elements += number
    if elements > limits.max_container_elements: raise _fail("container element limit exceeded", pos)
  def put_memo(index: int, value: Any, pos: int) -> None:
    if index < 0 or index >= limits.max_memo_entries: raise _fail("memo entry limit exceeded", pos)
    if index not in memo and len(memo) >= limits.max_memo_entries: raise _fail("memo entry limit exceeded", pos)
    memo[index] = value

  try:
    operations = pickletools.genops(data)
    for opcode, arg, pos in operations:
      if pos is None: raise _fail("malformed pickle")
      position = pos
      count += 1
      if count > limits.max_opcodes: raise _fail("opcode count limit exceeded", position)
      name = opcode.name
      if count == 1 and (name != "PROTO" or position != 0): raise _fail("pickle must start with PROTO", position)
      if frame_end is not None:
        if position > frame_end: raise _fail("malformed FRAME", position)
        if position == frame_end: frame_end = None
        elif name == "FRAME": raise _fail("nested FRAME", position)
      if name in {"INT", "LONG"}: raise _fail("malformed pickle")
      if name not in _ALLOWED: raise _fail(f"unsupported opcode {name}", position)
      if name in _PASSIVE:
        if name == "PROTO":
          if count != 1 or not isinstance(arg, int) or arg < 2 or arg > 5: raise _fail("unsupported pickle protocol", position)
        elif not isinstance(arg, int) or arg < 0:
          raise _fail("malformed FRAME", position)
        else:
          frame_end = position + 9 + arg
          if frame_end > len(data): raise _fail("malformed FRAME", position)
      elif name == "MARK":
        marks += 1
        if marks > limits.max_depth: raise _fail("depth limit exceeded", position)
        push(_MARK)
      elif name == "NONE": push(None)
      elif name == "NEWTRUE": push(True)
      elif name == "NEWFALSE": push(False)
      elif name in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4"}:
        if not isinstance(arg, int) or arg.bit_length() > limits.max_integer_bits: raise _fail("integer size limit exceeded", position)
        push(arg)
      elif name in {"BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"}:
        if not isinstance(arg, str) or len(arg.encode("utf-8")) > limits.max_string_bytes: raise _fail("string/blob size limit exceeded", position)
        push(arg)
      elif name in {"BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8"}:
        if not isinstance(arg, (bytes, bytearray)):
          raise _fail("malformed pickle", position)
        blob = bytes(arg)
        if len(blob) > limits.max_blob_bytes: raise _fail("string/blob size limit exceeded", position)
        push(blob)
      elif name == "EMPTY_LIST": push([])
      elif name == "EMPTY_TUPLE": push(())
      elif name == "EMPTY_DICT": push({})
      elif name == "TUPLE":
        tuple_vals = tuple(marked(position))
        add_elements(len(tuple_vals), position)
        push(tuple_vals)
      elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
        number = int(name[-1])
        tuple1_vals = [pop(position) for _ in range(number)][::-1]
        add_elements(number, position)
        push(tuple(tuple1_vals))
      elif name == "APPEND":
        value, target = pop(position), pop(position)
        if not isinstance(target, list): raise _fail("APPEND target is not a list", position)
        add_elements(1, position)
        target.append(value)
        push(target)
      elif name == "APPENDS":
        append_vals = marked(position)
        if not stack or not isinstance(stack[-1], list): raise _fail("APPENDS target is not a list", position)
        add_elements(len(append_vals), position)
        stack[-1].extend(append_vals)
      elif name == "SETITEM":
        value, key, target = pop(position), pop(position), pop(position)
        if not isinstance(target, dict): raise _fail("SETITEM target is not a dict", position)
        add_elements(1, position)
        target[key] = value
        push(target)
      elif name == "SETITEMS":
        setitem_vals = marked(position)
        if len(setitem_vals) % 2 or not stack or not isinstance(stack[-1], dict): raise _fail("malformed SETITEMS", position)
        add_elements(len(setitem_vals) // 2, position)
        for i in range(0, len(setitem_vals), 2):
          stack[-1][setitem_vals[i]] = setitem_vals[i + 1]
      elif name in {"MEMOIZE", "BINPUT", "LONG_BINPUT"}:
        if not stack: raise _fail("memoize with empty stack", position)
        if name == "MEMOIZE":
          index = len(memo)
        else:
          if not isinstance(arg, int): raise _fail("malformed pickle", position)
          index = arg
        put_memo(index, stack[-1], position)
      elif name in {"BINGET", "LONG_BINGET"}:
        if not isinstance(arg, int) or arg not in memo: raise _fail("missing memo entry", position)
        push(memo[arg])
      elif name == "GLOBAL":
        module, sep, global_name = str(arg).partition(" ")
        if not sep or (module, global_name) not in _GLOBALS: raise _fail("unsupported pickle global", pos)
        push(_Global(module, global_name))
      elif name == "STACK_GLOBAL":
        global_name, module = pop(position), pop(position)
        if not isinstance(module, str) or not isinstance(global_name, str) or (module, global_name) not in _GLOBALS:
          raise _fail("unsupported pickle global", position)
        push(_Global(module, global_name))
      elif name in {"NEWOBJ", "REDUCE"}:
        args, constructor = pop(position), pop(position)
        if not isinstance(constructor, _Global) or not isinstance(args, tuple): raise _fail(f"malformed inert {name}", position)
        # This is data modelling only: no lookup, import, constructor, reducer, or state hook is invoked.
        push(_Record(constructor, args))
      elif name == "BUILD":
        state, target = pop(position), pop(position)
        if not isinstance(target, _Record) or not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
          raise _fail("malformed inert BUILD", position)
        target.state.update(state)
        push(target)
      elif name == "STOP":
        if marks or len(stack) != 1: raise _fail("malformed final stack", position)
        if frame_end is not None and frame_end != len(data): raise _fail("malformed FRAME", position)
        if position != len(data) - 1: raise _fail("trailing pickle data", position)
        stopped = True
        break
  except ExtractError: raise
  except Exception: raise _fail("malformed pickle") from None
  if not stopped: raise _fail("pickle has no STOP")
  active: set[int] = set()
  def check_depth(value: Any, depth: int) -> None:
    if depth > limits.max_depth: raise _fail("depth limit exceeded")
    if not isinstance(value, (list, tuple, dict, _Record)): return
    ident = id(value)
    if ident in active: raise _fail("cyclic pickle graph")
    active.add(ident)
    if isinstance(value, _Record):
      children = (*value.args, value.state)
    elif isinstance(value, dict):
      children = tuple(value.keys()) + tuple(value.values())
    else:
      children = tuple(value)
    for child in children: check_depth(child, depth + 1)
    active.remove(ident)
  check_depth(stack[0], 0)
  return stack[0]


def _digest(blob: bytes) -> dict[str, int | str]:
  return {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def _field(record: _Record, name: str, expected: type | tuple[type, ...], optional: bool = False) -> Any:
  value = record.state.get(name)
  if optional and value is None: return None
  integer_only = expected is int or isinstance(expected, tuple) and int in expected and bool not in expected
  if not isinstance(value, expected) or isinstance(value, bool) and integer_only: raise ExtractError(f"invalid {record.kind.name}.{name}")
  return value


def _is_amd_device(device: str) -> bool:
  return device == "AMD" or device.startswith("AMD:") and all(part.isascii() and part.isdigit() for part in device.split(":")[1:])


def extract_profile(data: bytes, limits: Limits = Limits()) -> dict[str, Any]:
  """Return schema-v2 metadata from a bounded legacy tinygrad profile pickle.

  Binary payloads occur once in ``blobs``, keyed by their SHA-256 digest. Event
  ``elf`` and ``blob`` fields are digest references into that table. This keeps
  serialized payload bytes bounded by unique content rather than memo aliases.
  """
  root = _parse(data, limits)
  if not isinstance(root, list): raise ExtractError("profile root is not a list")
  if len(root) > limits.max_events: raise ExtractError("event count limit exceeded")
  devices, programs, streams = [], [], []
  raw_blobs: dict[str, bytes] = {}
  aggregate_blob_bytes = 0

  def blob_reference(blob: bytes) -> str:
    nonlocal aggregate_blob_bytes
    digest = hashlib.sha256(blob).hexdigest()
    if digest in raw_blobs:
      if raw_blobs[digest] != blob: raise ExtractError("SHA-256 collision")
      return digest
    aggregate_blob_bytes += len(blob)
    if aggregate_blob_bytes > limits.max_aggregate_blob_bytes: raise ExtractError("aggregate blob bytes limit exceeded")
    raw_blobs[digest] = blob
    return digest

  for event in root:
    if not isinstance(event, _Record): continue
    if event.kind == _Global("tinygrad.device", "ProfileDeviceEvent"):
      device = _field(event, "device", str)
      if not _is_amd_device(device): continue
      props = _field(event, "props", (dict, type(None)), optional=True) or {}
      if not all(isinstance(k, str) and isinstance(v, (str, int, bool, type(None))) for k, v in props.items()):
        raise ExtractError("invalid ProfileDeviceEvent.props")
      devices.append({"device": device, "props": dict(sorted(props.items()))})
    elif event.kind == _Global("tinygrad.device", "ProfileProgramEvent"):
      device = _field(event, "device", str)
      if not _is_amd_device(device): continue
      elf = _field(event, "lib", (bytes, type(None)), optional=True)
      programs.append({"device": device, "tag": _field(event, "tag", (int, type(None)), optional=True),
                        "name": _field(event, "name", str), "base": _field(event, "base", (int, type(None)), optional=True),
                        "elf": None if elf is None else blob_reference(elf)})
    elif event.kind == _Global("tinygrad.runtime.ops_amd", "ProfileSQTTEvent"):
      device = _field(event, "device", str)
      if not _is_amd_device(device): continue
      blob = _field(event, "blob", bytes)
      streams.append({"device": device, "kern": _field(event, "kern", int),
                      "exec_tag": _field(event, "exec_tag", int), "se": _field(event, "se", int),
                      "itrace": _field(event, "itrace", bool), "blob": blob_reference(blob)})
  devices.sort(key=lambda x: x["device"])
  programs.sort(key=lambda x: (x["device"], x["tag"] is None, x["tag"] or 0, x["name"], x["base"] or 0))
  streams.sort(key=lambda x: (x["device"], x["exec_tag"], x["se"], x["kern"]))
  schema = {"name": "tinygrad.sqtt-extract", "version": 2, "blob_references": "SHA-256 keys into top-level blobs"}
  # Size the complete structure before Base64 allocation. The metadata walk is
  # streaming, and each payload contributes an exact 12-byte field overhead
  # plus its independently computable Base64 length.
  blobs: dict[str, dict[str, int | str]] = {digest: {"bytes": len(blob)} for digest, blob in sorted(raw_blobs.items())}
  profile = {"schema": schema, "capture": _digest(data), "blobs": blobs,
             "devices": devices, "programs": programs, "sqtt": streams}
  projected_size = _serialized_size(profile, limits.max_output_bytes)
  projected_size += sum(12 + 4 * ((len(blob) + 2) // 3) for blob in raw_blobs.values())
  if projected_size > limits.max_output_bytes: raise ExtractError("output bytes limit exceeded")
  for digest, blob in raw_blobs.items(): blobs[digest]["base64"] = base64.b64encode(blob).decode("ascii")
  return profile


def _serialized_size(profile: dict[str, Any], limit: int) -> int:
  size = 0
  encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False)
  for chunk in encoder.iterencode(profile):
    size += len(chunk.encode("utf-8"))
    if size > limit: raise ExtractError("output bytes limit exceeded")
  return size


def canonical_json(profile: dict[str, Any]) -> str:
  return json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input", type=Path)
  parser.add_argument("-o", "--output", type=Path)
  args = parser.parse_args()
  limits = Limits()
  with args.input.open("rb") as input_file: data = input_file.read(limits.max_input_bytes + 1)
  if len(data) > limits.max_input_bytes: raise ExtractError("input bytes limit exceeded")
  output_text = canonical_json(extract_profile(data, limits)) + "\n"
  if len(output_text.encode("utf-8")) > limits.max_output_bytes: raise ExtractError("output bytes limit exceeded")
  if args.output is None: print(output_text, end="")
  else: args.output.write_text(output_text)


if __name__ == "__main__": main()
