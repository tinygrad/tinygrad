import base64, builtins, hashlib, io, json, struct, sys, unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import extra.sqtt.extract_profile as extract_module
from extra.sqtt.extract_profile import ExtractError, Limits, canonical_json, extract_profile


EXAMPLE = Path(__file__).parents[2] / "extra/sqtt/examples/gfx1100/profile_empty_run_0.pkl"


def _global(module: str, name: str) -> bytes:
  return b"c" + module.encode() + b"\n" + name.encode() + b"\n"


def _empty_profile(device: bytes = b"AMD") -> bytes:
  # An inert ProfileDeviceEvent carrying only the fields the extractor needs.
  state = b"(X\x06\x00\x00\x00deviceX" + struct.pack("<I", len(device)) + device + b"X\x05\x00\x00\x00props}u"
  return b"\x80\x02]" + _global("tinygrad.device", "ProfileDeviceEvent") + b")\x81}" + state + b"b" + b"a."


def _unicode(value: str) -> bytes:
  encoded = value.encode()
  return b"X" + struct.pack("<I", len(encoded)) + encoded


def _sqtt_profile(blobs: list[bytes], memo_blob: bool = False, device: str = "AMD") -> bytes:
  events = []
  for index, blob in enumerate(blobs):
    if memo_blob and index: encoded_blob = b"h\x00"
    else:
      assert len(blob) < 256
      encoded_blob = b"C" + bytes([len(blob)]) + blob + (b"q\x00" if memo_blob else b"")
    state = (b"(" + _unicode("device") + _unicode(device) + _unicode("kern") + b"K\x01" +
             _unicode("exec_tag") + b"K\x02" + _unicode("se") + b"K\x03" + _unicode("itrace") + b"\x88" +
             _unicode("blob") + encoded_blob + b"u")
    events.append(_global("tinygrad.runtime.ops_amd", "ProfileSQTTEvent") + b")\x81}" + state + b"ba")
  return b"\x80\x03]" + b"".join(events) + b"."


def _program_profile(device: str, blob: bytes) -> bytes:
  assert len(blob) < 256
  state = (b"(" + _unicode("device") + _unicode(device) + _unicode("name") + _unicode("kernel") +
           _unicode("lib") + b"C" + bytes([len(blob)]) + blob + _unicode("base") + b"N" + _unicode("tag") + b"Nu")
  event = _global("tinygrad.device", "ProfileProgramEvent") + b")\x81}" + state + b"ba"
  return b"\x80\x03]" + event + b"."


class TestSQTTExtract(unittest.TestCase):
  def test_rejects_malicious_global_without_execution(self):
    called = False
    def boom(*_args, **_kwargs):
      nonlocal called
      called = True
      raise AssertionError("executed")
    with patch("os.system", boom), self.assertRaisesRegex(ExtractError, "global"):
      extract_profile(b"\x80\x02" + _global("posix", "system") + b"X\x02\x00\x00\x00id\x85R.")
    assert not called

  def test_allowed_constructor_is_never_imported_or_executed(self):
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
      if name.startswith("tinygrad"): raise AssertionError("pickle global imported")
      return real_import(name, *args, **kwargs)
    with patch.object(builtins, "__import__", guarded):
      self.assertEqual(extract_profile(_empty_profile())["devices"], [{"device": "AMD", "props": {}}])
      reduced = b"\x80\x02]" + _global("decimal", "Decimal") + b"X\x01\x00\x00\x001\x85Ra."
      self.assertEqual(extract_profile(reduced)["devices"], [])

  def test_rejects_unsupported_or_stateful_opcodes(self):
    for payload in [b"\x80\x04N0.", b"\x80\x04Pabc\n.", b"\x80\x04\x82\x01."]:
      with self.subTest(payload=payload), self.assertRaisesRegex(ExtractError, "unsupported opcode"): extract_profile(payload)

  def test_resource_limits(self):
    cases = [(b"0" * 9, Limits(max_input_bytes=8), "input bytes"), (b"\x80\x04NNN.", Limits(max_opcodes=3), "opcode count"),
             (b"\x80\x04C\x05abcde.", Limits(max_blob_bytes=4), "string/blob"),
             (b"\x80\x04\x8a\x09" + b"\x00" * 8 + b"\x01.", Limits(max_integer_bits=32), "integer"),
             (b"\x80\x04N" + b"\x94" * 3 + b".", Limits(max_memo_entries=2), "memo"),
             (b"\x80\x04](NNNe.", Limits(max_container_elements=2), "container"),
             (b"\x80\x04](](](NNNeee.", Limits(max_depth=2), "depth")]
    for payload, limits, message in cases:
      with self.subTest(message=message), self.assertRaisesRegex(ExtractError, message): extract_profile(payload, limits)

  def test_protocol_memo_and_input_boundaries(self):
    raw = _empty_profile()
    self.assertEqual(extract_profile(raw, Limits(max_input_bytes=len(raw)))["devices"][0]["device"], "AMD")
    with self.assertRaisesRegex(ExtractError, "input bytes limit exceeded"):
      extract_profile(raw + b"x", Limits(max_input_bytes=len(raw)))
    cases = [(b"N\x80\x02.", "pickle must start with PROTO"), (b"\x80\x01N.", "unsupported pickle protocol"),
             (b"\x80\x02\x80\x02N.", "unsupported pickle protocol"), (b"\x80\x02]h\x00a.", "missing memo entry")]
    for payload, message in cases:
      with self.subTest(message=message), self.assertRaisesRegex(ExtractError, message): extract_profile(payload)
    # Reassigning an explicit memo slot must make subsequent GETs observe the replacement.
    self.assertEqual(extract_profile(b"\x80\x02]q\x00Nq\x00ah\x00a.")["devices"], [])

  def test_cli_reads_once_with_a_hard_bound(self):
    class OpenStream(io.BytesIO):
      def close(self): pass
    stream = OpenStream(b"x" * 9)
    with patch.object(extract_module, "Limits", return_value=Limits(max_input_bytes=8)), \
         patch.object(Path, "stat", side_effect=AssertionError("metadata must not be trusted")), \
         patch.object(Path, "open", return_value=stream) as opened, patch.object(sys, "argv", ["extract_profile.py", "input.pkl"]), \
         self.assertRaisesRegex(ExtractError, "input bytes limit exceeded"):
      extract_module.main()
    opened.assert_called_once_with("rb")
    self.assertEqual(stream.tell(), 9)
    raw, stdout = _empty_profile(), io.StringIO()
    with patch.object(extract_module, "Limits", return_value=Limits(max_input_bytes=len(raw))), \
         patch.object(Path, "stat", side_effect=AssertionError("metadata must not be trusted")), \
         patch.object(Path, "open", return_value=OpenStream(raw)), patch.object(sys, "argv", ["extract_profile.py", "input.pkl"]), \
         patch.object(sys, "stdout", stdout):
      extract_module.main()
    self.assertEqual(json.loads(stdout.getvalue())["devices"], [{"device": "AMD", "props": {}}])

  def test_cli_writes_output(self):
    raw = _empty_profile()
    with tempfile.TemporaryDirectory() as tempdir:
      input_path = Path(tempdir) / "input.pkl"
      output_path = Path(tempdir) / "out.json"
      input_path.write_bytes(raw)
      with patch.object(extract_module, "Limits", return_value=Limits(max_input_bytes=len(raw))), \
           patch.object(sys, "argv", ["extract_profile.py", str(input_path), "-o", str(output_path)]):
        extract_module.main()
      self.assertEqual(json.loads(output_path.read_text())["devices"], [{"device": "AMD", "props": {}}])

  def test_non_amd_program_blobs_are_ignored_before_limits(self):
    for device in ("CPU", "AMDfake", "AMD:", "AMD:²", "AMD:١"):
      with self.subTest(device=device):
        profile = extract_profile(_program_profile(device, b"not an SQTT program"), Limits(max_aggregate_blob_bytes=0))
        self.assertEqual(profile["programs"], [])
        self.assertEqual(profile["blobs"], {})

  def test_non_amd_sqtt_blobs_are_ignored_before_limits(self):
    for device in ("CPU", "NV", "AMDfake", "AMD:", "AMD:²", "AMD:١"):
      with self.subTest(device=device):
        profile = extract_profile(_sqtt_profile([b"not an AMD SQTT trace"], device=device), Limits(max_aggregate_blob_bytes=0))
        self.assertEqual(profile["sqtt"], [])
        self.assertEqual(profile["blobs"], {})

  def test_boolean_integer_fields_are_rejected(self):
    raw = _sqtt_profile([b"trace"])
    for field, encoded in (("kern", b"K\x01"), ("exec_tag", b"K\x02"), ("se", b"K\x03")):
      malformed = raw.replace(_unicode(field) + encoded, _unicode(field) + b"\x88", 1)
      with self.subTest(field=field), self.assertRaisesRegex(ExtractError, rf"invalid ProfileSQTTEvent\.{field}"):
        extract_profile(malformed)

  def test_hash_collision_is_rejected(self):
    class CollidingHash:
      def hexdigest(self): return "0" * 64
    with patch.object(extract_module.hashlib, "sha256", return_value=CollidingHash()), \
         self.assertRaisesRegex(ExtractError, "SHA-256 collision"):
      extract_profile(_sqtt_profile([b"first", b"second"]))

  def test_canonical_json_orders_keys_and_preserves_unicode(self):
    self.assertEqual(canonical_json({"z": 1, "雪": "☃", "a": 2}), '{"a":2,"z":1,"雪":"☃"}')
    raw = b"\x80\x03]" + _unicode("雪") + b"a."
    self.assertEqual(extract_profile(raw, Limits(max_string_bytes=3))["devices"], [])
    with self.assertRaisesRegex(ExtractError, "string/blob size limit exceeded"):
      extract_profile(raw, Limits(max_string_bytes=2))

  def test_thousands_of_memo_blob_references_are_deduplicated(self):
    blob = b"repeated payload" * 8
    raw = _sqtt_profile([blob] * 3000, memo_blob=True)
    first, second = extract_profile(raw), extract_profile(raw)
    self.assertEqual(first, second)
    digest = hashlib.sha256(blob).hexdigest()
    self.assertEqual(first["schema"], {"name": "tinygrad.sqtt-extract", "version": 2,
      "blob_references": "SHA-256 keys into top-level blobs"})
    self.assertEqual(first["blobs"], {digest: {"bytes": len(blob), "base64": base64.b64encode(blob).decode()}})
    self.assertTrue(all(stream["blob"] == digest for stream in first["sqtt"]))
    self.assertLess(len(canonical_json(first)), 500_000)
    with self.assertRaisesRegex(ExtractError, "event count limit exceeded"):
      extract_profile(raw, Limits(max_events=2999))

  def test_aggregate_payload_and_output_boundaries(self):
    raw = _sqtt_profile([b"aaaa", b"bbbb", b"aaaa"])
    profile = extract_profile(raw, Limits(max_aggregate_blob_bytes=8))
    self.assertEqual(len(profile["blobs"]), 2)
    with self.assertRaisesRegex(ExtractError, "aggregate blob bytes limit exceeded"):
      extract_profile(raw, Limits(max_aggregate_blob_bytes=7))
    output_bytes = len(canonical_json(profile).encode())
    self.assertEqual(extract_profile(raw, Limits(max_aggregate_blob_bytes=8, max_output_bytes=output_bytes)), profile)
    with self.assertRaisesRegex(ExtractError, "output bytes limit exceeded"):
      extract_profile(raw, Limits(max_aggregate_blob_bytes=8, max_output_bytes=output_bytes - 1))

  def test_text_integer_failures_are_deterministic(self):
    old_limit = sys.get_int_max_str_digits()
    try:
      for runtime_limit in (640, 0):
        sys.set_int_max_str_digits(runtime_limit)
        for payload in (b"\x80\x02I" + b"9" * 1000 + b"\n.", b"\x80\x02I12x\n.",
                        b"\x80\x02L" + b"9" * 1000 + b"L\n.", b"\x80\x02L12xL\n."):
          with self.subTest(runtime_limit=runtime_limit, payload=payload[:16]), self.assertRaises(ExtractError) as raised:
            extract_profile(payload)
          self.assertEqual(str(raised.exception), "malformed pickle")
    finally:
      sys.set_int_max_str_digits(old_limit)

  def test_malformed_input(self):
    for payload in [b"", b"\x80\x04", b"\x80\x04h\xff.", b"\x80\x04}X\x01\x00\x00\x00x.", b"\x80\x04N.N",
                    b"\x80\x04\x95\xff\xff\xff\xff\xff\xff\xff\xffN."]:
      with self.subTest(payload=payload), self.assertRaises(ExtractError): extract_profile(payload)

  def test_completed_frame_allows_unframed_tail(self):
    payload = b"\x80\x04\x95" + struct.pack("<Q", 1) + b"](Ne."
    self.assertEqual(extract_profile(payload)["devices"], [])

  def test_real_gfx1100_fixture_is_deterministic_and_hashed(self):
    raw = EXAMPLE.read_bytes()
    first, second = extract_profile(raw), extract_profile(raw)
    assert first == second
    encoded = canonical_json(first)
    assert encoded == canonical_json(second)
    assert json.loads(encoded) == first
    assert first["schema"]["version"] == 2
    assert first["capture"] == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    assert first["devices"] == [{"device": "AMD", "props": {"array_count": 12, "cu_per_simd_array": 8, "gfx_target_version": 110000,
      "lds_size_in_kb": 64, "max_slots_scratch_cu": 32, "max_waves_per_simd": 16, "num_xcc": 1, "simd_arrays_per_engine": 2,
      "simd_count": 192, "simd_per_cu": 2}}]
    assert len(first["sqtt"]) == 6
    assert [stream["se"] for stream in first["sqtt"]] == list(range(6))
    assert all(stream["blob"] in first["blobs"] for stream in first["sqtt"])
    assert first["programs"] and all(program["elf"] in first["blobs"] for program in first["programs"] if program["elf"] is not None)
    for digest, payload in first["blobs"].items():
      decoded = base64.b64decode(payload["base64"], validate=True)
      assert len(decoded) == payload["bytes"] and hashlib.sha256(decoded).hexdigest() == digest


if __name__ == "__main__": unittest.main()
