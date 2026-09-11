"""SQTT timing corpus and evidence checks.

Capture with DEV=AMD PROFILE=1 SQTT=1 SQTT_TIMING_CAPTURE_DIR=/new/path and -n0.
Compare with DEV=MOCK+AMD SQTT_TIMING_REFERENCE_DIR=/path and -n0.
The reference mode reports exact observed packet-tick differences; it does not calibrate the emulator.
Without either variable, only CPU/mock tests run. Explicit capture requests never skip for missing hardware.
"""
import ctypes, json, os, pickle, struct, subprocess, tempfile, unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch
from tinygrad import Context, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv
from tinygrad.renderer.amd.dsl import s, v, NULL
from tinygrad.runtime.autogen.amd.rdna3 import ins as isa
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from test.amd.sqtt_timing_helpers import CaptureError, compare, digest, measured_region, observations, read_capture, validate_program, write_capture

@dataclass(frozen=True)
class Case:
  name:str
  body:tuple
  expected:tuple[int, ...]
  lds:int = 0
  split:str = "calibration"

  @property
  def instructions(self):
    return [isa.s_load_b64(s[0:1], s[0:1], soffset=NULL), isa.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0), *self.body,
            isa.v_lshlrev_b32_e32(v[0], 2, v[0]), isa.global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]), isa.s_endpgm()]

  @property
  def code(self): return b"".join(x.to_bytes() for x in self.instructions)

  @property
  def measurement_contract(self) -> dict:
    return {"local_size": [len(self.expected), 1, 1], "workgroups": [1, 1, 1], "wave_size": 32,
            "submitted_waves": len(self.expected)//32, "se": 0, "lds_bytes": self.lds,
            "measured_code_start": sum(i.size() for i in self.instructions[:2]),
            "measured_code_end": sum(i.size() for i in self.instructions[:-3])}

  def program(self, output:UOp) -> UOp:
    lds = (UOp.placeholder((self.lds,), dtypes.uint8, addrspace=AddrSpace.LOCAL),) if self.lds else ()
    sink = UOp.sink(output, *lds, UOp.special(len(self.expected), "lidx0"), arg=KernelInfo(f"sqtt_timing_{self.name}"))
    linear = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in self.instructions))
    return UOp(Ops.PROGRAM, src=(sink, linear))

  def library(self, architecture="gfx1100"):
    from tinygrad.renderer.amd.elf import assemble_linear
    program = self.program(UOp.param(0, dtypes.uint32, (len(self.expected),)))
    return assemble_linear(program, program.src[1], architecture)

def corpus() -> list[Case]:
  cases = []
  for n, split in ((16, "calibration"), (29, "held-out")):
    cases.append(Case(f"scalar_{n}", tuple([isa.s_mov_b32(s[4], 0)] + [isa.s_add_u32(s[4], s[4], 1)] * n +
                                        [isa.v_mov_b32_e32(v[1], s[4])]), (n,) * 32, split=split))
    cases.append(Case(f"vector_{n}", tuple([isa.v_mov_b32_e32(v[1], 1)] + [isa.v_add_nc_u32_e32(v[1], 1, v[1])] * n),
                      (n+1,) * 32, split=split))
  cases.append(Case("independent", tuple([isa.s_mov_b32(s[4], 0), isa.s_mov_b32(s[5], 0)] +
                    [x for _ in range(16) for x in (isa.s_add_u32(s[4], s[4], 1), isa.s_add_u32(s[5], s[5], 1))] +
                    [isa.s_add_u32(s[4], s[4], s[5]), isa.v_mov_b32_e32(v[1], s[4])]), (32,) * 32))
  cases.append(Case("mixed", tuple([isa.s_mov_b32(s[4], 0), isa.v_mov_b32_e32(v[1], 1)] +
                    [x for _ in range(16) for x in (isa.s_add_u32(s[4], s[4], 1), isa.v_add_nc_u32_e32(v[1], 1, v[1]))] +
                    [isa.v_add_nc_u32_e32(v[1], s[4], v[1])]), (33,) * 32, split="held-out"))
  cases.append(Case("branch", (isa.s_mov_b32(s[4], 0), isa.s_mov_b32(s[5], 8), isa.s_add_u32(s[4], s[4], 1),
                    isa.s_sub_u32(s[5], s[5], 1), isa.s_cmp_lg_u32(s[5], 0), isa.s_cbranch_scc1(-4),
                    isa.v_mov_b32_e32(v[1], s[4])), (8,) * 32))
  cases.append(Case("dual", tuple([isa.VOPD(isa.VOPDOp.V_DUAL_MOV_B32, isa.VOPDOp.V_DUAL_MOV_B32,
                    vdstx=v[2], vdsty=v[3], srcx0=3, srcy0=4)] * 16 + [isa.v_add_nc_u32_e32(v[1], v[2], v[3])]), (7,) * 32))
  for shift, threads in ((2, 32), (7, 32), (2, 64)):
    cases.append(Case(f"lds_{shift}_{threads}", (isa.v_lshlrev_b32_e32(v[2], shift, v[0]),
                      isa.ds_store_b32(addr=v[2], data0=v[0]), isa.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0), isa.s_barrier(),
                      isa.ds_load_b32(vdst=v[1], addr=v[2]), isa.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)),
                      tuple(range(threads)), lds=threads << shift, split="held-out" if threads == 64 else "calibration"))
  return cases

def emulate(case:Case) -> tuple[bytes, tuple[int, ...]]:
  from test.mockgpu.amd.emu import run_asm, sqtt_traces
  code = ctypes.create_string_buffer(case.code)
  output = (ctypes.c_uint32 * len(case.expected))(*([0xdeadbeef] * len(case.expected)))
  args = (ctypes.c_uint64 * 1)(ctypes.addressof(output))
  sqtt_traces.clear()
  with Context(PROFILE=1):
    run_asm(ctypes.addressof(code), len(case.code), 1, 1, 1, len(case.expected), 1, 1, ctypes.addressof(args),
            rsrc2=4 | ((case.lds + 511) // 512 << 15))
  if len(sqtt_traces) != 1: raise CaptureError("expected exactly one emulated trace")
  return sqtt_traces.pop(), tuple(output)

class _FixtureRecord: pass

class _FixtureUnpickler(pickle.Unpickler):
  """Only passive data records from the checked-in upstream fixtures; never import classes from a pickle."""
  def find_class(self, module, name):
    if (module == "tinygrad.device" and name == "ProfileProgramEvent") or (
      module == "tinygrad.runtime.ops_amd" and name in {"ProfileSQTTEvent", "ProfilePMCEvent", "PMCSample"}): return _FixtureRecord
    raise pickle.UnpicklingError(f"unsupported fixture class {module}.{name}")

def upstream_traces():
  from tinygrad import __file__ as tinygrad_file
  paths = sorted((Path(tinygrad_file).parent.parent / "extra/sqtt/examples/gfx1100").glob("*.pkl"))
  if len(paths) != 8: raise CaptureError("expected eight checked-in gfx1100 fixtures")
  for path in paths:
    with path.open("rb") as source: records = _FixtureUnpickler(source).load()
    programs = {e.tag: e for e in records if hasattr(e, "lib")}
    for event in records:
      if getattr(event, "itrace", False): yield path.name, event, programs[event.kern]

def validate_corpus(records:list[dict], cases:dict[str, Case]) -> None:
  expected = {(name, repetition) for name in cases for repetition in range(20)}
  if len(records) != len(expected): raise CaptureError(f"incomplete corpus: expected {len(expected)} captures, got {len(records)}")
  seen, builds = set(), set()
  for record in records:
    metadata = record["metadata"]
    key = (metadata["case"], metadata["repetition"])
    if key not in expected or key in seen: raise CaptureError("unexpected or duplicate case/repetition")
    seen.add(key)
    case = cases[key[0]]
    if record["code"] != case.code.hex(): raise CaptureError("reference kernel differs from the corpus")
    if tuple(metadata["outputs"]) != case.expected: raise CaptureError("hardware output differs from the independent expected result")
    if metadata["split"] != case.split: raise CaptureError("calibration/held-out split changed")
    for field, value in case.measurement_contract.items():
      if metadata[field] != value: raise CaptureError(f"corpus measurement contract changed: {field}")
    builds.add(tuple(metadata[k] for k in ("architecture", "repository_sha", "tracked_diff_sha256", "harness_sha256")))
  if len(builds) != 1: raise CaptureError("corpus mixes architectures or source identities")

class TestTimingEvidence(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.case = corpus()[0]
    cls.blob, _ = emulate(cls.case)
    cls.lib = cls.case.library()

  def test_corpus_outputs_and_complete_traces(self):
    for case in corpus():
      with self.subTest(case=case.name):
        blob, output = emulate(case)
        self.assertEqual(output, case.expected)
        self.assertEqual(observations(blob, case.library(), "gfx1100", 1 << 20)["waves"], len(case.expected) // 32)

  def test_round_trip(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "capture.json"
      metadata = {"source": "hardware", "architecture": "gfx1100", "buffer_capacity": 1 << 20, "case": self.case.name,
                  "repository_sha": "0" * 40, "tracked_diff_sha256": "0" * 64, "harness_sha256": "0" * 64, "repetition": 0,
                  "split": "calibration", "local_size": [32, 1, 1], "workgroups": [1, 1, 1], "wave_size": 32, "submitted_waves": 1,
                  "se": 0, "rsrc1": 0, "rsrc2": 0, "rsrc3": 0, "lds_bytes": 0, "outputs": list(self.case.expected),
                  "measured_code_start": 12, "measured_code_end": len(self.case.code)-16}
      write_capture(path, metadata, self.lib, self.case.code, self.blob)
      record = read_capture(path)
      self.assertEqual(record["sha256"]["trace"], digest(self.blob))
      with self.assertRaises(FileExistsError): write_capture(path, metadata, self.lib, self.case.code, self.blob)
      for program in (b"not ELF", b"\x7fELF", b"\x7fELF\x02" + b"\0"*59):
        changed = {**record, "program": program.hex(), "sha256": {**record["sha256"], "program": digest(program)}}
        path.write_text(json.dumps(changed))
        with self.subTest(program=program), self.assertRaises(CaptureError): read_capture(path)
      for field, value in (("workgroups", [True, 1, 1]), ("repetition", False), ("se", False), ("wave_size", 32.0)):
        changed = {**record, "metadata": {**metadata, field: value}}
        path.write_text(json.dumps(changed))
        with self.subTest(field=field), self.assertRaises(CaptureError): read_capture(path)
      record["trace"] = "00" * len(self.blob)
      path.write_text(json.dumps(record))
      with self.assertRaisesRegex(CaptureError, "digest mismatch"): read_capture(path)

  def test_elf_allocation_bounds(self):
    validate_program(self.lib)
    section = struct.unpack_from("<Q", self.lib, 40)[0]
    for offset, value, size in ((section+16, 1 << 62, "Q"), (section+32, 1 << 62, "Q"), (section+48, 1 << 62, "Q"),
                                (section+4, 4, "I"), (40, len(self.lib), "Q")):
      lib = bytearray(self.lib)
      struct.pack_into("<"+size, lib, offset, value)
      with self.subTest(offset=offset), self.assertRaises(CaptureError): validate_program(bytes(lib))

  def test_missing_truncated_saturated(self):
    for blob, capacity in ((b"", 1024), (self.blob[:-1], 1 << 20), (self.blob, len(self.blob)+32), (b"\0"*64, 1024)):
      with self.subTest(size=len(blob), capacity=capacity), self.assertRaises(CaptureError):
        observations(blob, self.lib, "gfx1100", capacity)

  def test_deleted_instructions_rejected(self):
    other = Case("short", (isa.s_mov_b32(s[4], 1), isa.v_mov_b32_e32(v[1], s[4])), (1,) * 32)
    blob, _ = emulate(other)
    with self.assertRaisesRegex(CaptureError, "end instruction"): observations(blob, self.lib, "gfx1100", 1 << 20)

  def test_no_per_wave_normalization(self):
    obs = observations(self.blob, self.lib, "gfx1100", 1 << 20)
    changed = json.loads(json.dumps(obs))
    changed["events"][-1]["tick"] += 7
    self.assertEqual(compare(obs, changed)["max_absolute_tick_difference"], 7)
    changed["waves"] += 1
    with self.assertRaisesRegex(CaptureError, "domains differ"): compare(obs, changed)

  def test_instruction_mismatch(self):
    obs = observations(self.blob, self.lib, "gfx1100", 1 << 20)
    changed = json.loads(json.dumps(obs))
    changed["events"][0]["pc"] += 4
    self.assertEqual(compare(obs, changed)["status"], "instruction-mismatch")

  def test_upstream_hardware_fixture_mapping(self):
    from tinygrad.renderer.amd.sqtt import map_insts, decode, LAYOUT_HEADER, WAVESTART, WAVEEND
    count, empty, populated = 0, 0, 0
    for name, trace, program in upstream_traces():
      with self.subTest(fixture=name, se=trace.se):
        mapped = [info for _, info in map_insts(trace.blob, program.lib, "gfx1100") if info is not None]
        packets = list(decode(trace.blob))
        self.assertIsInstance(packets[0], LAYOUT_HEADER)
        if not mapped:
          # An itrace flag alone does not establish that the selected CU/SIMD actually observed this kernel.
          self.assertFalse(any(isinstance(p, WAVESTART) and p.cu == 0 and p.simd == packets[0].simd for p in packets))
          empty += 1
          continue
        self.assertTrue(any(isinstance(packet, WAVEEND) for packet, _ in map_insts(trace.blob, program.lib, "gfx1100")))
        populated += 1
        count += len(mapped)
    self.assertEqual((empty, populated), (10, 10))
    self.assertGreater(count, 1000)

  def test_missing_malformed_duplicate_json(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "capture.json"
      with self.assertRaises(CaptureError): read_capture(path)
      for text in ("{", "[]", '{"schema":1,"schema":1}', '{"schema":true}', '{"schema":2}', '{"schema":1,"program":"XX"}'):
        with self.subTest(text=text):
          path.write_text(text)
          with self.assertRaises(CaptureError): read_capture(path)

  def test_incomplete_wave_rejected(self):
    from test.mockgpu.amd.sqtt_enc import make_encoder
    emit, _, finalize = make_encoder()
    emit(0, isa.s_mov_b32(s[0], 0), None)
    with self.assertRaisesRegex(CaptureError, "incomplete wave"): observations(finalize(), self.lib, "gfx1100", 1 << 20)

  def test_complete_corpus_required(self):
    cases = {c.name: c for c in corpus()}
    records = [{"code": c.code.hex(), "metadata": {"case": c.name, "repetition": r, "outputs": list(c.expected), "split": c.split,
                "architecture": "gfx1100", "repository_sha": "a"*40, "tracked_diff_sha256": "b"*64, "harness_sha256": "c"*64,
                **c.measurement_contract}}
                for c in cases.values() for r in range(20)]
    validate_corpus(records, cases)
    with self.assertRaisesRegex(CaptureError, "incomplete corpus"): validate_corpus(records[:-1], cases)
    with self.assertRaisesRegex(CaptureError, "duplicate"): validate_corpus(records[:-1] + [records[0]], cases)
    changed = json.loads(json.dumps(records))
    changed[-1]["metadata"]["outputs"][0] ^= 1
    with self.assertRaisesRegex(CaptureError, "independent expected"): validate_corpus(changed, cases)
    changed = json.loads(json.dumps(records))
    changed[-1]["metadata"]["split"] = "calibration"
    with self.assertRaisesRegex(CaptureError, "split changed"): validate_corpus(changed, cases)
    for field, value in (("measured_code_start", 16), ("measured_code_end", 16), ("local_size", [1, 32, 1]),
                         ("wave_size", 64), ("submitted_waves", 2), ("se", 1), ("lds_bytes", 512)):
      changed = json.loads(json.dumps(records))
      changed[0]["metadata"][field] = value
      with self.subTest(field=field), self.assertRaisesRegex(CaptureError, "measurement contract changed"):
        validate_corpus(changed, cases)

  def test_capture_request_never_silently_skips(self):
    environment = {"DEV": "MOCK+AMD", "SQTT_TIMING_CAPTURE_DIR": "/not-created", "SQTT_TIMING_REFERENCE_DIR": ""}
    with patch.dict(os.environ, environment):
      os.environ.pop("PYTEST_XDIST_WORKER", None)
      with self.assertRaisesRegex(AssertionError, "explicit DEV=AMD"): TestTimingHardware().test_capture_or_compare()

  def test_missing_reference_never_silently_skips(self):
    with tempfile.TemporaryDirectory() as directory:
      environment = {"DEV": "MOCK+AMD", "SQTT_TIMING_CAPTURE_DIR": "", "SQTT_TIMING_REFERENCE_DIR": directory}
      with patch.dict(os.environ, environment):
        os.environ.pop("PYTEST_XDIST_WORKER", None)
        with self.assertRaisesRegex(AssertionError, "no captures"): TestTimingHardware().test_capture_or_compare()

  def test_capture_and_reference_are_mutually_exclusive(self):
    environment = {"SQTT_TIMING_CAPTURE_DIR": "/not-created", "SQTT_TIMING_REFERENCE_DIR": "/not-created"}
    with patch.dict(os.environ, environment), self.assertRaisesRegex(AssertionError, "mutually exclusive"):
      TestTimingHardware().test_capture_or_compare()

  def test_sram_region_excludes_transfer_instructions(self):
    obs = observations(self.blob, self.lib, "gfx1100", 1 << 20)
    start, end = sum(i.size() for i in self.case.instructions[:2]), sum(i.size() for i in self.case.instructions[:-3])
    region = measured_region(obs, start, end)
    self.assertTrue(all(start <= e["pc"] < end for e in region["events"]))
    self.assertEqual(region["events"][0]["tick"], 0)
    self.assertLess(len(region["events"]), len(obs["events"]))

class TestTimingHardware(unittest.TestCase):
  def test_capture_or_compare(self):
    capture, reference = os.getenv("SQTT_TIMING_CAPTURE_DIR", ""), os.getenv("SQTT_TIMING_REFERENCE_DIR", "")
    if not capture and not reference: self.skipTest("explicit hardware capture/reference directory required")
    if capture and reference: self.fail("capture and reference modes are mutually exclusive")
    self.assertNotIn("PYTEST_XDIST_WORKER", os.environ, "hardware evidence requires -n0")
    if reference:
      self.assertTrue(getenv("DEV", "").startswith("MOCK+AMD"), "reference comparison requires DEV=MOCK+AMD")
      paths = sorted(Path(reference).glob("*.json"))
      self.assertTrue(paths, "reference directory contains no captures")
      cases = {c.name: c for c in corpus()}
      records = [read_capture(path) for path in paths]
      validate_corpus(records, cases)
      reports = []
      for path, record in zip(paths, records):
        case = cases[record["metadata"]["case"]]
        self.assertEqual(record["code"], case.code.hex(), "reference kernel differs from this corpus")
        blob, output = emulate(case)
        if output != case.expected:
          reports.append({"capture": path.name, "status": "functional-mismatch", "outputs": output})
          continue
        try:
          actual = observations(blob, bytes.fromhex(record["program"]), record["metadata"]["architecture"], 1 << 20)
          region = [record["metadata"][k] for k in ("measured_code_start", "measured_code_end")]
          reports.append({"capture": path.name, **compare(measured_region(record["observations"], *region), measured_region(actual, *region))})
        except CaptureError as exc: reports.append({"capture": path.name, "status": "invalid-observation", "error": str(exc)})
      print(json.dumps(reports, sort_keys=True))
      self.assertTrue(all(r["status"] == "match" for r in reports), "observed SQTT instructions/ticks differ; see report")
      return
    self.assertEqual(getenv("DEV", ""), "AMD", "hardware capture requires explicit DEV=AMD")
    self.assertTrue(getenv("PROFILE") and getenv("SQTT"), "hardware capture requires PROFILE=1 SQTT=1")
    from tinygrad import Device, Tensor
    from tinygrad.device import ProfileProgramEvent
    from tinygrad.runtime.ops_amd import ProfileSQTTEvent, _amd_program_image
    from test.amd.test_sqtt_profiler import save_sqtt
    device = Device["AMD"]
    self.assertTrue(device.arch.startswith("gfx11"), "RDNA3 hardware required")
    self.assertTrue(device.sqtt_enabled, "hardware SQTT profiling not enabled")
    directory = Path(capture)
    directory.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
    harness_digest = digest(Path(__file__).read_bytes() + Path(__file__).with_name("sqtt_timing_helpers.py").read_bytes())
    for case in corpus():
      for repetition in range(20):
        output = Tensor.empty(len(case.expected), dtype=dtypes.uint32, device="AMD").realize()
        with Context(SQTT_LIMIT_SE=1, SQTT_ITRACE_SE_MASK=1, SQTT_SIMD_SEL=0), save_sqtt() as data:
          output = output.custom_kernel(fxn=case.program)[0].realize()
        result = output.tolist()
        self.assertEqual(tuple(result), case.expected)
        traces = [e for e in data if isinstance(e, ProfileSQTTEvent) and e.itrace and e.se == 0]
        self.assertEqual(len(traces), 1, "missing or ambiguous hardware instruction capture")
        trace = traces[0]
        programs = [e for e in data if isinstance(e, ProfileProgramEvent) and e.tag == trace.kern]
        self.assertTrue(programs and programs[-1].lib, "missing executed kernel image")
        lib = programs[-1].lib
        resource, _ = _amd_program_image(device, lib)
        metadata = {"source": "hardware", "case": case.name, "split": case.split, "repetition": repetition, "repository_sha": revision,
                    "tracked_diff_sha256": digest(changes), "harness_sha256": harness_digest, "architecture": device.arch, "workgroups": [1, 1, 1],
                    "local_size": [len(case.expected), 1, 1], "wave_size": 32, "submitted_waves": len(case.expected)//32,
                    "se": trace.se, "buffer_capacity": device.sqtt_win, "rsrc1": resource.rsrc1, "rsrc2": resource.rsrc2,
                    "rsrc3": resource.rsrc3, "lds_bytes": resource.group_segment_size, "inputs": "instruction immediates and local lane IDs",
                    "outputs": result, "measured_code_start": sum(i.size() for i in case.instructions[:2]),
                    "measured_code_end": sum(i.size() for i in case.instructions[:-3])}
        write_capture(directory / f"{case.name}-{repetition:02d}.json", metadata, lib, case.code, trace.blob)

if __name__ == "__main__": unittest.main()
