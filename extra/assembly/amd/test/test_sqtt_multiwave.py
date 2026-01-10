import os
os.environ["SQTT"] = "1"
os.environ["PROFILE"] = "1"
os.environ["SQTT_LIMIT_SE"] = "2"
os.environ["SQTT_TOKEN_EXCLUDE"] = "3784"  # Exclude WAVERDY, REG, EVENT, UTILCTR, WAVEALLOC, PERF

import unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.sqtt import decode
from extra.assembly.amd.test.test_sqtt_hw import compile_asm_sqtt, run_prg_sqtt_batch, format_packet
from extra.assembly.amd.test.test_sqtt_compare import filter_noise_packets
from tinygrad.uop.ops import UOp
from tinygrad.engine.realize import get_runner

class SQTTMultiwave(unittest.TestCase):
  def test_simple_multiwave(self):
    ins = [
      v_mov_b32_e32(v[0], v[1]),
      s_endpgm(),
    ]
    #prg = get_runner("AMD", UOp.sink())._prg
    prg = compile_asm_sqtt(ins, alu_only=True)
    print(prg)
    blobs = run_prg_sqtt_batch(prg, n_runs=2, n_lanes=32*4)
    for blob in blobs:
      packets = decode(blob)
      for p in filter_noise_packets(packets):
        print(f"  {p._time:8d}: {format_packet(p)}")

if __name__ == "__main__":
  unittest.main()