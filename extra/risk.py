#!/usr/bin/env python3

# RISK architecture is going to change everything
# implement on S7t-VG6

import numpy as np

# 16x32 * 32x32 -> 16x32 matmul = 32768 FLOPS @ 1 GHz = 32 TOPS
# 16x32 (aka 512 element) ALU
# 512 wide permute
# 512 wide load/store (1 cycle to SRAM)
# all in elements, aka TF32 (19 bits)

# targets:
#   matmul input
#   matmul weights(0-16)
#   matmul weights(16-32)
#   ALU
#   permute

# 1024x1024x4x19 bits = 10MB
# fully strided
# load512 <target>, <address>, <stride x (16)>, <stride y (32)>

# 4 slots
# <input> <weight> <output> <empty>
# <empty> <output> <input> <weight>
# <weight> <input> <empty> <output>

sram = np.zeros((1024*1024*4), dtype=np.float32)
regfile = {}
SLOT = lambda x: x*1024*1024

from enum import Enum
class Target(Enum):
  MATMUL_INPUT = 0
  MATMUL_WEIGHTS_LO = 1
  MATMUL_WEIGHTS_HI = 2
  MATMUL_OUTPUT = 3
  ALU = 4

for t in Target:
  regfile[t] = np.zeros((16, 32), dtype=np.float32)

def riski_matmul():
  w = np.vstack([
    regfile[Target.MATMUL_WEIGHTS_LO],
    regfile[Target.MATMUL_WEIGHTS_HI]])
  regfile[Target.MATMUL_OUTPUT][:] = regfile[Target.MATMUL_INPUT] @ w

def riski_load512(target, address, stride_x=32, stride_y=1):
  d = regfile[target]
  for x in range(0, 16):
    for y in range(0, 32):
      d[x, y] = sram[address + x*stride_x + y*stride_y]

def riski_store512(target, address, stride_x=32, stride_y=1):
  d = regfile[target]
  for x in range(0, 16):
    for y in range(0, 32):
      sram[address + x*stride_x + y*stride_y] = d[x, y]

def riski_dmar(address, arr):
  arr = arr.reshape(-1)
  sram[address:address+arr.shape[0]] = arr

def riski_dmaw(address, shp):
  return sram[address:address+np.prod(shp)].reshape(shp)

def risk_matmul(x, w):
  # copy matrices into SRAM
  riski_dmar(SLOT(0), x)
  riski_dmar(SLOT(1), w)

  # do matmul
  riski_load512(Target.MATMUL_INPUT, SLOT(0))
  riski_load512(Target.MATMUL_WEIGHTS_LO, SLOT(1))
  riski_load512(Target.MATMUL_WEIGHTS_HI, SLOT(1)+512)
  riski_matmul()
  riski_store512(Target.MATMUL_OUTPUT, SLOT(2))

  # copy back from SRAM
  return riski_dmaw(SLOT(2), (x.shape[0], w.shape[1]))

import unittest
class TestRisk(unittest.TestCase):
  def test_matmul(self):
    x = np.random.uniform(size=(16, 32)).astype(np.float32)
    w = np.random.uniform(size=(32, 32)).astype(np.float32)

    np.testing.assert_allclose(x @ w, risk_matmul(x, w))

if __name__ == "__main__":
  unittest.main(verbosity=2)

