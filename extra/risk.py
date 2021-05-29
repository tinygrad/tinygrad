#!/usr/bin/env python3

# RISK architecture is going to change everything
# implement on S7t-VG6

import functools
import numpy as np
from collections import defaultdict

# 32x32 * 32x32 -> 32x32 matmul = 65536 FLOPS @ 1 GHz = 64 TOPS
# 32x32 (aka 1024 element) ALU
# 1024 wide permute
# 1024 wide load/store (1 cycle to SRAM)
# all in elements, aka TF32 (19 bits)

# targets:
#   matmul input
#   matmul weights
#   ALU
#   permute

# 1024x1024x4x19 bits = 10MB
# fully strided
# load1024 <target>, <address>, <stride x (32)>, <stride y (32)>

# 4 slots
# <input> <weight> <output> <empty>
# <empty> <output> <input> <weight>
# <weight> <input> <empty> <output>

SZ = 32

sram = np.zeros((1024*1024*4), dtype=np.float32)
regfile = {}
SLOT = lambda x: x*1024*1024

from enum import Enum
class Reg(Enum):
  ZERO = 0
  MATMUL_INPUT = 1
  MATMUL_WEIGHTS = 2
  MATMUL_OUTPUT = 3
  ALU = 4

for t in Reg:
  regfile[t] = np.zeros((SZ, SZ), dtype=np.float32)

# *** profiler ***

cnts = defaultdict(int)
def count(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    cnts[func.__name__] += 1
    func(*args, **kwargs)
  return wrapper

import atexit
@atexit.register
def debug():
  print(cnts)
  print("ran in %.2f us" % (sum(cnts.values())*1e-3))

# *** instructions ***

@count
def riski_matmul():
  #print("LLL:\n",regfile[Reg.MATMUL_INPUT],"\n",regfile[Reg.MATMUL_WEIGHTS])
  regfile[Reg.MATMUL_OUTPUT] += \
    regfile[Reg.MATMUL_INPUT] @ \
    regfile[Reg.MATMUL_WEIGHTS]

@count
def riski_mov(tout, tin):
  regfile[tout][:] = regfile[tin]

@count
def riski_load(target, address, stride_y=SZ, stride_x=1, len_y=SZ, len_x=SZ):
  d = regfile[target]
  d[:] = 0
  for y in range(0, len_y):
    for x in range(0, len_x):
      d[y, x] = sram[address + y*stride_y + x*stride_x]

@count
def riski_store(target, address, stride_y=SZ, stride_x=1, len_y=SZ, len_x=SZ):
  d = regfile[target]
  for y in range(0, len_y):
    for x in range(0, len_x):
      sram[address + y*stride_y + x*stride_x] = d[y, x]

def riski_dmar(address, arr):
  arr = arr.reshape(-1)
  sram[address:address+arr.shape[0]] = arr

def riski_dmaw(address, shp):
  return sram[address:address+np.prod(shp)].reshape(shp)

# *** RISK-5 code ***

def risk_matmul(x, w):
  # copy matrices into SRAM
  # x is M x K
  # w is K x N
  # out is M x N
  riski_dmar(SLOT(0), x)
  riski_dmar(SLOT(1), w)
  M,K,N = x.shape[0], x.shape[1], w.shape[1]
  assert x.shape == (M,K)
  assert w.shape == (K,N)

  # do matmul
  for m in range(0, M, SZ):
    for n in range(0, N, SZ):
      riski_mov(Reg.MATMUL_OUTPUT, Reg.ZERO)
      for k in range(0, K, SZ):
        riski_load(Reg.MATMUL_INPUT, SLOT(0)+m*K+k, K, 1, min(SZ, M-m), min(SZ, K-k))
        riski_load(Reg.MATMUL_WEIGHTS, SLOT(1)+k*N+n, N, 1, min(SZ, K-k), min(SZ, N-n))
        riski_matmul()
      riski_store(Reg.MATMUL_OUTPUT, SLOT(2)+m*N+n, N, 1, min(SZ, M-m), min(SZ, N-n))

  # copy back from SRAM
  return riski_dmaw(SLOT(2), (x.shape[0], w.shape[1]))

import unittest
class TestRisk(unittest.TestCase):
  def test_matmul_even(self):
    x = np.random.uniform(size=(SZ*8, SZ*8)).astype(np.float32)
    w = np.random.uniform(size=(SZ*8, SZ*8)).astype(np.float32)
    np.testing.assert_allclose(x @ w, risk_matmul(x, w), rtol=1e-5)

  def test_matmul_small(self):
    x = np.array([[1,2,3],[4,5,6],[7,8,9]])
    w = np.array([[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]])
    np.testing.assert_allclose(x @ w, risk_matmul(x, w), rtol=1e-5)

  def test_matmul_uneven(self):
    x = np.random.uniform(size=(47, 79)).astype(np.float32)
    w = np.random.uniform(size=(79, 42)).astype(np.float32)
    np.testing.assert_allclose(x @ w, risk_matmul(x, w), rtol=1e-5)

if __name__ == "__main__":
  np.random.seed(1337)
  unittest.main(verbosity=2)

