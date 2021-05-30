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
  # can the ALU use the same registers?
  MATMUL_INPUT = 1
  MATMUL_WEIGHTS = 2
  MATMUL_OUTPUT = 3

for t in Reg:
  regfile[t] = np.zeros((SZ, SZ), dtype=np.float32)

# *** profiler ***

cnts = defaultdict(int)
tcnts = defaultdict(int)
utils = defaultdict(int)
def count(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    cnts[func.__name__] += 1
    tcnts[func.__name__] += 1
    return func(*args, **kwargs)
  return wrapper

import atexit
@atexit.register
def risk_print_counts():
  print(cnts)
  print(tcnts)
  print(utils)
  util_n = sum([k[0]*k[1]*v for k,v in utils.items()])
  util_d = sum([SZ*SZ*v for k,v in utils.items()])
  print("ran in %.2f us with util %.2f%% total %.2f us" % (sum(cnts.values())*1e-3, util_n*100/(util_d+1), sum(tcnts.values())*1e-3))

def risk_reset_counts():
  global cnts, utils
  cnts = defaultdict(int)
  utils = defaultdict(int)

def risk_regdump():
  print("\n***** regdump *****")
  print(regfile[Reg.MATMUL_INPUT])
  print(regfile[Reg.MATMUL_WEIGHTS])
  print(regfile[Reg.MATMUL_OUTPUT])

# *** instructions ***

@count
def riski_mulacc():
  regfile[Reg.MATMUL_OUTPUT] += regfile[Reg.MATMUL_INPUT] * regfile[Reg.MATMUL_WEIGHTS]

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
  global util_n, util_d
  utils[(len_y, len_x)] += 1
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

@count
def riski_dmar(address, arr):
  arr = arr.reshape(-1)
  sram[address:address+arr.shape[0]] = arr

@count
def riski_dmaw(address, shp):
  return np.copy(sram[address:address+np.prod(shp)].reshape(shp))

# *** RISK-5 code ***

def risk_matmul(x, w, transpose_x=False, transpose_w=False):
  # copy matrices into SRAM
  # x is M x K
  # w is K x N
  # out is M x N
  riski_dmar(SLOT(0), x)
  riski_dmar(SLOT(1), w)

  if transpose_x:
    K,M = x.shape[-2], x.shape[-1]
  else:
    M,K = x.shape[-2], x.shape[-1]
  if transpose_w:
    N = w.shape[-2]
    assert w.shape[-1] == K
  else:
    N = w.shape[-1]
    assert w.shape[-2] == K
  cnt = np.prod(x.shape[0:-2]) if len(x.shape) > 2 else 1

  # do matmul
  for c in range(cnt):
    for m in range(0, M, SZ):
      for n in range(0, N, SZ):
        riski_mov(Reg.MATMUL_OUTPUT, Reg.ZERO)
        for k in range(0, K, SZ):
          if transpose_x:
            riski_load(Reg.MATMUL_INPUT, SLOT(0)+c*M*K + k*M+m, 1, M, min(SZ, M-m), min(SZ, K-k))
          else:
            riski_load(Reg.MATMUL_INPUT, SLOT(0)+c*M*K + m*K+k, K, 1, min(SZ, M-m), min(SZ, K-k))
          if transpose_w:
            riski_load(Reg.MATMUL_WEIGHTS, SLOT(1)+c*K*N + n*K+k, 1, K, min(SZ, K-k), min(SZ, N-n))
          else:
            riski_load(Reg.MATMUL_WEIGHTS, SLOT(1)+c*K*N + k*N+n, N, 1, min(SZ, K-k), min(SZ, N-n))
          riski_matmul()
        riski_store(Reg.MATMUL_OUTPUT, SLOT(2)+c*M*N + m*N+n, N, 1, min(SZ, M-m), min(SZ, N-n))

  # copy back from SRAM
  return riski_dmaw(SLOT(2), (*x.shape[0:-2],M,N))

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

  def test_matmul_transpose(self):
    x = np.random.uniform(size=(33, 33)).astype(np.float32)
    w = np.random.uniform(size=(33, 33)).astype(np.float32)
    np.testing.assert_allclose(x @ w, risk_matmul(x, w), rtol=1e-5)
    np.testing.assert_allclose(x.T @ w, risk_matmul(x, w, True), rtol=1e-5)
    np.testing.assert_allclose(x @ w.T, risk_matmul(x, w, False, True), rtol=1e-5)
    np.testing.assert_allclose(x.T @ w.T, risk_matmul(x, w, True, True), rtol=1e-5)

  def test_matmul_transpose_uneven_w(self):
    x = np.random.uniform(size=(47, 79)).astype(np.float32)
    w = np.random.uniform(size=(42, 79)).astype(np.float32)
    np.testing.assert_allclose(x @ w.T, risk_matmul(x, w, transpose_w=True), rtol=1e-5)

  def test_matmul_transpose_uneven_x(self):
    x = np.random.uniform(size=(79, 47)).astype(np.float32)
    w = np.random.uniform(size=(79, 42)).astype(np.float32)
    np.testing.assert_allclose(x.T @ w, risk_matmul(x, w, transpose_x=True), rtol=1e-5)

if __name__ == "__main__":
  np.random.seed(1337)
  unittest.main(verbosity=2)

