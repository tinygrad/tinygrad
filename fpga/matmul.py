#!/usr/bin/env python3
import numpy as np
np.set_printoptions(linewidth=200)

# https://www.youtube.com/watch?v=cmy7LBaWuZ8&t=74s

D = 8

N = D
A = np.random.rand(N,N)
B = np.random.rand(N,N)
#A = np.arange(1,N*N+1).reshape(N,N)
#B = np.ones((N,N))*2
#B = np.arange(1,17).reshape(4,4) + 10
C = A @ B

def reset():
  global acc, acache, wcache
  acc = np.zeros((D,D))
  acache = np.zeros((D,D))
  wcache = np.zeros((D,D))

def mxu(a):
  global acc, acache, wcache
  assert a.shape == (D,)
  acache[:, 1:] = acache[:, :-1]
  acache[:, 0] = a
  ret = np.copy(acc[0])
  acc[0:-1] = acc[1:]
  acc[-1] = 0
  acc += acache * wcache
  """
  print("****")
  print(acache)
  print(acc)
  print(ret)
  """
  return ret

def apad(a):
  ret = np.zeros((a.shape[0], a.shape[1]+a.shape[1]-1))
  for i in range(0, a.shape[0]):
    ret[i, i:i+a.shape[1]] = a[i]
  return ret

def unapad(a):
  ret = np.zeros(((a.shape[0]+1)//2, a.shape[1]))
  #print(a.shape, ret.shape)
  for i in range(0, a.shape[1]):
    ret[:, i] = a[ret.shape[0]-i-1:a.shape[0]-i, i]
  return ret
  


AA = apad(A.T)
print(AA)
print(A)
print(B)
print(C)
print("**************************")

reset()
wcache = B
out = []
for n in range(N+N-1):
  r = mxu(AA[:, -1-n])
  if n >= N:
    out.append(r)
for n in range(N):
  r = mxu(np.zeros(N))
  out.append(r)
ret = unapad(np.array(out)[::-1])
assert np.allclose(C, ret)
print(ret)

