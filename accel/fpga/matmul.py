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
  ret = np.zeros((a.shape[0], a.shape[1]+(2*a.shape[0])-1)) # length of the flowing dim can be theoretically infinite (though not in this impl)
  for i in range(0, a.shape[0]):                            # A.shape = (X, D), B.shape = (D, D) -> C.shape = (X, D)
    ret[i, i+a.shape[0]:i+(2*a.shape[0])] = a[i]            # padding needed for a.shape[0] steps on each side
  return ret

def unapad(a):
  ret = np.zeros(((a.shape[0]+1)//2, a.shape[1]))
  for i in range(0, a.shape[1]):
    ret[:, i] = a[ret.shape[0]-i-1:a.shape[0]-i, i]
  return ret
  


print(A)
print(B)
print(C)
print("**************************")
print("Weight Stationary Systolic Array") # static weights inside systolic array units

reset()
AA = apad(A.T)
wcache = B.copy()
out = []
for n in range(3*N - 1): # since we padded on both sides we have N zeros on the end of the array to finish accumulation
  r = mxu(AA[:, -1-n])
  if n >= N:
    out.append(r)

ret = unapad(np.array(out)[::-1])
print(ret)
assert np.allclose(C, ret)


def reset_mxm(N, X):
  global acc, acache, bcache
  acc = np.zeros((N,N))
  acache = np.zeros((N,N))
  bcache = np.zeros((N,N))

def mxm(a_col, b_row):
  global acc, acache, bcache

  # length of the flowing dim can be theoretically infinite
  # A.shape = (D, X) , B.shape = (X, D) -> C.shape = (D, D)
  # usually more useful than fixed weight matrix since you can chain matmuls easier that way
  # but supporting both ways in hardware shouldnt be too hard (just a simple "if" to decide what to pass on)
  
  acache = np.concatenate([a_col[:, None], acache[:, :-1]], axis=1) # flow the cols of A
  bcache = np.concatenate([b_row[None, :], bcache[:-1, :]], axis=0) # flow the cols of B
  acc += acache * bcache                                            # FMAC

def apad_mxm(a):
  ret = np.zeros((a.shape[0], a.shape[1]+(2*a.shape[0])-1))
  for i in range(0, a.shape[0]):
    ret[i, i:i+a.shape[1]] = a[i]
  return ret

print("**************************")
print("Output Stationary Systolic Array") # static accumulators inside systolic array units

reset_mxm(D,D)
AA = apad_mxm(A).T
BB = apad_mxm(B.T)

for n in range(3*N-1):
  mxm(AA[n, :], BB[:, n]) # no need for unpad since acc has results

ret2 = acc
print(ret2)
assert np.allclose(C, ret2)

print("**************************")
print("")
print("**************************")
print("Output Stationary Systolic Array - Dynamic Flow Size")


D = 8
X = 16 # Free dimension, matmul efficiency goes to 1 as X->inf (X / (X + 2D))

N = D
A = np.random.rand(N,X)
B = np.random.rand(X,N)
C = A @ B
print(C)

reset_mxm(N,X)
AA = apad_mxm(A).T
BB = apad_mxm(B.T)
print(AA.shape, BB.shape)
for n in range(2*N+X-1):
  mxm(AA[n, :], BB[:, n]) # no need for unpad since acc has results

ret3 = acc

print("**************************")
print(ret3)
assert np.allclose(C, ret3)
