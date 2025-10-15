from itertools import product

# ---- tiny helpers ----
def shape(T):
  s=[]; x=T
  while isinstance(x,list): s.append(len(x)); x=x[0] if x else 0
  return s

def get(T, I):
  for i in I: T=T[i]
  return T

def build(shp, f):
  return f(()) if not shp else [build(shp[1:], lambda tail, i=i: f((i,)+tail)) for i in range(shp[0])]

# ---- n-mode product: (T ×_k M) ----
def nmode(T, M, k):
  S = shape(T); R = len(M); Sk = S[k]
  S2 = S[:k] + [R] + S[k+1:]
  return build(S2, lambda J:
           sum(M[J[k]][r] * get(T, J[:k] + (r,) + J[k+1:]) for r in range(Sk)))

# ---- apply same M on all modes ----
def kron_all_modes(T, M):
  X, S = T, shape(T)
  for k in range(len(S)):
    X = nmode(X, M, k)
  return X


def matmul(A,B):
  return [[sum(A[i][k]*B[k][j] for k in range(len(A[0]))) for j in range(len(B[0]))] for i in range(len(A))]

# ---- demo vs “Bt^T @ T @ Bt” (lists) ----
if __name__ == "__main__":
  Bt = [
    [4,0,-5,0,1,0],
    [0,-4,-4,1,1,0],
    [0,4,-4,-1,1,0],
    [0,-2,-1,2,1,0],
    [0,2,-1,-2,1,0],
    [0,4,0,-5,0,1],
  ]
  BtT = [list(row) for row in zip(*Bt)]

  # 6x6 test tensor with easy values
  T = [[r*6+c+1 for c in range(6)] for r in range(6)]

  # n-mode on both axes with M = Bt^T  <=>  Bt^T @ T @ Bt
  out_nm = kron_all_modes(T, BtT)
  ref = matmul(BtT, matmul(T, Bt))
  nm_winograd = nmode_np(nmode_np(X, Bt, 0), Bt.T, 1)

  # quick max-abs diff
  def max_abs(A,B):
    return max(abs(a-b) for a_row,b_row in zip(A,B) for a,b in zip(a_row,b_row))

  print("max|nmode - (Bt^T @ T @ Bt)| =", max_abs(out_nm, ref))

  # Winograd-style: Bt @ T @ Bt^T  via two n-modes (mode-0 with Bt, mode-1 with Bt^T)
  out_wino = nmode(nmode(T, Bt, 0), BtT, 1)
  ref_wino = matmul(Bt, matmul(T, BtT))
  print("max|two-step nmode - (Bt @ T @ Bt^T)| =", max_abs(out_wino, ref_wino))