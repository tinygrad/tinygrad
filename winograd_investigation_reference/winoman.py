# mini_wino_bench.py
import time, itertools
from tinygrad import Tensor
from tinygrad.dtype import dtypes

# ---- your helpers (kept verbatim except imports) ----
def _get_winograd_matcols(mat, dims:int, shp:tuple[int, ...], device:str|tuple[str, ...], dtype):
  return [[Tensor.cat(*[Tensor.full(shp[:dim] + (1,) + shp[dim+1:], float(m[k]), device=device, dtype=dtype)
                         for m in mat], dim=dim)
           for k in range(len(mat[0]))] for dim in range(dims)]

def _apply_winograd_matrix(mat, t:Tensor, dims:int) -> Tensor:
  t_ = t.reshape(t.shape[:dims] + (1,)*dims + t.shape[dims:]) \
        .expand(t.shape[:dims] + (len(mat),)*dims + t.shape[dims:])
  matcols = _get_winograd_matcols(mat, dims, t_.shape[dims:], t_.device, t_.dtype)
  ret = sum(
      Tensor.prod([col[idx] for col, idx in zip(matcols, mat_is)]) * t_[mat_is]
      for mat_is in itertools.product(range(len(mat[0])), repeat=dims)
  )
  return ret
# -----------------------------------------------------

# Winograd F(2x2, 3x3): α = m+r-1 = 4
# One common (Cook–Toom) choice (float32-friendly)
Bt = [
  [1,   0,  -1,  0],
  [0,   1,   1,  0],
  [0,  -1,   1,  0],
  [0,   1,   0, -1],
]  # shape (4 x 4)^T will be used dimension-wise (we’ll feed its columns), see below

# We actually want B^T of shape (α x r) = (4 x 3). Take the first 3 columns:
B_t = [row[:3] for row in Bt]  # 4x3

# A^T (m x α) = (2 x 4)
A_t = [
  [1,  1,  1,  0],
  [0,  1, -1, -1],
]  # 2x4

# G (α x r) = (4 x 3)
G = [
  [ 1/4,     0,      0  ],
  [ -1/6, -1/6,  -1/6  ],
  [ -1/6,  1/6,  -1/6  ],
  [ 1/24, 1/12, 1/6   ],
]  # 4x3

def time_it(name, fn, iters=50):
  # warmup
  for _ in range(5): fn().realize()
  t0 = time.perf_counter()
  for _ in range(iters): fn().realize()
  dt = (time.perf_counter() - t0)*1e3/iters
  print(f"{name:<18s}: {dt:8.3f} ms/iter")
  return dt

def main():
  device = "METAL" if Tensor.default_device == "METAL" else Tensor.default_device
  dtype = dtypes.float32

  # toy tile U in R^{3x3}; keep it simple but non-trivial so fusable math still happens
  U = Tensor([[1,2,3],
              [4,5,6],
              [7,8,9]], device=device, dtype=dtype)

  # weight tile g in R^{3x3}
  g = Tensor([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
              [0.7, 0.8, 0.9]], device=device, dtype=dtype)

  # --- forward Winograd recipe (2D) for a single tile ---
  # V = B^T ⊗ B^T @ U
  V_fn = lambda: _apply_winograd_matrix(B_t, U, dims=2)
  # Ĝ = G ⊗ G @ g
  GHAT_fn = lambda: _apply_winograd_matrix(G, g, dims=2)
  # M = V ⊙ Ĝ
  def M_fn():
    V = _apply_winograd_matrix(B_t, U, dims=2)
    GH = _apply_winograd_matrix(G, g, dims=2)
    return V * GH
  # Y = A^T ⊗ A^T @ M   (yields 2x2 block)
  def Y_fn():
    M = M_fn()
    return _apply_winograd_matrix(A_t, M, dims=2)

  print("Devices:", device)
  print("Shapes: U(3x3), g(3x3), B^T(4x3), G(4x3), A^T(2x4)\n")

  # correctness snapshot (once)
  V = V_fn().realize()
  GH = GHAT_fn().realize()
  M = (V * GH).realize()
  Y = Y_fn().realize()
  print("V shape:", tuple(V.shape), "GHAT shape:", tuple(GH.shape), "M shape:", tuple(M.shape), "Y shape:", tuple(Y.shape))
  print("Y:\n", Y.numpy(), "\n")

  print("Timing (averaged):")
  time_it("V = B^T⊗B^T @ U", V_fn)
  time_it("Ĝ = G⊗G @ g",    GHAT_fn)
  time_it("M = V ⊙ Ĝ",      M_fn)
  time_it("Y = A^T⊗A^T@M",  Y_fn)

  # Optional: scale up the batch to see amortization/compile vs run
  # Repeat the tile N times (fake batch/spatial) to highlight runtime vs compile
  N = 128
  U_big = U.reshape(1,1,3,3).expand(N,1,3,3)   # [N, 1, 3, 3]
  g_big = g  # weights usually shared; leave small to isolate transform cost

  def V_big_fn():
    return _apply_winograd_matrix(B_t, U_big, dims=2)   # acts on the last two dims (3,3)->(4,4)

  print("\nScaled batch timing (N=128 tiles):")
  time_it("V (N=128 tiles)", V_big_fn, iters=20)

if __name__ == "__main__":
  main()