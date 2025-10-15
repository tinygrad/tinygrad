from tinygrad import Tensor, dtypes
from tinygrad.helpers import GlobalCounters
from tinygrad.tensor import prod, sint, DType
import itertools

# --- the original helper functions ---
def _get_winograd_matcols(mat, dims:int, shp:tuple[int, ...], device:str, dtype):
  # for each dimension, create column tensors from the matrix constants
  return [
    [
      Tensor.cat(
        *[
          Tensor.full(shp[:dim] + (1,) + shp[dim+1:], float(m[k]), device=device, dtype=dtype)
          for m in mat
        ],
        dim=dim
      )
      for k in range(len(mat[0]))
    ]
    for dim in range(dims)
  ]

def _apply_winograd_matrix(mat, t:Tensor, dims:int) -> Tensor:
  # reshape → expand → multiply with folded constants
  t_ = t.reshape(t.shape[:dims] + (1,) * dims + t.shape[dims:]) \
        .expand(t.shape[:dims] + (len(mat),) * dims + t.shape[dims:])

  matcols = _get_winograd_matcols(mat, dims, t_.shape[dims:], t_.device, t_.dtype)
  ret = sum(
    prod(col[idx] for col, idx in zip(matcols, mat_is)) * t_[mat_is]
    for mat_is in itertools.product(range(len(mat[0])), repeat=dims)
  )
  return ret

winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]

# --- minimal test setup ---
if __name__ == "__main__":
  # A tiny 2×2 Winograd-style matrix for testing (normally 4x3, 3x3, etc.)
  winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]


  # input tensor of shape (2,2)
  rng = Tensor.rand(())                     # not foldable at compile time
  t = Tensor([[ 1,  2,  3,  4,  5,  6],
              [ 7,  8,  9, 10, 11, 12],
              [13, 14, 15, 16, 17, 18],
              [19, 20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35, 36]], dtype=dtypes.float32)
  Bt = Tensor([                          # (6,6)
  [4,  0, -5,  0, 1, 0],
  [0, -4, -4,  1, 1, 0],
  [0,  4, -4, -1, 1, 0],
  [0, -2, -1,  2, 1, 0],
  [0,  2, -1, -2, 1, 0],
  [0,  4,  0, -5, 0, 1],
], dtype=dtypes.float32)


  def _apply_winograd_matrix(mat, t: Tensor, dims: int) -> Tensor:
    """
    Drop-in, faster version.
    Assumes `mat` is a list of N columns of B^T, as you pass with: [list(row) for row in zip(*winograd_Bt)].
    Then M[u,p] := mat[u][p] equals B[u,p]. Applying M along axis 0 gives left-multiply by B.
    Applying the same M along axis 1 gives right-multiply by B^T (since per-column combos = @ M^T).
    For dims>2 this applies the same 1D stencil along each of the first `dims` axes.
    """
    N = len(mat)
    assert all(len(col) == N for col in mat), "mat must be N columns of length N"
    out = t

    for ax in range(dims):
      rows_acc = []
      for u in range(N):                 # output index on this axis
        acc = None
        for p in range(N):               # input index on this axis
          coeff = float(mat[u][p])       # IMPORTANT: use mat[u][p] on ALL axes (M = B)
          if coeff == 0.0:
            continue

          sl = [slice(None)] * out.ndim
          sl[ax] = slice(p, p+1)         # pick the p-th slice along this axis
          term = out[tuple(sl)]

          # tiny strength reduction for common constants
          # if   coeff ==  1.0: pass
          # elif coeff == -1.0: term = -term
          # elif coeff ==  2.0: term = term + term
          # elif coeff == -2.0: term = -(term + term)
          # elif coeff ==  0.5: term = term * 0.5
          # elif coeff == -0.5: term = term * -0.5
          # else:                term = term * coeff
          term = term * coeff
          acc = term if acc is None else (acc + term)

        if acc is None:
          # (shouldn't happen for Winograd matrices, but keep it safe)
          zsl = [slice(None)] * out.ndim
          zsl[ax] = slice(0, 1)
          acc = out[tuple(zsl)] * 0.0

        rows_acc.append(acc)

      out = Tensor.cat(*rows_acc, dim=ax)

    return out

  def _apply_winograd_matrix_old(mat, t:Tensor, dims:int) -> Tensor:
    # multiply mat_1 @ mat_2 @ t with foldable constants, where mat_i acts on vector t along dimension i; roughly kron(mat, mat) @ t
    # due to realize-before-expand rule in lazy.py, we must operate in this order: reshape -> expand -> arithmetic
    t_ = t.reshape(t.shape[:dims] + (1,) * dims + t.shape[dims:]).expand(t.shape[:dims] + (len(mat),) * dims + t.shape[dims:])  # add output dims
    # precalculate mat columns for each dim; prod(itertools.product(matcols)) gives the columns of kron(mat, mat, ...)
    matcols = _get_winograd_matcols(mat, dims, t_.shape[dims:], t_.device, t_.dtype)
    # multiply each element of t_ by the corresponding stacked column of kron(mat, mat), producing only one view for each element of t
    ret = sum(prod(col[idx] for col, idx in zip(matcols, mat_is)) * t_[mat_is] for mat_is in itertools.product(range(len(mat[0])), repeat=dims))
    assert isinstance(ret, Tensor), "sum didn't return a Tensor"
    return ret

  # apply fake Winograd transform
  GlobalCounters.reset()
  #t = Tensor.arange(3*3).reshape(3,3)
  # t = Tensor([[ 1,  2,  3],
  #             [ 4,  5,  6],
  #             [ 7,  8,  9]], dtype=dtypes.float32)
  t = Tensor.arange(3*3).reshape(3,3)
  #out = _apply_winograd_matrix([[1,1,-1],[1,1,-1],[1,1,-1]], t, dims=2)
  #print("\nOutput Tensor:\n", out.numpy())
  print("ops", GlobalCounters.global_ops)
  print("mem", GlobalCounters.global_mem)
  GlobalCounters.reset()
  out2 = _apply_winograd_matrix_old([[1,1,-1],[1,1,-1],[1,1,-1]], t, dims=2)
  print("Output Tensor - old:\n", out2.numpy())
  # out2 = Bt.transpose()@t@Bt
  # print("Output Tensor - matmul:\n", out2.numpy())
  print("ops", GlobalCounters.global_ops)
  print("mem", GlobalCounters.global_mem)

  # show the internal lazy graph for debugging (what Rangeify sees)
  print("\n--- Rangeify-style UOp dump ---")
