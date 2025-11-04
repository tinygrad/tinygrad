from tinygrad.tensor import Tensor

def cumsum(self, dim=None):
  if dim is None:
    return self.reshape(-1).cumsum(0)
  # Usa reduce SUM cumulativo
  return self._movement_op("CUMSUM", dim)