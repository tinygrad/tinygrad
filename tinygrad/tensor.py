from tinygrad import Tensor
from typing import Optional

class Tensor:
    # ... (altri metodi esistenti) ...

    def std(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        """
        Compute standard deviation along dimension.
        
        Args:
            dim: Dimension to reduce. If None, compute over flattened tensor.
            keepdim: Whether to keep the dimension.
        
        Returns:
            Tensor with standard deviation.
        """
        if dim is None:
            return self.reshape(-1).std(0, keepdim)
        
        mean = self.mean(dim, keepdim=True)
        var = ((self - mean) ** 2).mean(dim, keepdim=keepdim)
        return var.sqrt()