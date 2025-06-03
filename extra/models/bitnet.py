from tinygrad import Tensor, dtypes
from typing import Tuple

class BitLinear:
    """
    BitNet Linear layer with 2-bit packed weights and int8 activations.
    Matches HuggingFace implementation but uses tinygrad operations
    https://github.com/huggingface/transformers/blob/096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4/src/transformers/integrations/bitnet.py#L126
    """

    VALUES_PER_ITEM = 4  # 4 ternary values packed per uint8 byte

    def __init__(self, in_features: int, out_features: int, bias: bool = False):

        # Packed weight tensor: (out_features//4, in_features)
        self.weight = Tensor.zeros(
            (out_features // self.VALUES_PER_ITEM, in_features),
            dtype=dtypes.uint8,
        )

        # Weight scale (single scalar)
        self.weight_scale = Tensor.ones(
            (1,),
            dtype=dtypes.float32,
        )

        # Optional bias
        if bias:
            self.bias = Tensor.zeros(
                (out_features,),
                dtype=dtypes.float32,
            )
        else:
            self.bias = None

        # Cache for unpacked weights
        self._unpacked_weights = None
        self._weight_hash = None

    @staticmethod
    def unpack_weights(packed: Tensor, dtype) -> Tensor:
        """
        Unpacks a tensor of quantized weights that were stored in a packed format using 2 bits per value.
        Parameters:
        -----------
        packed : Tensor
            A tensor containing packed weights where each element represents 4 quantized values (using 2 bits per value).
        dtype : dtypes
            The dtype of the returned Tensor
        Returns:
        --------
        Tensor
            A tensor of unpacked weights, where each value is converted from its packed 2-bit representation.
        Example:
        --------
        packed = Tensor([[0b10100001, 0b00011000],
                        [0b10010000, 0b00001010]], dtype=dtypes.uint8)
        # Unpack the values
        unpacked = unpack_weights(packed, dtypes.float32)
        # Resulting unpacked tensor
        print(unpacked)
        # Output: tensor([[ 0, -1],
                          [-1,  1],
                          [-1,  1],
                          [-1,  1],
                          [ 1,  0],
                          [ 0, -1],
                          [ 1, -1],
                          [ 1, -1]])
        Explanation of the example:
        ---------------------------
        Let's take the first value for example 0b10100001, we we will only focus on the first column,
        because every element is unpacked across the first dimension
        - First 2 bits: `01` → 0 at [0][0]
        - Second 2 bits: `00` → -1 at [0][2]
        - Third 2 bits: `10` → 1 at [0][4]
        - Fourth 2 bits: `10` → 1 at [0][6]
        the second value of the same row (0b10010000) will give the values for [0][1], [0][3], [0][5], [0][7]
        We subtract 1 because during the packing process, it's easier to work with values like 0, 1, and 2. To make this possible,
        we add 1 to the original ternary weights (which are typically -1, 0, and 1) when packing them. When unpacking, we reverse
        this by subtracting 1 to restore the original ternary values.
        """
        packed_shape = packed.shape

        if len(packed_shape) == 1:
            original_row_dim = packed_shape[0] * BitLinear.VALUES_PER_ITEM
            unpacked_shape = (original_row_dim,)
        else:
            original_row_dim = packed_shape[0] * BitLinear.VALUES_PER_ITEM
            unpacked_shape = (original_row_dim, *packed_shape[1:])

        unpacked = Tensor.zeros(unpacked_shape, device=packed.device, dtype=dtypes.uint8)

        for i in range(BitLinear.VALUES_PER_ITEM):
            start = i * packed_shape[0]
            end = start + packed_shape[0]
            mask = 3 << (2 * i)
            unpacked[start:end] = (packed & mask) >> (2 * i)

        return unpacked.cast(dtype) - 1

    def _get_unpacked_weights(self) -> Tensor:
        """Get unpacked weights with caching."""
        # Simple hash for cache invalidation
        current_hash = hash(str(self.weight.lazydata))

        if self._unpacked_weights is None or self._weight_hash != current_hash:
            self._unpacked_weights = self.unpack_weights(self.weight, dtypes.float32)
            self._weight_hash = current_hash

        return self._unpacked_weights

    def activation_quant(self, x: Tensor, num_bits: int = 8) -> Tuple[Tensor, Tensor]:
        """
        Activation function : Performs symmetric, per-token quantization on the input activations.
        Parameters:
        -----------
        x : Tensor
            Input activations to be quantized.
        num_bits : int, optional (default=8)
            Number of bits to use for quantization, determining the quantization range.
        Returns:
        --------
        result : Tensor
            Quantized activation tensor, with values mapped to an `int8` range.
        scale : Tensor
            The per-channel scaling factors used to quantize the tensor.
        """
        Qn = -(2 ** (num_bits - 1))  # -128
        Qp = 2 ** (num_bits - 1) - 1  # 127

        # Per-token scaling
        scale = Qp / x.abs().max(axis=-1, keepdim=True).clamp(min_=1e-5)

        # Quantize and clamp
        result = (x * scale).round().clip(Qn, Qp)

        return result.cast(dtypes.int8), scale

    def post_quant_process(self, x: Tensor, input_scale: Tensor, weight_scale: Tensor) -> Tensor:
        """Apply post-quantization scaling."""
        out = x / (input_scale * weight_scale)
        return out

    def __call__(self, x: Tensor) -> Tensor:
        # Get unpacked ternary weights
        w_quant = self._get_unpacked_weights()

        # Apply activation quantization
        input_quant, input_scale = self.activation_quant(x)

        # Cast to computation dtype for matrix multiplication
        input_float = input_quant.cast(dtypes.float32)

        # Use tinygrad's linear implementation pattern: x.linear(weight.transpose(), bias)
        # Note: tinygrad's linear expects weights to be transposed, so we pass w_quant.transpose()
        if len(w_quant.shape) == 1:
            # Handle 1D case by reshaping appropriately
            y = input_float * w_quant
        else:
            # Use tinygrad's linear method instead of manual matrix multiplication
            y = input_float.linear(w_quant.transpose(), None)

        # Apply post-quantization processing
        y = self.post_quant_process(y, input_scale, self.weight_scale)

        # Add bias if present (handled separately since we need special scaling)
        if self.bias is not None:
            y = y + self.bias.reshape(1, -1).expand(*y.shape)

        return y