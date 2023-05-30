import math
from tinygrad.tensor import Tensor, dtypes
from tinygrad.helpers import make_pair
from scipy.signal import get_window
import numpy as np
import math

def pad_center(data, size, axis=-1, **kwargs):
    kwargs.setdefault("mode", "constant")
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))
    assert lpad <= 0
    return np.pad(data,lengths, **kwargs)

def create_fourier_kernels(
    n_fft,
    win_length=None,
    freq_bins=None,
    fmin=50,
    fmax=6000,
    sr=44100,
    freq_scale="linear",
    window="hann",
    verbose=True,
):
    if freq_bins == None:
        freq_bins = n_fft // 2 + 1
    if win_length == None:
        win_length = n_fft

    s = np.arange(0, n_fft, 1.0)
    wsin = np.empty((freq_bins, 1, n_fft))
    wcos = np.empty((freq_bins, 1, n_fft))
    start_freq = fmin
    end_freq = fmax
    bins2freq = []
    binslist = []

    # num_cycles = start_freq*d/44000.
    # scaling_ind = np.log(end_freq/start_freq)/k

    # Choosing window shape
    window_mask = get_window(window, int(win_length), fftbins=True)
    window_mask = pad_center(window_mask, n_fft)

    if freq_scale == "linear":
        if verbose == True:
            print(
                f"sampling rate = {sr}. Please make sure the sampling rate is correct in order to"
                f"get a valid freq range"
            )
        start_bin = start_freq * n_fft / sr
        scaling_ind = (end_freq - start_freq) * (n_fft / sr) / freq_bins

        for k in range(freq_bins):  # Only half of the bins contain useful info
            # print("linear freq = {}".format((k*scaling_ind+start_bin)*sr/n_fft))
            bins2freq.append((k * scaling_ind + start_bin) * sr / n_fft)
            binslist.append((k * scaling_ind + start_bin))
            wsin[k, 0, :] = np.sin(
                2 * np.pi * (k * scaling_ind + start_bin) * s / n_fft
            )
            wcos[k, 0, :] = np.cos(
                2 * np.pi * (k * scaling_ind + start_bin) * s / n_fft
            )

    elif freq_scale == "log":
        if verbose == True:
            print(
                f"sampling rate = {sr}. Please make sure the sampling rate is correct in order to"
                f"get a valid freq range"
            )
        start_bin = start_freq * n_fft / sr
        scaling_ind = np.log(end_freq / start_freq) / freq_bins

        for k in range(freq_bins):  # Only half of the bins contain useful info
            # print("log freq = {}".format(np.exp(k*scaling_ind)*start_bin*sr/n_fft))
            bins2freq.append(np.exp(k * scaling_ind) * start_bin * sr / n_fft)
            binslist.append((np.exp(k * scaling_ind) * start_bin))
            wsin[k, 0, :] = np.sin(
                2 * np.pi * (np.exp(k * scaling_ind) * start_bin) * s / n_fft
            )
            wcos[k, 0, :] = np.cos(
                2 * np.pi * (np.exp(k * scaling_ind) * start_bin) * s / n_fft
            )

    elif freq_scale == "log2":
        if verbose == True:
            print(
                f"sampling rate = {sr}. Please make sure the sampling rate is correct in order to"
                f"get a valid freq range"
            )
        start_bin = start_freq * n_fft / sr
        scaling_ind = np.log2(end_freq / start_freq) / freq_bins

        for k in range(freq_bins):  # Only half of the bins contain useful info
            # print("log freq = {}".format(np.exp(k*scaling_ind)*start_bin*sr/n_fft))
            bins2freq.append(2**(k * scaling_ind) * start_bin * sr / n_fft)
            binslist.append((2**(k * scaling_ind) * start_bin))
            wsin[k, 0, :] = np.sin(
                2 * np.pi * (2**(k * scaling_ind) * start_bin) * s / n_fft
            )
            wcos[k, 0, :] = np.cos(
                2 * np.pi * (2**(k * scaling_ind) * start_bin) * s / n_fft
            )

    elif freq_scale == "no":
        for k in range(freq_bins):  # Only half of the bins contain useful info
            bins2freq.append(k * sr / n_fft)
            binslist.append(k)
            wsin[k, 0, :] = np.sin(2 * np.pi * k * s / n_fft)
            wcos[k, 0, :] = np.cos(2 * np.pi * k * s / n_fft)
    else:
        print("Please select the correct frequency scale, 'linear' or 'log'")
    return (
        wsin.astype(np.float32),
        wcos.astype(np.float32),
        bins2freq,
        binslist,
        window_mask.astype(np.float32),
    )

class STFT:
    def __init__(
        self,
        n_fft=128,
        win_length=128,
        freq_bins=None,
        hop_length=64,
        window="hann",
        freq_scale="no",
        center=True,
        fmin=50,
        fmax=6000,
        sr=22050,
        trainable=False,
        verbose=True,
        eps=1e-10
    ):

        super().__init__()

        # Trying to make the default setting same as librosa
        if win_length == None:
            win_length = n_fft
        if hop_length == None:
            hop_length = int(win_length // 4)

        self.trainable = trainable
        self.stride = hop_length
        self.center = center
        self.n_fft = n_fft
        self.freq_bins = freq_bins
        self.trainable = trainable
        self.pad_amount = self.n_fft // 2
        self.window = window
        self.win_length = win_length
        self.eps = eps

        # Create filter windows for stft
        (
            kernel_sin,
            kernel_cos,
            self.bins2freq,
            self.bin_list,
            window_mask,
        ) = create_fourier_kernels(
            n_fft,
            win_length=win_length,
            freq_bins=freq_bins,
            window=window,
            freq_scale=freq_scale,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            verbose=verbose,
        )

        kernel_sin = Tensor(kernel_sin, dtype=dtypes.float32)
        kernel_cos = Tensor(kernel_cos, dtype=dtypes.float32)

        self.kernel_sin_inv = kernel_sin.cat(-kernel_sin[1:-1].flip(0), dim=0).unsqueeze(-1)
        self.kernel_cos_inv = kernel_cos.cat(kernel_cos[1:-1].flip(0), dim=0).unsqueeze(-1)

        # Applying window functions to the Fourier kernels
        window_mask = Tensor(window_mask)
        self.wsin = kernel_sin * window_mask
        self.wcos = kernel_cos * window_mask
        self.window_mask = window_mask.unsqueeze(0).unsqueeze(-1)
    
    def __call__(self, x, inverse=False, *args, **kwargs):
        return self.forward(x, *args, **kwargs) if not inverse else self.inverse(x, *args, **kwargs)

    def forward(self, x, return_spec=False):
        self.num_samples = x.shape[-1]
        
        assert len(x.shape) == 2, "Input shape must be (batch, len) "
        if self.center:
            x = x.pad(((0, 0),(self.pad_amount, self.pad_amount)),)
        x = x[:,None,:]

        # spec_imag = x.conv2d(self.wsin[:,:,:,None], stride=self.stride)[:,:,:,0] #(batch, freq_bins, time)
        # spec_real = x.conv2d(self.wcos[:,:,:,None], stride=self.stride)[:,:,:,0]
        spec_imag = x.conv2d(self.wsin, stride=self.stride)
        spec_real = x.conv2d(self.wcos, stride=self.stride)
        # remove redundant parts
        spec_real = spec_real[:, : self.freq_bins, :]
        spec_imag = spec_imag[:, : self.freq_bins, :]

        if return_spec:
            spec = (spec_real.pow(2) + spec_imag.pow(2)).sqrt()
            spec = spec + self.eps if self.trainable else spec
            return spec
        else:
            return Tensor.stack((spec_real, -spec_imag), -1) 
    
        

    def inverse(
        self, X, onesided=True, length=None
    ):
        old = X
        assert len(X.shape)== 4, (
            "Inverse iSTFT only works for complex number,"
            "make sure our Tensor is in the shape of (batch, freq_bins, timesteps, 2)."
            "\nIf you have a magnitude spectrogram, please consider using Griffin-Lim."
        )
        # If the input spectrogram contains only half of the n_fft
        # Use extend_fbins function to get back another half
        if onesided:
            # Extending the number of frequency bins from `n_fft//2+1` back to `n_fft` by
            # reversing all bins except DC and Nyquist and append it on top of existing spectrogram"""
            X_ = X[:, 1:-1].flip(1)
            X_upper1 = X_[:, :, :, 0]
            X_upper2 = -X_[:, :, :, 1]
            X_upper = Tensor.stack([X_upper1,X_upper2], dim=3)
            X= X.cat(X_upper, dim=1)
        X_real, X_imag = X[:, :, :, 0][:,None], X[:, :, :, 1][:,None]
        a1 = X_real.conv2d(self.kernel_cos_inv, stride=(1,1))
        b2 = X_imag.conv2d(self.kernel_sin_inv, stride=(1,1))
        real = a1 - b2
        real = real[:,:,0,:] * self.window_mask

        real = real / self.n_fft
        import torch
        

    
        # Overlap and Add algorithm to connect all the frames
        n_fft = real.shape[1]
        output_len = n_fft + self.stride * (real.shape[2] - 1)
        real = torch.nn.functional.fold(torch.from_numpy(real.numpy()), (1, output_len), kernel_size=(1, n_fft), stride=self.stride).flatten(1)
        real = Tensor(real.numpy())
        def torch_window_sumsquare(win, n_frames, stride, n_fft, power=2):
            win_stacks = win.unsqueeze(-1).repeat((1, n_frames)).unsqueeze(0)
            output_len = win_stacks.shape[1] + stride * (win_stacks.shape[2] - 1)
            out = torch.nn.functional.fold(torch.from_numpy(win_stacks.numpy()) ** power,(1, output_len), (1, n_fft), stride=self.stride)
            return out
        w_sum = torch_window_sumsquare(self.window_mask.flatten(), X.shape[2], self.stride, self.n_fft).flatten()
        w_sum = Tensor(w_sum.numpy())
        real = real / w_sum
        real = real[:, self.pad_amount : -self.pad_amount]
        breakpoint()
        return real


def _im2col_col2im_indices_along_dim(
    input_d, kernel_d, dilation_d, padding_d, stride_d, device
):
    """Utility function to implement im2col and col2im"""
    blocks_d = input_d + padding_d * 2 - dilation_d * (kernel_d - 1)

    arange_kw = partial(torch.arange, dtype=torch.int64, device=device)

    # Stride kernel over input and find starting indices along dim d
    blocks_d_indices = arange_kw(0, blocks_d, stride_d).unsqueeze(0)

    # Apply dilation on kernel and find its indices along dim d
    kernel_grid = arange_kw(0, kernel_d * dilation_d, dilation_d).unsqueeze(-1)

    # Broadcast and add kernel staring positions (indices) with
    # kernel_grid along dim d, to get block indices along dim d
    return blocks_d_indices + kernel_grid

def col2im(
    input: Tensor,
    output_size: List[int],
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
) -> Tensor:
    utils.check(len(output_size) == 2, lambda: "only 2D output_size supported")
    utils.check(len(kernel_size) == 2, lambda: "only 2D kernel supported")
    utils.check(len(dilation) == 2, lambda: "only 2D dilation supported")
    utils.check(len(padding) == 2, lambda: "only 2D padding supported")
    utils.check(len(stride) == 2, lambda: "only 2D stride supported")

    def check_positive(param, param_name, strict=True):
        cond = all(p > 0 for p in param) if strict else all(p >= 0 for p in param)
        utils.check(
            cond, lambda: "{param_name} should be greater than zero, but got {param}"
        )

    check_positive(kernel_size, "kernel_size")
    check_positive(dilation, "dilation")
    check_positive(padding, "padding", strict=False)
    check_positive(stride, "stride")
    check_positive(output_size, "output_size")

    shape = input.shape
    ndim = len(shape)
    utils.check(
        ndim in (2, 3) and all(d != 0 for d in shape[-2:]),
        lambda: "Expected 2D or 3D (batch mode) tensor for input with possible 0 batch size "
        f"and non-zero dimensions, but got: {tuple(shape)}",
    )
    prod_kernel_size = kernel_size[0] * kernel_size[1]
    utils.check(
        shape[-2] % prod_kernel_size == 0,
        lambda: "Expected size of input's first non-batch dimension to be divisible by the "
        f"product of kernel_size, but got input.shape[-2] = {shape[-2]} and "
        f"kernel_size={kernel_size}",
    )
    col = [
        1 + (out + 2 * pad - dil * (ker - 1) - 1) // st
        for out, pad, dil, ker, st in zip(
            output_size, padding, dilation, kernel_size, stride
        )
    ]
    L = col[0] * col[1]
    utils.check(
        shape[-1] == L,
        lambda: f"Given output_size={output_size}, kernel_size={kernel_size}, "
        f"dilation={dilation}, padding={padding}, stride={stride}, "
        f"expected input.size(-1) to be {L} but got {shape[-1]}.",
    )
    utils.check(
        L > 0,
        lambda: f"Given output_size={output_size}, kernel_size={kernel_size}, "
        f"dilation={dilation}, padding={padding}, stride={stride}, "
        f"expected input.size(-1) to be {L} but got {shape[-1]}.",
    )
    batched_input = ndim == 3
    if not batched_input:
        input = input.unsqueeze(0)

    shape = input.shape

    out_h, out_w = output_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    kernel_h, kernel_w = kernel_size

    # col2im is defined as the backwards of im2col, so we differentiate its decomposition by hand
    input = input.reshape([shape[0], shape[1] // prod_kernel_size] + kernel_size + col)
    input = input.permute(0, 1, 2, 4, 3, 5)

    indices_row = _im2col_col2im_indices_along_dim(
        out_h, kernel_h, dilation_h, padding_h, stride_h, input.device
    )
    indices_row = _unsqueeze_to_dim(indices_row, 4)
    indices_col = _im2col_col2im_indices_along_dim(
        out_w, kernel_w, dilation_w, padding_w, stride_w, input.device
    )

    output_padded_size = [o + 2 * p for o, p in zip(output_size, padding)]
    output = input.new_zeros(
        [shape[0], shape[1] // prod(kernel_size)] + output_padded_size
    )
    idx = (None, None, indices_row, indices_col)
    output = aten._unsafe_index_put(output, idx, input, accumulate=True)
    output = F.pad(output, (-padding_w, -padding_w, -padding_h, -padding_h))

    if not batched_input:
        output = output.squeeze(0)
    return output


def _im2col_col2im_indices_along_dim(
    input_d, kernel_d, dilation_d, padding_d, stride_d, device
):
    """Utility function to implement im2col and col2im"""
    blocks_d = input_d + padding_d * 2 - dilation_d * (kernel_d - 1)

    arange_kw = partial(torch.arange, dtype=torch.int64, device=device)

    # Stride kernel over input and find starting indices along dim d
    blocks_d_indices = arange_kw(0, blocks_d, stride_d).unsqueeze(0)

    # Apply dilation on kernel and find its indices along dim d
    kernel_grid = arange_kw(0, kernel_d * dilation_d, dilation_d).unsqueeze(-1)

    # Broadcast and add kernel staring positions (indices) with
    # kernel_grid along dim d, to get block indices along dim d
    return blocks_d_indices + kernel_grid

# This expands x until x.dim() == dim. Might be useful as an operator
def _unsqueeze_to_dim(x: Tensor, dim: int) -> Tensor:
    for _ in range(dim - x.dim()):
        x = x.unsqueeze(-1)
    return x



def _unsafe_index_put(x, indices, values, accumulate=False):
    return index_put_impl_(clone(x), indices, values, accumulate, check=False)



def index_put_impl_(self, indices, values, accumulate, check):
    # Dispatch to masked fill for single boolean index with single value
    if (
        values.get_numel() == 1
        and len(indices) == 1
        and indices[0].get_dtype() in {torch.bool, torch.uint8}
    ):
        mask = indices[0]
        for _ in range(len(mask.get_size()), len(self.get_size())):
            mask = unsqueeze(mask, -1)
        return index_put_as_masked_fill(self, [mask], values, accumulate)

    # Fallback in torch deterministic mode
    if torch.are_deterministic_algorithms_enabled():
        return index_put_fallback(self, indices, values, accumulate)

    # Fallback if there is a boolean index
    for index in indices:
        if index is not None and index.get_dtype() in {torch.bool, torch.uint8}:
            return index_put_fallback(self, indices, values, accumulate)

    x_size = self.get_size()
    x_ndim = len(x_size)

    # fallback to aten.index_put_, as tl.atomic_add does NOT support int64 or bool
    if self.get_dtype() in {torch.int64, torch.bool}:
        # self is an scalar Tensor
        if x_ndim == 0:
            self = view(self, [1])
        self = index_put_fallback(self, indices, values, accumulate)
        if x_ndim == 0:
            self = view(self, [])
        return self

    values = to_dtype(values, self.get_dtype())
    try:
        indices, start_offset, end_offset = check_and_broadcast_indices(
            indices, self.get_device()
        )
    except NotImplementedError:
        return index_put_fallback(self, indices, values, accumulate)
    indices_sizes = [i.get_size() for i in indices if i is not None]
    indices_loaders = [i.make_loader() for i in indices if i is not None]

    assert isinstance(self, TensorBox)
    self.realize()

    # self is an scalar Tensor
    if x_ndim == 0:
        self = view(self, [1])

    output_size = list(indices_sizes[0])
    expected_vals_size = [
        *x_size[:start_offset],
        *output_size,
        *x_size[start_offset + len(indices_sizes) :],
    ]
    indexed_size = [x_size[i] for i in range(len(indices)) if indices[i] is not None]

    values = expand(values, expected_vals_size)
    # all guards are set above during broadcast_tensors and expand

    def output_indexer(index):
        assert len(index) == len(expected_vals_size)
        new_index = [
            ops.indirect_indexing(
                loader(index[start_offset:end_offset]), size, check=check
            )
            for loader, size in zip(indices_loaders, indexed_size)
        ]
        new_index = [*index[:start_offset], *new_index, *index[end_offset:]]
        return new_index

    scatter = ir.Scatter(
        device=self.get_device(),
        dtype=self.get_dtype(),
        inner_fn=values.make_loader(),
        ranges=expected_vals_size,  # iter_ranges,
        output_indexer=output_indexer,
        scatter_mode="atomic_add" if accumulate else None,
    )
    buffer = ir.ComputedBuffer(
        None,
        ir.MutationLayout(self),
        scatter,
    )
    buffer.name = V.graph.register_buffer(buffer)

    if x_ndim == 0:
        self = view(self, [])
    return self