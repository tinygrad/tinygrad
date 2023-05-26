# from nn.functional import conv1d, conv2d, fold
# import nn as nn
# import torch
# import numpy as np
# from time import time
# from scipy.signal import get_window
from tinygrad.tensor import Tensor, dtypes
from scipy.signal import get_window
import numpy as np


# def overlap_add(X, stride):
#     n_fft = X.shape[1]
#     output_len = n_fft + stride * (X.shape[2] - 1)

#     return X._par((1, output_len), kernel_size=(1, n_fft), stride=stride).flatten(1)


def extend_fbins(X):
    """Extending the number of frequency bins from `n_fft//2+1` back to `n_fft` by
    reversing all bins except DC and Nyquist and append it on top of existing spectrogram"""
    X_upper = X[:, 1:-1].flip(1)
    X_upper[:, :, :, 1] = -X_upper[
        :, :, :, 1
    ]  # For the imaganinry part, it is an odd function
    return X[:, :, :].cat(X_upper, dim=1)


def pad_center(data, size, axis=-1, **kwargs):
    """Wrapper for np.pad to automatically center an array prior to padding.
    This is analogous to `str.center()`

    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])

    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered

    size : int >= len(data) [scalar]
        Length to pad `data`

    axis : int
        Axis along which to pad and center the data

    kwargs : additional keyword arguments
      arguments passed to `np.pad()`

    Returns
    -------
    data_padded : np.ndarray
        `data` centered and padded to length `size` along the
        specified axis

    Raises
    ------
    ParameterError
        If `size < data.shape[axis]`

    See Also
    --------
    numpy.pad
    """

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    assert lpad < 0

    return data.pad(lengths, **kwargs)


def broadcast_dim(x):
    """
    Auto broadcast input so that it can fits into a Conv1d
    """

    if x.dim() == 2:
        x = x[:, None, :]
    elif x.dim() == 1:
        # If nn.DataParallel is used, this broadcast doesn't work
        x = x[None, None, :]
    elif x.dim() == 3:
        pass
    else:
        raise ValueError(
            "Only support input with shape = (batch, len) or shape = (len)"
        )
    return x

## Kernal generation functions ##
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
    """This function creates the Fourier Kernel for STFT, Melspectrogram and CQT.
    Most of the parameters follow librosa conventions. Part of the code comes from
    pytorch_musicnet. https://github.com/jthickstun/pytorch_musicnet

    Parameters
    ----------
    n_fft : int
        The window size

    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins

    fmin : int
        The starting frequency for the lowest frequency bin.
        If freq_scale is ``no``, this argument does nothing.

    fmax : int
        The ending frequency for the highest frequency bin.
        If freq_scale is ``no``, this ~argument does nothing.

    sr : int
        The sampling rate for the input audio. It is used to calculate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    freq_scale: 'linear', 'log', 'log2', or 'no'
        Determine the spacing between each frequency bin.
        When 'linear', 'log' or 'log2' is used, the bin spacing can be controlled by ``fmin`` and ``fmax``.
        If 'no' is used, the bin will start at 0Hz and end at Nyquist frequency with linear spacing.

    Returns
    -------
    wsin : numpy.array
        Imaginary Fourier Kernel with the shape ``(freq_bins, 1, n_fft)``

    wcos : numpy.array
        Real Fourier Kernel with the shape ``(freq_bins, 1, n_fft)``

    bins2freq : list
        Mapping each frequency bin to frequency in Hz.

    binslist : list
        The normalized frequency ``k`` in digital domain.
        This ``k`` is in the Discrete Fourier Transform equation $$

    """

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
        n_fft=2048,
        win_length=None,
        freq_bins=None,
        hop_length=None,
        window="hann",
        freq_scale="no",
        center=True,
        pad_mode="reflect",
        fmin=50,
        fmax=6000,
        sr=22050,
        trainable=False,
        output_format="Complex",
        verbose=True,
    ):

        super().__init__()

        # Trying to make the default setting same as librosa
        if win_length == None:
            win_length = n_fft
        if hop_length == None:
            hop_length = int(win_length // 4)

        self.output_format = output_format
        self.trainable = trainable
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.freq_bins = freq_bins
        self.trainable = trainable
        self.pad_amount = self.n_fft // 2
        self.window = window
        self.win_length = win_length
        self.trainable = trainable

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

        kernel_sin = Tensor(kernel_sin, dtype=float)
        kernel_cos = Tensor(kernel_cos, dtype=float)

        # In this way, the inverse kernel and the forward kernel do not share the same memory...
        self.kernel_sin_inv = Tensor.cat((kernel_sin, [-kernel_sin[1:-1].flip(0)]), 0).unsqueeze(-1)
        self.kernel_cos_inv = Tensor.cat((kernel_cos, [kernel_cos[1:-1].flip(0)]), 0).unsqueeze(-1)

        # Applying window functions to the Fourier kernels
        window_mask = Tensor(window_mask)
        self.wsin = kernel_sin * window_mask
        self.wcos = kernel_cos * window_mask
        self.window_mask = window_mask.unsqueeze(0).unsqueeze(-1)

    
    def __call__(self, x, inverse=False, *args, **kwargs):
        return self.forward(x, *args, **kwargs) if not inverse else self.inverse(x, *args, **kwargs)

    def forward(self, x, output_format=None):
        """
        Convert a batch of waveforms to spectrograms.

        Parameters
        ----------
        x : torch Tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape

        output_format : str
            Control the type of spectrogram to be return. Can be either ``Magnitude`` or ``Complex`` or ``Phase``.
            Default value is ``Complex``.

        """
        output_format = output_format or self.output_format
        self.num_samples = x.shape[-1]

        

        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == "constant":
                padding = nn.ConstantPad1d(self.pad_amount, 0)

            elif self.pad_mode == "reflect":
                if self.num_samples < self.pad_amount:
                    raise AssertionError(
                        "Signal length shorter than reflect padding length (n_fft // 2)."
                    )
                padding = nn.ReflectionPad1d(self.pad_amount)

            x = padding(x)
        spec_imag = x.conv2d(self.wsin, stride=self.stride)
        spec_real = x.conv2d(self.wcos, stride=self.stride)  # Doing STFT by using conv1d

        # remove redundant parts
        spec_real = spec_real[:, : self.freq_bins, :]
        spec_imag = spec_imag[:, : self.freq_bins, :]

        return Tensor.stack(
                (spec_real, -spec_imag), -1
            ) 
    
        if output_format == "Magnitude":
            spec = (spec_real.pow(2) + spec_imag.pow(2)).sqrt()
            spec = spec + 1e-8 if self.trainable else spec
            return spec

        # elif output_format == "Complex":
        #     return Tensor.stack(
        #         (spec_real, -spec_imag), -1
        #     )  # Remember the minus sign for imaginary part

        # elif output_format == "Phase":
        #     return atan2(
        #         -spec_imag + 0.0, spec_real
        #     )  # +0.0 removes -0.0 elements, which leads to error in calculating phase


    def inverse(
        self, X, onesided=True, length=None, refresh_win=True
    ):
        assert X.dim() == 4, (
            "Inverse iSTFT only works for complex number,"
            "make sure our Tensor is in the shape of (batch, freq_bins, timesteps, 2)."
            "\nIf you have a magnitude spectrogram, please consider using Griffin-Lim."
        )
        # return self.inverse_stft(
        #     X, self.kernel_cos_inv, self.kernel_sin_inv, onesided, length, refresh_win
        # )
        # If the input spectrogram contains only half of the n_fft
        # Use extend_fbins function to get back another half
        if onesided:
            X = extend_fbins(X)  # extend freq
        X_real, X_imag = X[:, :, :, 0], X[:, :, :, 1]

        # broadcast dimensions to support 2D convolution
        X_real_bc = X_real.unsqueeze(1)
        X_imag_bc = X_imag.unsqueeze(1)
        a1 = X_real_bc.conv2d(self.kernel_cos_inv, stride=(1, 1))
        b2 = X_imag_bc.conv2d(self.kernel_sin_inv, stride=(1, 1))
        # compute real and imag part. signal lies in the real part
        real = a1 - b2
        real = real.squeeze(-2) * self.window_mask

        # Normalize the amplitude with n_fft
        real /= self.n_fft

        # Overlap and Add algorithm to connect all the frames
        # real = overlap_add(real, self.stride)
        n_fft = real.shape[1]
        output_len = n_fft + self.stride * (X.shape[2] - 1)
        real = real._pair((1, output_len), kernel_size=(1, n_fft), stride=self.stride).flatten(1)


        # Prepare the window sumsqure for division
        # Only need to create this window once to save time
        # Unless the input spectrograms have different time steps
        real[:, self.nonzero_indices] = real[:, self.nonzero_indices].div(
            self.w_sum[self.nonzero_indices]
        )
        # Remove padding
        if length is None:
            if self.center:
                real = real[:, self.pad_amount : -self.pad_amount]

        else:
            if self.center:
                real = real[:, self.pad_amount : self.pad_amount + length]
            else:
                real = real[:, :length]

        return real