"""
This file contains functions for (Quantized) Neural Networks
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.1.0"
__date__ = "2020/01/23"

import numpy as np

def batch_norm(x, scale, bias):
    """
    Applies BatchNorm with scale and bias obtained from convert.batch_norm
    
    Parameters:
    - x: np.array(shape: [D, ...])
    - scale: np.array(shape: [D])
    - bias: np.array(shape: [D])

    Returns: np.array, same shape as x, same dtype as x
    """
    assert scale.shape == bias.shape
    assert len(scale.shape) == 1
    assert scale.shape[0] == x.shape[0]

    y = np.zeros(x.shape, dtype=x.dtype)
    for k in range(x.shape[0]):
        y[k, :] = x[k, :] * scale[k] + bias[k]

    return y


def apply_factor_offset(x, factor, offset=None):
    """
    Scales x according to the factor and offset.
    Factor and Offset should be obtained from convert.div_factor or convert.div_factor_batch_norm

    Parameters:
    - x: np.array(dtype=int)
    - factor: int

    - y: np.array(dtype=int)
    """

    if not isinstance(factor, np.ndarray):
        factor = np.ones((1, ), dtype=type(factor)) * factor
    if offset is None:
        offset = np.zeros(factor.shape, dtype=int)

    assert offset.shape == factor.shape
    assert len(factor.shape) == 1

    y = np.zeros(x.shape, dtype=x.dtype)

    if factor.shape[0] == 1:
        y = (x + offset + factor // 2) // factor
    else:
        for k in range(factor.shape[0]):
            y[k] = (x[k] + offset[k] + (factor[k] // 2)) // factor[k]
    return np.clip(y, -127, 127)


def relu(x, threshold=0):
    """
    Applies ReLU operation: max(x, threshold)

    Parameters:
    x: np.array(size=[D, ...])
    threshold: either float or np.array(size=[D])
    """
    # convert threshold to an np.ndarray of shape (1, )
    if isinstance(threshold, float):
        threshold = np.array(threshold)
    assert len(threshold.shape) == 1
    # if the shape of the threshold is (1, ), then convert it to shape(D, )
    if threshold.shape[0] == 1:
        threshold = (np.ones((x.shape[0], )) * threshold).astype(x.dtype)
    assert threshold.shape[0] == x.shape[0]

    y = np.zeros(x.shape, dtype=x.dtype)
    for k in range(x.shape[0]):
        y[k] = np.maximum(x[k], threshold[k])

    return y


def pool(x, shape, reduction="sum"):
    """
    Applies pooling

    Parameters:
    - x: np.array(size=[K, T])
    - shape: tuple of same dimensionality as x
    - reduction: str, either "sum", "mean" or "max"

    Returns: np.array
    """
    assert len(x.shape) == len(shape)
    assert len(x.shape) == 2
    do_round = False
    if reduction == "sum":
        func = np.sum
    elif reduction == "mean":
        func = np.mean
        if x.dtype == int:
            do_round = True
    elif reduction == "max":
        func = np.max
    else:
        raise TypeError("Parameter \"reduction\" must be either \"sum\", \"mean\" or \"max\"!")

    out_shape = tuple(d // s for d, s in zip(x.shape, shape))
    y = np.zeros(out_shape, dtype=x.dtype)

    for k in range(y.shape[0]):
        for t in range(y.shape[1]):
            y[k, t] = func(x[k * shape[0]:(k + 1) * shape[0], t * shape[1]:(t + 1) * shape[1]])

    if do_round:
        y = np.round(y).astype(int)

    return y


def conv_time(x, w):
    """
    Applies a Convolution in Time, where all channels are convolved with the same filter, with mode=same
    Used in Layer 1

    Parameters:
    - x: np.array(shape: [CH, T])
    - w: np.array(shape: [K, T'])

    Returns: np.array(shape: [K, CH, T]), same dtype as x
    """
    assert len(x.shape) == 2
    assert len(w.shape) == 2

    # determine padding
    if w.shape[1] % 2 == 0: # even
        padding = (w.shape[1] // 2 - 1, w.shape[1] // 2)
    else: #odd
        padding = ((w.shape[1] - 1) // 2, (w.shape[1] - 1) // 2)

    y = np.zeros((w.shape[0], x.shape[0], x.shape[1]), dtype=x.dtype)
    x = np.pad(x, ((0, 0), padding))
    for k in range(w.shape[0]):
        for ch in range(x.shape[0]):
            y[k, ch, :] = np.convolve(x[ch, :], w[k, :], mode="valid")

    return y


def depthwise_conv_space(x, w):
    """
    Applies a Depthwise Convolution in Space, where each filter is applied at all time steps., with mode=valid
    Used in Layer 2

    Parameters:
    - x: np.array(shape: [K1, CH, T])
    - w: np.array(shape: [K2, CH])

    Returns: np.array(shape: [K2, T]), same dtype as x
    """
    assert len(w.shape) == 2
    assert len(x.shape) == 3
    assert x.shape[1] == w.shape[1]
    assert w.shape[0] % x.shape[0] == 0 # K2 must be divisible by K1

    D = w.shape[0] // x.shape[0]

    y = np.zeros((w.shape[0], x.shape[2]), dtype=x.dtype)

    for k in range(w.shape[0]):
        for t in range(x.shape[2]):
            y[k, t] = np.convolve(x[k // D, :, t], w[k], mode="valid")

    return y


def depthwise_conv_time(x, w):
    """
    Applies a Depthwise Convolution in Time, where each channel has it's own filter, with mode=same
    Used in Layer 3

    Parameters:
    - x: np.array(shape: [K, T])
    - w: np.array(shape: [K, T'])

    Returns: np.array(shape: [K, T]), same dtype as x
    """
    assert len(x.shape) == 2
    assert len(w.shape) == 2
    assert x.shape[0] == w.shape[0]

    # determine padding
    if w.shape[1] % 2 == 0: # even
        padding = (w.shape[1] // 2 - 1, w.shape[1] // 2)
    else: #odd
        padding = ((w.shape[1] - 1) // 2, (w.shape[1] - 1) // 2)

    y = np.zeros(x.shape, dtype=x.dtype)
    x = np.pad(x, ((0, 0), padding))
    for k in range(x.shape[0]):
            y[k] = np.convolve(x[k], w[k], mode="valid")

    return y


def pointwise_conv(x, w):
    """
    Applies a pointwise convolution.
    Used in Layer4

    Parameters:
    - x: np.array(shape: [K, T])
    - w: np.array(shape: [K, K])

    Returns: np.array(shape: [K, T]), same dtype as x
    """

    assert len(x.shape) == 2
    assert len(w.shape) == 2
    assert x.shape[0] == w.shape[0]
    assert x.shape[0] == w.shape[1]

    y = np.zeros(x.shape, dtype=x.dtype)

    for k_outer in range(w.shape[0]):
        for k_inner in range(w.shape[1]):
            y[k_outer] += x[k_inner] * w[k_outer, k_inner]

    return y

def linear(x, w, b):
    """
    Applies a set of dot products, corresponding to a linear (FC) layer
    Used in layer 5

    Parameters:
    - x: np.array(shape: [K])
    - w: np.array(shape: [N, K])
    - b: np.array(shape: [N])
    
    Returns: np.array(shape: [N]), same dtype as x
    """

    assert len(w.shape) == 2
    assert len(x.shape) == 1
    assert len(b.shape) == 1
    assert w.shape[1] == x.shape[0]
    assert b.shape[0] == w.shape[0]

    y = np.zeros((w.shape[0], ), dtype=x.dtype)

    for n in range(w.shape[0]):
        y[n] = np.dot(x, w[n]) + b[n]

    return y


def quantize(x, scale_factor, num_levels=255):
    """
    Quantizes the input linearly (without offset) with the given number of levels.
    The quantization levels will be: 
        np.linspace(-scale_factor, scale_facotr, num_levels)
    The output will contain only quantized values (not the integer representation)

    Parameters:
    - x: np.array(dtype=float), original vector
    - scale_factor: float, the output will be quantized to range [-s, s]
    - num_levels: int, number of quantization levels

    Returns: np.array(dtype=float), where all values are within the quantized grid
    """
    x_q = quantize_to_int(x, scale_factor, num_levels)
    return dequantize(x_q, scale_factor, num_levels)


def quantize_to_int(x, scale_factor, num_levels=255):
    """
    Quantizes the input linearly (without offset) with the given number of levels.
    The quantization levels will be: 
        np.linspace(-scale_factor, scale_facotr, num_levels)
    The output values will be one of:
        [-(num_levels-1)/2, ..., -1, 0, 1, ..., (num_levels-1)/2]
    As an example, num_levels = 255, the output range will be int8_t without -128
        [-127, -126, ..., -1, 0, 1, ..., 126, 127]

    Parameters:
    - x: np.array(dtype=float), original vector
    - scale_factor: float, the output will be quantized to range [-s, s]
    - num_levels: int, number of quantization levels, must be odd

    Returns: np.array(dtype=int), where all values will be in the integer representation
    """

    # num_levels must be odd!
    assert num_levels % 2
    
    x = x / scale_factor
    x = np.clip(x, -1, 1)
    x = x * (num_levels - 1) / 2
    x = x.round()
    x = x.astype(np.int)
    return x


def dequantize(x, scale_factor, num_levels=255):
    """
    Reverse operation of quantize_to_int

    Parameters:
    - x: np.array(dtype=int), quantized vector in integer representation
    - scale factor: float input will be mapped to this range
    - num_levels: int: number of quantization levels, must be odd
    
    Returns: np.array(dtype=float), in float representation
    """
    assert num_levels % 2

    x = x / ((num_levels - 1) / 2)
    x = x * scale_factor
    return x
