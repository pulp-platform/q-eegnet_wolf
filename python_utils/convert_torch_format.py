"""
This file contains functions to convert layer parameters from torch (or QuantLab) into standard 
format
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.0.1"
__date__ = "2020/01/19"

import numpy as np

def inq_conv2d(net, layer_name, num_levels=255, store_reversed=False):
    """
    Converts a INQConv2d layer into a quantized array
    
    Parameters
    - net: dict, contains all network parameters
    - layer_name: str, name of the layer
    - num_levels: int, Number of quantization layers
    - store_reversed: bool, if True, then return the weights in reversed order (Cross Correlation)

    Returns: weights, scale_factor
     - weights: np.array(dtype=int)
     - scale_factor: float
    """

    weights = net["{}.weightFrozen".format(layer_name)]
    scale_factor = net["{}.sParam".format(layer_name)][0]

    # Torch implements conv2d as a cross-correlation. Thus, we need to flip the dimension
    if not store_reversed:
        weights = np.flip(weights, (-2, -1))

    # quantize the weights
    weights = quantize_to_int(weights, scale_factor, num_levels)
    return weights, scale_factor


def inq_linear(net, layer_name, num_levels=255):
    """
    Converts a INQLinear layer into a quantized array
    
    Parameters
    - net: dict, contains all network parameters
    - layer_name: str, name of the layer
    - num_levels: int, Number of quantization layers

    Returns: weights, scale_factor
     - weights: np.array(dtype=int)
     - bias: np.array(dtype=int)
     - scale_factor: float
    """

    weights = net["{}.weightFrozen".format(layer_name)]
    bias = net["{}.bias".format(layer_name)]
    scale_factor = net["{}.sParam".format(layer_name)][0]

    weights = quantize_to_int(weights, scale_factor, num_levels)
    bias = quantize_to_int(bias, scale_factor, num_levels)

    return weights, bias, scale_factor


def batch_norm(net, layer_name):
    """
    Converts a batch_norm layer into a scale factor and an offset

    Parameters
    - net: dict, contains all network parameters
    - layer_name: str, name of the layer

    Returns: scale, offset
     - scale: np.array of size [num_channels]
     - offset: np.array of size [num_channels]
    """

    mean = net["{}.running_mean".format(layer_name)]
    var = net["{}.running_var".format(layer_name)]
    gamma = net["{}.weight".format(layer_name)]
    beta = net["{}.bias".format(layer_name)]

    scale = gamma / np.sqrt(var)
    offset = beta - mean / np.sqrt(var) * gamma

    return scale, offset


def ste_quant(net, layer_name):
    """
    extracts the scaling factor of STEActivation

    Parameters
    - net: dict, contains all network parameters
    - layer_name: str, name of the layer

    Returns: scale_factor: float
    """

    try:
        return net["{}.absMaxValue".format(layer_name)][0]
    except KeyError:
        return net["{}.quant.absMaxValue".format(layer_name)][0]


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
    x = x / scale_factor    # [-1, 1]
    x = np.clip(x, -1, 1)   # [-1, 1]
    x = (x + 1) / 2         # [0, 1]
    x = x * (num_levels-1)  # [0, n]
    x = x.round()           # [0, n]
    x = x / (nnum_levels-1) # [0, 1]
    x = 2*x - 1             # [-1, 1]
    x = x * scale_factor    # [-s, s]
    return x


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
    
    x = x / scale_factor     # Range: [-1, 1]
    x = np.clip(x, -1, 1)
    x = x * (num_levels-1)/2 # Range: [-(num_levels-1)/2, (num_levels-1)/2]
    x = x.round()
    x = x.astype(np.int)
    return x


def div_factor(input_scale, weight_scale, output_scale, num_levels=255):
    """
    Returns the division factor to rescale the output of a layer.
    Notation: x: real value, x': quantized integer value, s_x, scale, R_x: range

        x' = x/s_x * R_x; w' = w/s_w * R_w; y' = y/s_y * R_y
        y = x * w

              x' * w'
        y' = ---------
              factor

                  s_y * R_x * R_w
        factor = -----------------
                  R_y * s_x * s_w

    Parameters:
    - input_scale: float
    - weight_scale: float
    - output_scale: float
    - num_levels: int, number of levels for all input, weight and scale

    Returns: int, scale factor
    """
    factor = output_scale * num_levels / (input_scale * weight_scale)
    return int(round(factor))


def div_factor_batch_norm(input_scale, weight_scale, output_scale, bn_scale, bn_offset,
                          num_levels=255):
    """
    Returns the division factor to rescale the output of a layer.
    Notation: x: real value, x': quantized integer value, s_x, scale, R_x: range

        x' = x/s_x * R_x; w' = w/s_w * R_w; y' = y/s_y * R_y
        y = (x * w) * s_bn + o_bn

              x' * w' + bias
        y' = ----------------
                 factor

                     s_y * R_x * R_w               o_bn * R_x * R_w
        factor = ------------------------, bias = ------------------
                  s_bn * R_y * s_x * s_w           s_bn * s_x * s_w

    Parameters:
    - input_scale: float
    - weight_scale: float
    - output_scale: float
    - bn_scale: np.array
    - bn_offset: np.array
    - num_levels: int, number of levels for all input, weight and scale

    Returns: 
    - factor: np.array(dtype=int)
    - bias: np.array(dtype=int)
    """
    factor = output_scale * num_levels / (bn_scale * input_scale * weight_scale)
    bias = bn_offset * num_levels * num_levels / (bn_scale * input_scale * weight_scale)
    factor = factor.round().astype(np.int)
    bias = bias.round().astype(np.int)
    return factor, bias
