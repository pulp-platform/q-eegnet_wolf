"""
This file contains functions to convert layer parameters from torch (or QuantLab) into standard 
format
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.1.0"
__date__ = "2020/01/23"
__license__ = "Apache 2.0"
__copyright__ = """
    Copyright (C) 2020 ETH Zurich. All rights reserved.

    Author: Tibor Schneider, ETH Zurich

    SPDX-License-Identifier: Apache-2.0

    Licensed under the Apache License, Version 2.0 (the License); you may
    not use this file except in compliance with the License.
    You may obtain a copy of the License at

    www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an AS IS BASIS, WITHOUT
    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


import numpy as np

import functional as F

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
    weights = F.quantize_to_int(weights, scale_factor, num_levels)
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

    weights = F.quantize_to_int(weights, scale_factor, num_levels)
    bias = F.quantize_to_int(bias, scale_factor, num_levels)

    return weights, bias, scale_factor


def batch_norm(net, layer_name, epsilon=0.001):
    """
    Converts a batch_norm layer into a scale factor and an offset
    https://pytorch.org/docs/stable/nn.html#batchnorm2d
    
    y = x * scale + offset

    Parameters
    - net: dict, contains all network parameters
    - layer_name: str, name of the layer
    - epsilon: numerical stability (from torch)

    Returns: scale, offset
     - scale: np.array of size [num_channels]
     - offset: np.array of size [num_channels]
    """

    mean = net["{}.running_mean".format(layer_name)]
    var = net["{}.running_var".format(layer_name)]
    gamma = net["{}.weight".format(layer_name)]
    beta = net["{}.bias".format(layer_name)]

    scale = gamma / np.sqrt(var + epsilon)
    offset = beta - mean * gamma / np.sqrt(var + epsilon)

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


def div_factor(input_scale, weight_scale, output_scale, num_levels=255, pool=1):
    """
    Returns the division factor to rescale the output of a layer.
    If pooling is used (pool > 1), then the factor must be applied after summing up the values
    Notation: x: real value, x': quantized integer value, s_x, scale, R_x: range

        x' = x/s_x * R_x; w' = w/s_w * R_w; y' = y/s_y * R_y
        y = x * w

              x' * w' + bias
        y' = ----------------
                  factor

                  s_y * R_x * R_w
        factor = -----------------
                  R_y * s_x * s_w

    Parameters:
    - input_scale: float
    - weight_scale: float
    - output_scale: float
    - num_levels: int, number of levels for all input, weight and scale
    - pool: int, number of samples avgPool'ed together. Multiplies factor by pool, such that factor
            can be applied after doing sumPool.

    Returns: scale factor: int
    """
    val_range = (num_levels-1)/2
    factor = output_scale * val_range / (input_scale * weight_scale)
    factor *= pool
    return int(round(factor))


def div_factor_batch_norm(input_scale, weight_scale, output_scale, bn_scale, bn_offset,
                          num_levels=255, pool=1):
    """
    Returns the division factor to rescale the output of a layer.
    If pooling is used (pool > 1), then the factor and bias must be applied after summing up all values.
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
    - pool: int, number of samples avgPool'ed together. Multiplies factor by pool, such that factor
            can be applied after doing sumPool.

    Returns: 
    - factor: np.array(dtype=int)
    - bias: np.array(dtype=int)
    """
    val_range = (num_levels-1)/2
    factor = output_scale * val_range / (bn_scale * input_scale * weight_scale)
    bias = bn_offset * val_range * val_range / (bn_scale * input_scale * weight_scale)
    factor *= pool
    bias *= pool
    factor = factor.round().astype(np.int)
    bias = bias.round().astype(np.int)
    return factor, bias
