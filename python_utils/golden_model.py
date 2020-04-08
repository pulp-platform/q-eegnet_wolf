"""
This file contains the golden model for the quantized EEGNet in integer representation.
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.0.1"
__date__ = "2020/01/20"
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


from operator import mul
from functools import reduce
import json
import numpy as np

import convert_torch_format as convert
import functional as F

class GoldenModel:
    """
    Golden EEGNet Model
    """
    def __init__(self, config_file, net_file, clip_balanced=True, no_scale_between_l1_l2=False, reorder_bn=True):
        """
        Initialize the model based on the config file and the npz file containing all weights

        Parameters:
        - config_file: filename of config.json (from QuantLab)
        - net_file: filename of net.npz (exported from QuantLab)
        """
        # load network parameters
        net = np.load(net_file)

        # load configuration file
        with open(config_file, "r") as _f:
            config = json.load(_f)
        # we only need the network parameters
        net_params = config["indiv"]["net"]["params"]

        # only allow nets with 255 levels
        assert net_params["weightInqNumLevels"] == 255
        assert net_params["actSTENumLevels"] == 255

        # only allow nets which are trained by floorToZero
        assert net_params["floorToZero"]

        # initialize dimensions
        self.num_levels = net_params["weightInqNumLevels"]
        self.F1 = net_params["F1"]
        self.F2 = net_params["F2"]
        self.D = net_params["D"]
        self.C = net_params["C"]
        self.T = net_params["T"]
        self.N = net_params["N"]

        self.reorder_bn = reorder_bn
        net_params["reorder_bn"] = reorder_bn

        if self.F2 is None:
            self.F2 = self.D * self.F1

        # only allow F2 = F1 * D
        assert self.F2 == self.F1 * self.D

        # load individual layers
        if no_scale_between_l1_l2:
            self.layers = [
                FusedLayer12(net, **net_params, clip_balanced=clip_balanced),
                Layer3(net, **net_params, clip_balanced=clip_balanced),
                Layer4(net, **net_params, clip_balanced=clip_balanced),
                Layer5(net, **net_params, clip_balanced=clip_balanced)
            ]
        else:
            self.layers = [
                Layer1(net, **net_params, clip_balanced=clip_balanced),
                Layer2(net, **net_params, clip_balanced=clip_balanced),
                Layer3(net, **net_params, clip_balanced=clip_balanced),
                Layer4(net, **net_params, clip_balanced=clip_balanced),
                Layer5(net, **net_params, clip_balanced=clip_balanced)
            ]

        self.input_scale = self.layers[0].input_scale
        self.output_scale = self.layers[-1].output_scale

        self.input_shape = self.layers[0].input_shape
        self.output_shape = self.layers[-1].output_shape

    def __str__(self):
        ret = "\n\n".join([str(l) for l in self.layers])
        ret += "\n\nTotal Memory: {} B".format(self.mem_size())
        return ret

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def num_params(self):
        return sum([l.num_params() for l in self.layers])

    def mem_size(self):
        return sum([l.mem_size() for l in self.layers])


class Layer:
    """
    Abstract Layer class 

    Important Fields:
    - input_shape, output_shape
    - input_scale, output_scale
    """
    def __init__(self, clip_balanced=False, **params):
        self.input_shape = (0,)
        self.output_shape = (0,)
        self.name = ""
        self.input_scale = 1
        self.output_scale = 1
        self.clip_balanced = clip_balanced

    def __call__(self, x):
        """ Executes the layer """
        return x

    def __str__(self):
        """ returns a formated string with a summary of the layer """
        ret = "{}\n".format(self.name)
        ret += "  Input:    {}\n".format(self.input_shape)
        ret += "  Output:   {}\n".format(self.output_shape)
        ret += "  n params: {}\n".format(self.num_params())
        ret += "  Memory:   {} B".format(self.mem_size())
        return ret

    def num_params(self):
        """ Returns the number of parameters """
        return 0

    def mem_size(self):
        """ Returns the number of bytes in memory """
        return 0


class FusedLayer12(Layer):
    """
    Convolution(time) + BN + Convolution(space) + BN + RELU + POOL, no scale in between
    """
    def __init__(self, net, C, T, F1, F2, clip_balanced=True, **params):
        self.name = "Layer 1: Convolution in Time + Batch Norm"
        self.C = C
        self.T = T
        self.F1 = F1
        self.F2 = F2
        self.input_shape = ((C, T))
        self.output_shape = ((F2, T // 8))
        self.clip_balanced = clip_balanced

        # fetch weights
        self.weights_1, self.weight_scale_1 = convert.inq_conv2d(net, "conv1")
        assert self.weights_1.shape == (self.F1, 1, 1, 64)
        self.weights_1 = np.reshape(self.weights_1, (self.F1, 64))

        self.weights_2, self.weight_scale_2 = convert.inq_conv2d(net, "conv2")
        assert self.weights_2.shape == (self.F2, 1, self.C, 1)
        self.weights_2 = np.reshape(self.weights_2, (self.F2, self.C))

        # fetch batch norm offset and scale
        self.input_scale = convert.ste_quant(net, "quant1")
        self.intermediate_scale = convert.ste_quant(net, "quant2")
        self.output_scale = convert.ste_quant(net, "quant3")
        self.bn_scale_1, self.bn_offset_1 = convert.batch_norm(net, "batch_norm1")
        self.bn_scale_2, self.bn_offset_2 = convert.batch_norm(net, "batch_norm2")
        self.factor_1, self.bias_1 = convert.div_factor_batch_norm(self.input_scale, self.weight_scale_1,
                                                                   self.intermediate_scale, self.bn_scale_1,
                                                                   self.bn_offset_1)
        self.factor_2, self.bias_2 = convert.div_factor_batch_norm(self.intermediate_scale, self.weight_scale_2,
                                                                   self.output_scale, self.bn_scale_2,
                                                                   self.bn_offset_2, pool=8)
        # update the factors and scales
        for k in range(16):
            self.factor_2[k] *= self.factor_1[k // 2]
            self.bias_2[k] *= self.factor_1[k // 2]

    def num_params(self):
        count = reduce(mul, self.weights_1.shape)
        count += reduce(mul, self.factor_1.shape)
        count += reduce(mul, self.bias_1.shape)
        count += reduce(mul, self.weights_2.shape)
        count += reduce(mul, self.factor_2.shape)
        count += reduce(mul, self.bias_2.shape)
        return count

    def mem_size(self):
        count = reduce(mul, self.weights_1.shape)
        count += 4 * reduce(mul, self.factor_1.shape)
        count += 4 * reduce(mul, self.bias_1.shape)
        count += 4 * reduce(mul, self.weights_2.shape)
        count += 4 * reduce(mul, self.factor_2.shape)
        count += 4 * reduce(mul, self.bias_2.shape)
        return count

    def __call__(self, x):
        assert x.shape == self.input_shape, "shape was {}".format(x.shape)
        y = F.conv_time(x, self.weights_1)
        # add the offset
        for k in range(self.F1):
            y[k] += self.bias_1[k]

        # do the second layer
        y = F.depthwise_conv_space(y, self.weights_2)
        y = F.relu(y, -(self.bias_2 // 8))
        y = F.pool(y, (1, 8))
        y = F.apply_factor_offset(y, self.factor_2, self.bias_2, clip_balanced=self.clip_balanced)
        return y


class Layer1(Layer):
    """
    Convolution(time) + BN
    """
    def __init__(self, net, C, T, F1, clip_balanced=True, **params):
        self.name = "Layer 1: Convolution in Time + Batch Norm"
        self.C = C
        self.T = T
        self.F1 = F1
        self.input_shape = ((C, T))
        self.output_shape = ((F1, C, T))
        self.clip_balanced = clip_balanced

        # fetch weights
        self.weights, self.weight_scale = convert.inq_conv2d(net, "conv1")
        assert self.weights.shape == (self.F1, 1, 1, 64)
        self.weights = np.reshape(self.weights, (self.F1, 64))

        # fetch batch norm offset and scale
        self.input_scale = convert.ste_quant(net, "quant1")
        self.output_scale = convert.ste_quant(net, "quant2")
        self.bn_scale, self.bn_offset = convert.batch_norm(net, "batch_norm1")
        self.factor, self.bias = convert.div_factor_batch_norm(self.input_scale, self.weight_scale,
                                                               self.output_scale, self.bn_scale,
                                                               self.bn_offset)

    def num_params(self):
        count = reduce(mul, self.weights.shape)
        count += reduce(mul, self.factor.shape)
        count += reduce(mul, self.bias.shape)
        return count

    def mem_size(self):
        count = reduce(mul, self.weights.shape)
        count += 4 * reduce(mul, self.factor.shape)
        count += 4 * reduce(mul, self.bias.shape)
        return count
        
    def __call__(self, x):
        assert x.shape == self.input_shape, "shape was {}".format(x.shape)
        y = F.conv_time(x, self.weights)
        y = F.apply_factor_offset(y, self.factor, self.bias, clip_balanced=self.clip_balanced)
        return y


class Layer2(Layer):
    """
    Convolution(channels) + BN + ReLU + Pool
    """
    def __init__(self, net, C, T, F1, F2, reorder_bn=True, clip_balanced=True, **params):
        self.name = "Layer 2: Convolution in Space + Batch Norm + ReLU + Pooling"
        self.C = C
        self.T = T
        self.F1 = F1
        self.F2 = F2
        self.input_shape = ((F1, C, T))
        self.output_shape = ((F2, T // 8))
        self.clip_balanced = clip_balanced
        self.reorder_bn = reorder_bn

        # fetch weights
        self.weights, self.weight_scale = convert.inq_conv2d(net, "conv2")
        self.float_weights = np.reshape(net["conv2.weightFrozen"], (self.F2, self.C))
        self.float_weights = np.flip(self.float_weights, (-1))
        assert self.weights.shape == (self.F2, 1, self.C, 1)
        self.weights = np.reshape(self.weights, (self.F2, self.C))

        # fetch batch norm offset and scale
        self.input_scale = convert.ste_quant(net, "quant2")
        self.output_scale = convert.ste_quant(net, "quant3")
        self.bn_scale, self.bn_offset = convert.batch_norm(net, "batch_norm2")
        self.factor, self.bias = convert.div_factor_batch_norm(self.input_scale, self.weight_scale,
                                                               self.output_scale, self.bn_scale,
                                                               self.bn_offset, pool=8)

    def num_params(self):
        count = reduce(mul, self.weights.shape)
        count += reduce(mul, self.factor.shape)
        count += reduce(mul, self.bias.shape)
        return count

    def mem_size(self):
        count = reduce(mul, self.weights.shape)
        count += 4 * reduce(mul, self.factor.shape)
        count += 4 * reduce(mul, self.bias.shape)
        return count

    def __call__(self, x):
        assert x.shape == self.input_shape
        y = F.depthwise_conv_space(x, self.weights)
        if self.reorder_bn:
            y = F.relu(y, -(self.bias // 8))
            y = F.pool(y, (1, 8))
            y = F.apply_factor_offset(y, self.factor, self.bias, clip_balanced=self.clip_balanced)
        else:
            y = F.apply_factor_offset(y, self.factor // 8, self.bias // 8,
                                      clip_balanced=self.clip_balanced)
            y = F.relu(y, (self.bias) * 0)
            y = F.pool(y, (1, 8)) // 8

        return y


class Layer3(Layer):
    """
    Convolution(T)
    """
    def __init__(self, net, T, F2, clip_balanced=True, **params):
        self.name = "Layer 3: Convolution in Time"
        self.T = T
        self.F2 = F2
        self.input_shape = ((F2, T // 8))
        self.output_shape = ((F2, T // 8))
        self.clip_balanced = clip_balanced

        # fetch weights
        self.weights, self.weight_scale = convert.inq_conv2d(net, "sep_conv1")
        assert self.weights.shape == (self.F2, 1, 1, 16)
        self.weights = np.reshape(self.weights, (self.F2, 16))

        # fetch batch norm offset and scale
        self.input_scale = convert.ste_quant(net, "quant3")
        self.output_scale = convert.ste_quant(net, "quant4")
        self.factor = convert.div_factor(self.input_scale, self.weight_scale, self.output_scale)

    def num_params(self):
        return reduce(mul, self.weights.shape) + 1

    def mem_size(self):
        return reduce(mul, self.weights.shape) + 4
        
    def __call__(self, x):
        assert x.shape == self.input_shape, "shape was {}".format(x.shape)
        y = F.depthwise_conv_time(x, self.weights)
        y = F.apply_factor_offset(y, self.factor, clip_balanced=self.clip_balanced)
        return y


class Layer4(Layer):
    """
    Convolution(1x1) + BN + ReLU + Pool
    """
    def __init__(self, net, T, F2, reorder_bn=True, clip_balanced=True, **params):
        self.name = "Layer 4: Point Convolution + Batch Norm + ReLU + Pooling"
        self.T = T
        self.F2 = F2
        self.input_shape = ((F2, T // 8))
        self.output_shape = ((F2, T // 64))
        self.clip_balanced = clip_balanced
        self.reorder_bn = reorder_bn

        # fetch weights
        self.weights, self.weight_scale = convert.inq_conv2d(net, "sep_conv2")
        assert self.weights.shape == (self.F2, self.F2, 1, 1)
        self.weights = np.reshape(self.weights, (self.F2, self.F2))

        # fetch batch norm offset and scale
        self.input_scale = convert.ste_quant(net, "quant4")
        self.output_scale = convert.ste_quant(net, "quant5")
        self.bn_scale, self.bn_offset = convert.batch_norm(net, "batch_norm3")
        self.factor, self.bias = convert.div_factor_batch_norm(self.input_scale, self.weight_scale,
                                                               self.output_scale, self.bn_scale,
                                                               self.bn_offset, pool=8)

    def num_params(self):
        count = reduce(mul, self.weights.shape)
        count += reduce(mul, self.factor.shape)
        count += reduce(mul, self.bias.shape)
        return count

    def mem_size(self):
        count = reduce(mul, self.weights.shape)
        count += 4 * reduce(mul, self.factor.shape)
        count += 4 * reduce(mul, self.bias.shape)
        return count
        
    def __call__(self, x):
        assert x.shape == self.input_shape, "shape was {}".format(x.shape)
        y = F.pointwise_conv(x, self.weights)
        if self.reorder_bn:
            y = F.relu(y, -(self.bias // 8))
            y = F.pool(y, (1, 8))
            y = F.apply_factor_offset(y, self.factor, self.bias, clip_balanced=self.clip_balanced)
        else:
            y = F.apply_factor_offset(y, self.factor // 8, self.bias // 8,
                                      clip_balanced=self.clip_balanced)
            y = F.relu(y, (self.bias) * 0)
            y = F.pool(y, (1, 8)) // 8
        return y


class Layer5(Layer):
    """
    Linear Layer
    """
    def __init__(self, net, T, F2, N, clip_balanced=True, **params):
        self.name = "Layer 5: Linear Layer"
        self.T = T
        self.F2 = F2
        self.N = N
        self.flatten_dim = self.F2 * (self.T // 64)
        self.input_shape = ((F2, T // 64))
        self.output_shape = ((N, ))
        self.clip_balanced = clip_balanced

        # fetch weights
        self.weights, self.bias, self.weight_scale = convert.inq_linear(net, "fc")
        assert self.weights.shape == (self.N, self.flatten_dim)
        self.input_scale = convert.ste_quant(net, "quant5")
        self.output_scale = convert.ste_quant(net, "quant6")
        self.factor = convert.div_factor(self.input_scale, self.weight_scale, self.output_scale)

    def num_params(self):
        count = reduce(mul, self.weights.shape)
        count += reduce(mul, self.bias.shape)
        return count

    def mem_size(self):
        return self.num_params()
        
    def __call__(self, x):
        assert x.shape == self.input_shape, "shape was {}".format(x.shape)
        x = x.ravel()
        y = F.linear(x, self.weights, self.bias)
        y = F.apply_factor_offset(y, self.factor, clip_balanced=self.clip_balanced)
        return y
