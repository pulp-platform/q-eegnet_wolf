"""
This File generates the header files defining the trained and quantized network.
The following files are required
- [project_root]/data/config.json containing the QuantLab configuration how the network was trained
- [project_root]/data/net.npz, containing the entire network
"""

import argparse
import json
import numpy as np

from header_file import HeaderFile, HeaderConstant, HeaderScalar, HeaderArray, HeaderComment
import convert_torch_format as convert

DEFAULT_HEADER_NAME = "../src/cl/net.h"
DEFAULT_CONFIG_JSON = "config.json"
DEFAULT_NET_NPZ = "net.npz"


def gen_net_header(net_file, config_file, output_file):

    # load network
    net = np.load(net_file)

    # load configuration file
    with open(config_file, "r") as _f:
        config = json.load(_f)
    # we only need the network parameters
    net_params = config["indiv"]["net"]["params"]

    # only allow nets with 255 levels
    assert net_params["weightInqNumLevels"] == 255
    assert net_params["actSTENumLevels"] == 255

    # prepare params
    if net_params["F2"] is None:
        net_params["F2"] = net_params["F1"] * net_params["D"]

    # only allow F2 = F1 * D
    assert net_params["F2"] == net_params["F1"] * net_params["D"]

    # start the header file
    header = HeaderFile(output_file, "__NET_H__")

    # add network dimensions
    header.add(HeaderComment("Network Dimensions", blank_line=False))
    header.add(HeaderConstant("NET_F1", net_params["F1"], blank_line=False))
    header.add(HeaderConstant("NET_F2", net_params["F2"], blank_line=False))
    header.add(HeaderConstant("NET_D", net_params["D"], blank_line=False))
    header.add(HeaderConstant("NET_C", net_params["C"], blank_line=False))
    header.add(HeaderConstant("NET_T", net_params["T"], blank_line=False))
    header.add(HeaderConstant("NET_N", net_params["N"], blank_line=True))

    # Layer 1
    input_scale = convert.ste_quant(net, "quant1")
    weight, weight_scale = convert.inq_conv2d(net, "conv1")
    bn_scale, bn_offset = convert.batch_norm(net, "batch_norm1")
    output_scale = convert.ste_quant(net, "quant2")
    factor, offset = convert.div_factor_batch_norm(input_scale, weight_scale, output_scale, bn_scale, bn_offset)

    header.add(HeaderComment("Layer 1\n"
                             "=======\n"
                             "Convolution + BN\n\n"
                             "Input:  [F1, C, T]\n"
                             "Weight: [F2, C]\n"
                             "Output: [F2, T // 8]",
                             mode="/*"))
    header.add(HeaderArray("net_layer1_factor", "int32_t", factor.ravel()))
    header.add(HeaderArray("net_layer1_offset", "int32_t", offset.ravel()))
    header.add(HeaderArray("net_layer1_weight", "int8_t", weight.ravel()))

    # layer2
    input_scale = convert.ste_quant(net, "quant2")
    weight, weight_scale = convert.inq_conv2d(net, "conv2")
    bn_scale, bn_offset = convert.batch_norm(net, "batch_norm2")
    output_scale = convert.ste_quant(net, "quant3")
    factor, offset = convert.div_factor_batch_norm(input_scale, weight_scale, output_scale, bn_scale, bn_offset)

    header.add(HeaderComment("Layer 2\n"
                             "=======\n"
                             "Convolution + BN + ReLU + Pooling\n\n"
                             "Input:  [F1, C, T]\n"
                             "Weight: [F2, C]\n"
                             "Output: [F2, T // 8]",
                             mode="/*"))
    header.add(HeaderArray("net_layer2_factor", "int32_t", factor.ravel()))
    header.add(HeaderArray("net_layer2_offset", "int32_t", offset.ravel()))
    header.add(HeaderArray("net_layer2_weight", "int8_t", weight.ravel()))

    # layer3
    input_scale = convert.ste_quant(net, "quant3")
    weight, weight_scale = convert.inq_conv2d(net, "sep_conv1")
    output_scale = convert.ste_quant(net, "quant4")
    factor = convert.div_factor(input_scale, weight_scale, output_scale)

    header.add(HeaderComment("Layer 3\n"
                             "=======\n"
                             "Convolution\n\n"
                             "Input:  [F2, T // 8]\n"
                             "Weight: [F2, 16]\n"
                             "Output: [F2, T // 8]",
                             mode="/*", blank_line=False))
    header.add(HeaderScalar("net_layer3_factor", "int32_t", factor))
    header.add(HeaderArray("net_layer3_weight", "int8_t", weight.ravel()))

    # layer4
    input_scale = convert.ste_quant(net, "quant4")
    weight, weight_scale = convert.inq_conv2d(net, "sep_conv2")
    output_scale = convert.ste_quant(net, "quant5")
    bn_scale, bn_offset = convert.batch_norm(net, "batch_norm3")
    factor, offset = convert.div_factor_batch_norm(input_scale, weight_scale, output_scale, bn_scale, bn_offset)

    header.add(HeaderComment("Layer 4\n"
                             "=======\n"
                             "Convolution + BN + ReLU + Pooling\n\n"
                             "Input:  [F2, T // 8]\n"
                             "Weight: [F2, 16]\n"
                             "Output: [F2, T // 64]",
                             mode="/*"))
    header.add(HeaderArray("net_layer4_factor", "int32_t", factor.ravel()))
    header.add(HeaderArray("net_layer4_offset", "int32_t", offset.ravel()))
    header.add(HeaderArray("net_layer4_weight", "int8_t", weight.ravel()))

    # layer5
    input_scale = convert.ste_quant(net, "quant5")
    weight, bias, weight_scale = convert.inq_linear(net, "fc")

    header.add(HeaderComment("Layer 5\n"
                             "=======\n"
                             "Linear Layer (without scaling in the end)\n\n"
                             "Input:  [F2, T // 64]\n"
                             "Weight: [N, T // 64]\n"
                             "Bias:   [N]\n"
                             "Output: [N]",
                             mode="/*"))
    header.add(HeaderArray("net_layer4_bias", "int8_t", bias.ravel()))
    header.add(HeaderArray("net_layer4_weight", "int8_t", weight.ravel()))

    # store the header file
    header.write()

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generates the header file defining the trained EEGNet")
    parser.add_argument("-o", "--output", help="Export header file name", default=DEFAULT_HEADER_NAME)
    parser.add_argument("-n", "--net",    help="numpy file containing the network", default=DEFAULT_NET_NPZ)
    parser.add_argument("-c", "--config", help="configuration file name", default=DEFAULT_CONFIG_JSON)
    args = parser.parse_args()

    gen_net_header(args.net, args.config, args.output)
