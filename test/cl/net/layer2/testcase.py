"""
This file will test the convolution implementation
"""

import random
import os
import numpy as np
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray, align_array, align_array_size
from makefile import Makefile
from golden_model import GoldenModel
import functional as F

TESTNAME = "cl::net::layer2"
RESULT_FILE = "result.out"

INPUT_FILENAME = "../../../../data/verification.npz"
NET_FILENAME = "../../../../data/net.npz"
CONFIG_FILENAME = "../../../../data/config.json"


def gen_stimuli(random_input):
    """
    This function generates the stimuli (input and output) for the test
    """
    model = GoldenModel(CONFIG_FILENAME, NET_FILENAME, clip_balanced=False)
    layer = model.layers[1]
    if random_input:
        x = np.random.randint(-60, 60, (model.F1, model.C, model.T))
    else:
        x = np.load(INPUT_FILENAME)["layer1_activ"][0, :, :, :]
        x = F.quantize_to_int(x, layer.input_scale)
    y_exp = layer(x)
    x_flip = np.transpose(x, (0, 2, 1))
    x_align = np.zeros((model.F1, align_array_size(model.T), align_array_size(model.C)), dtype=int)
    x_align[:, :model.T, :model.C] = x_flip
    y_exp_align = align_array(y_exp)
    return x, x_align, y_exp, y_exp_align


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME, show_title=False)

    # generate makefile
    mkf = Makefile()
    mkf.add_fc_test_source("test.c")
    mkf.add_cl_test_source("cluster.c")
    mkf.add_cl_prog_source("net/layer2.c")
    mkf.add_cl_prog_source("net/net.c")
    mkf.add_cl_prog_source("func/transform.c")
    mkf.add_cl_prog_source("func/dotp.c")
    mkf.write()

    random_input = False

    # generate the stimuli
    _, x_align, _, y_exp_align = gen_stimuli(random_input)

    # prepare header file
    header = HeaderFile("test_stimuli.h")
    header.add(HeaderArray("x_vec", "int8_t", x_align.ravel()))
    header.add(HeaderArray("y_exp_vec", "int8_t", y_exp_align.ravel()))
    header.write()

    # compile and run
    os.system("make clean all run > {}".format(RESULT_FILE))

    # parse output
    result = parse_output(RESULT_FILE)

    # log the result
    subcase_name = "Layer 2 naive"
    logger.show_subcase_result(subcase_name, result)

    # return summary
    return logger.summary()