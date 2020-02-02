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

TESTNAME = "cl::net::layer3_flip_inplace"
RESULT_FILE = "result.out"

INPUT_FILENAME = "../../../../data/input.npz"
NET_FILENAME = "../../../../data/net.npz"
CONFIG_FILENAME = "../../../../data/config.json"


def gen_stimuli():
    """
    This function generates the stimuli (input and output) for the test
    """
    model = GoldenModel(CONFIG_FILENAME, NET_FILENAME, clip_balanced=False)
    x = np.random.randint(-60, 60, (model.F2, model.T // 8)).astype(int)
    x_align = align_array(x)
    y_exp = np.transpose(x)
    y_exp_align = np.transpose(x_align)
    y_exp_align = align_array(y_exp_align)
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
    mkf.add_cl_prog_source("net/layer3.c")
    mkf.add_cl_prog_source("net/net.c")
    mkf.add_cl_prog_source("func/flip.c")
    mkf.write()

    # generate the stimuli
    _, x_align, _, y_exp_align = gen_stimuli()

    # prepare header file
    header = HeaderFile("test_stimuli.h")
    header.add(HeaderArray("x_vec", "int8_t", x_align.ravel(), const=False))
    header.add(HeaderArray("y_exp", "int8_t", y_exp_align.ravel(), const=False))
    header.write()

    # compile and run
    os.system("make clean all run > {}".format(RESULT_FILE))

    # parse output
    result = parse_output(RESULT_FILE)

    # log the result
    subcase_name = "Layer 3 flip"
    logger.show_subcase_result(subcase_name, result)

    # return summary
    return logger.summary()